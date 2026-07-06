"""
Build the training dataset for wildfire prediction.

Creates a grid-cell × day tabular dataset by:
1. Defining a spatial grid over the target region
2. Assigning historical fires (FPA-FOD) to grid cells
3. Joining gridMET weather/fire-danger data to each cell × day
4. Engineering derived features (rolling averages, dry streaks, etc.)
5. Labeling each row as fire (1) or no fire (0)

Output: a single parquet file ready for model training.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from src.data_acquisition.config import RAW_DIR, PROCESSED_DIR, REGIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Spatial grid
# ---------------------------------------------------------------------------

def build_grid(bbox: dict, resolution_deg: float = 0.1) -> pd.DataFrame:
    """Create a regular lat/lon grid over a bounding box.

    Args:
        bbox: Dict with west, south, east, north.
        resolution_deg: Grid spacing in degrees (~0.1° ≈ 10 km).

    Returns:
        DataFrame with columns: grid_id, lat_center, lon_center, lat_min,
        lat_max, lon_min, lon_max.
    """
    lats = np.arange(bbox["south"], bbox["north"], resolution_deg)
    lons = np.arange(bbox["west"], bbox["east"], resolution_deg)

    records = []
    grid_id = 0
    for lat in lats:
        for lon in lons:
            records.append({
                "grid_id": grid_id,
                "lat_center": round(lat + resolution_deg / 2, 4),
                "lon_center": round(lon + resolution_deg / 2, 4),
                "lat_min": round(lat, 4),
                "lat_max": round(lat + resolution_deg, 4),
                "lon_min": round(lon, 4),
                "lon_max": round(lon + resolution_deg, 4),
            })
            grid_id += 1

    grid = pd.DataFrame(records)
    logger.info(f"Built grid: {len(grid)} cells, {len(lats)} lat × {len(lons)} lon, res={resolution_deg}°")
    return grid


def assign_fires_to_grid(
    fires: pd.DataFrame,
    grid: pd.DataFrame,
    resolution_deg: float = 0.1,
    bbox: dict | None = None,
) -> pd.DataFrame:
    """Assign fire records to grid cells using fast binning.

    Args:
        fires: FPA-FOD DataFrame with LATITUDE, LONGITUDE, discovery_date.
        grid: Grid DataFrame from build_grid().
        resolution_deg: Must match the grid resolution.
        bbox: Bounding box to filter fires.

    Returns:
        fires DataFrame with added grid_id column.
    """
    if bbox:
        fires = fires[
            (fires["LATITUDE"] >= bbox["south"]) & (fires["LATITUDE"] < bbox["north"]) &
            (fires["LONGITUDE"] >= bbox["west"]) & (fires["LONGITUDE"] < bbox["east"])
        ].copy()

    # Compute grid cell via floor division
    fires["lat_bin"] = np.floor(fires["LATITUDE"] / resolution_deg) * resolution_deg
    fires["lon_bin"] = np.floor(fires["LONGITUDE"] / resolution_deg) * resolution_deg

    # Merge to get grid_id
    grid_lookup = grid[["grid_id", "lat_min", "lon_min"]].copy()
    fires = fires.merge(
        grid_lookup,
        left_on=["lat_bin", "lon_bin"],
        right_on=["lat_min", "lon_min"],
        how="left",
    )
    fires = fires.drop(columns=["lat_bin", "lon_bin", "lat_min", "lon_min"])

    assigned = fires["grid_id"].notna().sum()
    logger.info(f"Assigned {assigned:,}/{len(fires):,} fires to grid cells")
    return fires


# ---------------------------------------------------------------------------
# 2. Fire labels
# ---------------------------------------------------------------------------

def create_fire_labels(
    fires: pd.DataFrame,
    grid: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Create a grid_id × date DataFrame with binary fire labels.

    Args:
        fires: Fire records with grid_id and discovery_date.
        grid: Grid DataFrame.
        start_date: Start of date range (YYYY-MM-DD).
        end_date: End of date range (YYYY-MM-DD).

    Returns:
        DataFrame with columns: grid_id, date, fire_count, total_acres, has_fire.
    """
    dates = pd.date_range(start_date, end_date, freq="D")

    # Aggregate fires per grid cell per day
    fires_valid = fires.dropna(subset=["grid_id", "discovery_date"]).copy()
    fires_valid["date"] = fires_valid["discovery_date"].dt.normalize()
    fires_valid["grid_id"] = fires_valid["grid_id"].astype(int)

    daily_fires = (
        fires_valid
        .groupby(["grid_id", "date"])
        .agg(fire_count=("FOD_ID", "count"), total_acres=("FIRE_SIZE", "sum"))
        .reset_index()
    )

    # Create the full grid_id × date index (only for cells that ever had fires,
    # plus a sample of non-fire cells to control dataset size)
    fire_cells = daily_fires["grid_id"].unique()
    all_cells = grid["grid_id"].values

    # Include all fire cells + a random sample of non-fire cells
    rng = np.random.default_rng(42)
    non_fire_cells = np.setdiff1d(all_cells, fire_cells)
    # Sample ~3x the number of fire cells from non-fire cells for balance
    n_sample = min(len(non_fire_cells), len(fire_cells) * 3)
    sampled_non_fire = rng.choice(non_fire_cells, size=n_sample, replace=False)
    selected_cells = np.concatenate([fire_cells, sampled_non_fire])

    logger.info(
        f"Selected {len(selected_cells)} cells: "
        f"{len(fire_cells)} fire + {len(sampled_non_fire)} non-fire"
    )

    # Build full index
    full_index = pd.MultiIndex.from_product(
        [selected_cells, dates], names=["grid_id", "date"]
    )
    labels = pd.DataFrame(index=full_index).reset_index()

    # Merge fire data
    labels = labels.merge(daily_fires, on=["grid_id", "date"], how="left")
    labels["fire_count"] = labels["fire_count"].fillna(0).astype(int)
    labels["total_acres"] = labels["total_acres"].fillna(0.0)
    labels["has_fire"] = (labels["fire_count"] > 0).astype(int)

    logger.info(
        f"Labels: {len(labels):,} rows, "
        f"{labels['has_fire'].sum():,} fire-days ({labels['has_fire'].mean()*100:.2f}%)"
    )
    return labels


# ---------------------------------------------------------------------------
# 3. Weather features from gridMET
# ---------------------------------------------------------------------------

def fetch_gridmet_for_grid(
    grid: pd.DataFrame,
    year: int,
    variables: list[str] | None = None,
    bbox: dict | None = None,
    refresh_current: bool = False,
) -> pd.DataFrame:
    """Fetch gridMET data as a spatial block and map to grid cells.

    Downloads the entire spatial block for the bounding box (fast single
    request per variable), then uses nearest-neighbor indexing locally
    to map gridMET pixels to our grid cells.

    Args:
        grid: Grid DataFrame with grid_id, lat_center, lon_center.
        year: Year to fetch.
        variables: gridMET variable names.
        bbox: Bounding box for spatial subsetting.
        refresh_current: Re-download the file when ``year`` is the current
            calendar year. The current-year gridMET file grows daily, so a
            cached copy goes stale — live scoring must re-fetch it or it
            silently keeps re-scoring an old day. Historical years are final,
            so they stay cached.

    Returns:
        DataFrame with columns: grid_id, date, and one column per variable.
    """
    import datetime as _dt
    is_current_year = year == _dt.date.today().year
    variables = variables or ["tmmx", "rmin", "vs", "pr", "erc", "vpd", "fm100", "bi"]

    if not bbox:
        bbox = {"west": -125.0, "south": 24.5, "east": -66.5, "north": 49.5}

    # Download full NetCDF files via HTTP (much faster than OPeNDAP)
    # then subset locally
    import requests as req
    from src.data_acquisition.config import RAW_DIR

    cache_dir = RAW_DIR / "gridmet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result_df = None

    for var in variables:
        local_path = cache_dir / f"{var}_{year}.nc"

        # Download if missing, or if it's the current (still-growing) year and a
        # refresh was requested — otherwise a stale cache freezes the latest day.
        stale = refresh_current and is_current_year and local_path.exists()
        if not local_path.exists() or stale:
            if stale:
                logger.info(f"Refreshing current-year {local_path.name} (was stale)")
            url = f"https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
            logger.info(f"Downloading gridMET {var}_{year}.nc (~25-80MB)...")
            resp = req.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 16):
                    f.write(chunk)
            logger.info(f"  Saved {local_path.stat().st_size / 1e6:.1f} MB")
        else:
            logger.info(f"Using cached {local_path.name}")

        # Open locally and subset
        try:
            ds = xr.open_dataset(local_path, engine="netcdf4")
            data_var = list(ds.data_vars)[0]

            block = ds[data_var].sel(
                lat=slice(bbox["north"], bbox["south"]),
                lon=slice(bbox["west"], bbox["east"]),
            ).load()

            gm_lats = block.lat.values
            gm_lons = block.lon.values
            days = pd.DatetimeIndex(block.day.values)

            # Map grid cells to nearest gridMET pixel
            lat_idx = np.abs(gm_lats[:, None] - grid["lat_center"].values[None, :]).argmin(axis=0)
            lon_idx = np.abs(gm_lons[:, None] - grid["lon_center"].values[None, :]).argmin(axis=0)

            values = block.values[:, lat_idx, lon_idx]

            n_days = len(days)
            n_cells = len(grid)
            var_df = pd.DataFrame({
                "date": np.repeat(days, n_cells),
                "grid_id": np.tile(grid["grid_id"].values, n_days),
                var: values.ravel(),
            })

            if result_df is None:
                result_df = var_df
            else:
                result_df[var] = var_df[var]

            ds.close()
            del block, values
            logger.info(f"  {var} done.")

        except Exception as e:
            logger.error(f"Failed to process {var} for {year}: {e}")

    if result_df is None:
        return pd.DataFrame()

    logger.info(f"Weather data for {year}: {len(result_df):,} rows, {len(variables)} variables")
    return result_df


# ---------------------------------------------------------------------------
# 3b. Air quality features (EPA AQS PM2.5)
# ---------------------------------------------------------------------------

def fetch_air_quality_for_grid(
    grid: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch EPA AQS PM2.5 data for California and map to grid cells.

    Downloads daily PM2.5 measurements, maps monitoring stations to the
    nearest grid cell, and aggregates per grid cell per day.

    Returns:
        DataFrame with columns: grid_id, date, pm25_mean, pm25_max.
    """
    from src.data_acquisition.fetch_air_quality import fetch_epa_daily
    from src.data_acquisition.config import EPA_AQS_EMAIL, EPA_AQS_KEY

    if not EPA_AQS_EMAIL or not EPA_AQS_KEY:
        logger.warning("EPA AQS credentials not set — skipping air quality")
        return pd.DataFrame()

    cache_dir = RAW_DIR / "air_quality" / "epa"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    state_fips = "06"  # California

    # Fetch quarterly to avoid EPA API timeouts on full-year requests
    quarters = [("0101", "0331"), ("0401", "0630"), ("0701", "0930"), ("1001", "1231")]

    for year in range(start_year, end_year + 1):
        cache_path = cache_dir / f"pm25_{state_fips}_{year}.csv"

        if cache_path.exists():
            logger.info(f"Loading cached PM2.5 for CA {year}")
            df = pd.read_csv(cache_path)
        else:
            logger.info(f"Fetching EPA PM2.5 for CA {year} (quarterly)...")
            year_dfs = []
            for q_start, q_end in quarters:
                try:
                    qdf = fetch_epa_daily(state_fips, "88101", f"{year}{q_start}", f"{year}{q_end}")
                    if not qdf.empty:
                        year_dfs.append(qdf)
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"EPA PM2.5 Q{quarters.index((q_start, q_end))+1} {year}: {e}")
            if year_dfs:
                df = pd.concat(year_dfs, ignore_index=True)
                df.to_csv(cache_path, index=False)
            else:
                logger.warning(f"No PM2.5 data for {year}")
                continue

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No air quality data retrieved")
        return pd.DataFrame()

    aq = pd.concat(all_dfs, ignore_index=True)

    # Normalize column names (EPA returns lowercase with spaces)
    col_map = {}
    for c in aq.columns:
        col_map[c] = c.lower().replace(" ", "_")
    aq = aq.rename(columns=col_map)

    aq["date"] = pd.to_datetime(aq["date_local"], errors="coerce")
    aq = aq.dropna(subset=["latitude", "longitude", "date"])

    # Deduplicate stations and map each to nearest grid cell
    stations = aq[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    s_lats = stations["latitude"].values[:, None]
    s_lons = stations["longitude"].values[:, None]
    g_lats = grid["lat_center"].values[None, :]
    g_lons = grid["lon_center"].values[None, :]

    dists = np.sqrt((s_lats - g_lats) ** 2 + (s_lons - g_lons) ** 2)
    stations["grid_id"] = grid["grid_id"].values[dists.argmin(axis=1)]

    aq = aq.merge(stations[["latitude", "longitude", "grid_id"]],
                  on=["latitude", "longitude"], how="left")

    # Aggregate per grid cell per day
    agg_cols = {}
    if "arithmetic_mean" in aq.columns:
        agg_cols["pm25_mean"] = ("arithmetic_mean", "mean")
    if "first_max_value" in aq.columns:
        agg_cols["pm25_max"] = ("first_max_value", "max")
    if "aqi" in aq.columns:
        agg_cols["aqi_mean"] = ("aqi", "mean")

    if not agg_cols:
        logger.warning("No recognized AQ columns found")
        return pd.DataFrame()

    daily_aq = aq.groupby(["grid_id", "date"]).agg(**agg_cols).reset_index()
    logger.info(f"Air quality: {len(daily_aq):,} grid-cell-day records, "
                f"{daily_aq['grid_id'].nunique()} cells with stations")
    return daily_aq


# ---------------------------------------------------------------------------
# 3c. Vegetation features (MODIS NDVI)
# ---------------------------------------------------------------------------

def fetch_vegetation_for_grid(
    grid: pd.DataFrame,
    start_year: int,
    end_year: int,
    bbox: dict,
) -> pd.DataFrame:
    """Fetch TerraClimate vegetation/dryness data and map to grid cells.

    Downloads actual evapotranspiration (aet) and climatic water deficit (def)
    from TerraClimate (monthly, ~4km resolution NetCDF). These are strong
    vegetation dryness indicators for fire prediction.

    Returns:
        DataFrame with columns: grid_id, date, aet, water_deficit.
    """
    cache_dir = RAW_DIR / "vegetation"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tc_vars = ["aet", "def"]  # TerraClimate variable names
    var_dfs = {}  # var_name -> list of yearly DataFrames

    for var_name in tc_vars:
        var_dfs[var_name] = []
        for year in range(start_year, end_year + 1):
            local_path = cache_dir / f"TerraClimate_{var_name}_{year}.nc"

            if not local_path.exists():
                url = f"https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{var_name}_{year}.nc"
                logger.info(f"Downloading TerraClimate {var_name}_{year}.nc...")
                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
                logger.info(f"  Saved {local_path.stat().st_size / 1e6:.1f} MB")
            else:
                logger.info(f"Using cached TerraClimate {var_name}_{year}.nc")

            try:
                ds = xr.open_dataset(local_path, engine="netcdf4")
                actual_var = list(ds.data_vars)[0]

                block = ds[actual_var].sel(
                    lat=slice(bbox["north"], bbox["south"]),
                    lon=slice(bbox["west"], bbox["east"]),
                ).load()

                tc_lats = block.lat.values
                tc_lons = block.lon.values
                times = pd.DatetimeIndex(block.time.values)

                # Map grid cells to nearest TerraClimate pixel
                lat_idx = np.abs(tc_lats[:, None] - grid["lat_center"].values[None, :]).argmin(axis=0)
                lon_idx = np.abs(tc_lons[:, None] - grid["lon_center"].values[None, :]).argmin(axis=0)

                values = block.values[:, lat_idx, lon_idx]

                n_times = len(times)
                n_cells = len(grid)
                var_df = pd.DataFrame({
                    "date": np.repeat(times, n_cells),
                    "grid_id": np.tile(grid["grid_id"].values, n_times),
                    var_name: values.ravel(),
                })
                var_dfs[var_name].append(var_df)

                ds.close()
                del block, values
                logger.info(f"  {var_name} {year} done.")

            except Exception as e:
                logger.error(f"Failed to process TerraClimate {var_name} {year}: {e}")

    # Concat years per variable, then merge variables together
    combined = {}
    for var_name in tc_vars:
        if var_dfs[var_name]:
            combined[var_name] = pd.concat(var_dfs[var_name], ignore_index=True)

    if not combined:
        return pd.DataFrame()

    result_df = list(combined.values())[0]
    for other_df in list(combined.values())[1:]:
        result_df = result_df.merge(other_df, on=["grid_id", "date"], how="outer")

    if result_df is None:
        return pd.DataFrame()

    # Rename for clarity
    if "def" in result_df.columns:
        result_df = result_df.rename(columns={"def": "water_deficit"})

    logger.info(f"Vegetation data: {len(result_df):,} rows, "
                f"{result_df['grid_id'].nunique()} cells")
    return result_df


# ---------------------------------------------------------------------------
# 3d. Lightning features
# ---------------------------------------------------------------------------

def fetch_lightning_for_grid(
    grid: pd.DataFrame,
    start_year: int,
    end_year: int,
    bbox: dict,
    resolution_deg: float = 0.1,
) -> pd.DataFrame:
    """Fetch lightning/storm data and map to grid cells.

    Uses NOAA Storm Events database (lightning + thunderstorm events).
    Assigns each event to its nearest grid cell and counts events per
    cell per day.

    Returns:
        DataFrame with columns: grid_id, date, lightning_count.
    """
    from src.data_acquisition.fetch_lightning import fetch_lightning_for_years

    lightning_raw = fetch_lightning_for_years(start_year, end_year, bbox)
    if lightning_raw.empty:
        logger.warning("No lightning data retrieved")
        return pd.DataFrame()

    # Filter to bbox
    lightning_raw = lightning_raw[
        (lightning_raw["latitude"] >= bbox["south"]) &
        (lightning_raw["latitude"] < bbox["north"]) &
        (lightning_raw["longitude"] >= bbox["west"]) &
        (lightning_raw["longitude"] < bbox["east"])
    ].copy()

    if lightning_raw.empty:
        return pd.DataFrame()

    # Map events to grid cells via nearest neighbor
    e_lats = lightning_raw["latitude"].values[:, None]
    e_lons = lightning_raw["longitude"].values[:, None]
    g_lats = grid["lat_center"].values[None, :]
    g_lons = grid["lon_center"].values[None, :]

    dists = np.sqrt((e_lats - g_lats) ** 2 + (e_lons - g_lons) ** 2)
    lightning_raw["grid_id"] = grid["grid_id"].values[dists.argmin(axis=1)]

    # Aggregate per grid cell per day
    daily_lightning = (
        lightning_raw
        .groupby(["grid_id", "date"])
        .agg(lightning_count=("event_type", "count"))
        .reset_index()
    )

    logger.info(f"Lightning: {len(daily_lightning):,} grid-cell-day records, "
                f"{daily_lightning['lightning_count'].sum():,} total events")
    return daily_lightning


# ---------------------------------------------------------------------------
# 4. Derived features
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataset.

    Operates per grid cell (groupby grid_id), adding rolling and lag features.
    """
    df = df.sort_values(["grid_id", "date"]).copy()

    # Temperature: convert K to °C
    if "tmmx" in df.columns:
        df["tmmx_c"] = df["tmmx"] - 273.15
    if "tmmn" in df.columns:
        df["tmmn_c"] = df["tmmn"] - 273.15

    # Group by grid cell for temporal features
    grouped = df.groupby("grid_id")

    # Rolling averages (7-day and 14-day)
    for var in ["erc", "vpd", "bi"]:
        if var in df.columns:
            df[f"{var}_7d"] = grouped[var].transform(lambda x: x.rolling(7, min_periods=1).mean())
            df[f"{var}_14d"] = grouped[var].transform(lambda x: x.rolling(14, min_periods=1).mean())

    # Rolling temperature and humidity
    if "tmmx_c" in df.columns:
        df["tmmx_7d"] = grouped["tmmx_c"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    if "rmin" in df.columns:
        df["rmin_7d"] = grouped["rmin"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Consecutive dry days (precipitation < 1mm)
    if "pr" in df.columns:
        df["is_dry"] = (df["pr"] < 1.0).astype(int)
        df["dry_streak"] = grouped["is_dry"].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        ) * df["is_dry"]
        df = df.drop(columns=["is_dry"])

    # Precipitation accumulation (7-day and 14-day)
    if "pr" in df.columns:
        df["pr_7d"] = grouped["pr"].transform(lambda x: x.rolling(7, min_periods=1).sum())
        df["pr_14d"] = grouped["pr"].transform(lambda x: x.rolling(14, min_periods=1).sum())

    # Fuel moisture change rate
    if "fm100" in df.columns:
        df["fm100_change_3d"] = grouped["fm100"].transform(lambda x: x.diff(3))

    # VPD change rate
    if "vpd" in df.columns:
        df["vpd_change_3d"] = grouped["vpd"].transform(lambda x: x.diff(3))

    # Temporal features
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        # Cyclical encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    logger.info(f"Engineered features: {len(df.columns)} total columns")
    return df


# ---------------------------------------------------------------------------
# 5. Grid metadata (static features)
# ---------------------------------------------------------------------------

def add_grid_metadata(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Add static grid cell features (lat, lon center) to the dataset."""
    df = df.merge(
        grid[["grid_id", "lat_center", "lon_center"]],
        on="grid_id",
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def build_california_dataset(
    start_year: int = 2015,
    end_year: int = 2020,
    resolution_deg: float = 0.1,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the complete California wildfire prediction dataset.

    Args:
        start_year: First year to include.
        end_year: Last year to include.
        resolution_deg: Grid cell size in degrees (~0.1° ≈ 10km).
        output_path: Where to save the parquet file.

    Returns:
        Complete training DataFrame.
    """
    from src.data_acquisition.fetch_fire_history import load_fpa_fod

    bbox = REGIONS["california"]
    output_path = output_path or PROCESSED_DIR / "california_dataset.parquet"

    # Step 1: Build grid
    print("Step 1/9: Building spatial grid...")
    grid = build_grid(bbox, resolution_deg)
    print(f"  Grid: {len(grid)} cells")

    # Step 2: Load and assign fires
    print("\nStep 2/9: Loading fire records...")
    fpa_path = RAW_DIR / "fpa_fod" / "FPA_FOD.sqlite"
    fires = load_fpa_fod(fpa_path)

    # Filter to date range and California
    fires = fires[
        (fires["STATE"] == "CA") &
        (fires["discovery_date"].dt.year >= start_year) &
        (fires["discovery_date"].dt.year <= end_year)
    ]
    print(f"  California fires ({start_year}-{end_year}): {len(fires):,}")

    fires = assign_fires_to_grid(fires, grid, resolution_deg, bbox)

    # Step 3: Create fire labels
    print("\nStep 3/9: Creating fire labels...")
    labels = create_fire_labels(
        fires, grid,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
    )
    print(f"  Label rows: {len(labels):,}")
    print(f"  Fire-days: {labels['has_fire'].sum():,} ({labels['has_fire'].mean()*100:.3f}%)")

    # Step 4: Fetch weather data year by year
    print("\nStep 4/9: Fetching gridMET weather data...")
    # Get only the grid cells that are in our labels
    selected_cells = labels["grid_id"].unique()
    grid_subset = grid[grid["grid_id"].isin(selected_cells)]

    weather_frames = []
    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year}...")
        wdf = fetch_gridmet_for_grid(grid_subset, year, bbox=bbox)
        if not wdf.empty:
            weather_frames.append(wdf)

    weather = pd.concat(weather_frames, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    print(f"  Weather rows: {len(weather):,}")

    # Step 5: Fetch additional data layers
    print("\nStep 5/9: Fetching air quality data...")
    try:
        aq_data = fetch_air_quality_for_grid(grid_subset, start_year, end_year)
        if not aq_data.empty:
            print(f"  Air quality: {len(aq_data):,} records")
        else:
            print("  Air quality: no data (skipped)")
    except Exception as e:
        print(f"  Air quality failed: {e}")
        aq_data = pd.DataFrame()

    print("\nStep 6/9: Fetching vegetation data (TerraClimate)...")
    try:
        veg_data = fetch_vegetation_for_grid(grid_subset, start_year, end_year, bbox)
        if not veg_data.empty:
            print(f"  Vegetation: {len(veg_data):,} records")
        else:
            print("  Vegetation: no data (skipped)")
    except Exception as e:
        print(f"  Vegetation failed: {e}")
        veg_data = pd.DataFrame()

    print("\nStep 7/9: Fetching lightning data...")
    try:
        lightning_data = fetch_lightning_for_grid(
            grid_subset, start_year, end_year, bbox, resolution_deg
        )
        if not lightning_data.empty:
            print(f"  Lightning: {len(lightning_data):,} records")
        else:
            print("  Lightning: no data (skipped)")
    except Exception as e:
        print(f"  Lightning failed: {e}")
        lightning_data = pd.DataFrame()

    # Step 8: Merge all data layers
    print("\nStep 8/9: Merging and engineering features...")
    dataset = labels.merge(weather, on=["grid_id", "date"], how="left")

    # Drop ocean/offshore cells that have no weather data (land-only datasets)
    weather_cols = ["tmmx", "rmin", "vs", "pr", "erc", "vpd", "fm100", "bi"]
    has_weather = dataset.groupby("grid_id")[weather_cols[0]].transform("count") > 0
    n_before = dataset["grid_id"].nunique()
    dataset = dataset[has_weather].copy()
    n_after = dataset["grid_id"].nunique()
    print(f"  Dropped {n_before - n_after} ocean cells ({n_before} → {n_after} grid cells)")

    # Merge air quality
    if not aq_data.empty:
        dataset = dataset.merge(aq_data, on=["grid_id", "date"], how="left")
        # Fill missing AQ values (cells without nearby stations)
        for col in ["pm25_mean", "pm25_max", "aqi_mean"]:
            if col in dataset.columns:
                dataset[col] = dataset[col].fillna(0.0)
        print(f"  Merged air quality ({aq_data.columns.tolist()})")

    # Merge vegetation data (monthly, forward-filled to daily)
    if not veg_data.empty:
        dataset = dataset.merge(veg_data, on=["grid_id", "date"], how="left")
        # Forward-fill monthly values to daily within each grid cell
        for col in ["aet", "water_deficit"]:
            if col in dataset.columns:
                dataset[col] = dataset.groupby("grid_id")[col].transform(
                    lambda x: x.ffill().bfill()
                )
        print(f"  Merged vegetation data (forward-filled to daily)")

    # Merge lightning
    if not lightning_data.empty:
        dataset = dataset.merge(lightning_data, on=["grid_id", "date"], how="left")
        dataset["lightning_count"] = dataset["lightning_count"].fillna(0).astype(int)
        print(f"  Merged lightning events")

    # Add static features
    dataset = add_grid_metadata(dataset, grid)

    # Engineer derived features
    dataset = engineer_features(dataset)

    # Step 9: Clean data
    print("\nStep 9/10: Cleaning data...")
    n_before = len(dataset)

    # 1. Drop sparse air quality columns (only 3% of cells have station data;
    #    TODO: replace with MERRA-2 gridded PM2.5 reanalysis for full coverage)
    aq_cols = [c for c in ["pm25_mean", "pm25_max", "aqi_mean"] if c in dataset.columns]
    if aq_cols:
        dataset = dataset.drop(columns=aq_cols)
        print(f"  Dropped sparse air quality columns: {aq_cols}")

    # 2. Fill missing vegetation (41 coastal cells not covered by TerraClimate)
    for col in ["aet", "water_deficit"]:
        n_miss = dataset[col].isna().sum()
        if n_miss > 0:
            dataset[col] = dataset[col].fillna(dataset[col].median())
            print(f"  Filled {n_miss:,} missing {col} with median ({dataset[col].median():.1f})")

    # 3. Fill missing 3-day change features (first 3 days of dataset)
    for col in ["fm100_change_3d", "vpd_change_3d"]:
        n_miss = dataset[col].isna().sum()
        if n_miss > 0:
            dataset[col] = dataset[col].fillna(0.0)
            print(f"  Filled {n_miss:,} missing {col} with 0")

    # 4. Drop redundant tmmx (Kelvin) — tmmx_c (Celsius) is more interpretable
    if "tmmx" in dataset.columns and "tmmx_c" in dataset.columns:
        dataset = dataset.drop(columns=["tmmx"])
        print("  Dropped tmmx (Kelvin) — keeping tmmx_c (Celsius)")

    # 5. Drop redundant duplicate encodings identified in EDA
    #    day_of_year ↔ month (r=0.997), doy_sin ↔ month_sin (r=0.955),
    #    month_cos ↔ doy_cos (r=0.955), erc ↔ bi_7d (r=0.936)
    eda_drops = [c for c in ["day_of_year", "doy_sin", "month_cos", "erc"] if c in dataset.columns]
    if eda_drops:
        dataset = dataset.drop(columns=eda_drops)
        print(f"  Dropped redundant features: {eda_drops}")

    assert dataset.isnull().sum().sum() == 0, f"Still have nulls: {dataset.isnull().sum()[dataset.isnull().sum()>0]}"
    print(f"  No missing values remaining ✓")
    print(f"  Final shape: {dataset.shape[0]:,} rows × {dataset.shape[1]} columns")

    # Step 10: Save
    print(f"\nStep 10/10: Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"  Rows: {len(dataset):,}")
    print(f"  Columns: {len(dataset.columns)}")
    print(f"  Column names: {sorted(dataset.columns.tolist())}")
    print(f"  Fire-days: {dataset['has_fire'].sum():,} ({dataset['has_fire'].mean()*100:.3f}%)")
    print(f"  Date range: {dataset['date'].min().date()} to {dataset['date'].max().date()}")
    print(f"  Grid cells: {dataset['grid_id'].nunique()}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Saved to: {output_path}")

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_california_dataset(start_year=2018, end_year=2020)
