"""
Fetch gridMET weather and fire danger data via OPeNDAP/THREDDS.

gridMET provides daily 4km-resolution surface weather and pre-computed fire danger
indices for the contiguous US (1979–present). This is the primary weather data source
because it's gridded (no station gaps), includes fire-specific indices, and requires
no API key.

Data source: https://www.climatologylab.org/gridmet.html
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    GRIDMET_VARS,
    RAW_DIR,
    CONUS_BBOX,
    REGIONS,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
)

logger = logging.getLogger(__name__)

THREDDS_BASE = "https://thredds.northwestknowledge.net/thredds/dodsC/MET"


def fetch_gridmet_variable(
    variable: str,
    year: int,
    bbox: dict | None = None,
    output_dir: Path | None = None,
) -> xr.Dataset:
    """Fetch a single gridMET variable for a given year and bounding box.

    Args:
        variable: gridMET variable name (e.g., 'tmmx', 'erc', 'vpd').
        year: Calendar year to fetch.
        bbox: Dict with keys west, south, east, north. Defaults to CONUS.
        output_dir: Directory to save the NetCDF file. Defaults to RAW_DIR/gridmet/.

    Returns:
        xarray.Dataset with the requested variable.
    """
    if variable not in GRIDMET_VARS:
        raise ValueError(f"Unknown variable '{variable}'. Choose from: {list(GRIDMET_VARS.keys())}")

    bbox = bbox or CONUS_BBOX
    output_dir = output_dir or RAW_DIR / "gridmet"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"{THREDDS_BASE}/{variable}/{variable}_{year}.nc"
    logger.info(f"Fetching gridMET {variable} for {year} from {url}")

    ds = xr.open_dataset(url, engine="netcdf4")

    # Subset to bounding box (gridMET uses lat/lon coordinates)
    ds = ds.sel(
        lat=slice(bbox["north"], bbox["south"]),  # gridMET lat is descending
        lon=slice(bbox["west"], bbox["east"]),
    )

    # Save to local file
    out_path = output_dir / f"{variable}_{year}.nc"
    ds.to_netcdf(out_path)
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return ds


def fetch_gridmet_all_weather(
    year: int,
    bbox: dict | None = None,
    variables: list[str] | None = None,
) -> dict[str, xr.Dataset]:
    """Fetch all weather and fire danger variables for a given year.

    Args:
        year: Calendar year.
        bbox: Geographic bounding box.
        variables: List of variable names. Defaults to all GRIDMET_VARS.

    Returns:
        Dict mapping variable name to xarray.Dataset.
    """
    variables = variables or list(GRIDMET_VARS.keys())
    results = {}

    for var in variables:
        try:
            results[var] = fetch_gridmet_variable(var, year, bbox)
        except Exception as e:
            logger.error(f"Failed to fetch {var} for {year}: {e}")

    return results


def fetch_gridmet_range(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    bbox: dict | None = None,
    variables: list[str] | None = None,
) -> None:
    """Fetch gridMET data for a range of years.

    This downloads and saves NetCDF files locally for all specified variables
    and years. Files are saved to data/raw/gridmet/{variable}_{year}.nc.
    """
    variables = variables or list(GRIDMET_VARS.keys())

    for year in range(start_year, end_year + 1):
        logger.info(f"--- Fetching gridMET data for {year} ---")
        fetch_gridmet_all_weather(year, bbox, variables)


def load_gridmet_as_dataframe(
    variable: str,
    year: int,
    bbox: dict | None = None,
) -> pd.DataFrame:
    """Load a gridMET variable as a flat pandas DataFrame.

    Useful for merging with point-based fire occurrence data.

    Returns:
        DataFrame with columns: lat, lon, time, {variable_name}
    """
    filepath = RAW_DIR / "gridmet" / f"{variable}_{year}.nc"

    if filepath.exists():
        ds = xr.open_dataset(filepath)
    else:
        ds = fetch_gridmet_variable(variable, year, bbox)

    # Convert to dataframe (this can be large — consider chunking for full CONUS)
    df = ds.to_dataframe().reset_index()
    df = df.dropna(subset=[list(ds.data_vars)[0]])

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: fetch fire danger indices for California, 2020
    print("Fetching gridMET ERC for California, 2020...")
    ds = fetch_gridmet_variable("erc", 2020, bbox=REGIONS["california"])
    print(f"Dataset shape: {dict(ds.dims)}")
    print(ds)
