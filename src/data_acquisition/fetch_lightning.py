"""
Fetch lightning data from NOAA Severe Weather Data Inventory (SWDI).

Uses NLDN (National Lightning Detection Network) tile summaries via the
SWDI API. Falls back to downloading NOAA Storm Events bulk CSVs if the
tile API is unavailable.

Sources:
- SWDI: https://www.ncdc.noaa.gov/swdi/
- Storm Events: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
"""

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import RAW_DIR

logger = logging.getLogger(__name__)

SWDI_BASE = "https://www.ncdc.noaa.gov/swdi"
STORM_EVENTS_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"


def fetch_swdi_lightning(
    bbox: dict,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch NLDN lightning tile counts from SWDI.

    Args:
        bbox: Dict with west, south, east, north.
        start_date: YYYYMMDD format.
        end_date: YYYYMMDD format.

    Returns:
        DataFrame with lightning flash locations and counts.
    """
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    url = f"{SWDI_BASE}/csv/nldn-tiles-15min/{start_date}:{end_date}"
    params = {"bbox": bbox_str}

    logger.info(f"Fetching SWDI lightning tiles {start_date} to {end_date}")
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    # SWDI returns CSV
    df = pd.read_csv(io.StringIO(resp.text))
    logger.info(f"Retrieved {len(df)} lightning tile records")
    return df


def fetch_storm_events_lightning(
    year: int,
    state: str = "CALIFORNIA",
) -> pd.DataFrame:
    """Download NOAA Storm Events and filter to lightning in a state.

    Storm Events provides damage/injury lightning reports (not comprehensive
    strike data), but serves as a useful supplementary feature.

    Args:
        year: Year to fetch.
        state: State name (uppercase).

    Returns:
        DataFrame of lightning events with lat/lon and date.
    """
    cache_dir = RAW_DIR / "lightning"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"storm_events_{year}.csv"

    if cache_path.exists():
        logger.info(f"Loading cached Storm Events for {year}")
        df = pd.read_csv(cache_path, low_memory=False)
    else:
        # Find the correct filename — it includes a version date suffix
        # Try the listing page to find it
        logger.info(f"Downloading Storm Events details for {year}...")
        listing_url = f"{STORM_EVENTS_BASE}/"
        try:
            resp = requests.get(listing_url, timeout=30)
            resp.raise_for_status()
            # Find the details file for this year
            import re
            pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
            matches = re.findall(pattern, resp.text)
            if not matches:
                logger.warning(f"No Storm Events file found for {year}")
                return pd.DataFrame()
            filename = matches[-1]  # latest version
        except Exception:
            # Fallback: try common naming
            filename = f"StormEvents_details-ftp_v1.0_d{year}_c20240117.csv.gz"

        url = f"{STORM_EVENTS_BASE}/{filename}"
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), compression="gzip", low_memory=False)
            df.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(df)} Storm Events records for {year}")
        except Exception as e:
            logger.warning(f"Failed to download Storm Events for {year}: {e}")
            return pd.DataFrame()

    # Filter to lightning and thunderstorm events in the target state
    lightning_types = ["Lightning", "Thunderstorm Wind", "Hail"]
    df_state = df[df["STATE"].str.upper() == state] if "STATE" in df.columns else df
    df_lightning = df_state[df_state["EVENT_TYPE"].isin(lightning_types)] if "EVENT_TYPE" in df.columns else df_state

    # Extract coordinates and dates
    records = []
    for _, row in df_lightning.iterrows():
        lat = row.get("BEGIN_LAT") or row.get("END_LAT")
        lon = row.get("BEGIN_LON") or row.get("END_LON")
        date_str = row.get("BEGIN_DATE_TIME")

        if pd.notna(lat) and pd.notna(lon) and pd.notna(date_str):
            records.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "date": date_str,
                "event_type": row.get("EVENT_TYPE", "Unknown"),
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"], format="mixed", errors="coerce")
        result["date"] = result["date"].dt.normalize()

    logger.info(f"Found {len(result)} lightning/storm events in {state} for {year}")
    return result


def fetch_lightning_for_years(
    start_year: int,
    end_year: int,
    bbox: dict,
    state: str = "CALIFORNIA",
) -> pd.DataFrame:
    """Fetch lightning data for multiple years.

    Tries SWDI NLDN tiles first, falls back to Storm Events.

    Args:
        start_year: First year.
        end_year: Last year.
        bbox: Bounding box.
        state: State name.

    Returns:
        DataFrame with latitude, longitude, date, event_type.
    """
    # Try SWDI first (more comprehensive)
    try:
        start = f"{start_year}0101"
        end = f"{start_year}0131"  # test with one month
        test = fetch_swdi_lightning(bbox, start, end)
        if not test.empty:
            logger.info("SWDI NLDN tiles available — using tile data")
            all_dfs = []
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    m_start = f"{year}{month:02d}01"
                    if month == 12:
                        m_end = f"{year}1231"
                    else:
                        m_end = f"{year}{month + 1:02d}01"
                    try:
                        chunk = fetch_swdi_lightning(bbox, m_start, m_end)
                        all_dfs.append(chunk)
                        time.sleep(0.5)
                    except Exception as e:
                        logger.warning(f"SWDI failed for {year}-{month:02d}: {e}")
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
    except Exception as e:
        logger.info(f"SWDI unavailable ({e}), falling back to Storm Events")

    # Fallback: Storm Events
    all_dfs = []
    for year in range(start_year, end_year + 1):
        df = fetch_storm_events_lightning(year, state)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .config import REGIONS

    bbox = REGIONS["california"]
    df = fetch_storm_events_lightning(2020, "CALIFORNIA")
    print(f"Lightning events: {len(df)}")
    if not df.empty:
        print(df.head())
