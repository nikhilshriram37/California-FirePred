"""
Fetch NASA FIRMS active fire detection data (MODIS / VIIRS).

FIRMS provides near-real-time and archival satellite fire detections at 375m (VIIRS)
and 1km (MODIS) resolution. Used both as validation data and as additional fire
occurrence labels.

Data source: https://firms.modaps.eosdis.nasa.gov/
API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
"""

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import NASA_FIRMS_MAP_KEY, RAW_DIR, CONUS_BBOX, REGIONS

logger = logging.getLogger(__name__)

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api"

# Available satellite sources
SOURCES = {
    "viirs_noaa20": "VIIRS_NOAA20_NRT",
    "viirs_noaa21": "VIIRS_NOAA21_NRT",
    "viirs_snpp": "VIIRS_SNPP_NRT",
    "modis": "MODIS_NRT",
}


def fetch_firms_area(
    source: str = "viirs_noaa20",
    bbox: dict | None = None,
    days: int = 1,
    map_key: str | None = None,
) -> pd.DataFrame:
    """Fetch active fire detections for a geographic area.

    Args:
        source: Satellite source key (see SOURCES dict).
        bbox: Dict with west, south, east, north. Defaults to CONUS.
        days: Number of days of data (1–10 for NRT).
        map_key: FIRMS MAP_KEY. Defaults to env variable.

    Returns:
        DataFrame with fire detection records.
    """
    map_key = map_key or NASA_FIRMS_MAP_KEY
    if not map_key:
        raise ValueError(
            "NASA FIRMS MAP_KEY required. Register at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ "
            "and set NASA_FIRMS_MAP_KEY in .env"
        )

    bbox = bbox or CONUS_BBOX
    source_id = SOURCES.get(source, source)
    area = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    url = f"{FIRMS_BASE}/area/csv/{map_key}/{source_id}/{area}/{days}"
    logger.info(f"Fetching FIRMS {source_id} data: {days} day(s), bbox={area}")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    logger.info(f"Retrieved {len(df)} fire detections")

    return df


def fetch_firms_country(
    country: str = "USA",
    source: str = "viirs_noaa20",
    days: int = 1,
    map_key: str | None = None,
) -> pd.DataFrame:
    """Fetch active fire detections for an entire country."""
    map_key = map_key or NASA_FIRMS_MAP_KEY
    if not map_key:
        raise ValueError("NASA FIRMS MAP_KEY required. Set NASA_FIRMS_MAP_KEY in .env")

    source_id = SOURCES.get(source, source)
    url = f"{FIRMS_BASE}/country/csv/{map_key}/{source_id}/{country}/{days}"
    logger.info(f"Fetching FIRMS {source_id} for {country}: {days} day(s)")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    logger.info(f"Retrieved {len(df)} fire detections")

    return df


def fetch_firms_archive(
    source: str = "viirs_noaa20",
    bbox: dict | None = None,
    date: str = "2023-08-01",
    map_key: str | None = None,
) -> pd.DataFrame:
    """Fetch archival FIRMS data for a specific date.

    For dates older than 10 days, use the archive endpoint.

    Args:
        date: Date string in YYYY-MM-DD format.
    """
    map_key = map_key or NASA_FIRMS_MAP_KEY
    if not map_key:
        raise ValueError("NASA FIRMS MAP_KEY required. Set NASA_FIRMS_MAP_KEY in .env")

    bbox = bbox or CONUS_BBOX
    source_id = SOURCES.get(source, source).replace("_NRT", "_SP")
    area = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    url = f"{FIRMS_BASE}/area/csv/{map_key}/{source_id}/{area}/1/{date}"
    logger.info(f"Fetching FIRMS archive {source_id} for {date}")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    logger.info(f"Retrieved {len(df)} fire detections for {date}")

    return df


def fetch_firms_date_range(
    start_date: str,
    end_date: str,
    source: str = "viirs_noaa20",
    bbox: dict | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS data for a range of dates, respecting rate limits.

    Saves intermediate results and returns combined DataFrame.
    """
    output_dir = output_dir or RAW_DIR / "firms"
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date, freq="D")
    all_dfs = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        cache_path = output_dir / f"firms_{source}_{date_str}.csv"

        if cache_path.exists():
            logger.info(f"Loading cached FIRMS data for {date_str}")
            df = pd.read_csv(cache_path)
        else:
            try:
                df = fetch_firms_archive(source, bbox, date_str)
                df.to_csv(cache_path, index=False)
                time.sleep(7)  # Rate limit: ~10 req/min
            except Exception as e:
                logger.error(f"Failed to fetch FIRMS for {date_str}: {e}")
                continue

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / f"firms_{source}_{start_date}_to_{end_date}.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"Combined {len(combined)} detections saved to {combined_path}")
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: fetch recent US fire detections
    print("Fetching recent FIRMS VIIRS data for California...")
    df = fetch_firms_area("viirs_noaa20", bbox=REGIONS["california"], days=2)
    print(f"Columns: {list(df.columns)}")
    print(df.head())
