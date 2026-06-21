"""
Fetch vegetation (NDVI) data from the MODIS Web Service at ORNL DAAC.

Uses MOD13Q1 (16-day, 250m NDVI) via the REST API. Samples at a coarser
resolution (0.5°) and maps to finer grid cells via nearest-neighbor to
keep API calls manageable.

Source: https://modis.ornl.gov/rst/api/v1/
No API key required.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import RAW_DIR

logger = logging.getLogger(__name__)

MODIS_BASE = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"  # 16-day 250m NDVI


def fetch_ndvi_point(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Fetch NDVI time series for a single point.

    Args:
        lat: Latitude.
        lon: Longitude.
        start_date: MODIS date string (e.g., 'A2018001').
        end_date: MODIS date string (e.g., 'A2020365').

    Returns:
        List of {date, ndvi} dicts.
    """
    url = f"{MODIS_BASE}/{PRODUCT}/subset"
    params = {
        "latitude": lat,
        "longitude": lon,
        "startDate": start_date,
        "endDate": end_date,
        "kmAboveBelow": 0,
        "kmLeftRight": 0,
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    records = []
    seen_dates = set()
    for item in data.get("subset", []):
        cal_date = item.get("calendar_date")
        values = item.get("data", [])
        # Take only the first (center) pixel per date to avoid duplicates
        if cal_date and values and cal_date not in seen_dates:
            seen_dates.add(cal_date)
            # Filter to valid NDVI values only
            valid = [v for v in values if -2000 <= v <= 10000]
            if valid:
                ndvi_raw = np.mean(valid)  # average if multiple pixels
                records.append({
                    "date": cal_date,
                    "ndvi": ndvi_raw / 10000.0,
                })

    return records


def fetch_ndvi_grid(
    bbox: dict,
    start_year: int,
    end_year: int,
    sample_resolution: float = 0.5,
) -> pd.DataFrame:
    """Fetch NDVI over a bounding box at sampled resolution.

    Queries the MODIS Web Service for a coarse grid of points, caching
    results to avoid repeated downloads.

    Args:
        bbox: Dict with west, south, east, north.
        start_year: First year.
        end_year: Last year.
        sample_resolution: Spacing between sample points in degrees.

    Returns:
        DataFrame with columns: sample_lat, sample_lon, date, ndvi.
    """
    cache_dir = RAW_DIR / "vegetation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ndvi_{start_year}_{end_year}.csv"

    if cache_path.exists():
        logger.info(f"Loading cached NDVI from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    sample_lats = np.arange(bbox["south"] + 0.25, bbox["north"], sample_resolution)
    sample_lons = np.arange(bbox["west"] + 0.25, bbox["east"], sample_resolution)

    total_points = len(sample_lats) * len(sample_lons)
    logger.info(f"Fetching NDVI for {total_points} sample points ({sample_resolution}° grid)")

    all_records = []
    count = 0

    for lat in sample_lats:
        for lon in sample_lons:
            count += 1
            if count % 10 == 0:
                logger.info(f"  NDVI progress: {count}/{total_points}")

            # Fetch in 90-day chunks (API rejects longer spans)
            point_failed = False
            for year in range(start_year, end_year + 1):
                if point_failed:
                    break
                # 4 quarters per year (days 1-90, 91-180, 181-270, 271-365)
                quarters = [(1, 90), (91, 180), (181, 270), (271, 365)]
                for q_start, q_end in quarters:
                    start_doy = f"A{year}{q_start:03d}"
                    end_doy = f"A{year}{q_end:03d}"
                    try:
                        records = fetch_ndvi_point(lat, lon, start_doy, end_doy)
                        for r in records:
                            r["sample_lat"] = round(lat, 4)
                            r["sample_lon"] = round(lon, 4)
                        all_records.extend(records)
                        time.sleep(0.2)
                    except Exception as e:
                        point_failed = True
                        break  # skip remaining quarters for this point

    ndvi_df = pd.DataFrame(all_records)
    if not ndvi_df.empty:
        ndvi_df["date"] = pd.to_datetime(ndvi_df["date"])
        ndvi_df.to_csv(cache_path, index=False)
        logger.info(f"Saved {len(ndvi_df)} NDVI records to {cache_path}")

    return ndvi_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test: single point in central CA
    records = fetch_ndvi_point(37.0, -120.0, "A2020001", "A2020365")
    print(f"Got {len(records)} NDVI observations")
    for r in records[:5]:
        print(f"  {r['date']}: NDVI={r['ndvi']:.3f}")
