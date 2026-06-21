"""
Fetch air quality data from OpenAQ and EPA AQS.

PM2.5 spikes are a strong early indicator of nearby wildfire activity.
Historical AQ data from EPA provides training features; OpenAQ provides
real-time feeds for the dashboard.

Sources:
- EPA AQS: https://aqs.epa.gov/aqsweb/documents/data_api.html
- OpenAQ: https://api.openaq.org/v3/
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import EPA_AQS_EMAIL, EPA_AQS_KEY, OPENAQ_API_KEY, RAW_DIR

logger = logging.getLogger(__name__)

EPA_BASE = "https://aqs.epa.gov/data/api"
OPENAQ_BASE = "https://api.openaq.org/v3"

# EPA AQS parameter codes
AQS_PARAMS = {
    "pm25_frm": "88101",     # PM2.5 Federal Reference Method
    "pm25_nonfrm": "88502",  # PM2.5 non-FRM
    "pm10": "81102",         # PM10
    "ozone": "44201",        # Ozone
    "co": "42101",           # Carbon monoxide
}

# US state FIPS codes for fire-prone states
FIRE_STATE_FIPS = {
    "CA": "06", "OR": "41", "WA": "53", "MT": "30", "ID": "16",
    "AZ": "04", "NM": "35", "CO": "08", "TX": "48", "FL": "12",
    "GA": "13", "NV": "32", "UT": "49",
}


def fetch_epa_daily(
    state_fips: str,
    parameter: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily AQ data from EPA AQS for a state.

    Args:
        state_fips: 2-digit FIPS code.
        parameter: AQS parameter code (e.g., '88101' for PM2.5).
        start_date: YYYYMMDD format.
        end_date: YYYYMMDD format (max 1 year from start).

    Returns:
        DataFrame of daily AQ observations.
    """
    if not EPA_AQS_EMAIL or not EPA_AQS_KEY:
        raise ValueError(
            "EPA AQS credentials required. Register at "
            "https://aqs.epa.gov/aqsweb/documents/data_api.html "
            "and set EPA_AQS_EMAIL and EPA_AQS_KEY in .env"
        )

    url = f"{EPA_BASE}/dailyData/byState"
    params = {
        "email": EPA_AQS_EMAIL,
        "key": EPA_AQS_KEY,
        "param": parameter,
        "bdate": start_date,
        "edate": end_date,
        "state": state_fips,
    }

    logger.info(f"Fetching EPA AQS param={parameter} for FIPS:{state_fips} {start_date}-{end_date}")
    response = requests.get(url, params=params, timeout=300)
    response.raise_for_status()

    data = response.json()
    results = data.get("Data", [])
    df = pd.DataFrame(results)
    logger.info(f"Retrieved {len(df)} daily observations")
    return df


def fetch_epa_pm25_bulk(
    state_fips: str,
    start_year: int,
    end_year: int,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch PM2.5 daily data for a state across multiple years.

    Handles the 1-year-max-span limitation by iterating year by year.
    """
    output_dir = output_dir or RAW_DIR / "air_quality" / "epa"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for year in range(start_year, end_year + 1):
        cache_path = output_dir / f"pm25_{state_fips}_{year}.csv"

        if cache_path.exists():
            logger.info(f"Loading cached EPA PM2.5 for FIPS:{state_fips} {year}")
            df = pd.read_csv(cache_path)
        else:
            try:
                df = fetch_epa_daily(state_fips, "88101", f"{year}0101", f"{year}1231")
                if not df.empty:
                    df.to_csv(cache_path, index=False)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed for FIPS:{state_fips} {year}: {e}")
                continue

        all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def fetch_openaq_locations(
    lat: float,
    lon: float,
    radius_m: int = 50000,
    parameter: str = "pm25",
) -> pd.DataFrame:
    """Find OpenAQ monitoring locations near a point.

    Args:
        lat, lon: Center point coordinates.
        radius_m: Search radius in meters.
        parameter: Pollutant to filter by.

    Returns:
        DataFrame of nearby monitoring locations.
    """
    headers = {}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY

    # Convert point + radius to a bounding box approximation
    # ~0.01 degrees latitude ≈ 1.11 km
    deg_offset = radius_m / 111_000
    bbox = f"{lon - deg_offset},{lat - deg_offset},{lon + deg_offset},{lat + deg_offset}"

    url = f"{OPENAQ_BASE}/locations"
    params = {
        "bbox": bbox,
        "limit": 100,
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    results = response.json().get("results", [])
    records = []
    for loc in results:
        coords = loc.get("coordinates", {})
        records.append({
            "id": loc.get("id"),
            "name": loc.get("name"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "country": loc.get("country", {}).get("code") if isinstance(loc.get("country"), dict) else loc.get("country"),
            "provider": loc.get("provider", {}).get("name") if isinstance(loc.get("provider"), dict) else loc.get("provider"),
        })

    return pd.DataFrame(records)


def fetch_openaq_measurements(
    location_id: int,
    date_from: str,
    date_to: str,
    parameter: str = "pm25",
) -> pd.DataFrame:
    """Fetch historical measurements from an OpenAQ location.

    Args:
        location_id: OpenAQ location ID.
        date_from: ISO date (YYYY-MM-DD).
        date_to: ISO date (YYYY-MM-DD).
        parameter: Pollutant name.

    Returns:
        DataFrame of measurements.
    """
    headers = {}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY

    url = f"{OPENAQ_BASE}/locations/{location_id}/measurements"
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "parameter": parameter,
        "limit": 1000,
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    results = response.json().get("results", [])
    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: find OpenAQ stations near Los Angeles
    print("Finding OpenAQ PM2.5 stations near Los Angeles...")
    locs = fetch_openaq_locations(34.05, -118.25, radius_m=100000)
    print(f"Found {len(locs)} locations")
    print(locs.head())
