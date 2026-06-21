"""
Fetch weather data from NOAA APIs.

Two sources:
1. NOAA CDO API (historical daily observations from GHCND stations)
2. NWS API (recent observations and forecasts — no key required)

Data source: https://www.ncdc.noaa.gov/cdo-web/api/v2/
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import NOAA_API_KEY, RAW_DIR

logger = logging.getLogger(__name__)

CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NWS_BASE = "https://api.weather.gov"

# GHCND data types relevant to fire weather
FIRE_WEATHER_DATATYPES = [
    "TMAX",   # Max temperature
    "TMIN",   # Min temperature
    "PRCP",   # Precipitation
    "AWND",   # Average wind speed
    "WSF2",   # Fastest 2-min wind speed
    "WSF5",   # Fastest 5-sec wind speed
]


def _cdo_headers() -> dict:
    if not NOAA_API_KEY:
        raise ValueError(
            "NOAA CDO API key required. Register at "
            "https://www.ncdc.noaa.gov/cdo-web/token and set NOAA_API_KEY in .env"
        )
    return {"token": NOAA_API_KEY}


def fetch_cdo_stations(
    state_fips: str = "06",
    dataset: str = "GHCND",
) -> pd.DataFrame:
    """Fetch weather station metadata for a state.

    Args:
        state_fips: 2-digit FIPS code (e.g., '06' for California).
        dataset: NOAA dataset ID.

    Returns:
        DataFrame of station metadata.
    """
    url = f"{CDO_BASE}/stations"
    params = {
        "datasetid": dataset,
        "locationid": f"FIPS:{state_fips}",
        "limit": 1000,
    }

    logger.info(f"Fetching CDO stations for FIPS:{state_fips}")
    response = requests.get(url, headers=_cdo_headers(), params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])
    return pd.DataFrame(results)


def fetch_cdo_daily(
    station_id: str,
    start_date: str,
    end_date: str,
    datatypes: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch daily weather observations from a GHCND station.

    Args:
        station_id: GHCND station ID (e.g., 'GHCND:USW00023174').
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD). Max 1-year span per request.
        datatypes: List of data type IDs. Defaults to fire weather types.

    Returns:
        DataFrame with columns: date, datatype, value, station.
    """
    datatypes = datatypes or FIRE_WEATHER_DATATYPES
    url = f"{CDO_BASE}/data"

    all_results = []
    offset = 1

    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": ",".join(datatypes),
            "units": "metric",
            "limit": 1000,
            "offset": offset,
        }

        response = requests.get(url, headers=_cdo_headers(), params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        offset += len(results)

        if offset > data.get("metadata", {}).get("resultset", {}).get("count", 0):
            break

        time.sleep(0.25)  # Rate limit: 5 req/sec

    df = pd.DataFrame(all_results)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_cdo_bulk(
    state_fips: str,
    start_date: str,
    end_date: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch daily weather data for all stations in a state.

    Iterates through stations and concatenates results. Saves to CSV.
    """
    output_dir = output_dir or RAW_DIR / "weather"
    output_dir.mkdir(parents=True, exist_ok=True)

    stations_df = fetch_cdo_stations(state_fips)
    if stations_df.empty:
        logger.warning(f"No stations found for FIPS:{state_fips}")
        return pd.DataFrame()

    logger.info(f"Fetching data from {len(stations_df)} stations in FIPS:{state_fips}")
    all_dfs = []

    for _, station in stations_df.iterrows():
        station_id = station["id"]
        cache_path = output_dir / f"{station_id}_{start_date}_{end_date}.csv"

        if cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=["date"])
        else:
            try:
                df = fetch_cdo_daily(station_id, start_date, end_date)
                if not df.empty:
                    df.to_csv(cache_path, index=False)
                time.sleep(0.25)
            except Exception as e:
                logger.error(f"Failed for station {station_id}: {e}")
                continue

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined

    return pd.DataFrame()


def fetch_nws_observations(
    station_id: str,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch recent observations from an NWS station (no API key needed).

    Args:
        station_id: NWS station ID (e.g., 'KLAX').
        limit: Max number of observations.

    Returns:
        DataFrame of recent observations.
    """
    url = f"{NWS_BASE}/stations/{station_id}/observations"
    headers = {"User-Agent": "FireProject (wildfire-prediction-research)"}
    params = {"limit": limit}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    features = response.json().get("features", [])
    records = []
    for f in features:
        props = f["properties"]
        records.append({
            "timestamp": props.get("timestamp"),
            "temperature_c": props.get("temperature", {}).get("value"),
            "dewpoint_c": props.get("dewpoint", {}).get("value"),
            "wind_speed_kmh": props.get("windSpeed", {}).get("value"),
            "wind_direction_deg": props.get("windDirection", {}).get("value"),
            "humidity_pct": props.get("relativeHumidity", {}).get("value"),
            "precip_1hr_mm": props.get("precipitationLastHour", {}).get("value"),
            "visibility_m": props.get("visibility", {}).get("value"),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fetch_nws_red_flag_warnings() -> pd.DataFrame:
    """Fetch active Red Flag Warnings from NWS (no API key needed).

    Red Flag Warnings indicate critical fire weather conditions.
    """
    url = f"{NWS_BASE}/alerts/active"
    headers = {"User-Agent": "FireProject (wildfire-prediction-research)"}
    params = {"event": "Red Flag Warning"}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    features = response.json().get("features", [])
    records = []
    for f in features:
        props = f["properties"]
        records.append({
            "id": props.get("id"),
            "area_desc": props.get("areaDesc"),
            "onset": props.get("onset"),
            "expires": props.get("expires"),
            "severity": props.get("severity"),
            "headline": props.get("headline"),
            "description": props.get("description"),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: fetch NWS observations (no key needed)
    print("Fetching NWS observations for KLAX (Los Angeles)...")
    df = fetch_nws_observations("KLAX", limit=10)
    print(df)

    # Example: check for Red Flag Warnings
    print("\nActive Red Flag Warnings:")
    rfw = fetch_nws_red_flag_warnings()
    print(f"{len(rfw)} active warnings")
