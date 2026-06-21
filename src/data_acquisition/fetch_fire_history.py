"""
Fetch historical fire occurrence data from NIFC and NASA EONET.

The primary training labels come from:
1. FPA-FOD (Fire Program Analysis Fire-Occurrence Database) — bulk download, SQLite
2. NIFC IRWIN / fire perimeters — ArcGIS REST API
3. NASA EONET — structured wildfire event feed

FPA-FOD (2.3M+ records, 1992–2020) is the definitive source for model training labels.
NIFC/IRWIN provides more recent data. EONET is a lightweight supplement.

Data sources:
- FPA-FOD: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
- NIFC: https://data-nifc.opendata.arcgis.com/
- EONET: https://eonet.gsfc.nasa.gov/api/v3/
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd
import requests

from .config import RAW_DIR, CONUS_BBOX

logger = logging.getLogger(__name__)

NIFC_BASE = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
EONET_BASE = "https://eonet.gsfc.nasa.gov/api/v3"


def load_fpa_fod(sqlite_path: Path | str) -> pd.DataFrame:
    """Load the FPA-FOD wildfire database from its SQLite file.

    The FPA-FOD database must be downloaded manually (~2GB):
    https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6

    Save to data/raw/fpa_fod/FPA_FOD.sqlite

    Args:
        sqlite_path: Path to the FPA_FOD.sqlite file.

    Returns:
        DataFrame with fire occurrence records.
    """
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"FPA-FOD database not found at {sqlite_path}. "
            "Download from https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6 "
            "and place the .sqlite file in data/raw/fpa_fod/"
        )

    logger.info(f"Loading FPA-FOD from {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)

    # Select the most relevant columns for wildfire prediction
    query = """
    SELECT
        FOD_ID,
        FPA_ID,
        FIRE_NAME,
        DISCOVERY_DATE,
        DISCOVERY_DOY,
        DISCOVERY_TIME,
        NWCG_CAUSE_CLASSIFICATION,
        NWCG_GENERAL_CAUSE,
        FIRE_SIZE,
        FIRE_SIZE_CLASS,
        LATITUDE,
        LONGITUDE,
        STATE,
        COUNTY,
        FIPS_CODE,
        FIPS_NAME,
        CONT_DATE,
        CONT_DOY
    FROM Fires
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if "DISCOVERY_DATE" in df.columns:
        df["discovery_date"] = pd.to_datetime(df["DISCOVERY_DATE"], errors="coerce")

    logger.info(f"Loaded {len(df)} fire records from FPA-FOD")
    return df


def fetch_nifc_perimeters(
    year: int | None = None,
    bbox: dict | None = None,
    max_records: int = 2000,
) -> pd.DataFrame:
    """Fetch fire perimeters from NIFC ArcGIS REST API.

    Args:
        year: Filter to a specific year. None for all available.
        bbox: Spatial bounding box filter.
        max_records: Maximum records to return.

    Returns:
        DataFrame of fire perimeter records.
    """
    service = "WFIGS_Interagency_Perimeters"
    url = f"{NIFC_BASE}/{service}/FeatureServer/0/query"

    where_clause = "1=1"
    if year:
        where_clause = f"EXTRACT(YEAR FROM poly_CreateDate) = {year}"

    params = {
        "where": where_clause,
        "outFields": "*",
        "f": "json",
        "resultRecordCount": max_records,
        "returnGeometry": "false",
    }

    if bbox:
        params["geometry"] = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["spatialRel"] = "esriSpatialRelIntersects"
        params["inSR"] = "4326"

    logger.info(f"Fetching NIFC perimeters (year={year})")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])
    records = [f["attributes"] for f in features]

    df = pd.DataFrame(records)
    logger.info(f"Retrieved {len(df)} fire perimeter records")
    return df


def fetch_nifc_incidents(
    max_records: int = 2000,
    active_only: bool = True,
) -> pd.DataFrame:
    """Fetch wildfire incident data from IRWIN via NIFC ArcGIS.

    Args:
        max_records: Maximum records to return.
        active_only: If True, fetch only active incidents.

    Returns:
        DataFrame of incident records.
    """
    service = "WFIGS_Incident_Locations"
    url = f"{NIFC_BASE}/{service}/FeatureServer/0/query"

    where_clause = "1=1"
    if active_only:
        where_clause = "IsActive = 'Y'"

    params = {
        "where": where_clause,
        "outFields": "*",
        "f": "json",
        "resultRecordCount": max_records,
        "returnGeometry": "false",
    }

    logger.info("Fetching NIFC IRWIN incidents")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])
    records = [f["attributes"] for f in features]

    df = pd.DataFrame(records)
    logger.info(f"Retrieved {len(df)} incident records")
    return df


def fetch_eonet_wildfires(
    status: str = "all",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch wildfire events from NASA EONET (no API key needed).

    Args:
        status: 'open', 'closed', or 'all'.
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string.
        limit: Max events to return.

    Returns:
        DataFrame of wildfire events.
    """
    url = f"{EONET_BASE}/events"
    params = {
        "category": "wildfires",
        "limit": limit,
    }
    if status != "all":
        params["status"] = status
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date

    logger.info(f"Fetching EONET wildfire events (status={status})")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    events = response.json().get("events", [])
    records = []
    for event in events:
        # Each event can have multiple geometry entries (tracking over time)
        for geom in event.get("geometry", []):
            coords = geom.get("coordinates", [None, None])
            records.append({
                "event_id": event.get("id"),
                "title": event.get("title"),
                "date": geom.get("date"),
                "longitude": coords[0] if len(coords) >= 2 else None,
                "latitude": coords[1] if len(coords) >= 2 else None,
                "closed": event.get("closed"),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Retrieved {len(df)} EONET wildfire event records")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # EONET doesn't require any API key — good for quick testing
    print("Fetching recent EONET wildfire events...")
    df = fetch_eonet_wildfires(status="open")
    print(f"Active wildfire events: {len(df)}")
    if not df.empty:
        print(df[["title", "date", "latitude", "longitude"]].head(10))

    # Check for FPA-FOD
    fpa_path = RAW_DIR / "fpa_fod" / "FPA_FOD.sqlite"
    if fpa_path.exists():
        print("\nLoading FPA-FOD database...")
        fires = load_fpa_fod(fpa_path)
        print(f"Total fire records: {len(fires)}")
        print(f"Date range: {fires['discovery_date'].min()} to {fires['discovery_date'].max()}")
        print(f"\nCause breakdown:")
        print(fires["NWCG_GENERAL_CAUSE"].value_counts().head(10))
    else:
        print(f"\nFPA-FOD not found at {fpa_path}")
        print("Download from: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6")
