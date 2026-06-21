"""
Orchestrate data fetching from all sources.

Run with: python -m src.data_acquisition.fetch_all

This script coordinates downloads across all data sources, respecting
rate limits and caching previously downloaded data.
"""

import argparse
import logging

from .config import REGIONS, DEFAULT_START_YEAR, DEFAULT_END_YEAR, RAW_DIR
from .fetch_fire_history import fetch_eonet_wildfires, fetch_nifc_incidents, load_fpa_fod
from .fetch_gridmet import fetch_gridmet_range
from .fetch_firms import fetch_firms_area
from .fetch_weather import fetch_nws_red_flag_warnings
from .fetch_air_quality import fetch_openaq_locations

logger = logging.getLogger(__name__)


def fetch_no_auth_sources():
    """Fetch data from sources that require no API keys.

    Good for initial setup and testing the pipeline.
    """
    print("=" * 60)
    print("FETCHING DATA FROM NO-AUTH SOURCES")
    print("=" * 60)

    # 1. EONET wildfire events
    print("\n[1/3] NASA EONET wildfire events...")
    try:
        df = fetch_eonet_wildfires(status="all", limit=500)
        out = RAW_DIR / "eonet" / "wildfire_events.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  -> {len(df)} events saved to {out}")
    except Exception as e:
        print(f"  -> Failed: {e}")

    # 2. NIFC active incidents
    print("\n[2/3] NIFC IRWIN incidents...")
    try:
        df = fetch_nifc_incidents(active_only=False, max_records=5000)
        out = RAW_DIR / "nifc" / "irwin_incidents.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  -> {len(df)} incidents saved to {out}")
    except Exception as e:
        print(f"  -> Failed: {e}")

    # 3. NWS Red Flag Warnings
    print("\n[3/3] NWS Red Flag Warnings...")
    try:
        df = fetch_nws_red_flag_warnings()
        out = RAW_DIR / "weather" / "red_flag_warnings.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  -> {len(df)} active warnings saved to {out}")
    except Exception as e:
        print(f"  -> Failed: {e}")


def fetch_gridmet_data(region: str = "california", start_year: int = 2020, end_year: int = 2023):
    """Fetch gridMET weather/fire-danger data (no API key needed, but large downloads)."""
    print("=" * 60)
    print(f"FETCHING gridMET DATA: {region} {start_year}-{end_year}")
    print("=" * 60)

    bbox = REGIONS.get(region)
    if not bbox:
        print(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")
        return

    # Start with fire danger indices (most directly useful)
    priority_vars = ["erc", "bi", "vpd", "fm100", "tmmx", "rmin", "vs", "pr"]
    fetch_gridmet_range(start_year, end_year, bbox, variables=priority_vars)


def fetch_all(region: str = "california"):
    """Fetch all available data sources."""
    # Phase 1: No-auth sources (always available)
    fetch_no_auth_sources()

    # Phase 2: gridMET (no auth but larger downloads)
    fetch_gridmet_data(region, 2020, 2023)

    # Phase 3: Sources requiring API keys (skip if not configured)
    print("\n" + "=" * 60)
    print("SOURCES REQUIRING API KEYS")
    print("=" * 60)

    from .config import NASA_FIRMS_MAP_KEY, NOAA_API_KEY, EPA_AQS_KEY

    if NASA_FIRMS_MAP_KEY:
        print("\n[FIRMS] Fetching active fire detections...")
        try:
            bbox = REGIONS.get(region, REGIONS["california"])
            df = fetch_firms_area("viirs_noaa20", bbox=bbox, days=10)
            out = RAW_DIR / "firms" / f"firms_recent_{region}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"  -> {len(df)} detections saved")
        except Exception as e:
            print(f"  -> Failed: {e}")
    else:
        print("\n[FIRMS] Skipped — set NASA_FIRMS_MAP_KEY in .env")

    if NOAA_API_KEY:
        print("[NOAA CDO] API key found — use fetch_weather.py for station data")
    else:
        print("[NOAA CDO] Skipped — set NOAA_API_KEY in .env")

    if EPA_AQS_KEY:
        print("[EPA AQS] API key found — use fetch_air_quality.py for PM2.5 data")
    else:
        print("[EPA AQS] Skipped — set EPA_AQS_EMAIL and EPA_AQS_KEY in .env")

    # Phase 4: Manual downloads
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOADS REQUIRED")
    print("=" * 60)
    fpa_path = RAW_DIR / "fpa_fod" / "FPA_FOD.sqlite"
    if fpa_path.exists():
        print(f"[FPA-FOD] Found at {fpa_path}")
    else:
        print("[FPA-FOD] Download the SQLite file from:")
        print("  https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6")
        print(f"  Save to: {fpa_path}")

    print("[LANDFIRE] Download fuel model rasters from:")
    print("  https://landfire.gov/viewer/")
    print(f"  Save to: {RAW_DIR / 'landfire/'}")

    print("[USGS 3DEP] Download elevation data from:")
    print("  https://apps.nationalmap.gov/downloader/")
    print(f"  Save to: {RAW_DIR / 'elevation/'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch wildfire prediction data")
    parser.add_argument("--region", default="california", choices=list(REGIONS.keys()))
    parser.add_argument("--no-auth-only", action="store_true",
                        help="Only fetch sources that don't require API keys")
    args = parser.parse_args()

    if args.no_auth_only:
        fetch_no_auth_sources()
    else:
        fetch_all(args.region)
