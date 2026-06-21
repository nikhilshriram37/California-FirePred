"""Configuration for data acquisition — API endpoints, default parameters, and regions."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env.local")

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API Keys (loaded from .env) ---
NOAA_API_KEY = os.getenv("NOAA_API_KEY", "")
NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")
NASA_FIRMS_MAP_KEY = os.getenv("NASA_FIRMS_MAP_KEY", "")
EPA_AQS_EMAIL = os.getenv("EPA_AQS_EMAIL", "")
EPA_AQS_KEY = os.getenv("EPA_AQS_KEY", "")
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")

# --- Default Geographic Bounds (Contiguous US) ---
CONUS_BBOX = {
    "west": -125.0,
    "south": 24.5,
    "east": -66.5,
    "north": 49.5,
}

# High-priority fire-prone regions for focused analysis
REGIONS = {
    "california": {"west": -124.5, "south": 32.5, "east": -114.0, "north": 42.0},
    "pacific_northwest": {"west": -125.0, "south": 42.0, "east": -116.5, "north": 49.0},
    "northern_rockies": {"west": -117.0, "south": 42.0, "east": -104.0, "north": 49.0},
    "southwest": {"west": -115.0, "south": 31.0, "east": -103.0, "north": 37.0},
    "southeast": {"west": -88.0, "south": 25.0, "east": -80.0, "north": 35.0},
    "great_plains": {"west": -104.0, "south": 26.0, "east": -94.0, "north": 40.0},
}

# --- Default Time Range for Historical Data ---
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2023

# --- API Endpoints ---
ENDPOINTS = {
    "noaa_cdo": "https://www.ncdc.noaa.gov/cdo-web/api/v2",
    "nws": "https://api.weather.gov",
    "gridmet_thredds": "https://thredds.northwestknowledge.net/thredds/dodsC/MET",
    "firms": "https://firms.modaps.eosdis.nasa.gov/api",
    "epa_aqs": "https://aqs.epa.gov/data/api",
    "openaq": "https://api.openaq.org/v3",
    "eonet": "https://eonet.gsfc.nasa.gov/api/v3",
    "nifc_arcgis": "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services",
    "goes_s3": "https://noaa-goes16.s3.amazonaws.com",
    "usgs_tnm": "https://tnmaccess.nationalmap.gov/api/v1",
    "gdelt": "https://api.gdeltproject.org/api/v2",
}

# --- gridMET Variable Names ---
GRIDMET_VARS = {
    "tmmx": "max_temperature",
    "tmmn": "min_temperature",
    "rmax": "max_relative_humidity",
    "rmin": "min_relative_humidity",
    "vs": "wind_speed",
    "pr": "precipitation",
    "erc": "energy_release_component",
    "bi": "burning_index",
    "fm100": "100hr_dead_fuel_moisture",
    "fm1000": "1000hr_dead_fuel_moisture",
    "vpd": "vapor_pressure_deficit",
}
