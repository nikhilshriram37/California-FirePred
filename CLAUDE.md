# FireProject — AI-Powered Wildfire Risk Prediction & Detection

## Project Overview

This project builds an end-to-end wildfire risk prediction system using machine learning to process and combine multiple data streams (weather, lightning, air quality, satellite imagery, historical fire records, and social reports). The system has two main deliverables:

1. **ML Prediction Model** — Trained on historical multi-source data to predict wildfire outbreak probability
2. **Live Dashboard** — Geographic map with real-time data feeds showing risk zones (red/yellow/green) for forward-looking fire prediction

## Architecture

```
FireProject/
├── CLAUDE.md              # This file — project context and conventions
├── docs/                  # Research, data source documentation, design docs
├── data/
│   ├── raw/               # Unprocessed data downloads (gitignored)
│   ├── processed/         # Cleaned, feature-engineered datasets
│   └── external/          # Third-party reference data
├── src/
│   ├── data_acquisition/  # Scripts to fetch data from APIs
│   ├── preprocessing/     # Data cleaning, normalization, feature engineering
│   ├── models/            # Model training, evaluation, comparison
│   ├── visualization/     # Heat maps, charts, EDA plots
│   └── dashboard/         # Frontend live dashboard (map + feeds)
├── notebooks/             # Jupyter notebooks for exploration and model dev
├── configs/               # API keys, model hyperparameters, settings
└── tests/                 # Unit and integration tests
```

## Data Sources (Free / Public)

| Category | Source | API/Format | Notes |
|----------|--------|------------|-------|
| Fire Weather | NOAA Weather API | REST / JSON | Temperature, humidity, wind, precip |
| Fire Weather Index | NOAA NFDRS / RAWS | Bulk CSV | Fire danger ratings |
| Lightning | Vaisala GLD360 / NLDN (via NOAA) | Bulk | Strike location/time |
| Historical Fires | NIFC / MTBS / IRWIN | Shapefile/GeoJSON | Fire perimeters, ignition causes |
| Active Fires | NASA FIRMS (MODIS/VIIRS) | REST CSV/JSON | Near-real-time hotspots |
| Soil Moisture | NASA SMAP | HDF5/NetCDF | Fuel dryness proxy |
| Air Quality | OpenAQ / EPA AQS | REST JSON | PM2.5, AQI, smoke detection |
| Vegetation | LANDFIRE / MODIS NDVI | GeoTIFF/HDF | Fuel load and greenness |
| Topography | USGS 3DEP | GeoTIFF | Elevation, slope, aspect |
| Social Reports | Twitter/X API, EONET | REST JSON | Early crowd-sourced signals |

## Model Strategy

Compare 3–4 candidate algorithms, then ensemble or select the best:

1. **Random Forest** — Baseline; handles mixed feature types well
2. **XGBoost / LightGBM** — Gradient boosting for tabular data; strong benchmark
3. **LSTM / Temporal CNN** — Capture time-series patterns in weather sequences
4. **Logistic Regression** — Interpretable baseline for comparison

### Evaluation Metrics
- **Primary**: AUC-ROC (handles class imbalance — fires are rare events)
- **Secondary**: Precision, Recall, F1-score
- **Operational**: False alarm rate, lead time before fire detection

## Key Conventions

### Python
- Python 3.10+
- Use `pyproject.toml` for dependency management
- Key libraries: pandas, scikit-learn, xgboost, lightgbm, tensorflow/pytorch, geopandas, folium, plotly, rasterio
- Type hints on public function signatures
- Docstrings on modules and non-trivial functions

### Notebooks
- Notebooks are for exploration, visualization, and model comparison
- Production logic lives in `src/` and is imported into notebooks
- Name notebooks with numeric prefix: `01_eda.ipynb`, `02_feature_engineering.ipynb`, etc.

### Data
- Raw data is never modified in place — always write to `processed/`
- Large data files are gitignored; document download steps in `docs/`
- API keys go in `.env` (gitignored) and are loaded via `python-dotenv`

### Dashboard
- Map-based frontend with geographic risk overlay
- Risk levels: Red (high risk), Yellow (potential risk), Green (clear)
- Forward-looking prediction window (configurable, e.g., 24h / 48h / 7d)

## Running the Project

```bash
# Install dependencies
pip install -e ".[dev]"

# Fetch data
python -m src.data_acquisition.fetch_all

# Run notebooks
jupyter lab notebooks/

# Start dashboard (once built)
python -m src.dashboard.app
```

## Environment Variables

Store in `.env` at project root (gitignored):
```
NOAA_API_KEY=...
NASA_EARTHDATA_TOKEN=...
OPENAQ_API_KEY=...
EPA_AQS_EMAIL=...
EPA_AQS_KEY=...
```
