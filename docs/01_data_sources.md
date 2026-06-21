# Data Sources — Wildfire Risk Prediction

This document catalogs all free, publicly available data sources used for wildfire prediction modeling. Each source includes access methods, resolution, coverage, and relevance to the prediction pipeline.

---

## 1. Fire Weather Data

### 1a. NOAA Weather API (api.weather.gov)
- **URL**: https://api.weather.gov
- **Data**: Temperature, relative humidity, wind speed/direction, precipitation, dewpoint
- **Coverage**: United States (NWS forecast zones)
- **Resolution**: Hourly observations, 1–7 day forecasts by grid point
- **Format**: JSON (GeoJSON)
- **Auth**: Free, no API key required (set User-Agent header)
- **Rate Limit**: ~60 requests/minute
- **Relevance**: Core weather variables for fire weather index calculation
- **Endpoints**:
  - `/points/{lat},{lon}` — get forecast office and grid
  - `/gridpoints/{office}/{gridX},{gridY}/forecast/hourly` — hourly forecast
  - `/stations/{stationId}/observations` — historical obs

### 1b. NOAA Integrated Surface Database (ISD)
- **URL**: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
- **Data**: Historical hourly weather observations from ~30,000 global stations
- **Coverage**: Global, dense in US; 1901–present
- **Format**: CSV (fixed-width, also available via BigQuery)
- **Auth**: Free bulk download
- **Relevance**: Historical weather data for model training

### 1c. RAWS (Remote Automated Weather Stations)
- **URL**: https://raws.dri.edu/ and https://fam.nwcg.gov/fam-web/weatherfirecd/
- **Data**: Fire-weather specific: temp, RH, wind, fuel moisture, solar radiation
- **Coverage**: ~2,200 stations in wildland areas across US
- **Format**: CSV
- **Auth**: Free
- **Relevance**: Purpose-built for fire weather; stations located in fire-prone areas

### 1d. gridMET (Climatology Lab)
- **URL**: https://www.climatologylab.org/gridmet.html
- **Data**: Gridded daily surface meteorological data: temp, precip, wind, humidity, fire danger indices (ERC, BI, FM100, FM1000)
- **Coverage**: Contiguous US, 1979–present
- **Resolution**: ~4km daily
- **Format**: NetCDF
- **Auth**: Free via THREDDS/OpenDAP
- **Relevance**: Pre-computed fire danger indices; gridded (no station gaps)

---

## 2. Lightning Data

### 2a. Vaisala GLD360 via NOAA NLDN
- **URL**: https://www.ncei.noaa.gov/products/lightning
- **Data**: Cloud-to-ground lightning strike locations, time, polarity, peak current
- **Coverage**: US, historical archive
- **Format**: CSV
- **Auth**: Free for research via NCEI
- **Relevance**: Lightning is the #1 natural ignition source for wildfires

### 2b. WWLLN (World Wide Lightning Location Network)
- **URL**: http://wwlln.net/
- **Data**: Global lightning stroke locations
- **Coverage**: Global
- **Auth**: Free for academic/research use (apply for access)
- **Relevance**: Backup lightning source with global coverage

### 2c. GLM (Geostationary Lightning Mapper) via GOES
- **URL**: https://www.goes-r.gov/spacesegment/glm.html
- **Data**: Total lightning (cloud-to-cloud and cloud-to-ground)
- **Coverage**: Western hemisphere
- **Resolution**: ~8km, continuous
- **Format**: NetCDF via NOAA CLASS or AWS
- **Auth**: Free
- **Relevance**: Near-real-time lightning data from GOES-16/17/18

---

## 3. Historical Fire Records

### 3a. NIFC Wildland Fire Locations (IRWIN)
- **URL**: https://data-nifc.opendata.arcgis.com/
- **Data**: Fire incidents: location, size, cause, discovery date, containment
- **Coverage**: US, 2014–present (IRWIN); historical back to 1980s via merged datasets
- **Format**: GeoJSON, Shapefile, CSV via ArcGIS REST API
- **Auth**: Free
- **Relevance**: Ground truth labels for model training

### 3b. MTBS (Monitoring Trends in Burn Severity)
- **URL**: https://www.mtbs.gov/
- **Data**: Fire perimeters and burn severity for fires >1,000 acres (West) or >500 acres (East)
- **Coverage**: US, 1984–present
- **Format**: Shapefile, GeoTIFF (burn severity)
- **Auth**: Free
- **Relevance**: High-quality fire boundary and severity data for larger fires

### 3c. FPA-FOD (Fire Program Analysis Fire-Occurrence Database)
- **URL**: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
- **Data**: 2.17M+ georeferenced fire records with cause, size, dates
- **Coverage**: US, 1992–2020
- **Format**: SQLite, CSV
- **Auth**: Free
- **Relevance**: Best single dataset for historical US wildfire occurrences; includes cause codes (lightning, arson, campfire, debris burning, etc.)

---

## 4. Active Fire Detection (Satellite)

### 4a. NASA FIRMS (MODIS / VIIRS)
- **URL**: https://firms.modaps.eosdis.nasa.gov/
- **API**: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{bbox}/{days}
- **Data**: Active fire detections — lat/lon, brightness temp, fire radiative power, confidence
- **Coverage**: Global, 2000–present (MODIS), 2012–present (VIIRS)
- **Resolution**: 375m (VIIRS), 1km (MODIS); multiple daily passes
- **Format**: CSV, SHP, GeoJSON, KML
- **Auth**: Free MAP_KEY (register at FIRMS)
- **Rate Limit**: 10 requests/minute for NRT data
- **Relevance**: Near-real-time fire hotspot detection; validation data for predictions

### 4b. GOES Active Fire Product
- **URL**: https://www.ospo.noaa.gov/products/land/fire.html
- **Data**: Fire detection and characterization from geostationary orbit
- **Coverage**: Western hemisphere
- **Resolution**: 2km, every 5–15 minutes
- **Format**: NetCDF
- **Auth**: Free via NOAA CLASS or AWS
- **Relevance**: Highest temporal resolution fire detection

---

## 5. Air Quality

### 5a. OpenAQ
- **URL**: https://api.openaq.org/v3/
- **Data**: Real-time and historical air quality — PM2.5, PM10, O3, NO2, SO2, CO
- **Coverage**: Global, 70,000+ locations
- **Format**: JSON REST API
- **Auth**: Free API key (register)
- **Relevance**: Smoke from fires dramatically increases PM2.5; early AQ changes can signal nearby fire activity

### 5b. EPA AQS (Air Quality System)
- **URL**: https://aqs.epa.gov/aqsweb/documents/data_api.html
- **Data**: Regulatory-grade AQ measurements from ~4,000 US monitoring stations
- **Coverage**: US, 1980–present
- **Format**: JSON REST API
- **Auth**: Free (register email + key)
- **Relevance**: High-quality historical PM2.5/AQI data for training

### 5c. AirNow API
- **URL**: https://docs.airnowapi.org/
- **Data**: Real-time AQI, forecasts, current observations
- **Coverage**: US, Canada, Mexico
- **Format**: JSON/XML
- **Auth**: Free API key
- **Relevance**: Real-time AQ feed for dashboard

### 5d. HMS (Hazard Mapping System) Smoke Product
- **URL**: https://www.ospo.noaa.gov/products/land/hms.html
- **Data**: Analyst-drawn smoke plume polygons from satellite imagery
- **Coverage**: US
- **Format**: Shapefile, KML (daily)
- **Auth**: Free
- **Relevance**: Direct smoke detection — bridges satellite fire data and AQ data

---

## 6. Soil Moisture & Drought

### 6a. NASA SMAP (Soil Moisture Active Passive)
- **URL**: https://nsidc.org/data/smap
- **Data**: Surface soil moisture (top 5cm) and root-zone soil moisture
- **Coverage**: Global, 2015–present
- **Resolution**: 9km, every 2–3 days
- **Format**: HDF5
- **Auth**: Free (NASA Earthdata login)
- **Relevance**: Dry soil = dry fuel = higher fire risk

### 6b. US Drought Monitor
- **URL**: https://droughtmonitor.unl.edu/DmData/DataDownload.aspx
- **Data**: Weekly drought severity classifications (D0–D4)
- **Coverage**: US, 2000–present
- **Format**: Shapefile, CSV
- **Auth**: Free
- **Relevance**: Drought conditions strongly correlate with fire risk

---

## 7. Vegetation & Fuel

### 7a. LANDFIRE
- **URL**: https://landfire.gov/
- **Data**: Vegetation type, fuel models (13/40 fuel models), canopy cover, height, bulk density
- **Coverage**: US
- **Resolution**: 30m
- **Format**: GeoTIFF
- **Auth**: Free
- **Relevance**: Fuel characteristics determine fire behavior and spread potential

### 7b. MODIS NDVI (MOD13A2)
- **URL**: https://lpdaac.usgs.gov/products/mod13a2v061/
- **Data**: Normalized Difference Vegetation Index (greenness/dryness)
- **Coverage**: Global, 2000–present
- **Resolution**: 1km, 16-day composites
- **Format**: HDF4/GeoTIFF via AppEEARS
- **Auth**: Free (NASA Earthdata login)
- **Relevance**: Vegetation stress/dryness indicates fire susceptibility

---

## 8. Topography

### 8a. USGS 3DEP (3D Elevation Program)
- **URL**: https://apps.nationalmap.gov/downloader/
- **Data**: Digital Elevation Model — elevation, from which slope and aspect are derived
- **Coverage**: US
- **Resolution**: 1/3 arc-second (~10m), 1 arc-second (~30m)
- **Format**: GeoTIFF
- **Auth**: Free
- **Relevance**: Slope and aspect affect fire spread; south-facing slopes dry faster

### 8b. SRTM (Shuttle Radar Topography Mission)
- **URL**: https://dwtkns.com/srtm30m/ or via USGS EarthExplorer
- **Data**: Global DEM
- **Resolution**: 30m
- **Format**: GeoTIFF/HDF
- **Auth**: Free
- **Relevance**: Backup global elevation source

---

## 9. Social / Early Reports

### 9a. NASA EONET (Earth Observatory Natural Event Tracker)
- **URL**: https://eonet.gsfc.nasa.gov/api/v3/events?category=wildfires
- **Data**: Curated natural event reports including wildfires
- **Format**: JSON
- **Auth**: Free, no key
- **Relevance**: Structured wildfire event feed

### 9b. InciWeb (Incident Information System)
- **URL**: https://inciweb.wildfire.gov/
- **Data**: Active large fire incident reports with narratives
- **Format**: Web scraping / RSS
- **Auth**: Free
- **Relevance**: Human-reported incident details

### 9c. GDELT Project
- **URL**: https://www.gdeltproject.org/
- **Data**: Global news event monitoring — can filter for fire-related articles
- **Format**: CSV (BigQuery)
- **Auth**: Free
- **Relevance**: Early news signals of fire outbreaks; proxy for social reports

---

## Priority Data Sources for Initial Model

For the first iteration, prioritize these based on accessibility and signal strength:

| Priority | Source | Variable | Why |
|----------|--------|----------|-----|
| 1 | FPA-FOD | Fire labels (target) | Best historical fire occurrence dataset |
| 2 | gridMET | Weather + fire indices | Pre-gridded, includes fire danger indices |
| 3 | NASA FIRMS | Active fire detections | Validation + additional labels |
| 4 | LANDFIRE | Fuel types | Static but critical fire behavior input |
| 5 | USGS 3DEP | Topography | Static terrain features |
| 6 | EPA AQS / OpenAQ | Air quality | Smoke as early signal |
| 7 | NOAA NLDN / GLM | Lightning | Natural ignition source |
| 8 | NASA SMAP | Soil moisture | Fuel dryness proxy |
| 9 | US Drought Monitor | Drought severity | Seasonal fire risk context |
