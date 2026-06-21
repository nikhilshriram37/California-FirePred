# API Reference — Data Source Endpoints & Access Details

Quick-reference for all API endpoints, authentication, and rate limits used in the data acquisition pipeline.

---

## Registration Checklist (Do This First)

| Service | Registration URL | What You Get |
|---------|-----------------|--------------|
| NOAA CDO | https://www.ncdc.noaa.gov/cdo-web/token | API token |
| NASA Earthdata | https://urs.earthdata.nasa.gov/users/new | Login + Bearer token |
| NASA FIRMS | https://firms.modaps.eosdis.nasa.gov/api/area/ | MAP_KEY |
| EPA AQS | https://aqs.epa.gov/aqsweb/documents/data_api.html | Email + key pair |
| OpenAQ | https://api.openaq.org/register | API key |
| AirNow | https://docs.airnowapi.org/ | API key |
| Synoptic (RAWS) | https://synopticdata.com/ | API token (free tier) |

---

## Endpoint Quick Reference

### NOAA CDO API
```
Base: https://www.ncdc.noaa.gov/cdo-web/api/v2/
Auth: Header "token: {NOAA_API_KEY}"
Rate: 5 req/sec, 10,000 req/day

GET /data?datasetid=GHCND&datatypeid=TMAX,TMIN,PRCP,AWND&locationid=FIPS:06&startdate=2020-01-01&enddate=2020-12-31
GET /stations?datasetid=GHCND&locationid=FIPS:06
```

### NWS API (No Key Required)
```
Base: https://api.weather.gov/
Auth: User-Agent header (e.g., "FireProject, contact@email.com")

GET /points/{lat},{lon}
GET /gridpoints/{office}/{gridX},{gridY}/forecast/hourly
GET /stations/{stationId}/observations?start=2024-01-01&end=2024-01-31
GET /alerts/active?event=Red+Flag+Warning
```

### gridMET (OPeNDAP — No Key)
```
Base: https://thredds.northwestknowledge.net/thredds/reacch_climate_MET_catalog.html
Variables: tmmx, tmmn, rmax, rmin, vs (wind), pr (precip), erc, bi, fm100, fm1000

Python (xarray):
  ds = xarray.open_dataset("https://thredds.northwestknowledge.net/thredds/dodsC/MET/tmmx/tmmx_2023.nc")
  ds.sel(lat=slice(42,38), lon=slice(-124,-120), day=slice("2023-06-01","2023-09-30"))
```

### NASA FIRMS
```
Base: https://firms.modaps.eosdis.nasa.gov/api/
Auth: MAP_KEY in URL path

GET /area/csv/{MAP_KEY}/VIIRS_NOAA20_NRT/{west},{south},{east},{north}/{days}
GET /country/csv/{MAP_KEY}/VIIRS_NOAA20_NRT/USA/1

Archive: https://firms.modaps.eosdis.nasa.gov/download/
Rate: 10 transactions/min
```

### NASA SMAP (via Earthdata)
```
NSIDC DAAC: https://n5eil02u.ecs.nsidc.org/
Product: SPL3SMP_E (9km enhanced daily soil moisture)
Auth: NASA Earthdata Bearer token

OPeNDAP subsetting available for server-side spatial/temporal filtering
```

### EPA AQS API
```
Base: https://aqs.epa.gov/data/api/
Auth: Query params email= & key=

GET /dailyData/byState?email={}&key={}&param=88101&bdate=20200101&edate=20201231&state=06
Parameter codes: 88101 (PM2.5 FRM), 44201 (O3), 42401 (SO2), 42101 (CO)
```

### OpenAQ v3
```
Base: https://api.openaq.org/v3/
Auth: Header "X-API-Key: {OPENAQ_API_KEY}"

GET /locations?coordinates={lat},{lon}&radius=50000&parameter=pm25
GET /locations/{id}/measurements?date_from=2024-01-01&date_to=2024-06-01
```

### GOES GLM Lightning (AWS S3 — No Key)
```
S3: s3://noaa-goes16/GLM-L2-LCFA/{year}/{dayofyear}/{hour}/
HTTP: https://noaa-goes16.s3.amazonaws.com/GLM-L2-LCFA/{year}/{dayofyear}/{hour}/

aws s3 ls --no-sign-request s3://noaa-goes16/GLM-L2-LCFA/2024/180/
Format: NetCDF4
```

### GOES ABI Fire Product (AWS S3 — No Key)
```
S3: s3://noaa-goes16/ABI-L2-FDCC/{year}/{dayofyear}/{hour}/
CONUS fire detection, 5-minute refresh
Format: NetCDF4
```

### NIFC / IRWIN (ArcGIS REST — No Key)
```
Base: https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/

Fire Perimeters:
GET /NIFC_Perimeters/FeatureServer/0/query?where=1=1&outFields=*&f=geojson&resultRecordCount=1000

IRWIN Incidents:
GET /IRWIN_Incidents/FeatureServer/0/query?where=1=1&outFields=*&f=geojson
```

### EONET (No Key)
```
Base: https://eonet.gsfc.nasa.gov/api/v3/

GET /events?category=wildfires&status=open
GET /events?category=wildfires&start=2024-01-01&end=2024-06-30
```

### GDELT (No Key)
```
Base: https://api.gdeltproject.org/api/v2/

GET /doc/doc?query=wildfire&mode=artlist&format=json&maxrecords=250&startdatetime=20240601000000&enddatetime=20240630235959
```

### USGS 3DEP / TNM
```
Base: https://tnmaccess.nationalmap.gov/api/v1/

GET /products?datasets=National+Elevation+Dataset+(NED)+1/3+arc-second&bbox=-120,37,-119,38&prodFormats=GeoTIFF
```

### LANDFIRE
```
Download: https://landfire.gov/viewer/ (interactive selection)
Bulk: https://landfire.gov/data (product selection)
LFPS: https://lfps.usgs.gov/arcgis/rest/services/
Products: FBFM40 (fuel models), EVC (canopy cover), EVT (veg type)
```

### FPA-FOD (Direct Download)
```
URL: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
Format: SQLite (~2GB), CSV
Contains: 2.3M+ wildfire records, 1992–2020
```

### US Drought Monitor
```
Download: https://droughtmonitor.unl.edu/DmData/DataDownload.aspx
Format: Shapefile (weekly), CSV (county/state level)
Coverage: 2000–present
```
