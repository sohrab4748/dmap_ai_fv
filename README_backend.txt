DMAP-AI / FV backend package

Files:
- main.py           FastAPI backend for Render
- requirements.txt  Python dependencies
- render.yaml       Render deployment file

Environment variables you should set on Render:
- GEE_PROJECT_ID=dmapaifv
- GEE_SERVICE_ACCOUNT_JSON={...service account json...}
- CLIMATE_ENGINE_API_KEY=...optional, only for 4-week forecast...
- CORS_ALLOW_ORIGINS=https://your-dotnet-site.com

What the backend returns:
- weather_daily      Historical/current daily GRIDMET weather from GEE
- drought_latest     Latest GRIDMET/DROUGHT indices from GEE
- noaa_monitoring    Current NOAA EDDI monitoring values (04wk / 01mn)
- forecast_4week     Optional Climate Engine CFS_GRIDMET rows, if key exists

Main endpoint:
POST /fv/point
Body example:
{
  "lat": 42.03,
  "lon": -93.63,
  "history_days": 60,
  "include_weather_history": true,
  "include_drought_latest": true,
  "include_noaa_monitoring": true,
  "include_forecast_4week": true,
  "workload_tag": "fv-point-request"
}
