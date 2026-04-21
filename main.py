import json
import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import ee
import google.auth
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmap_ai_fv_backend")

APP_TITLE = "DMAP-AI / FV Backend"
APP_VERSION = "0.1.0"

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "dmapaifv")
GEE_SERVICE_ACCOUNT_JSON = os.getenv("GEE_SERVICE_ACCOUNT_JSON", "").strip()
CLIMATE_ENGINE_API_KEY = os.getenv("CLIMATE_ENGINE_API_KEY", "").strip()
CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]

WEATHER_BANDS = [
    "pr", "tmmx", "tmmn", "eto", "vpd", "vs", "srad", "rmax", "rmin", "sph"
]
DROUGHT_BANDS = [
    "eddi14d", "eddi30d", "eddi90d",
    "spi14d", "spi30d", "spi90d",
    "pdsi", "z"
]
FORECAST_VARIABLES = ["eddi", "pr", "tmmx", "tmmn", "eto"]
FORECAST_WEEKS = [
    ("week1", "day01", "day07"),
    ("week2", "day08", "day14"),
    ("week3", "day15", "day21"),
    ("week4", "day22", "day28"),
]

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_GEE_INITIALIZED = False
_GEE_INIT_ERROR: Optional[str] = None


class FVPointRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    history_days: int = Field(60, ge=7, le=3650)
    include_weather_history: bool = True
    include_drought_latest: bool = True
    include_noaa_monitoring: bool = True
    include_forecast_4week: bool = True
    workload_tag: Optional[str] = None


class FVPointResponse(BaseModel):
    meta: Dict[str, Any]
    weather_daily: Optional[List[Dict[str, Any]]] = None
    drought_latest: Optional[Dict[str, Any]] = None
    noaa_monitoring: Optional[Dict[str, Any]] = None
    forecast_4week: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = []


def _normalize_climate_engine_key(value: str) -> str:
    key = value.strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def initialize_earth_engine() -> None:
    global _GEE_INITIALIZED, _GEE_INIT_ERROR
    if _GEE_INITIALIZED:
        return
    try:
        if GEE_SERVICE_ACCOUNT_JSON:
            info = json.loads(GEE_SERVICE_ACCOUNT_JSON)
            creds = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            ee.Initialize(creds, project=GEE_PROJECT_ID)
            logger.info("Earth Engine initialized with service account for project %s", GEE_PROJECT_ID)
        else:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            ee.Initialize(creds, project=GEE_PROJECT_ID)
            logger.info("Earth Engine initialized with ADC for project %s", GEE_PROJECT_ID)
        _GEE_INITIALIZED = True
        _GEE_INIT_ERROR = None
    except Exception as exc:  # pragma: no cover
        _GEE_INITIALIZED = False
        _GEE_INIT_ERROR = str(exc)
        logger.exception("Earth Engine initialization failed")
        raise


@app.on_event("startup")
def on_startup() -> None:
    try:
        initialize_earth_engine()
    except Exception:
        # Keep app alive so /health can surface the error.
        pass


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "gee_project_id": GEE_PROJECT_ID,
        "gee_initialized": _GEE_INITIALIZED,
        "forecast_provider": "Climate Engine" if _normalize_climate_engine_key(CLIMATE_ENGINE_API_KEY) else None,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _GEE_INITIALIZED else "degraded",
        "gee_initialized": _GEE_INITIALIZED,
        "gee_error": _GEE_INIT_ERROR,
    }


@app.post("/fv/point", response_model=FVPointResponse)
def fv_point(req: FVPointRequest) -> FVPointResponse:
    initialize_earth_engine()
    if not _GEE_INITIALIZED:
        raise HTTPException(status_code=500, detail=f"Earth Engine not initialized: {_GEE_INIT_ERROR}")

    workload_tag = req.workload_tag or "fv-point-request"
    ee.data.setDefaultWorkloadTag(workload_tag)

    warnings: List[str] = []
    weather_daily = None
    drought_latest = None
    noaa_monitoring = None
    forecast_4week = None

    try:
        if req.include_weather_history:
            weather_daily = extract_weather_timeseries(req.lat, req.lon, req.history_days)
        if req.include_drought_latest:
            drought_latest = extract_latest_drought(req.lat, req.lon)
        if req.include_noaa_monitoring:
            noaa_monitoring = fetch_noaa_monitoring_eddi(req.lat, req.lon)
        if req.include_forecast_4week:
            key = _normalize_climate_engine_key(CLIMATE_ENGINE_API_KEY)
            if key:
                forecast_4week = fetch_climate_engine_forecast(req.lat, req.lon, key)
            else:
                warnings.append("CLIMATE_ENGINE_API_KEY is not set. forecast_4week was skipped.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("FV point request failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return FVPointResponse(
        meta={
            "lat": req.lat,
            "lon": req.lon,
            "history_days": req.history_days,
            "gee_project_id": GEE_PROJECT_ID,
            "workload_tag": workload_tag,
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        },
        weather_daily=weather_daily,
        drought_latest=drought_latest,
        noaa_monitoring=noaa_monitoring,
        forecast_4week=forecast_4week,
        warnings=warnings,
    )


def _point_geometry(lat: float, lon: float) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat])


def _kelvin_to_celsius(v: Any) -> Optional[float]:
    try:
        return round(float(v) - 273.15, 3)
    except Exception:
        return None


def extract_weather_timeseries(lat: float, lon: float, history_days: int) -> List[Dict[str, Any]]:
    point = _point_geometry(lat, lon)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=history_days)

    ic = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(str(start_date), str(end_date + timedelta(days=1)))
        .select(WEATHER_BANDS)
    )

    def image_to_feature(img):
        vals = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=4000,
            maxPixels=1_000_000,
        )
        vals = vals.set("date", img.date().format("YYYY-MM-dd"))
        return ee.Feature(None, vals)

    fc = ee.FeatureCollection(ic.map(image_to_feature))
    features = fc.getInfo().get("features", [])
    rows = [feat.get("properties", {}) for feat in features]
    rows.sort(key=lambda r: r.get("date", ""))

    for row in rows:
        if "tmmx" in row:
            row["tmmx_c"] = _kelvin_to_celsius(row.get("tmmx"))
        if "tmmn" in row:
            row["tmmn_c"] = _kelvin_to_celsius(row.get("tmmn"))
    return rows


def extract_latest_drought(lat: float, lon: float) -> Dict[str, Any]:
    point = _point_geometry(lat, lon)
    ic = ee.ImageCollection("GRIDMET/DROUGHT").select(DROUGHT_BANDS)
    latest = ic.sort("system:time_start", False).first()
    values = latest.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=4000,
        maxPixels=1_000_000,
    ).getInfo()
    image_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    out = {"date": image_date}
    out.update(values or {})
    return out


def fetch_noaa_monitoring_eddi(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://mapservices.weather.noaa.gov/experimental/rest/services/eddi_monitoring/MapServer/identify"
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "all",
        "tolerance": "3",
        "mapExtent": f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}",
        "imageDisplay": "800,600,96",
        "returnGeometry": "false",
        "f": "json",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    out: Dict[str, Any] = {}
    for result in data.get("results", []):
        if result.get("layerName") != "Image":
            continue
        attr = result.get("attributes", {})
        name = attr.get("name")
        val = attr.get("Service Pixel Value")
        if name and ("04wk" in name or "01mn" in name):
            out[name] = val
    return out


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _flatten_climate_engine_payload(payload: Any, variable: str, week_name: str, start_day: str, end_day: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_row(item: Dict[str, Any]) -> None:
        row = dict(item)
        row["variable"] = variable
        row["week"] = week_name
        row["start_day"] = start_day
        row["end_day"] = end_day
        rows.append(row)

    top_items = payload if isinstance(payload, list) else [payload]
    for item in top_items:
        if isinstance(item, dict) and "Data" in item and isinstance(item["Data"], list):
            for sub in item["Data"]:
                if isinstance(sub, dict):
                    add_row(sub)
        elif isinstance(item, dict):
            add_row(item)
        else:
            rows.append({
                "value": item,
                "variable": variable,
                "week": week_name,
                "start_day": start_day,
                "end_day": end_day,
            })
    return rows


def fetch_climate_engine_forecast(lat: float, lon: float, api_key: str) -> List[Dict[str, Any]]:
    url = "https://api.climateengine.org/timeseries/native/forecasts/coordinates"
    headers = {
        "Authorization": api_key,
        "Accept": "application/json",
        "User-Agent": "dmap-ai-fv-backend/1.0",
    }

    all_rows: List[Dict[str, Any]] = []
    for variable in FORECAST_VARIABLES:
        for week_name, start_day, end_day in FORECAST_WEEKS:
            params = {
                "dataset": "CFS_GRIDMET",
                "variable": variable,
                "start_day": start_day,
                "end_day": end_day,
                "coordinates": _compact_json([[lon, lat]]),
                "model": "ens_mean",
                "export_format": "json",
            }
            r = requests.get(url, params=params, headers=headers, timeout=120)
            if r.status_code == 401:
                raise HTTPException(status_code=500, detail="Climate Engine API key is invalid.")
            r.raise_for_status()
            payload = r.json()
            rows = _flatten_climate_engine_payload(payload, variable, week_name, start_day, end_day)
            all_rows.extend(rows)

    # Normalize row keys across variables for easier frontend use.
    if not all_rows:
        return []

    df = pd.DataFrame(all_rows)
    if df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict(orient="records")
