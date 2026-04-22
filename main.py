import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import ee
import google.auth
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmap_ai_fv_backend")

APP_TITLE = "DMAP-AI / FV Backend"
APP_VERSION = "0.4.2-gee-only-cfsv2-nullsafe"

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "dmapaifv")
GEE_SERVICE_ACCOUNT_JSON = os.getenv("GEE_SERVICE_ACCOUNT_JSON", "").strip()

_raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
if _raw_origins == "*":
    CORS_ALLOW_ORIGINS = ["*"]
else:
    CORS_ALLOW_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

WEATHER_BANDS = [
    "pr", "tmmx", "tmmn", "eto", "vpd", "vs", "srad", "rmax", "rmin", "sph"
]
DROUGHT_BANDS = [
    "eddi14d", "eddi30d", "eddi90d",
    "spi14d", "spi30d", "spi90d",
    "pdsi", "z"
]
CFS_BANDS = [
    "Temperature_height_above_ground",
    "Maximum_temperature_height_above_ground_6_Hour_Interval",
    "Minimum_temperature_height_above_ground_6_Hour_Interval",
    "Precipitation_rate_surface_6_Hour_Average",
    "Potential_Evaporation_Rate_surface_6_Hour_Average",
    "u-component_of_wind_height_above_ground",
    "v-component_of_wind_height_above_ground",
    "Volumetric_Soil_Moisture_Content_depth_below_surface_layer_5_cm",
    "Volumetric_Soil_Moisture_Content_depth_below_surface_layer_25_cm",
    "Volumetric_Soil_Moisture_Content_depth_below_surface_layer_70_cm",
    "Volumetric_Soil_Moisture_Content_depth_below_surface_layer_150_cm",
]
SECONDS_PER_6H = 6 * 60 * 60
LATENT_HEAT_VAPORIZATION_J_PER_KG = 2.45e6
MAX_FORECAST_DAYS = 28

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False if CORS_ALLOW_ORIGINS == ["*"] else True,
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
    forecast_4week: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)


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
    except Exception as exc:
        _GEE_INITIALIZED = False
        _GEE_INIT_ERROR = str(exc)
        logger.exception("Earth Engine initialization failed")
        raise


@app.on_event("startup")
def on_startup() -> None:
    try:
        initialize_earth_engine()
    except Exception:
        pass


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "gee_project_id": GEE_PROJECT_ID,
        "gee_initialized": _GEE_INITIALIZED,
        "forecast_provider": "NOAA/CFSV2/FOR6H_HARMONIZED",
        "backend_mode": "gee-only",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _GEE_INITIALIZED else "degraded",
        "gee_initialized": _GEE_INITIALIZED,
        "gee_error": _GEE_INIT_ERROR,
        "backend_mode": "gee-only",
        "forecast_provider": "NOAA/CFSV2/FOR6H_HARMONIZED",
    }


@app.post("/fv/point", response_model=FVPointResponse)
def fv_point(req: FVPointRequest) -> FVPointResponse:
    initialize_earth_engine()
    if not _GEE_INITIALIZED:
        raise HTTPException(status_code=500, detail=f"Earth Engine not initialized: {_GEE_INIT_ERROR}")

    workload_tag = req.workload_tag or "fv-point-request-ui"
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
            forecast_4week = extract_forecast_4week(req.lat, req.lon)
            warnings.extend(forecast_4week.pop("warnings", []))

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
            "backend_mode": "gee-only",
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


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _round(v: Optional[float], digits: int = 3) -> Optional[float]:
    if v is None:
        return None
    return round(v, digits)


def _kelvin_to_celsius(v: Any) -> Optional[float]:
    x = _coerce_float(v)
    return None if x is None else round(x - 273.15, 3)


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _safe_sum(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals)


def _safe_min(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return min(vals)


def _safe_max(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(vals)


def _kg_m2_s_to_mm_over_6h(rate: Any) -> Optional[float]:
    x = _coerce_float(rate)
    if x is None:
        return None
    return x * SECONDS_PER_6H


def _w_m2_to_mm_equiv_over_6h(flux: Any) -> Optional[float]:
    x = _coerce_float(flux)
    if x is None:
        return None
    if x < 0:
        x = 0.0
    return (x * SECONDS_PER_6H) / LATENT_HEAT_VAPORIZATION_J_PER_KG


def _wind_speed(u: Any, v: Any) -> Optional[float]:
    uu = _coerce_float(u)
    vv = _coerce_float(v)
    if uu is None or vv is None:
        return None
    return math.sqrt((uu ** 2) + (vv ** 2))


def _derive_daily_cfs_step(props: Dict[str, Any]) -> Dict[str, Any]:
    tmean_c = _kelvin_to_celsius(props.get("Temperature_height_above_ground"))
    tmax_c = _kelvin_to_celsius(props.get("Maximum_temperature_height_above_ground_6_Hour_Interval"))
    tmin_c = _kelvin_to_celsius(props.get("Minimum_temperature_height_above_ground_6_Hour_Interval"))
    precip_mm_step = _kg_m2_s_to_mm_over_6h(props.get("Precipitation_rate_surface_6_Hour_Average"))
    pet_proxy_mm_step = _w_m2_to_mm_equiv_over_6h(props.get("Potential_Evaporation_Rate_surface_6_Hour_Average"))
    wind_mps = _wind_speed(
        props.get("u-component_of_wind_height_above_ground"),
        props.get("v-component_of_wind_height_above_ground"),
    )
    soil_5 = _coerce_float(props.get("Volumetric_Soil_Moisture_Content_depth_below_surface_layer_5_cm"))
    soil_25 = _coerce_float(props.get("Volumetric_Soil_Moisture_Content_depth_below_surface_layer_25_cm"))
    soil_70 = _coerce_float(props.get("Volumetric_Soil_Moisture_Content_depth_below_surface_layer_70_cm"))
    soil_150 = _coerce_float(props.get("Volumetric_Soil_Moisture_Content_depth_below_surface_layer_150_cm"))
    rootzone = _safe_mean([soil_25, soil_70, soil_150])

    return {
        "date": (props.get("forecast_start_utc") or "")[:10],
        "run_time_utc": props.get("run_time_utc"),
        "forecast_start_utc": props.get("forecast_start_utc"),
        "forecast_end_utc": props.get("forecast_end_utc"),
        "lead_start_hour": _coerce_float(props.get("start_hour")),
        "lead_end_hour": _coerce_float(props.get("end_hour")),
        "tmean_c": tmean_c,
        "tmax_c": tmax_c,
        "tmin_c": tmin_c,
        "precip_mm_step": precip_mm_step,
        "pet_proxy_mm_step": pet_proxy_mm_step,
        "wind_mps": wind_mps,
        "soil_moisture_surface": soil_5,
        "soil_moisture_rootzone": rootzone,
    }


def _aggregate_daily_forecast(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        if row.get("date"):
            grouped[row["date"]].append(row)

    out: List[Dict[str, Any]] = []
    for date in sorted(grouped.keys())[:MAX_FORECAST_DAYS]:
        rows = grouped[date]
        precip_total = _safe_sum([r.get("precip_mm_step") for r in rows])
        pet_total = _safe_sum([r.get("pet_proxy_mm_step") for r in rows])
        water_balance = None
        if precip_total is not None or pet_total is not None:
            water_balance = (precip_total or 0.0) - (pet_total or 0.0)

        out.append({
            "date": date,
            "step_count": len(rows),
            "tmean_c": _round(_safe_mean([r.get("tmean_c") for r in rows])),
            "tmax_c": _round(_safe_max([r.get("tmax_c") for r in rows])),
            "tmin_c": _round(_safe_min([r.get("tmin_c") for r in rows])),
            "precip_mm": _round(precip_total),
            "pet_proxy_mm": _round(pet_total),
            "water_balance_mm": _round(water_balance),
            "wind_mps": _round(_safe_mean([r.get("wind_mps") for r in rows])),
            "soil_moisture_surface": _round(_safe_mean([r.get("soil_moisture_surface") for r in rows])),
            "soil_moisture_rootzone": _round(_safe_mean([r.get("soil_moisture_rootzone") for r in rows])),
        })

    return out


def _score_week_risk(week_row: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    flags: List[str] = []

    precip = week_row.get("precip_total_mm")
    pet = week_row.get("pet_proxy_total_mm")
    balance = week_row.get("water_balance_mm")
    tmean = week_row.get("tmean_avg_c")
    soil_surface = week_row.get("soil_moisture_surface_avg")
    soil_root = week_row.get("soil_moisture_rootzone_avg")
    wind = week_row.get("wind_avg_mps")

    if balance is not None and balance <= -25:
        score += 2
        flags.append("strong_moisture_deficit_signal")
    elif balance is not None and balance <= -10:
        score += 1
        flags.append("moderate_moisture_deficit_signal")

    if soil_surface is not None and soil_surface < 0.20:
        score += 1
        flags.append("low_surface_soil_moisture")

    if soil_root is not None and soil_root < 0.25:
        score += 1
        flags.append("low_rootzone_soil_moisture")

    if precip is not None and precip < 10:
        score += 1
        flags.append("low_weekly_precipitation")

    if pet is not None and pet > 25:
        score += 1
        flags.append("high_evaporative_demand")

    if tmean is not None and tmean > 20:
        score += 1
        flags.append("warm_week")

    if wind is not None and wind > 5:
        score += 1
        flags.append("windy_week")

    if score <= 1:
        level = "low"
    elif score <= 3:
        level = "moderate"
    elif score <= 5:
        level = "high"
    else:
        level = "very_high"

    return {"risk_score": score, "risk_level": level, "risk_flags": flags}


def _aggregate_weekly_forecast(daily_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    weekly: List[Dict[str, Any]] = []
    for i in range(4):
        chunk = daily_rows[i * 7:(i + 1) * 7]
        if not chunk:
            continue

        precip_total = _safe_sum([r.get("precip_mm") for r in chunk])
        pet_total = _safe_sum([r.get("pet_proxy_mm") for r in chunk])
        balance = None
        if precip_total is not None or pet_total is not None:
            balance = (precip_total or 0.0) - (pet_total or 0.0)

        row = {
            "week": i + 1,
            "start": chunk[0]["date"],
            "end": chunk[-1]["date"],
            "day_count": len(chunk),
            "precip_total_mm": _round(precip_total),
            "pet_proxy_total_mm": _round(pet_total),
            "water_balance_mm": _round(balance),
            "tmean_avg_c": _round(_safe_mean([r.get("tmean_c") for r in chunk])),
            "tmax_max_c": _round(_safe_max([r.get("tmax_c") for r in chunk])),
            "tmin_min_c": _round(_safe_min([r.get("tmin_c") for r in chunk])),
            "wind_avg_mps": _round(_safe_mean([r.get("wind_mps") for r in chunk])),
            "soil_moisture_surface_avg": _round(_safe_mean([r.get("soil_moisture_surface") for r in chunk])),
            "soil_moisture_rootzone_avg": _round(_safe_mean([r.get("soil_moisture_rootzone") for r in chunk])),
        }
        row.update(_score_week_risk(row))
        weekly.append(row)

    return weekly


def _classify_signal(weekly_rows: List[Dict[str, Any]]) -> str:
    if not weekly_rows:
        return "unknown"

    balances = [r.get("water_balance_mm") for r in weekly_rows if r.get("water_balance_mm") is not None]
    if not balances:
        return "unknown"

    mean_balance = sum(balances) / len(balances)
    if mean_balance <= -20:
        return "strong_drying"
    if mean_balance <= -5:
        return "drying"
    if mean_balance < 5:
        return "near_neutral"
    return "wetting"


def _extract_recent_observed_summary(lat: float, lon: float, days: int) -> Dict[str, Any]:
    point = _point_geometry(lat, lon)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    ic = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(str(start_date), str(end_date + timedelta(days=1)))
        .select(["pr", "eto", "tmmx", "tmmn"])
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

    precip_total = _safe_sum([_coerce_float(r.get("pr")) for r in rows])
    eto_total = _safe_sum([_coerce_float(r.get("eto")) for r in rows])
    balance = None
    if precip_total is not None or eto_total is not None:
        balance = (precip_total or 0.0) - (eto_total or 0.0)

    tmax_vals = [_kelvin_to_celsius(r.get("tmmx")) for r in rows]
    tmin_vals = [_kelvin_to_celsius(r.get("tmmn")) for r in rows]
    tmean_vals = []
    for tx, tn in zip(tmax_vals, tmin_vals):
        if tx is not None and tn is not None:
            tmean_vals.append((tx + tn) / 2.0)

    return {
        "days": days,
        "start": str(start_date),
        "end": str(end_date),
        "precip_total_mm": _round(precip_total),
        "eto_total_mm": _round(eto_total),
        "water_balance_mm": _round(balance),
        "tmean_avg_c": _round(_safe_mean(tmean_vals)),
    }


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


def extract_forecast_4week(lat: float, lon: float) -> Dict[str, Any]:
    point = _point_geometry(lat, lon)
    warnings: List[str] = []

    today = datetime.utcnow().date()
    search_start = today - timedelta(days=3)
    search_end = today + timedelta(days=MAX_FORECAST_DAYS + 3)

    ic = (
        ee.ImageCollection("NOAA/CFSV2/FOR6H_HARMONIZED")
        .filterDate(str(search_start), str(search_end))
        .filter(ee.Filter.notNull(["start_hour", "end_hour"]))
        .select(CFS_BANDS)
    )

    collection_size = ic.size().getInfo()
    if not collection_size:
        return {
            "provider": "NOAA/CFSV2/FOR6H_HARMONIZED",
            "status": "no_data",
            "message": "No CFSv2 forecast images were found in the recent search window.",
            "warnings": ["Forecast block is empty because no recent CFSv2 forecast images were available in GEE."],
        }

    def add_run_properties(img):
        valid_start = ee.Date(img.get("system:time_start"))
        start_hour = ee.Number(img.get("start_hour"))
        end_hour = ee.Number(img.get("end_hour"))
        run_time = valid_start.advance(start_hour.multiply(-1), "hour")
        step_hours = end_hour.subtract(start_hour)
        forecast_end = valid_start.advance(step_hours, "hour")
        return img.set({
            "run_time_millis": run_time.millis(),
            "run_time_utc": run_time.format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            "forecast_start_utc": valid_start.format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            "forecast_end_utc": forecast_end.format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
        })

    ic_with_run = ic.map(add_run_properties)
    latest_run_img = ic_with_run.sort("run_time_millis", False).first()
    latest_run_millis = latest_run_img.get("run_time_millis")
    latest_run = (
        ic_with_run.filter(ee.Filter.eq("run_time_millis", latest_run_millis))
        .filter(ee.Filter.lte("start_hour", MAX_FORECAST_DAYS * 24))
        .sort("start_hour")
    )

    def image_to_feature(img):
        vals = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=22264,
            maxPixels=1_000_000,
        )
        vals = vals.set("run_time_utc", img.get("run_time_utc"))
        vals = vals.set("start_hour", ee.Number(img.get("start_hour")))
        vals = vals.set("end_hour", ee.Number(img.get("end_hour")))
        vals = vals.set("forecast_start_utc", img.get("forecast_start_utc"))
        vals = vals.set("forecast_end_utc", img.get("forecast_end_utc"))
        return ee.Feature(None, vals)

    fc = ee.FeatureCollection(latest_run.map(image_to_feature))
    features = fc.getInfo().get("features", [])
    step_rows = [feat.get("properties", {}) for feat in features]
    step_rows.sort(key=lambda r: (r.get("start_hour", 0), r.get("end_hour", 0)))

    if not step_rows:
        return {
            "provider": "NOAA/CFSV2/FOR6H_HARMONIZED",
            "status": "no_data",
            "message": "No forecast images were returned after grouping the latest CFSv2 initialization.",
            "warnings": ["Forecast block is empty because the latest CFSv2 initialization returned no usable images."],
        }

    derived_steps = [_derive_daily_cfs_step(r) for r in step_rows if r.get("forecast_start_utc")]
    daily = _aggregate_daily_forecast(derived_steps)
    weekly = _aggregate_weekly_forecast(daily)

    if len(daily) < MAX_FORECAST_DAYS:
        warnings.append(
            f"Forecast returned {len(daily)} daily rows instead of {MAX_FORECAST_DAYS}; later weeks may be partial."
        )

    obs14 = _extract_recent_observed_summary(lat, lon, 14)
    obs28 = _extract_recent_observed_summary(lat, lon, 28)

    forecast14_precip = _safe_sum([r.get("precip_mm") for r in daily[:14]])
    forecast14_pet = _safe_sum([r.get("pet_proxy_mm") for r in daily[:14]])
    forecast14_balance = None
    if forecast14_precip is not None or forecast14_pet is not None:
        forecast14_balance = (forecast14_precip or 0.0) - (forecast14_pet or 0.0)

    forecast28_precip = _safe_sum([r.get("precip_mm") for r in daily[:28]])
    forecast28_pet = _safe_sum([r.get("pet_proxy_mm") for r in daily[:28]])
    forecast28_balance = None
    if forecast28_precip is not None or forecast28_pet is not None:
        forecast28_balance = (forecast28_precip or 0.0) - (forecast28_pet or 0.0)

    blended_14 = None
    if obs14.get("water_balance_mm") is not None or forecast14_balance is not None:
        blended_14 = (obs14.get("water_balance_mm") or 0.0) + (forecast14_balance or 0.0)

    blended_28 = None
    if obs28.get("water_balance_mm") is not None or forecast28_balance is not None:
        blended_28 = (obs28.get("water_balance_mm") or 0.0) + (forecast28_balance or 0.0)

    highest_risk_week = None
    if weekly:
        highest_risk_week = max(weekly, key=lambda r: (r.get("risk_score", -1), -(r.get("week", 0))))["week"]

    forecast_status = "ok"
    if not weekly:
        forecast_status = "partial"
        warnings.append("Weekly forecast summaries could not be built from the CFSv2 run.")

    return {
        "provider": "NOAA/CFSV2/FOR6H_HARMONIZED",
        "status": forecast_status,
        "message": "4-week forecast block generated from the latest available CFSv2 run in Google Earth Engine.",
        "run_time_utc": step_rows[0].get("run_time_utc"),
        "daily_step_source": "6-hour CFSv2 images aggregated to daily rows",
        "daily_days_returned": len(daily),
        "weekly_weeks_returned": len(weekly),
        "daily": daily,
        "weekly": weekly,
        "observed_context": {
            "last_14_days": obs14,
            "last_28_days": obs28,
        },
        "summary": {
            "highest_risk_week": highest_risk_week,
            "weeks_high_or_above": len([r for r in weekly if r.get("risk_level") in {"high", "very_high"}]),
            "two_to_four_week_signal": _classify_signal(weekly[1:]) if len(weekly) > 1 else _classify_signal(weekly),
            "forecast_14d_precip_mm": _round(forecast14_precip),
            "forecast_14d_pet_proxy_mm": _round(forecast14_pet),
            "forecast_14d_water_balance_mm": _round(forecast14_balance),
            "forecast_28d_precip_mm": _round(forecast28_precip),
            "forecast_28d_pet_proxy_mm": _round(forecast28_pet),
            "forecast_28d_water_balance_mm": _round(forecast28_balance),
            "blended_observed_plus_forecast_14d_water_balance_mm": _round(blended_14),
            "blended_observed_plus_forecast_28d_water_balance_mm": _round(blended_28),
        },
        "warnings": warnings,
    }


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
