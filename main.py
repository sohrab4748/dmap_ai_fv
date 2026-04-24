import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmap_ai_fv_backend")

APP_TITLE = "DMAP-AI / FV Backend"
APP_VERSION = "0.7.1-thredds-eddi-farmer-risk-fileserver-fixed"

_raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
if _raw_origins == "*":
    CORS_ALLOW_ORIGINS = ["*"]
else:
    CORS_ALLOW_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

FORECAST_PERIODS_ALL = ["wk1", "wk2", "wk3", "wk4", "wk12", "wk34", "wk123", "wk1234"]
FORECAST_PERIOD_FILES = {
    "wk1": "cfsv2_eddi_wk1_ensembles.nc",
    "wk2": "cfsv2_eddi_wk2_ensembles.nc",
    "wk3": "cfsv2_eddi_wk3_ensembles.nc",
    "wk4": "cfsv2_eddi_wk4_ensembles.nc",
    "wk12": "cfsv2_eddi_wk12_ensembles.nc",
    "wk34": "cfsv2_eddi_wk34_ensembles.nc",
    "wk123": "cfsv2_eddi_wk123_ensembles.nc",
    "wk1234": "cfsv2_eddi_wk1234_ensembles.nc",
}
FORECAST_PERIOD_WINDOWS = {
    "wk1": (1, 7),
    "wk2": (8, 14),
    "wk3": (15, 21),
    "wk4": (22, 28),
    "wk12": (1, 14),
    "wk34": (15, 28),
    "wk123": (1, 21),
    "wk1234": (1, 28),
}
FORECAST_BASE_ODAP = os.getenv(
    "CFSGRIDMET_EDDI_ODAP_BASE",
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
    "NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/cfsv2_metdata_90day",
).rstrip("/")
FORECAST_BASE_FILESERVER = os.getenv(
    "CFSGRIDMET_EDDI_FILESERVER_BASE",
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/"
    "NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/cfsv2_metdata_90day",
).rstrip("/")
THREDDS_TIMEOUT_SECONDS = int(os.getenv("THREDDS_TIMEOUT_SECONDS", "90"))
THREDDS_CONNECT_TIMEOUT_SECONDS = int(os.getenv("THREDDS_CONNECT_TIMEOUT_SECONDS", "20"))
THREDDS_READ_TIMEOUT_SECONDS = int(os.getenv("THREDDS_READ_TIMEOUT_SECONDS", "90"))

SUPPORTED_CROPS = ["corn", "soybean"]
CROP_STAGE_OPTIONS = {
    "corn": [
        "planting_emergence",
        "vegetative",
        "tasseling_silking",
        "grain_fill",
        "maturity_late",
    ],
    "soybean": [
        "planting_emergence",
        "vegetative",
        "flowering",
        "pod_set_pod_fill",
        "seed_fill",
        "maturity_late",
    ],
}
STAGE_SENSITIVITY = {
    "corn": {
        "planting_emergence": 1.0,
        "vegetative": 1.1,
        "tasseling_silking": 1.5,
        "grain_fill": 1.3,
        "maturity_late": 0.7,
    },
    "soybean": {
        "planting_emergence": 1.0,
        "vegetative": 1.0,
        "flowering": 1.3,
        "pod_set_pod_fill": 1.5,
        "seed_fill": 1.3,
        "maturity_late": 0.7,
    },
}
STAGE_SENSITIVITY_LABELS = {
    1.45: "very_high",
    1.20: "high",
    0.90: "moderate",
    0.0: "low",
}
EDDI_DRY_DIRECTION = os.getenv("EDDI_DRY_DIRECTION", "negative").strip().lower()
DRY_THRESHOLD = float(os.getenv("EDDI_DRY_THRESHOLD", "-1.0"))
SEVERE_DRY_THRESHOLD = float(os.getenv("EDDI_SEVERE_DRY_THRESHOLD", "-1.5"))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False if CORS_ALLOW_ORIGINS == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FVPointRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    history_days: int = Field(60, ge=7, le=3650)
    include_weather_history: bool = True
    include_drought_latest: bool = True
    include_noaa_monitoring: bool = True
    include_forecast_4week: bool = True
    forecast_periods: Optional[List[str]] = None
    include_forecast_ensembles: bool = True
    workload_tag: Optional[str] = None
    crop: Optional[str] = None
    stage: Optional[str] = None

    @field_validator("forecast_periods")
    @classmethod
    def validate_forecast_periods(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        cleaned: List[str] = []
        seen = set()
        for item in v:
            key = str(item).strip().lower()
            if not key:
                continue
            if key not in FORECAST_PERIOD_FILES:
                raise ValueError(
                    f"Unsupported forecast period '{item}'. Supported values: {', '.join(FORECAST_PERIODS_ALL)}"
                )
            if key not in seen:
                cleaned.append(key)
                seen.add(key)
        return cleaned or None

    @field_validator("crop")
    @classmethod
    def validate_crop(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip().lower()
        if not value:
            return None
        if value not in SUPPORTED_CROPS:
            raise ValueError(f"Unsupported crop '{v}'. Supported values: {', '.join(SUPPORTED_CROPS)}")
        return value

    @field_validator("stage")
    @classmethod
    def normalize_stage(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip().lower()
        return value or None


class FVPointResponse(BaseModel):
    meta: Dict[str, Any]
    weather_daily: Optional[List[Dict[str, Any]]] = None
    drought_latest: Optional[Dict[str, Any]] = None
    noaa_monitoring: Optional[Dict[str, Any]] = None
    forecast_4week: Optional[Dict[str, Any]] = None
    risk_analysis: Optional[Dict[str, Any]] = None
    farmer_advice: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = Field(default_factory=list)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "gee_initialized": False,
        "gee_enabled": False,
        "forecast_provider": "official-thredds-eddi-fileserver",
        "backend_mode": "thredds-eddi-fileserver-no-gee",
        "supported_forecast_periods": FORECAST_PERIODS_ALL,
        "supported_crops": SUPPORTED_CROPS,
        "crop_stage_options": CROP_STAGE_OPTIONS,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "gee_initialized": False,
        "gee_enabled": False,
        "gee_error": None,
        "backend_mode": "thredds-eddi-fileserver-no-gee",
        "forecast_provider": "official-thredds-eddi-fileserver",
        "forecast_periods_supported": FORECAST_PERIODS_ALL,
        "supported_crops": SUPPORTED_CROPS,
    }


@app.post("/fv/point", response_model=FVPointResponse)
def fv_point(req: FVPointRequest) -> FVPointResponse:
    warnings: List[str] = []
    weather_daily = None
    drought_latest = None
    noaa_monitoring = None
    forecast_4week = None
    risk_analysis = None
    farmer_advice = None

    try:
        if req.crop and not req.stage:
            warnings.append("Crop was provided but current stage was not provided, so farmer risk analysis was skipped.")
        if req.stage and not req.crop:
            warnings.append("Current stage was provided but crop was not provided, so farmer risk analysis was skipped.")
        if req.crop and req.stage and req.stage not in CROP_STAGE_OPTIONS.get(req.crop, []):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported stage '{req.stage}' for crop '{req.crop}'. "
                    f"Supported stages: {', '.join(CROP_STAGE_OPTIONS[req.crop])}"
                ),
            )

        if req.include_weather_history:
            warnings.append(
                "Weather history is not included in this backend because GEE has been removed."
            )

        if req.include_drought_latest:
            warnings.append(
                "Latest drought indices are not included in this backend because GEE has been removed."
            )

        if req.include_noaa_monitoring:
            noaa_monitoring = fetch_noaa_monitoring_eddi(req.lat, req.lon)

        if req.include_forecast_4week:
            periods = req.forecast_periods or FORECAST_PERIODS_ALL
            need_ensembles_for_analysis = bool(req.crop and req.stage)
            forecast_4week = extract_official_forecast_eddi(
                lat=req.lat,
                lon=req.lon,
                periods=periods,
                include_ensembles=(req.include_forecast_ensembles or need_ensembles_for_analysis),
            )
            if forecast_4week.get("status") != "ok":
                warnings.append(str(forecast_4week.get("message") or "Forecast download did not complete normally."))
            if req.crop and req.stage and forecast_4week.get("summary"):
                risk_analysis = analyze_farmer_risk(
                    forecast_4week=forecast_4week,
                    noaa_monitoring=noaa_monitoring,
                    crop=req.crop,
                    stage=req.stage,
                )
                farmer_advice = build_farmer_advice(risk_analysis)
                if not req.include_forecast_ensembles:
                    forecast_4week["ensembles"] = None

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
            "workload_tag": req.workload_tag,
            "backend_mode": "thredds-eddi-fileserver-no-gee",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "crop": req.crop,
            "stage": req.stage,
        },
        weather_daily=weather_daily,
        drought_latest=drought_latest,
        noaa_monitoring=noaa_monitoring,
        forecast_4week=forecast_4week,
        risk_analysis=risk_analysis,
        farmer_advice=farmer_advice,
        warnings=warnings,
    )


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


def _download_forecast_file(period: str) -> str:
    url = f"{FORECAST_BASE_FILESERVER}/{FORECAST_PERIOD_FILES[period]}"
    suffix = f"_{FORECAST_PERIOD_FILES[period]}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(THREDDS_CONNECT_TIMEOUT_SECONDS, THREDDS_READ_TIMEOUT_SECONDS),
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    tmp.write(chunk)
        tmp.close()
        return tmp_path
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def _open_thredds_dataset(local_path: str):
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "xarray is required for forecast download. Install xarray and h5netcdf in the backend environment."
        ) from exc

    last_exc = None
    for engine in ("h5netcdf", None):
        try:
            if engine:
                return xr.open_dataset(local_path, engine=engine, decode_times=False)
            return xr.open_dataset(local_path, decode_times=False)
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"Could not open downloaded forecast dataset: {local_path}. Error: {last_exc}"
    ) from last_exc


def _parse_http_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _fetch_file_issue_datetime(period: str) -> Optional[datetime]:
    url = f"{FORECAST_BASE_FILESERVER}/{FORECAST_PERIOD_FILES[period]}"
    try:
        resp = requests.head(url, timeout=(THREDDS_CONNECT_TIMEOUT_SECONDS, THREDDS_READ_TIMEOUT_SECONDS), allow_redirects=True)
        resp.raise_for_status()
        dt = _parse_http_datetime(resp.headers.get("Last-Modified"))
        if dt is not None:
            return dt
    except Exception as exc:
        logger.warning("HEAD request failed for %s: %s", url, exc)

    try:
        resp = requests.get(url, timeout=(THREDDS_CONNECT_TIMEOUT_SECONDS, THREDDS_READ_TIMEOUT_SECONDS), stream=True)
        resp.raise_for_status()
        dt = _parse_http_datetime(resp.headers.get("Last-Modified"))
        resp.close()
        return dt
    except Exception as exc:
        logger.warning("GET request failed for %s: %s", url, exc)
        return None


def _find_data_var_name(ds) -> str:
    if "eddi" in getattr(ds, "data_vars", {}):
        return "eddi"
    data_vars = list(getattr(ds, "data_vars", {}).keys())
    if not data_vars:
        raise RuntimeError("No data variables found in forecast dataset.")
    return data_vars[0]


def _find_coord_name(ds, candidates: List[str]) -> Optional[str]:
    names = list(getattr(ds, "coords", {}).keys()) + list(getattr(ds, "variables", {}).keys())
    for wanted in candidates:
        for name in names:
            if name.lower() == wanted.lower():
                return name
    for wanted in candidates:
        for name in names:
            if wanted.lower() in name.lower():
                return name
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def extract_official_forecast_eddi(
    lat: float,
    lon: float,
    periods: List[str],
    include_ensembles: bool = True,
) -> Dict[str, Any]:
    import numpy as np

    summary_rows: List[Dict[str, Any]] = []
    ensemble_rows: List[Dict[str, Any]] = []
    file_rows: List[Dict[str, Any]] = []
    issue_dates: List[str] = []
    period_errors: List[Dict[str, str]] = []

    for period in periods:
        if period not in FORECAST_PERIOD_FILES:
            raise RuntimeError(f"Unsupported forecast period: {period}")

        url = f"{FORECAST_BASE_ODAP}/{FORECAST_PERIOD_FILES[period]}"
        file_url = f"{FORECAST_BASE_FILESERVER}/{FORECAST_PERIOD_FILES[period]}"

        try:
            local_path = _download_forecast_file(period)
            ds = _open_thredds_dataset(local_path)
            try:
                var_name = _find_data_var_name(ds)
                lat_name = _find_coord_name(ds, ["lat", "latitude", "y"])
                lon_name = _find_coord_name(ds, ["lon", "longitude", "x"])
                if not lat_name or not lon_name:
                    raise RuntimeError(f"Could not detect lat/lon coordinate names in {url}")

                da = ds[var_name].sel({lat_name: lat, lon_name: lon}, method="nearest")
                values = np.asarray(da.values, dtype=float).reshape(-1)
                values_finite = values[np.isfinite(values)]
                n_members = int(values.size)
                n_finite = int(values_finite.size)

                lat_grid = _safe_float(da[lat_name].values)
                lon_grid = _safe_float(da[lon_name].values)

                issue_dt = _fetch_file_issue_datetime(period)
                issue_date = (issue_dt.date() if issue_dt else datetime.now(timezone.utc).date())
                issue_date_iso = issue_date.isoformat()
                issue_dates.append(issue_date_iso)

                start_day, end_day = FORECAST_PERIOD_WINDOWS[period]
                valid_start = (issue_date + timedelta(days=start_day)).isoformat()
                valid_end = (issue_date + timedelta(days=end_day)).isoformat()

                summary_rows.append({
                    "period": period,
                    "issue_date": issue_date_iso,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "n_members": n_members,
                    "n_finite": n_finite,
                    "eddi_mean": float(np.mean(values_finite)) if n_finite else None,
                    "eddi_median": float(np.median(values_finite)) if n_finite else None,
                    "eddi_min": float(np.min(values_finite)) if n_finite else None,
                    "eddi_max": float(np.max(values_finite)) if n_finite else None,
                    "eddi_p25": float(np.percentile(values_finite, 25)) if n_finite else None,
                    "eddi_p75": float(np.percentile(values_finite, 75)) if n_finite else None,
                    "lat_req": lat,
                    "lon_req": lon,
                    "lat_grid": lat_grid,
                    "lon_grid": lon_grid,
                })

                file_rows.append({
                    "period": period,
                    "url": url,
                    "file_url": file_url,
                    "filename": FORECAST_PERIOD_FILES[period],
                    "issue_date": issue_date_iso,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "n_members": n_members,
                })

                if include_ensembles:
                    for idx, val in enumerate(values, start=1):
                        ensemble_rows.append({
                            "period": period,
                            "issue_date": issue_date_iso,
                            "valid_start": valid_start,
                            "valid_end": valid_end,
                            "ensemble_member": idx,
                            "eddi": float(val) if np.isfinite(val) else None,
                            "lat_req": lat,
                            "lon_req": lon,
                            "lat_grid": lat_grid,
                            "lon_grid": lon_grid,
                        })
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Skipping forecast period %s due to remote access error: %s", period, exc)
            period_errors.append({
                "period": period,
                "url": url,
                "file_url": file_url,
                "error": str(exc),
            })
            continue

    if not summary_rows:
        return {
            "provider": "official-thredds-eddi",
            "status": "no_data",
            "message": "No official THREDDS EDDI forecast rows were returned.",
            "requested_periods": periods,
            "period_errors": period_errors,
        }

    status = "partial" if period_errors else "ok"
    message = None
    if period_errors:
        failed = ", ".join(err["period"] for err in period_errors)
        message = f"Some forecast periods could not be loaded from THREDDS: {failed}"

    return {
        "provider": "official-thredds-eddi",
        "status": status,
        "base_odap_url": FORECAST_BASE_ODAP,
        "base_fileserver_url": FORECAST_BASE_FILESERVER,
        "requested_periods": periods,
        "issue_dates": sorted(set(issue_dates)),
        "summary": summary_rows,
        "ensembles": ensemble_rows if include_ensembles else None,
        "files": file_rows,
        "period_errors": period_errors,
        "message": message,
    }


def _stage_sensitivity_label(weight: float) -> str:
    for min_weight, label in STAGE_SENSITIVITY_LABELS.items():
        if weight >= min_weight:
            return label
    return "low"


def _is_dry_signal(value: Optional[float], threshold: float) -> bool:
    if value is None:
        return False
    if EDDI_DRY_DIRECTION == "positive":
        return value >= abs(threshold)
    return value <= threshold


def _fraction_matching(values: List[Optional[float]], threshold: float) -> float:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return 0.0
    hits = sum(1 for v in finite if _is_dry_signal(v, threshold))
    return hits / len(finite)


def _confidence_from_spread(p25: Optional[float], p75: Optional[float]) -> str:
    if p25 is None or p75 is None:
        return "low"
    spread = abs(float(p75) - float(p25))
    if spread <= 0.75:
        return "high"
    if spread <= 1.5:
        return "moderate"
    return "low"


def _signal_label(dry_fraction: float, severe_fraction: float, confidence: str) -> str:
    if severe_fraction >= 0.40:
        return "very_dry_week"
    if dry_fraction >= 0.55:
        return "dry_week"
    if dry_fraction >= 0.30:
        return "watch_week" if confidence == "low" else "dry_watch_week"
    if dry_fraction <= 0.10:
        return "near_normal_week"
    return "mixed_signal_week"


def _risk_from_score(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "moderate"
    if score < 0.90:
        return "high"
    return "very_high"


def _current_context_label(current_04wk_value: Optional[float]) -> str:
    if current_04wk_value is None:
        return "monitoring_unavailable"
    if _is_dry_signal(current_04wk_value, DRY_THRESHOLD):
        return "current_dry_signal_present"
    return "current_signal_not_dry"


def _trajectory_from_weeklies(weeklies: List[Dict[str, Any]]) -> Dict[str, str]:
    week_order = {"wk1": 1, "wk2": 2, "wk3": 3, "wk4": 4}
    rows = [w for w in weeklies if w.get("period") in week_order]
    rows.sort(key=lambda r: week_order[r["period"]])
    if not rows:
        return {
            "label": "not_available",
            "summary": "Week-by-week trajectory is not available from the selected periods.",
        }

    dry_weeks = [r for r in rows if r.get("dry_member_fraction", 0) >= 0.55]
    if len(dry_weeks) >= 3:
        return {
            "label": "persistent_dry_risk",
            "summary": "The dryness signal persists across most of the week-by-week outlook.",
        }
    if rows[0].get("dry_member_fraction", 0) >= 0.55 and all(r.get("dry_member_fraction", 0) < 0.55 for r in rows[1:]):
        return {
            "label": "early_dry_risk",
            "summary": "The strongest dryness risk appears early in the outlook and then weakens.",
        }
    if rows[-1].get("dry_member_fraction", 0) >= 0.55 and all(r.get("dry_member_fraction", 0) < 0.55 for r in rows[:-1]):
        return {
            "label": "late_dry_risk",
            "summary": "The dryness signal becomes stronger later in the outlook.",
        }
    if all(r.get("dry_member_fraction", 0) < 0.30 for r in rows):
        return {
            "label": "limited_dry_risk",
            "summary": "The week-by-week outlook shows limited dryness risk overall.",
        }
    return {
        "label": "mixed_signal",
        "summary": "The week-by-week outlook is mixed, so the dryness signal should be watched closely.",
    }


def analyze_farmer_risk(
    forecast_4week: Dict[str, Any],
    noaa_monitoring: Optional[Dict[str, Any]],
    crop: str,
    stage: str,
) -> Dict[str, Any]:
    summary_rows = forecast_4week.get("summary") or []
    ensemble_rows = forecast_4week.get("ensembles") or []
    stage_weight = STAGE_SENSITIVITY[crop][stage]
    sensitivity_label = _stage_sensitivity_label(stage_weight)

    ensemble_by_period: Dict[str, List[Optional[float]]] = {}
    for row in ensemble_rows:
        period = row.get("period")
        if not period:
            continue
        ensemble_by_period.setdefault(period, []).append(row.get("eddi"))

    weekly_rows: List[Dict[str, Any]] = []
    combined_rows: Dict[str, Dict[str, Any]] = {}

    for row in summary_rows:
        period = str(row.get("period") or "")
        values = ensemble_by_period.get(period, [])
        dry_fraction = _fraction_matching(values, DRY_THRESHOLD) if values else 0.0
        severe_fraction = _fraction_matching(values, SEVERE_DRY_THRESHOLD) if values else 0.0
        confidence = _confidence_from_spread(row.get("eddi_p25"), row.get("eddi_p75"))
        signal_label = _signal_label(dry_fraction, severe_fraction, confidence)
        base_score = dry_fraction * 0.6 + severe_fraction * 0.4
        adjusted_score = base_score * stage_weight
        overall_risk = _risk_from_score(adjusted_score)

        out_row = {
            "period": period,
            "issue_date": row.get("issue_date"),
            "valid_start": row.get("valid_start"),
            "valid_end": row.get("valid_end"),
            "eddi_mean": row.get("eddi_mean"),
            "eddi_median": row.get("eddi_median"),
            "eddi_p25": row.get("eddi_p25"),
            "eddi_p75": row.get("eddi_p75"),
            "eddi_min": row.get("eddi_min"),
            "eddi_max": row.get("eddi_max"),
            "dry_member_fraction": round(dry_fraction, 4),
            "severe_dry_member_fraction": round(severe_fraction, 4),
            "confidence": confidence,
            "signal_label": signal_label,
            "overall_risk": overall_risk,
            "stage_weight": round(stage_weight, 3),
            "lat_grid": row.get("lat_grid"),
            "lon_grid": row.get("lon_grid"),
        }

        if period in {"wk1", "wk2", "wk3", "wk4"}:
            weekly_rows.append(out_row)
        else:
            combined_rows[period] = out_row

    weekly_rows.sort(key=lambda r: {"wk1": 1, "wk2": 2, "wk3": 3, "wk4": 4}.get(r["period"], 99))
    current_04wk = _safe_float((noaa_monitoring or {}).get("EDDI_ETrs_04wk"))
    current_01mn = _safe_float((noaa_monitoring or {}).get("EDDI_ETrs_01mn"))

    return {
        "crop": crop,
        "stage": stage,
        "stage_sensitivity": sensitivity_label,
        "stage_weight": round(stage_weight, 3),
        "current_context": {
            "noaa_monitoring_eddi_04wk": current_04wk,
            "noaa_monitoring_eddi_01mn": current_01mn,
            "status_label": _current_context_label(current_04wk),
        },
        "thresholds": {
            "dry_threshold": DRY_THRESHOLD,
            "severe_dry_threshold": SEVERE_DRY_THRESHOLD,
            "dry_direction": EDDI_DRY_DIRECTION,
        },
        "weekly": weekly_rows,
        "combined": combined_rows,
        "trajectory": _trajectory_from_weeklies(weekly_rows),
    }


def build_farmer_advice(risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    crop = risk_analysis.get("crop")
    stage = risk_analysis.get("stage")
    advice: List[Dict[str, Any]] = []

    label_messages = {
        "very_dry_week": "This period shows a strong dryness signal across many ensemble members.",
        "dry_week": "This period shows elevated atmospheric moisture-stress risk.",
        "dry_watch_week": "This period leans dry, although the signal is not fully settled.",
        "watch_week": "The signal is mixed, so this period should be watched closely.",
        "near_normal_week": "This period does not show a strong dryness signal.",
        "mixed_signal_week": "This period has mixed ensemble outcomes, so confidence is limited.",
    }

    for row in risk_analysis.get("weekly", []):
        period = row.get("period")
        signal_label = row.get("signal_label")
        overall_risk = row.get("overall_risk")
        base_msg = label_messages.get(signal_label, "This period should be monitored.")
        message = f"{base_msg} For {crop} in the {stage} stage, overall field stress risk is {overall_risk}."
        advice.append({
            "period": period,
            "label": signal_label,
            "overall_risk": overall_risk,
            "message": message,
        })

    wk12 = (risk_analysis.get("combined") or {}).get("wk12")
    if wk12:
        advice.append({
            "period": "wk12",
            "label": "two_week_view",
            "overall_risk": wk12.get("overall_risk"),
            "message": (
                f"The 2-week outlook for {crop} in the {stage} stage is {wk12.get('overall_risk')} risk, "
                "which is useful for short-term planning."
            ),
        })

    wk1234 = (risk_analysis.get("combined") or {}).get("wk1234")
    if wk1234:
        advice.append({
            "period": "wk1234",
            "label": "four_week_view",
            "overall_risk": wk1234.get("overall_risk"),
            "message": (
                f"The 4-week integrated outlook for {crop} in the {stage} stage is {wk1234.get('overall_risk')} risk."
            ),
        })

    trajectory = risk_analysis.get("trajectory") or {}
    if trajectory.get("label") and trajectory.get("label") != "not_available":
        advice.append({
            "period": "trajectory",
            "label": trajectory.get("label"),
            "overall_risk": None,
            "message": trajectory.get("summary"),
        })

    return advice
