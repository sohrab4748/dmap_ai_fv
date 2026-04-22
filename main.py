import logging
import os
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
APP_VERSION = "0.5.0-thredds-eddi-no-gee"

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


class FVPointResponse(BaseModel):
    meta: Dict[str, Any]
    weather_daily: Optional[List[Dict[str, Any]]] = None
    drought_latest: Optional[Dict[str, Any]] = None
    noaa_monitoring: Optional[Dict[str, Any]] = None
    forecast_4week: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "gee_initialized": False,
        "gee_enabled": False,
        "forecast_provider": "official-thredds-eddi",
        "backend_mode": "thredds-eddi-no-gee",
        "supported_forecast_periods": FORECAST_PERIODS_ALL,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "gee_initialized": False,
        "gee_enabled": False,
        "gee_error": None,
        "backend_mode": "thredds-eddi-no-gee",
        "forecast_provider": "official-thredds-eddi",
        "forecast_periods_supported": FORECAST_PERIODS_ALL,
    }


@app.post("/fv/point", response_model=FVPointResponse)
def fv_point(req: FVPointRequest) -> FVPointResponse:
    warnings: List[str] = []
    weather_daily = None
    drought_latest = None
    noaa_monitoring = None
    forecast_4week = None

    try:
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
            forecast_4week = extract_official_forecast_eddi(
                lat=req.lat,
                lon=req.lon,
                periods=periods,
                include_ensembles=req.include_forecast_ensembles,
            )
            if forecast_4week.get("status") != "ok":
                warnings.append(str(forecast_4week.get("message") or "Forecast download did not complete normally."))

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
            "backend_mode": "thredds-eddi-no-gee",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        weather_daily=weather_daily,
        drought_latest=drought_latest,
        noaa_monitoring=noaa_monitoring,
        forecast_4week=forecast_4week,
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


def _open_thredds_dataset(url: str):
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError(
            "xarray is required for forecast download. Install xarray and pydap in the backend environment."
        ) from exc

    errors: List[str] = []
    for engine in ("pydap", None):
        try:
            if engine is None:
                return xr.open_dataset(url, decode_times=False)
            return xr.open_dataset(url, engine=engine, decode_times=False)
        except Exception as exc:
            errors.append(f"engine={engine or 'default'}: {exc}")
    raise RuntimeError(f"Could not open THREDDS dataset {url}. Errors: {' | '.join(errors)}")


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
        resp = requests.head(url, timeout=THREDDS_TIMEOUT_SECONDS, allow_redirects=True)
        resp.raise_for_status()
        dt = _parse_http_datetime(resp.headers.get("Last-Modified"))
        if dt is not None:
            return dt
    except Exception as exc:
        logger.warning("HEAD request failed for %s: %s", url, exc)

    try:
        resp = requests.get(url, timeout=THREDDS_TIMEOUT_SECONDS, stream=True)
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

    for period in periods:
        if period not in FORECAST_PERIOD_FILES:
            raise RuntimeError(f"Unsupported forecast period: {period}")

        url = f"{FORECAST_BASE_ODAP}/{FORECAST_PERIOD_FILES[period]}"
        ds = _open_thredds_dataset(url)
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

    if not summary_rows:
        return {
            "provider": "official-thredds-eddi",
            "status": "no_data",
            "message": "No official THREDDS EDDI forecast rows were returned.",
            "requested_periods": periods,
        }

    return {
        "provider": "official-thredds-eddi",
        "status": "ok",
        "base_odap_url": FORECAST_BASE_ODAP,
        "requested_periods": periods,
        "issue_dates": sorted(set(issue_dates)),
        "summary": summary_rows,
        "ensembles": ensemble_rows if include_ensembles else None,
        "files": file_rows,
    }
