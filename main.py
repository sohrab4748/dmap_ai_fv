import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ee
import pandas as pd

PROJECT_ID = "dmapaifv"
WORKLOAD_TAG = "fv-forecast-probe"
LAT = 42.03
LON = -93.63
OUT_DIR = Path("gee_fv_request_output")
NOW_UTC = datetime.now(UTC)

COLLECTIONS = {
    "cfs_harmonized": "NOAA/CFSV2/FOR6H_HARMONIZED",
    "gfs": "NOAA/GFS0P25",
    "ecmwf_ifs": "ECMWF/NRT_FORECAST/IFS/OPER",
}

PROPERTY_CANDIDATES = [
    "system:time_start",
    "system:time_end",
    "creation_time",
    "forecast_time",
    "forecast_hours",
    "forecast_hour",
    "forecast_step",
    "start_hour",
    "end_hour",
]


def authenticate_and_initialize() -> None:
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def set_workload_tag() -> None:
    ee.data.setDefaultWorkloadTag(WORKLOAD_TAG)


def get_point() -> ee.Geometry:
    return ee.Geometry.Point([LON, LAT])


def dt_to_str(v):
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def ms_to_datetime_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s > 0) & (s < 32503680000000))
    return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")


def hours_to_timedelta_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s >= 0) & (s < 10000))
    return pd.to_timedelta(s, unit="h", errors="coerce")


def get_selected_property_rows(collection_id: str, limit: int = 12) -> list[dict]:
    ic = ee.ImageCollection(collection_id).sort("system:time_start", False)
    count = int(ic.size().getInfo())
    n = min(limit, count)
    rows: list[dict] = []
    if n <= 0:
        return rows

    lst = ic.toList(n)
    for i in range(n):
        img = ee.Image(lst.get(i))
        row = {}
        for prop in PROPERTY_CANDIDATES:
            try:
                row[prop] = img.get(prop).getInfo()
            except Exception:
                row[prop] = None
        try:
            row["_band_count"] = int(img.bandNames().size().getInfo())
        except Exception:
            row["_band_count"] = None
        rows.append(row)
    return rows


def summarize_collection(collection_key: str, collection_id: str) -> dict:
    ic = ee.ImageCollection(collection_id)
    recent_start = (NOW_UTC - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
    recent_ic = ic.filterDate(recent_start, NOW_UTC.strftime("%Y-%m-%dT%H:%M:%S"))
    recent_count = int(recent_ic.size().getInfo())
    latest_band_count = None
    latest_time = None
    if recent_count > 0:
        latest = ee.Image(recent_ic.sort("system:time_start", False).first())
        latest_band_count = int(latest.bandNames().size().getInfo())
        latest_time = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd HH:mm").getInfo()
    return {
        "source": collection_key,
        "collection_id": collection_id,
        "recent_count": recent_count,
        "latest_system_time_start_utc": latest_time,
        "latest_band_count": latest_band_count,
    }


def derive_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["system:time_start", "system:time_end", "creation_time", "forecast_time"]:
        if col in out.columns:
            out[col + "_dt"] = ms_to_datetime_safe(out[col])

    if "forecast_hours" in out.columns:
        out["forecast_hours_td"] = hours_to_timedelta_safe(out["forecast_hours"])
    if "forecast_hour" in out.columns:
        out["forecast_hour_td"] = hours_to_timedelta_safe(out["forecast_hour"])
    if "forecast_step" in out.columns:
        out["forecast_step_td"] = hours_to_timedelta_safe(out["forecast_step"])
    if "start_hour" in out.columns:
        out["start_hour_td"] = hours_to_timedelta_safe(out["start_hour"])
    if "end_hour" in out.columns:
        out["end_hour_td"] = hours_to_timedelta_safe(out["end_hour"])

    out["issue_time_utc"] = pd.NaT
    if "creation_time_dt" in out.columns and out["creation_time_dt"].notna().any():
        out["issue_time_utc"] = out["creation_time_dt"]
    elif "system:time_start_dt" in out.columns:
        out["issue_time_utc"] = out["system:time_start_dt"]

    out["valid_time_utc"] = pd.NaT
    if "forecast_time_dt" in out.columns and out["forecast_time_dt"].notna().any():
        out["valid_time_utc"] = out["forecast_time_dt"]
    elif "issue_time_utc" in out.columns:
        if "forecast_hours_td" in out.columns and out["forecast_hours_td"].notna().any():
            out["valid_time_utc"] = out["issue_time_utc"] + out["forecast_hours_td"]
        elif "forecast_hour_td" in out.columns and out["forecast_hour_td"].notna().any():
            out["valid_time_utc"] = out["issue_time_utc"] + out["forecast_hour_td"]
        elif "forecast_step_td" in out.columns and out["forecast_step_td"].notna().any():
            out["valid_time_utc"] = out["issue_time_utc"] + out["forecast_step_td"]
        elif "end_hour_td" in out.columns and out["end_hour_td"].notna().any():
            out["valid_time_utc"] = out["issue_time_utc"] + out["end_hour_td"]
        elif "system:time_end_dt" in out.columns and out["system:time_end_dt"].notna().any():
            out["valid_time_utc"] = out["system:time_end_dt"]
        elif "system:time_start_dt" in out.columns:
            out["valid_time_utc"] = out["system:time_start_dt"]

    return out


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def dataframe_to_csv(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    out.to_csv(path, index=False)


def main():
    print("=" * 80)
    print("DMAP-AI / FV forecast source probe")
    print(f"Project ID  : {PROJECT_ID}")
    print(f"Workload tag: {WORKLOAD_TAG}")
    print(f"Point       : ({LAT}, {LON})")
    print(f"Now UTC     : {NOW_UTC.isoformat()}")
    print("=" * 80)

    authenticate_and_initialize()
    set_workload_tag()
    get_point()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overview_rows = []
    summary = {
        "project_id": PROJECT_ID,
        "workload_tag": WORKLOAD_TAG,
        "now_utc": NOW_UTC.isoformat(),
        "collections": {},
    }

    for key, collection_id in COLLECTIONS.items():
        print(f"\nInspecting {key}: {collection_id}")
        overview = summarize_collection(key, collection_id)
        overview_rows.append(overview)
        print(f"  recent_count={overview['recent_count']}")
        print(f"  latest_system_time_start_utc={overview['latest_system_time_start_utc']}")

        rows = get_selected_property_rows(collection_id, limit=12)
        save_json(OUT_DIR / f"{key}_property_samples_light.json", rows)

        df = pd.DataFrame(rows)
        if df.empty:
            summary["collections"][key] = {
                "collection_id": collection_id,
                "sample_rows": 0,
                "future_rows": 0,
                "max_valid_time_utc": None,
            }
            continue

        df = derive_time_columns(df)
        future_df = df[df["valid_time_utc"] > pd.Timestamp(NOW_UTC)] if "valid_time_utc" in df.columns else df.iloc[0:0].copy()
        max_valid = df["valid_time_utc"].max() if "valid_time_utc" in df.columns else None

        export_cols = [c for c in PROPERTY_CANDIDATES if c in df.columns]
        export_cols += [c for c in [
            "_band_count",
            "system:time_start_dt", "system:time_end_dt", "creation_time_dt", "forecast_time_dt",
            "issue_time_utc", "valid_time_utc"
        ] if c in df.columns]

        dataframe_to_csv(OUT_DIR / f"{key}_property_samples_light_derived.csv", df[export_cols])
        dataframe_to_csv(OUT_DIR / f"{key}_future_rows_light.csv", future_df[export_cols] if len(future_df) else pd.DataFrame(columns=export_cols))

        print(f"  sample_rows={len(df)}")
        print(f"  future_rows={len(future_df)}")
        print(f"  max_valid_time_utc={dt_to_str(max_valid)}")

        summary["collections"][key] = {
            "collection_id": collection_id,
            "sample_rows": int(len(df)),
            "future_rows": int(len(future_df)),
            "max_valid_time_utc": dt_to_str(max_valid),
            "property_columns": list(df.columns),
        }

    pd.DataFrame(overview_rows).to_csv(OUT_DIR / "forecast_source_overview_v3.csv", index=False)
    save_json(OUT_DIR / "forecast_test_summary_v3.json", summary)

    print("\nSaved files:")
    print(f"  {OUT_DIR / 'forecast_source_overview_v3.csv'}")
    print(f"  {OUT_DIR / 'forecast_test_summary_v3.json'}")
    for key in COLLECTIONS:
        print(f"  {OUT_DIR / (key + '_property_samples_light.json')}")
        print(f"  {OUT_DIR / (key + '_property_samples_light_derived.csv')}")
        print(f"  {OUT_DIR / (key + '_future_rows_light.csv')}")


if __name__ == "__main__":
    main()
