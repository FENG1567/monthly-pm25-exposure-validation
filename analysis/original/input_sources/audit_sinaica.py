"""Audit and re-aggregate the official SINAICA validated hourly PM2.5 files.

This is a source-QC product only. It does not fit or score a prediction model.
Raw per-station CSV files and the archived station metadata are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(r"F:\cand1_20261105\incoming\sinaica")
METADATA_PATH = Path(r"F:\cand1_20261105\work\sinaica\station_meta.json")
DOWNLOAD_ERRORS_PATH = Path(r"F:\cand1_20261105\work\sinaica\download_errors.json")
OUT = ROOT / "outputs" / "repair_v2" / "mexico_sinaica"
YEARS = list(range(2018, 2023))
MIN_VALID_HOURS_PER_DAY = 18
MIN_VALID_DAYS_PER_MONTH = 20
EXPECTED_FIELDS = ["date", "hour", "pm25", "valid"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_numeric(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    return values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))


def add_station_metadata(frame: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["iso"] = "MX"
    out["region"] = "mex"
    out["site_id"] = "MX_" + str(meta.get("station_id", ""))
    out["station_id"] = str(meta.get("station_id", ""))
    out["station_name"] = meta.get("station_name")
    out["state"] = meta.get("state")
    out["municipality"] = meta.get("municipio")
    out["network"] = meta.get("red")
    out["lat"] = pd.to_numeric(meta.get("lat"), errors="coerce")
    out["lon"] = pd.to_numeric(meta.get("lon"), errors="coerce")
    out["metadata_date_limits"] = "|".join(map(str, meta.get("limits_v") or []))
    out["coordinate_source"] = "SINAICA infoEstacion metadata snapshot; historical station moves are not supplied"
    return out


def aggregate_daily(valid_rows: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    if valid_rows.empty:
        return pd.DataFrame(columns=[
            "iso", "region", "site_id", "station_id", "station_name", "state", "municipality",
            "network", "lat", "lon", "date", "year", "month", "pm25", "n_valid_hours",
            "n_source_rows", "n_negative_source_rows", "n_duplicate_hour_rows", "metadata_date_limits",
            "coordinate_source",
        ])
    hourly = (valid_rows.groupby(["date", "hour"], as_index=False, dropna=False)
              .agg(pm25=("pm25", "mean"), n_source_rows=("pm25", "size"),
                   n_negative_source_rows=("is_negative", "sum")))
    hourly["n_duplicate_hour_rows"] = (hourly["n_source_rows"] - 1).clip(lower=0)
    daily = (hourly.groupby("date", as_index=False, dropna=False)
             .agg(pm25=("pm25", "mean"), n_valid_hours=("hour", "nunique"),
                  n_source_rows=("n_source_rows", "sum"),
                  n_negative_source_rows=("n_negative_source_rows", "sum"),
                  n_duplicate_hour_rows=("n_duplicate_hour_rows", "sum")))
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily.date.dt.year.astype("int16")
    daily["month"] = daily.date.dt.month.astype("int8")
    daily["day_eligible_18h"] = daily.n_valid_hours.ge(MIN_VALID_HOURS_PER_DAY)
    return add_station_metadata(daily, meta)


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=[
            "iso", "region", "site_id", "station_id", "station_name", "state", "municipality", "network",
            "year", "month", "pm25", "ndays", "n_valid_hours", "n_source_rows",
            "n_negative_source_rows", "n_duplicate_hour_rows", "lat", "lon", "metadata_date_limits",
            "coordinate_source", "month_eligible_20d",
        ])
    eligible_days = daily.loc[daily.n_valid_hours.ge(MIN_VALID_HOURS_PER_DAY)].copy()
    monthly = (eligible_days.groupby([
        "iso", "region", "site_id", "station_id", "station_name", "state", "municipality", "network",
        "year", "month", "lat", "lon", "metadata_date_limits", "coordinate_source",
    ], as_index=False, dropna=False)
      .agg(pm25=("pm25", "mean"), ndays=("date", "nunique"),
           n_valid_hours=("n_valid_hours", "sum"), n_source_rows=("n_source_rows", "sum"),
           n_negative_source_rows=("n_negative_source_rows", "sum"),
           n_duplicate_hour_rows=("n_duplicate_hour_rows", "sum")))
    monthly["month_eligible_20d"] = monthly.ndays.ge(MIN_VALID_DAYS_PER_MONTH)
    return monthly


def main(argv: list[str] | None = None) -> None:
    global SOURCE_ROOT, METADATA_PATH, DOWNLOAD_ERRORS_PATH, OUT
    parser = argparse.ArgumentParser(description="Audit the saved SINAICA PM2.5 hourly files.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--download-errors-path", type=Path, default=DOWNLOAD_ERRORS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    SOURCE_ROOT = args.source_root
    METADATA_PATH = args.metadata_path
    DOWNLOAD_ERRORS_PATH = args.download_errors_path
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise TypeError("SINAICA station metadata must be a JSON list")
    meta_by_id: dict[str, dict[str, Any]] = {}
    duplicate_meta_ids: list[str] = []
    for row in metadata:
        sid = str(row.get("station_id", "")).strip()
        if sid in meta_by_id:
            duplicate_meta_ids.append(sid)
        else:
            meta_by_id[sid] = row

    file_manifest: list[dict[str, Any]] = []
    flag_parts: list[pd.DataFrame] = []
    raw_rows_by_file: list[dict[str, Any]] = []
    source_valid_daily_parts: list[pd.DataFrame] = []
    nonnegative_daily_parts: list[pd.DataFrame] = []
    unmatched_files: list[dict[str, Any]] = []

    pattern = re.compile(r"sinaica_(.+)_(\d{4})_pm25_hourly\.csv$", re.IGNORECASE)
    paths = sorted(p for p in SOURCE_ROOT.glob("*/sinaica_*_pm25_hourly.csv") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No SINAICA hourly CSV files under {SOURCE_ROOT}")

    for path in paths:
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        sid, file_year = match.group(1), int(match.group(2))
        directory_year = int(path.parent.name) if path.parent.name.isdigit() else None
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        schema_ok = list(raw.columns) == EXPECTED_FIELDS
        if not schema_ok:
            raise ValueError(f"Unexpected CSV schema in {path}: {list(raw.columns)}")

        date = pd.to_datetime(raw["date"].str.strip(), format="%Y-%m-%d", errors="coerce")
        hour = pd.to_numeric(raw["hour"].str.strip(), errors="coerce")
        value = pd.to_numeric(raw["pm25"].str.strip(), errors="coerce")
        flag = raw["valid"].astype(str).str.strip()
        valid_flag = pd.to_numeric(flag, errors="coerce").eq(1)
        finite = finite_numeric(raw["pm25"])
        date_valid = date.notna()
        file_year_match = date_valid & date.dt.year.eq(file_year)
        hour_integral = hour.notna() & np.isfinite(hour.to_numpy(dtype=float, na_value=np.nan)) & hour.mod(1).eq(0)
        hour_valid = hour_integral & hour.between(0, 23)
        date_hour_value_ok = date_valid & file_year_match & hour_valid & finite
        source_valid = date_hour_value_ok & valid_flag
        negative_source_valid = source_valid & value.lt(0)

        dup_key = pd.DataFrame({"date": date, "hour": hour})
        duplicate_rows = int(dup_key.duplicated(["date", "hour"], keep="first").sum())
        raw["_year"] = file_year
        raw["_valid_flag"] = flag.replace("", "<blank>")
        flag_parts.append(raw[["_year", "_valid_flag"]].rename(columns={"_year": "year", "_valid_flag": "valid_flag"}))

        metadata_row = meta_by_id.get(sid)
        if metadata_row is None:
            unmatched_files.append({"station_id": sid, "year": file_year, "file": str(path)})
            metadata_row = {"station_id": sid}

        if source_valid.any():
            core = pd.DataFrame({
                "date": date.loc[source_valid].dt.date,
                "hour": hour.loc[source_valid].astype("int16"),
                "pm25": value.loc[source_valid].astype(float),
                "is_negative": value.loc[source_valid].lt(0).astype("int8"),
            })
            source_daily = aggregate_daily(core, metadata_row)
            source_valid_daily_parts.append(source_daily)
            nonnegative_daily_parts.append(aggregate_daily(core.loc[core.pm25.ge(0)].copy(), metadata_row))
        else:
            source_daily = aggregate_daily(pd.DataFrame(), metadata_row)
            nonnegative_daily_parts.append(source_daily)

        raw_rows_by_file.append({
            "year": file_year,
            "directory_year": directory_year,
            "station_id": sid,
            "file": path.name,
            "source_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "raw_rows": int(len(raw)),
            "valid_flag_1_rows": int(valid_flag.sum()),
            "valid_flag_0_rows": int(pd.to_numeric(flag, errors="coerce").eq(0).sum()),
            "other_or_blank_valid_flag_rows": int((~pd.to_numeric(flag, errors="coerce").isin([0, 1])).sum()),
            "invalid_date_rows": int((~date_valid).sum()),
            "date_not_in_filename_year_rows": int((date_valid & ~file_year_match).sum()),
            "invalid_hour_rows": int((~hour_valid).sum()),
            "nonnumeric_or_nonfinite_pm25_rows": int((~finite).sum()),
            "valid_flag_1_candidate_rows": int(source_valid.sum()),
            "negative_valid_flag_1_rows": int(negative_source_valid.sum()),
            "valid_flag_1_gt500_rows": int((source_valid & value.gt(500)).sum()),
            "valid_flag_1_gt1000_rows": int((source_valid & value.gt(1000)).sum()),
            "duplicate_date_hour_rows_all_flags": duplicate_rows,
            "distinct_valid_flag_1_date_hours": int(core.shape[0] - core.duplicated(["date", "hour"]).sum()) if source_valid.any() else 0,
            "min_source_date": date.loc[date_valid].min().date().isoformat() if date_valid.any() else "",
            "max_source_date": date.loc[date_valid].max().date().isoformat() if date_valid.any() else "",
            "metadata_present": metadata_row.get("station_name") is not None,
            "metadata_lists_pm25": "PM2.5" in (metadata_row.get("params_v") or []),
            "metadata_lat": metadata_row.get("lat"),
            "metadata_lon": metadata_row.get("lon"),
        })

    inventory = pd.DataFrame(raw_rows_by_file)
    if inventory.empty:
        raise RuntimeError("No SINAICA files were processed")
    inventory = inventory.sort_values(["year", "station_id"]).reset_index(drop=True)
    inventory.to_csv(OUT / "raw_file_inventory_2018_2022.csv", index=False)

    all_meta = pd.DataFrame(metadata)
    pm_meta = all_meta.loc[all_meta.params_v.map(lambda x: "PM2.5" in (x or []))].copy()
    pm_meta["station_id"] = pm_meta.station_id.astype(str)
    pm_meta["lat"] = pd.to_numeric(pm_meta.lat, errors="coerce")
    pm_meta["lon"] = pd.to_numeric(pm_meta.lon, errors="coerce")
    pm_meta["has_valid_coordinate_pair"] = pm_meta.lat.between(-90, 90) & pm_meta.lon.between(-180, 180)
    pm_meta["metadata_date_limits"] = pm_meta.limits_v.map(lambda x: "|".join(map(str, x or [])))
    pm_meta[["station_id", "station_name", "state", "municipio", "red", "lat", "lon",
             "has_valid_coordinate_pair", "metadata_date_limits", "error"]].to_csv(
        OUT / "station_metadata_snapshot.csv", index=False)

    flags = pd.concat(flag_parts, ignore_index=True)
    flag_counts = flags.groupby(["year", "valid_flag"], as_index=False).size().rename(columns={"size": "rows"})
    flag_counts.to_csv(OUT / "valid_flag_counts_2018_2022.csv", index=False)

    file_year_flow = (inventory.groupby("year", as_index=False)
                      .agg(files=("file", "size"), raw_rows=("raw_rows", "sum"),
                           valid_flag_1_rows=("valid_flag_1_rows", "sum"),
                           valid_flag_0_rows=("valid_flag_0_rows", "sum"),
                           other_flag_rows=("other_or_blank_valid_flag_rows", "sum"),
                           invalid_date_rows=("invalid_date_rows", "sum"),
                           date_not_in_filename_year_rows=("date_not_in_filename_year_rows", "sum"),
                           invalid_hour_rows=("invalid_hour_rows", "sum"),
                           nonnumeric_or_nonfinite_pm25_rows=("nonnumeric_or_nonfinite_pm25_rows", "sum"),
                           source_valid_candidate_rows=("valid_flag_1_candidate_rows", "sum"),
                           negative_valid_rows=("negative_valid_flag_1_rows", "sum"),
                           gt500_valid_rows=("valid_flag_1_gt500_rows", "sum"),
                           gt1000_valid_rows=("valid_flag_1_gt1000_rows", "sum"),
                           duplicate_date_hour_rows=("duplicate_date_hour_rows_all_flags", "sum"),
                           station_files_missing_metadata=("metadata_present", lambda s: int((~s).sum())),
                           station_files_metadata_not_pm25=("metadata_lists_pm25", lambda s: int((~s).sum()))))

    source_daily_all = pd.concat(source_valid_daily_parts, ignore_index=True) if source_valid_daily_parts else pd.DataFrame()
    nonnegative_daily_all = pd.concat(nonnegative_daily_parts, ignore_index=True) if nonnegative_daily_parts else pd.DataFrame()
    source_daily_all.to_parquet(OUT / "station_day_source_valid_2018_2022.parquet", index=False)
    source_month_all = aggregate_monthly(source_daily_all)
    nonnegative_month_all = aggregate_monthly(nonnegative_daily_all)
    primary_month = source_month_all.loc[source_month_all.month_eligible_20d].copy()
    nonnegative_month = nonnegative_month_all.loc[nonnegative_month_all.month_eligible_20d].copy()
    primary_month.to_parquet(OUT / "station_month_2018_2022.parquet", index=False)
    nonnegative_month.to_parquet(OUT / "station_month_nonnegative_sensitivity_2018_2022.parquet", index=False)
    source_month_all.to_parquet(OUT / "station_month_candidates_all_days_2018_2022.parquet", index=False)
    nonnegative_month_all.to_parquet(OUT / "station_month_nonnegative_candidates_all_days_2018_2022.parquet", index=False)

    pcols = ["site_id", "year", "month", "pm25", "ndays"]
    ncols = ["site_id", "year", "month", "pm25", "ndays"]
    comparison = source_month_all[pcols].rename(columns={"pm25": "source_valid_pm25", "ndays": "source_valid_ndays"}).merge(
        nonnegative_month_all[ncols].rename(columns={"pm25": "nonnegative_pm25", "ndays": "nonnegative_ndays"}),
        on=["site_id", "year", "month"], how="outer", indicator=True)
    comparison["sensitivity_delta_ug_m3"] = comparison.nonnegative_pm25 - comparison.source_valid_pm25
    comparison["month_eligibility_change"] = comparison["_merge"].astype(str).map({
        "both": "present_in_both", "left_only": "source_valid_only", "right_only": "nonnegative_only"})
    comparison.drop(columns="_merge").to_csv(OUT / "negative_value_sensitivity_comparison_2018_2022.csv", index=False)

    # Station-year coverage after the raw row checks and temporal aggregation.
    if not source_daily_all.empty:
        day_summary = (source_daily_all.groupby(["station_id", "year"], as_index=False)
                       .agg(source_valid_days=("date", "nunique"),
                            days_ge18_valid_hours=("day_eligible_18h", "sum"),
                            valid_hour_count=("n_valid_hours", "sum"),
                            min_hours_per_day=("n_valid_hours", "min"),
                            median_hours_per_day=("n_valid_hours", "median"),
                            max_hours_per_day=("n_valid_hours", "max"),
                            negative_source_rows_in_days=("n_negative_source_rows", "sum"),
                            duplicate_hour_rows_in_days=("n_duplicate_hour_rows", "sum")))
    else:
        day_summary = pd.DataFrame(columns=["station_id", "year", "source_valid_days", "days_ge18_valid_hours"])
    month_summary = (source_month_all.groupby(["station_id", "year"], as_index=False)
                     .agg(candidate_months=("month", "nunique"), months_ge20_days=("month_eligible_20d", "sum")))
    station_year = inventory.merge(day_summary, on=["station_id", "year"], how="left").merge(
        month_summary, on=["station_id", "year"], how="left")
    station_year.to_csv(OUT / "station_year_coverage_2018_2022.csv", index=False)
    file_year_flow = file_year_flow.merge(
        primary_month.groupby("year", as_index=False).agg(primary_station_months=("site_id", "size"),
                                                           primary_stations=("site_id", "nunique")),
        on="year", how="left")
    file_year_flow = file_year_flow.merge(
        nonnegative_month.groupby("year", as_index=False).agg(nonnegative_station_months=("site_id", "size")),
        on="year", how="left")
    file_year_flow.to_csv(OUT / "qc_flow_2018_2022.csv", index=False)

    # Expected station-years are based only on the saved official PM2.5 metadata date limits.
    expected: set[tuple[str, int]] = set()
    for row in pm_meta.to_dict("records"):
        limits = row.get("limits_v") or []
        lo = str(limits[0])[:10] if limits else ""
        hi = str(limits[1])[:10] if len(limits) > 1 else ""
        for year in YEARS:
            if (not lo or lo <= f"{year}-12-31") and (not hi or hi >= f"{year}-01-01"):
                expected.add((str(row["station_id"]), year))
    actual = set(zip(inventory.station_id.astype(str), inventory.year.astype(int)))
    missing_expected = sorted(expected - actual)
    unexpected_actual = sorted(actual - expected)
    meta_no_coordinate = pm_meta.loc[~pm_meta.has_valid_coordinate_pair, "station_id"].astype(str).tolist()
    metadata_duplicate_ids = sorted(set(duplicate_meta_ids))

    year_primary = primary_month.groupby("year").size().to_dict()
    year_sites = primary_month.groupby("year").site_id.nunique().to_dict()
    year_nonnegative = nonnegative_month.groupby("year").size().to_dict()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "data_type": "SINAICA validated historical PM2.5 observations",
            "request_provenance_script": str(Path(r"F:\cand1_20261105\work\scripts\download_sinaica.py")),
            "request_semantics_from_saved_script": "getDatos; paramId=PM2.5; fechaInit/fechaFin by year; nivel=1000; limit=100000; soloVal=1; output valid is API validoAct",
            "source_root": str(SOURCE_ROOT),
            "metadata_path": str(METADATA_PATH),
            "metadata_sha256": sha256(METADATA_PATH),
            "download_errors_path": str(DOWNLOAD_ERRORS_PATH),
            "download_errors": json.loads(DOWNLOAD_ERRORS_PATH.read_text(encoding="utf-8")) if DOWNLOAD_ERRORS_PATH.exists() else None,
            "official_validated_page": "https://sinaica.inecc.gob.mx/data.php?tipo=V",
            "official_source_description": "https://sinaica.inecc.gob.mx/index.php",
            "official_pm25_unit_reference": "https://sinaica.inecc.gob.mx/archivo/noms/NOM-172-SEMARNAT-2023-Indice-AIRE-y-SALUD.pdf",
        },
        "scope": {"years": YEARS, "source_files": int(len(inventory)), "source_rows": int(inventory.raw_rows.sum()),
                  "pm25_station_metadata_rows": int(len(pm_meta)), "source_file_station_ids": int(inventory.station_id.nunique()),
                  "source_file_hashes_in_raw_file_inventory": True},
        "station_year_coverage": {
            "expected_station_years_from_metadata_limits": int(len(expected)),
            "observed_station_year_files": int(len(actual)),
            "missing_expected_station_year_files": [{"station_id": sid, "year": year} for sid, year in missing_expected],
            "observed_station_years_outside_metadata_limits": [{"station_id": sid, "year": year} for sid, year in unexpected_actual],
            "metadata_without_valid_coordinate_pair": meta_no_coordinate,
            "duplicate_metadata_station_ids": metadata_duplicate_ids,
        },
        "qc_rules": {
            "timestamps": "Preserve date and hour exactly as returned in the SINAICA API CSV; no UTC conversion or invented timezone shift.",
            "date_rule": "Parse YYYY-MM-DD and require the year to equal the station-year filename.",
            "hour_rule": "Require integral hour 0..23; report and exclude other values.",
            "source_status_rule": "Canonical source-valid series retains rows where numeric valid == 1, PM2.5 is finite, and date/hour pass checks; no upper cap or winsorization.",
            "negative_values": "Retained in the source-valid canonical values because the source labels them validated; reported explicitly, with a separate nonnegative-only sensitivity that excludes rather than clips them.",
            "duplicates": "If multiple accepted rows map to the same station-date-hour, average within hour and report duplicate rows.",
            "day_rule": f"Require at least {MIN_VALID_HOURS_PER_DAY} distinct accepted hours per station-day.",
            "month_rule": f"Average retained station-day means and require at least {MIN_VALID_DAYS_PER_MONTH} distinct retained days per station-month.",
            "units": "Use µg/m³ based on the official Mexican PM2.5 concentration standard; do not numerically convert values.",
            "coordinates": "Use the saved SINAICA infoEstacion station snapshot; no historical coordinate version is present in the hourly files.",
        },
        "primary_station_months_by_year": {str(k): int(v) for k, v in year_primary.items()},
        "primary_stations_by_year": {str(k): int(v) for k, v in year_sites.items()},
        "nonnegative_sensitivity_months_by_year": {str(k): int(v) for k, v in year_nonnegative.items()},
        "yearly_flow_csv": str(OUT / "qc_flow_2018_2022.csv"),
        "no_final_model_scoring_performed": True,
    }
    (OUT / "qc_summary_2018_2022.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    flow_lines = [
        "| Year | Files | Raw rows | valid=1 rows | Valid source candidates | Negative candidate rows | Primary station-months | Sites |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in file_year_flow.sort_values("year").to_dict("records"):
        primary_months = row.get("primary_station_months")
        primary_sites = row.get("primary_stations")
        primary_months = int(primary_months) if pd.notna(primary_months) else 0
        primary_sites = int(primary_sites) if pd.notna(primary_sites) else 0
        flow_lines.append(
            f"| {int(row['year'])} | {int(row['files'])} | {int(row['raw_rows'])} | {int(row['valid_flag_1_rows'])} "
            f"| {int(row['source_valid_candidate_rows'])} | {int(row['negative_valid_rows'])} "
            f"| {primary_months} | {primary_sites} |"
        )
    report = "\n".join([
        "# SINAICA PM2.5 source QC, 2018–2022",
        "",
        f"Generated {summary['generated_utc']}. This is a source-QC and aggregation artifact; no model was fit or scored.",
        "",
        "## Source and fields",
        "",
        f"Inputs are the official per-station/year CSVs under `{SOURCE_ROOT}` and station metadata from `{METADATA_PATH}`. "
        "The saved downloader code requests SINAICA `getDatos` PM2.5 for each calendar year with `soloVal=1`; it writes `fecha`, `hora`, `valorAct`, and `validoAct` as `date`, `hour`, `pm25`, and `valid`. "
        "The official `Datos validados` page says these values passed validation mechanisms and may be considered definitive. The SINAICA site describes validation as a process specific to each monitoring system and quality-assurance plan, so the tag is source status rather than independent validation. "
        "PM2.5 mass concentration is kept in µg/m³ per the official Mexican standard. [Validated data page](https://sinaica.inecc.gob.mx/data.php?tipo=V); "
        "[SINAICA validation definitions](https://sinaica.inecc.gob.mx/index.php); "
        "[NOM-172-SEMARNAT-2023](https://sinaica.inecc.gob.mx/archivo/noms/NOM-172-SEMARNAT-2023-Indice-AIRE-y-SALUD.pdf).",
        "",
        "Date and hour remain exactly as returned by the API; no UTC shift is invented. Source coordinates are from the saved `infoEstacion` metadata snapshot, not historical station coordinates.",
        "",
        "## QC and aggregation",
        "",
        "The source-valid monthly table requires numeric `valid == 1`, parseable `YYYY-MM-DD` whose year matches the source filename, integral hours 0–23, and finite PM2.5. It preserves negative and high values as supplied (no clipping or upper cap), averages duplicate records within station-date-hour, requires at least 18 distinct accepted hours per day, then averages eligible daily means and requires at least 20 days per station-month. A separate nonnegative-only sensitivity removes negative rows before the same aggregation rules.",
        "These 18-hour/20-day support cutoffs are analysis rules for consistent monthly coverage, not SINAICA validation standards. Detailed `valid` codes, numeric outliers, duplicate timestamps, date/hours, negative readings, station metadata coverage, and file-level SHA-256s are in the CSV inventory and QC flow outputs.",
        "",
        "## Yearly flow",
        "",
        *flow_lines,
        "",
        "## Outputs",
        "",
        "- `station_month_2018_2022.parquet`: source-valid monthly observations, suitable as the canonical clean source table pending downstream eligibility review.",
        "- `station_month_nonnegative_sensitivity_2018_2022.parquet`: same aggregation with negative source values excluded, never clipped.",
        "- `station_month_candidates_all_days_2018_2022.parquet`: all months with at least one day having 18+ accepted hours, with `month_eligible_20d` for audit.",
        "- `station_day_source_valid_2018_2022.parquet`, `station_year_coverage_2018_2022.csv`, `qc_flow_2018_2022.csv`, `valid_flag_counts_2018_2022.csv`, `raw_file_inventory_2018_2022.csv`, and `station_metadata_snapshot.csv`.",
        "- `negative_value_sensitivity_comparison_2018_2022.csv`, `qc_summary_2018_2022.json`, and `provenance.json`.",
        "",
        "This audit does not repair or overwrite any inherited model-target table and contains no final model outputs.",
        "",
    ])
    (OUT / "README.md").write_text(report, encoding="utf-8")

    script_hash = sha256(Path(__file__))
    outputs = {}
    for name in [
        "raw_file_inventory_2018_2022.csv", "station_metadata_snapshot.csv", "valid_flag_counts_2018_2022.csv",
        "station_day_source_valid_2018_2022.parquet", "station_month_2018_2022.parquet",
        "station_month_nonnegative_sensitivity_2018_2022.parquet", "station_year_coverage_2018_2022.csv",
        "qc_flow_2018_2022.csv", "negative_value_sensitivity_comparison_2018_2022.csv",
        "qc_summary_2018_2022.json", "README.md",
    ]:
        path = OUT / name
        outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    provenance = {"script": str(Path(__file__)), "script_sha256": script_hash,
                  "source_root": str(SOURCE_ROOT), "metadata_sha256": sha256(METADATA_PATH),
                  "outputs": outputs, "generated_utc": summary["generated_utc"]}
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "complete", "out": str(OUT), "source_files": len(inventory),
                      "source_rows": int(inventory.raw_rows.sum()), "primary_months": int(len(primary_month)),
                      "primary_stations": int(primary_month.site_id.nunique()), "yearly_flow": file_year_flow.to_dict("records"),
                      "summary": str(OUT / "qc_summary_2018_2022.json"), "provenance": str(OUT / "provenance.json")},
                     ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
