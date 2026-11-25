"""Build a versioned, provisional Mexico station-month eligibility cohort.

This script reads the frozen SINAICA QC artifacts and current saved source
metadata. It does not overwrite the source-valid or nonnegative sensitivity
tables and does not fit or score a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "repair_v2" / "mexico_sinaica"
OUT = BASE / "model_eligibility_v1"
METADATA_PATH = Path(r"F:\cand1_20261105\work\sinaica\station_meta.json")
SOURCE_ROOT = Path(r"F:\cand1_20261105\incoming\sinaica")
PROBES = Path(r"F:\cand1_repair_v2_data\sinaica\probes")
WORK_SCRIPTS = Path(r"F:\cand1_20261105\work\scripts")
YEARS = list(range(2018, 2023))
MIN_DAYS = 20
QUARANTINE_IDS = {"503", "504"}
QUARANTINE_YEAR = 2022
KEYS = ["country", "site_id", "year", "month"]
SOURCE_VALID_ALL = "station_month_candidates_all_days_2018_2022.parquet"
NONNEGATIVE_ALL = "station_month_nonnegative_candidates_all_days_2018_2022.parquet"
SOURCE_VALID_ELIGIBLE = "station_month_2018_2022.parquet"
NONNEGATIVE_ELIGIBLE = "station_month_nonnegative_sensitivity_2018_2022.parquet"

PRIMARY_POLICY = "nonnegative_hourly_source_valid_quarantine_503_504_2022"
SECONDARY_POLICY = "source_valid_nonnegative_month_quarantine_503_504_2022"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def assert_month_keys(frame: pd.DataFrame, label: str, unique: bool = True) -> None:
    absent = [column for column in KEYS if column not in frame.columns]
    if absent:
        raise ValueError(f"{label} is missing key columns: {absent}")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{label} has null month keys")
    if unique and frame.duplicated(KEYS).any():
        raise ValueError(f"{label} has duplicated month keys")


def _station_key(frame: pd.DataFrame) -> pd.Series:
    if "station_id" in frame:
        return frame.station_id.fillna("").astype(str).str.strip()
    return frame.site_id.fillna("").astype(str).str.replace("^MX_", "", regex=True)


def annotate_policy(candidates: pd.DataFrame, policy_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the stated day, coordinate, outcome, and quarantine rules."""
    frame = candidates.copy()
    frame["country"] = "MEX"
    frame["iso"] = "MX"
    frame["region"] = "mex"
    frame["station_id"] = _station_key(frame)
    frame["pm25"] = pd.to_numeric(frame.pm25, errors="coerce")
    frame["ndays"] = pd.to_numeric(frame.ndays, errors="coerce")
    frame["year"] = pd.to_numeric(frame.year, errors="coerce").astype("Int64")
    frame["month"] = pd.to_numeric(frame.month, errors="coerce").astype("Int64")
    frame["lat"] = pd.to_numeric(frame.lat, errors="coerce")
    frame["lon"] = pd.to_numeric(frame.lon, errors="coerce")

    month_rule = frame.get("month_eligible_20d", pd.Series(False, index=frame.index))
    frame["support_ok"] = month_rule.fillna(False).astype(bool) & frame.ndays.ge(MIN_DAYS)
    lat_finite = np.isfinite(frame.lat.to_numpy(dtype=float, na_value=np.nan))
    lon_finite = np.isfinite(frame.lon.to_numpy(dtype=float, na_value=np.nan))
    frame["coordinate_pair_valid"] = (
        lat_finite & lon_finite & frame.lat.between(-90, 90).to_numpy()
        & frame.lon.between(-180, 180).to_numpy()
    )
    frame["target_finite"] = np.isfinite(frame.pm25.to_numpy(dtype=float, na_value=np.nan))
    frame["negative_monthly_target"] = frame.pm25.lt(0).fillna(False)
    frame["quarantined_station_year"] = (
        frame.station_id.isin(QUARANTINE_IDS) & frame.year.eq(QUARANTINE_YEAR)
    )

    reason_columns = [
        ("fewer_than_20_eligible_days", ~frame.support_ok),
        ("missing_or_out_of_range_source_coordinates", ~frame.coordinate_pair_valid),
        ("nonfinite_monthly_target", ~frame.target_finite),
        ("negative_monthly_outcome", frame.negative_monthly_target),
        ("duplicate_implausible_source_station_year_quarantine", frame.quarantined_station_year),
    ]
    frame["rejection_reasons"] = [
        ";".join(name for name, mask in reason_columns if bool(mask.loc[index]))
        for index in frame.index
    ]
    frame["policy_id"] = policy_id
    frame["source_policy"] = (
        "nonnegative-hourly source-valid sensitivity; negative hourly observations excluded before aggregation"
        if policy_id == PRIMARY_POLICY else
        "source-valid monthly aggregation; negative monthly target outcomes excluded from this model cohort"
    )
    frame["source_status_semantics"] = "API validoAct mapped to CSV valid; request soloVal=1"
    frame["eligible"] = frame.rejection_reasons.eq("")
    frame["target_extreme_gt500_flag"] = frame.pm25.gt(500)
    frame["target_extreme_gt1000_flag"] = frame.pm25.gt(1000)
    frame["coordinate_validation"] = "finite source-snapshot pair within latitude/longitude bounds"
    frame["eligibility_status"] = "provisional; broad source QC is not certification"

    accepted = frame.loc[frame.eligible].copy()
    rejected = frame.loc[~frame.eligible].copy()
    return accepted, rejected


def add_metadata_limits(candidate_audit: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise TypeError("SINAICA station metadata must be a JSON list")
    meta_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for item in metadata:
        station_id = str(item.get("station_id", "")).strip()
        if station_id in meta_by_id:
            duplicate_ids.add(station_id)
        else:
            meta_by_id[station_id] = item
    if duplicate_ids:
        raise ValueError(f"Duplicate station metadata IDs: {sorted(duplicate_ids)}")

    rows: list[dict[str, Any]] = []
    for station_id in candidate_audit.station_id.astype(str):
        item = meta_by_id.get(station_id, {})
        limits = item.get("limits_v") or []
        rows.append({
            "limits_v_start": limits[0] if len(limits) > 0 else None,
            "limits_v_end": limits[1] if len(limits) > 1 else None,
            "metadata_lists_pm25": "PM2.5" in (item.get("params_v") or []),
            "metadata_present": bool(item),
        })
    limit_meta = pd.DataFrame(rows, index=candidate_audit.index)
    out = pd.concat([candidate_audit.copy(), limit_meta], axis=1)
    out["month_start"] = pd.to_datetime(
        out.year.astype(str) + "-" + out.month.astype(str).str.zfill(2) + "-01", errors="coerce")
    out["month_end"] = out.month_start + pd.offsets.MonthEnd(0)
    lower = pd.to_datetime(out.limits_v_start, errors="coerce")
    upper = pd.to_datetime(out.limits_v_end, errors="coerce")
    out["outside_limits_v"] = (
        (lower.notna() & out.month_end.lt(lower))
        | (upper.notna() & out.month_start.gt(upper))
    )
    out["partial_boundary_month"] = (
        (lower.notna() & out.month_start.lt(lower) & out.month_end.ge(lower))
        | (upper.notna() & out.month_start.le(upper) & out.month_end.gt(upper))
    )
    out["limits_v_relationship"] = np.select(
        [~out.metadata_present, ~out.metadata_lists_pm25, out.outside_limits_v, out.partial_boundary_month],
        ["metadata_missing", "PM2.5_not_in_params_v", "outside_limits_v", "partial_boundary_month"],
        default="within_or_open_limits_v",
    )
    return out


def make_flow(annotated: pd.DataFrame) -> pd.DataFrame:
    flow: list[dict[str, Any]] = []
    for (policy_id, year), group in annotated.groupby(["policy_id", "year"], dropna=False):
        after_support = group.support_ok
        after_coordinates = after_support & group.coordinate_pair_valid
        after_target = after_coordinates & group.target_finite & ~group.negative_monthly_target
        after_quarantine = after_target & ~group.quarantined_station_year
        flow.append({
            "policy_id": policy_id,
            "year": int(year),
            "candidate_months": int(len(group)),
            "after_20_day_support": int(after_support.sum()),
            "after_coordinate_validation": int(after_coordinates.sum()),
            "after_finite_nonnegative_target": int(after_target.sum()),
            "after_duplicate_station_year_quarantine": int(after_quarantine.sum()),
            "final_eligible_months": int(group.eligible.sum()),
            "rejected_candidates": int((~group.eligible).sum()),
        })
    return pd.DataFrame(flow).sort_values(["policy_id", "year"]).reset_index(drop=True)


def build_policy_products(source_valid_candidates: pd.DataFrame,
                          nonnegative_candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    assert_month_keys(source_valid_candidates.assign(country="MEX"), "source-valid candidates")
    assert_month_keys(nonnegative_candidates.assign(country="MEX"), "nonnegative candidates")
    primary, primary_rejections = annotate_policy(nonnegative_candidates, PRIMARY_POLICY)
    secondary, secondary_rejections = annotate_policy(source_valid_candidates, SECONDARY_POLICY)
    for name, frame in [("primary", primary), ("secondary", secondary)]:
        assert_month_keys(frame, name)
        if frame.duplicated(["site_id", "year", "month"]).any():
            raise ValueError(f"{name} has duplicate station-months")
        if not frame.country.eq("MEX").all():
            raise ValueError(f"{name} country is not MEX throughout")
        if not frame.ndays.ge(MIN_DAYS).all() or not frame.pm25.ge(0).all():
            raise ValueError(f"{name} violates day support or nonnegative outcome rules")
        if not frame.coordinate_pair_valid.all():
            raise ValueError(f"{name} has missing or invalid coordinates")
        if frame.quarantined_station_year.any():
            raise ValueError(f"{name} retained quarantined station-year rows")

    comparison = primary.merge(
        secondary,
        on=["country", "iso", "region", "site_id", "station_id", "year", "month"],
        how="outer", suffixes=("_primary", "_secondary"), indicator=True,
        validate="one_to_one",
    )
    comparison["primary_eligible"] = comparison._merge.ne("right_only")
    comparison["secondary_eligible"] = comparison._merge.ne("left_only")
    comparison["policy_presence"] = comparison._merge.map({
        "both": "both_policies", "left_only": "primary_only", "right_only": "secondary_only"})
    comparison["target_delta_primary_minus_secondary"] = (
        comparison.pm25_primary - comparison.pm25_secondary)
    comparison["same_station_month_coordinate"] = (
        comparison.lat_primary.eq(comparison.lat_secondary)
        & comparison.lon_primary.eq(comparison.lon_secondary)
    )
    comparison.drop(columns="_merge", inplace=True)
    common = comparison.loc[comparison.policy_presence.eq("both_policies"), [
        "country", "iso", "region", "site_id", "station_id", "station_name_primary",
        "state_primary", "municipality_primary", "year", "month", "lat_primary", "lon_primary",
        "ndays_primary", "ndays_secondary", "pm25_primary", "pm25_secondary",
        "target_delta_primary_minus_secondary", "same_station_month_coordinate",
    ]].rename(columns={
        "station_name_primary": "station_name", "state_primary": "state",
        "municipality_primary": "municipality", "lat_primary": "lat", "lon_primary": "lon",
    })
    if common.duplicated(["country", "site_id", "year", "month"]).any():
        raise ValueError("Common-key resampling manifest has duplicated keys")
    flow = pd.concat([
        make_flow(pd.concat([primary, primary_rejections], ignore_index=True)),
        make_flow(pd.concat([secondary, secondary_rejections], ignore_index=True)),
    ], ignore_index=True)
    return {
        "primary": primary,
        "secondary": secondary,
        "primary_rejections": primary_rejections,
        "secondary_rejections": secondary_rejections,
        "comparison": comparison,
        "common_resampling_manifest": common,
        "flow": flow,
    }


def run_fixture() -> dict[str, int]:
    """Exercise all eligibility output keys before processing real artifacts."""
    base_rows = [
        ("MX_1", "1", 2020, 1, 20.0, 20, True, 20.0, -100.0),
        ("MX_2", "2", 2020, 2, -3.0, 22, True, 21.0, -101.0),
        ("MX_503", "503", 2022, 11, 2000.0, 22, True, 32.56, -116.62),
        ("MX_3", "3", 2020, 3, 30.0, 19, False, 22.0, -102.0),
        ("MX_4", "4", 2020, 4, 40.0, 22, True, np.nan, -103.0),
        ("MX_5", "5", 2020, 5, 800.0, 21, True, 23.0, -104.0),
    ]
    columns = ["site_id", "station_id", "year", "month", "pm25", "ndays",
               "month_eligible_20d", "lat", "lon"]
    source_valid = pd.DataFrame(base_rows, columns=columns)
    source_valid["station_name"] = ["one", "two", "Tecate", "three", "four", "five"]
    source_valid["state"] = "fixture"
    source_valid["municipality"] = "fixture"
    nonnegative = source_valid.copy()
    nonnegative.loc[nonnegative.station_id.eq("2"), "pm25"] = 0.0
    products = build_policy_products(source_valid, nonnegative)
    expected_names = {
        "primary", "secondary", "primary_rejections", "secondary_rejections",
        "comparison", "common_resampling_manifest", "flow",
    }
    if set(products) != expected_names:
        raise AssertionError(f"Fixture output names mismatch: {set(products)}")
    for name in ["primary", "secondary", "primary_rejections", "secondary_rejections",
                 "comparison", "common_resampling_manifest"]:
        if not set(KEYS).issubset(products[name].columns):
            raise AssertionError(f"Fixture {name} missing common output keys")
    if set(products["primary"].site_id) != {"MX_1", "MX_2", "MX_5"}:
        raise AssertionError("Fixture primary eligibility result is incorrect")
    if set(products["secondary"].site_id) != {"MX_1", "MX_5"}:
        raise AssertionError("Fixture secondary eligibility result is incorrect")
    if len(products["common_resampling_manifest"]) != 2:
        raise AssertionError("Fixture common-key intersection is incorrect")
    if not products["primary"].loc[products["primary"].site_id.eq("MX_5"), "target_extreme_gt500_flag"].all():
        raise AssertionError("Fixture positive high value was not retained and flagged")
    if not products["secondary_rejections"].rejection_reasons.str.contains("negative_monthly_outcome").any():
        raise AssertionError("Fixture did not record negative-target rejection")
    return {name: int(len(frame)) for name, frame in products.items()}


def get_source_field_review(metadata_path: Path) -> dict[str, Any]:
    downloader = WORK_SCRIPTS / "download_sinaica.py"
    valid_js = PROBES / "valid.js"
    probe_files = [PROBES / f"station_{sid}_20221108_response.json" for sid in ["503", "504", "102"]]
    responses = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in probe_files}
    band_summary = {}
    distinct_api_ids = {}
    for name, rows in responses.items():
        band_summary[name] = sorted({"<null>" if row.get("band") is None else str(row.get("band")) for row in rows})
        distinct_api_ids[name] = sorted({str(row.get("id", "")) for row in rows})
    json_row_fields = sorted(set().union(*(set(row) for row in responses["station_503_20221108_response"])))
    js = valid_js.read_text(encoding="utf-8")
    js_lines = js.splitlines()
    pm25_unit_line = next((
        js_lines[index + 1].strip()
        for index, line in enumerate(js_lines[:-1])
        if line.strip() == 'case "PM2.5":'
    ), "")
    unit_phrase = "m&sup3;" in pm25_unit_line and ("µg" in pm25_unit_line or "Âµg" in pm25_unit_line)
    downloader_text = downloader.read_text(encoding="utf-8")
    csv_drops_band = 'w.writerow([r.get("fecha"), r.get("hora"), v, r.get("validoAct")])' in downloader_text
    if not unit_phrase or not csv_drops_band:
        raise ValueError("Official unit/export convention could not be confirmed from saved source files")
    return {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "official_sources": {
            "validated_observations_page": "https://sinaica.inecc.gob.mx/data.php?tipo=V",
            "official_validation_definition": "https://sinaica.inecc.gob.mx/index.php",
            "official_indicators_page": "https://sinaica.inecc.gob.mx/scica/",
            "official_pm25_standard": "https://sinaica.inecc.gob.mx/archivo/noms/NOM-172-SEMARNAT-2023-Indice-AIRE-y-SALUD.pdf",
            "official_historical_qc_report": "https://sinaica.inecc.gob.mx/archivo/informes/Informe2016.pdf",
            "api_endpoint": "https://sinaica.inecc.gob.mx/lib/libd/cnxn.php",
        },
        "official_saved_source_files": {
            "download_script": str(downloader), "download_script_sha256": sha256(downloader),
            "valid_js": str(valid_js), "valid_js_sha256": sha256(valid_js),
            "station_metadata": str(metadata_path), "station_metadata_sha256": sha256(metadata_path),
            "api_probe_responses": [{"path": str(path), "sha256": sha256(path)} for path in probe_files],
        },
        "source_fields": {
            "current_getDatos_response_fields": json_row_fields,
            "archived_hourly_csv_fields": ["date", "hour", "pm25", "valid"],
            "csv_valid_field_maps_from_api": "validoAct; downloader request uses soloVal=1",
            "api_band_field_dropped_from_archived_csv": True,
            "band_values_in_saved_probes": band_summary,
            "api_ids_in_saved_probes": distinct_api_ids,
            "historical_band_cannot_be_recovered_from_archived_csv": True,
            "interpretation": "The API returns a band field, but the downloader omitted it. The saved 503/504 2022-11-08 probes contain blank band values; the 102 probe has null band. No band value is inferred for historical archived rows.",
        },
        "unit_convention": {
            "official_valid_js_getUnidad_pm25_micrograms_per_cubic_metre": True,
            "saved_valid_js_unit_line": pm25_unit_line,
            "unit_encoding_note": "The saved UTF-8 copy may display the micro symbol as a mojibake sequence; the live SINAICA indicators page and official standard specify PM2.5 concentrations in µg/m³.",
            "conversion_applied": False,
            "reason": "Official display code specifies µg/m³, and no source evidence justifies factor-1000 conversion.",
        },
        "validated_status_interpretation": {
            "soloVal_request": "1",
            "saved_query_maps_valid_from": "API validoAct",
            "caveat": "SINAICA defines validation according to each monitoring system and its quality-assurance plan; it is source validation, not independent revalidation by this audit.",
            "extreme_values": "Official historical QC description separates anomaly detection from subsequent validation or invalidation by responsible monitoring agencies. No local upper cutoff is applied.",
        },
    }


def duplicate_quarantine_evidence(metadata_path: Path) -> dict[str, Any]:
    station_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_id = {str(row.get("station_id")): row for row in station_meta}
    paths = [SOURCE_ROOT / "2022" / f"sinaica_{station_id}_2022_pm25_hourly.csv" for station_id in ["503", "504"]]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Required raw station-year file for quarantine evidence is missing")
    raw = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    hashes = [sha256(path) for path in paths]
    if raw[0].shape[0] != 1281 or raw[1].shape[0] != 1281:
        raise ValueError("503/504 raw source row counts differ from reviewed evidence")
    if hashes[0] != hashes[1] or paths[0].read_bytes() != paths[1].read_bytes():
        raise ValueError("503/504 station-year raw CSVs are no longer byte-identical")
    finite = [int(pd.to_numeric(frame.pm25, errors="coerce").notna().sum()) for frame in raw]
    if finite != [1281, 1281]:
        raise ValueError(f"Unexpected finite hourly counts in station-year duplicate: {finite}")

    probe_rows = {}
    for station_id in ["503", "504"]:
        probe = json.loads((PROBES / f"station_{station_id}_20221108_response.json").read_text(encoding="utf-8"))
        probe_rows[station_id] = probe
    if len(probe_rows["503"]) != len(probe_rows["504"]):
        raise ValueError("Official probe response row counts disagree")
    comparable = lambda rows: [
        {key: row.get(key) for key in ["parametro", "fecha", "hora", "valorAct", "band", "validoAct"]}
        for row in rows
    ]
    if comparable(probe_rows["503"]) != comparable(probe_rows["504"]):
        raise ValueError("Current API probe values for 503 and 504 no longer match")

    return {
        "quarantine_rule": "Exclude station IDs 503 and 504 for source year 2022 as a pair; do not exclude other years or stations.",
        "quarantine_site_ids": ["MX_503", "MX_504"],
        "quarantine_year": 2022,
        "raw_station_year_files": [
            {"station_id": station_id, "site_id": f"MX_{station_id}", "path": str(path),
             "bytes": path.stat().st_size, "sha256": digest, "raw_rows": len(frame),
             "finite_hourly_values": finite_count, "first_date": frame.date.min(),
             "last_date": frame.date.max(), "station_name": by_id[station_id].get("station_name"),
             "state": by_id[station_id].get("state"), "municipality": by_id[station_id].get("municipio"),
             "lat": by_id[station_id].get("lat"), "lon": by_id[station_id].get("lon"),
             "limits_v": by_id[station_id].get("limits_v")}
            for station_id, path, digest, frame, finite_count in zip(["503", "504"], paths, hashes, raw, finite)
        ],
        "byte_identical_pair": True,
        "official_api_probe": {
            "request_date": "2022-11-08", "paramId": "PM2.5", "soloVal": 1,
            "same_values_and_status_across_distinct_station_ids": True,
            "station_503_api_ids": sorted({row.get("id") for row in probe_rows["503"]}),
            "station_504_api_ids": sorted({row.get("id") for row in probe_rows["504"]}),
            "rows_per_station": len(probe_rows["503"]),
            "band_values": sorted({str(row.get("band")) for row in probe_rows["503"]}),
            "response_files_sha256": {station_id: sha256(PROBES / f"station_{station_id}_20221108_response.json") for station_id in ["503", "504"]},
        },
        "interpretive_limits": [
            "The current API reproduces identical same-date values for distinct station IDs, but does not identify the cause.",
            "Station 504 metadata name Ensenada with coordinates 32.60318, -115.39135 may have a location conflict; this is not independently adjudicated.",
            "Station 503 published limits_v starts in 2023, so its 2022 observations are outside that snapshot period; station 504 limits_v has no lower bound.",
            "No unit conversion or numerical clipping is applied.",
        ],
    }


def metadata_numeric_audit(source_candidates: pd.DataFrame, nonnegative_candidates: pd.DataFrame,
                           metadata_path: Path, primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    left = source_candidates.copy()
    right = nonnegative_candidates.copy()
    for frame in [left, right]:
        frame["station_id"] = _station_key(frame)
    left = left.rename(columns={
        "pm25": "pm25_source_valid", "ndays": "ndays_source_valid",
        "month_eligible_20d": "source_valid_month_eligible_20d",
        "n_negative_source_rows": "n_negative_source_rows_source_valid",
    })
    right = right.rename(columns={
        "pm25": "pm25_nonnegative_hourly", "ndays": "ndays_nonnegative_hourly",
        "month_eligible_20d": "nonnegative_month_eligible_20d",
        "n_negative_source_rows": "n_negative_source_rows_after_filter",
    })
    keep_left = ["site_id", "station_id", "year", "month", "station_name", "state", "municipality",
                 "lat", "lon", "metadata_date_limits", "pm25_source_valid", "ndays_source_valid",
                 "source_valid_month_eligible_20d", "n_negative_source_rows_source_valid"]
    keep_right = ["site_id", "year", "month", "pm25_nonnegative_hourly", "ndays_nonnegative_hourly",
                  "nonnegative_month_eligible_20d", "n_negative_source_rows_after_filter"]
    audit = left[keep_left].merge(right[keep_right], on=["site_id", "year", "month"], how="outer",
                                  validate="one_to_one", indicator="candidate_presence")
    audit = add_metadata_limits(audit, metadata_path)
    audit["source_valid_negative_month"] = audit.pm25_source_valid.lt(0)
    audit["source_valid_gt500_flag"] = audit.pm25_source_valid.gt(500)
    audit["source_valid_gt1000_flag"] = audit.pm25_source_valid.gt(1000)
    audit["nonnegative_hourly_gt500_flag"] = audit.pm25_nonnegative_hourly.gt(500)
    audit["nonnegative_hourly_gt1000_flag"] = audit.pm25_nonnegative_hourly.gt(1000)
    audit["quarantined_station_year"] = audit.station_id.isin(QUARANTINE_IDS) & audit.year.eq(QUARANTINE_YEAR)
    primary_keys = set(map(tuple, primary[["site_id", "year", "month"]].itertuples(index=False, name=None)))
    secondary_keys = set(map(tuple, secondary[["site_id", "year", "month"]].itertuples(index=False, name=None)))
    keys = list(map(tuple, audit[["site_id", "year", "month"]].itertuples(index=False, name=None)))
    audit["primary_policy_eligible"] = [key in primary_keys for key in keys]
    audit["secondary_policy_eligible"] = [key in secondary_keys for key in keys]
    audit["anomaly_flags"] = [
        ";".join(flag for flag, condition in [
            ("negative_source_valid_month", bool(negative)),
            ("source_valid_gt500", bool(gt500)),
            ("source_valid_gt1000", bool(gt1000)),
            ("nonnegative_hourly_gt500", bool(nn_gt500)),
            ("nonnegative_hourly_gt1000", bool(nn_gt1000)),
            ("quarantined_duplicate_station_year", bool(quarantine)),
        ] if condition)
        for negative, gt500, gt1000, nn_gt500, nn_gt1000, quarantine in zip(
            audit.source_valid_negative_month.fillna(False),
            audit.source_valid_gt500_flag.fillna(False),
            audit.source_valid_gt1000_flag.fillna(False),
            audit.nonnegative_hourly_gt500_flag.fillna(False),
            audit.nonnegative_hourly_gt1000_flag.fillna(False),
            audit.quarantined_station_year.fillna(False),
        )
    ]
    audit["upper_threshold_applied"] = False
    return audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build provisional Mexico station-month eligibility artifacts")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args(argv)

    fixture_counts = run_fixture()
    if args.fixture_only:
        print(json.dumps({"status": "fixture_complete", "fixture_output_rows": fixture_counts}, indent=2))
        return

    base, output = args.base, args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    source_valid_candidates = pd.read_parquet(base / SOURCE_VALID_ALL)
    nonnegative_candidates = pd.read_parquet(base / NONNEGATIVE_ALL)
    source_valid_eligible_reference = pd.read_parquet(base / SOURCE_VALID_ELIGIBLE)
    nonnegative_eligible_reference = pd.read_parquet(base / NONNEGATIVE_ELIGIBLE)
    products = build_policy_products(source_valid_candidates, nonnegative_candidates)
    primary, secondary = products["primary"], products["secondary"]

    # Verify the eligibility rules reproduce the frozen upstream tables after
    # applying the explicitly paired quarantine.
    def month_keyset(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
        return set(map(tuple, frame[["site_id", "year", "month"]].itertuples(index=False, name=None)))

    primary_reference = nonnegative_eligible_reference.copy()
    primary_reference["station_id"] = _station_key(primary_reference)
    primary_reference = primary_reference.loc[
        ~(primary_reference.station_id.isin(QUARANTINE_IDS) & primary_reference.year.eq(QUARANTINE_YEAR))]
    if month_keyset(primary) != month_keyset(primary_reference):
        raise ValueError("Primary cohort differs from the frozen nonnegative-hourly artifact after the paired quarantine")
    secondary_reference = source_valid_eligible_reference.loc[
        source_valid_eligible_reference.pm25.ge(0)].copy()
    secondary_reference["station_id"] = _station_key(secondary_reference)
    secondary_reference = secondary_reference.loc[
        ~(secondary_reference.station_id.isin(QUARANTINE_IDS) & secondary_reference.year.eq(QUARANTINE_YEAR))]
    if month_keyset(secondary) != month_keyset(secondary_reference):
        raise ValueError("Secondary cohort differs from the frozen source-valid table after negative-month and quarantine exclusions")

    # Ensure primary, secondary, common-key, and reject tables share explicit MEX keys.
    assert_month_keys(primary, "primary output")
    assert_month_keys(secondary, "secondary output")
    assert_month_keys(products["comparison"], "policy comparison")
    assert_month_keys(products["common_resampling_manifest"], "common-key resampling manifest")
    if not products["common_resampling_manifest"].same_station_month_coordinate.all():
        raise ValueError("Common policy keys disagree on station-month coordinates")
    for policy in ["primary", "secondary"]:
        table = products[policy]
        if table.pm25.gt(500).sum() and not table.loc[table.pm25.gt(500), "target_extreme_gt500_flag"].all():
            raise ValueError(f"{policy} high positive outcomes are not flagged")

    primary_name = "station_month_operational_primary_v1_2018_2022.parquet"
    secondary_name = "station_month_operational_secondary_v1_2018_2022.parquet"
    products["primary"].to_parquet(output / primary_name, index=False)
    products["secondary"].to_parquet(output / secondary_name, index=False)
    products["primary_rejections"].to_csv(output / "primary_policy_rejections_v1_2018_2022.csv", index=False)
    products["secondary_rejections"].to_csv(output / "secondary_policy_rejections_v1_2018_2022.csv", index=False)
    products["comparison"].to_csv(output / "policy_month_comparison_v1_2018_2022.csv", index=False)
    products["common_resampling_manifest"].to_csv(output / "common_key_resampling_manifest_v1_2018_2022.csv", index=False)
    products["flow"].to_csv(output / "eligibility_flow_v1_2018_2022.csv", index=False)

    numeric_audit = metadata_numeric_audit(source_valid_candidates, nonnegative_candidates,
                                           args.metadata_path, primary, secondary)
    numeric_audit.to_csv(output / "numeric_anomaly_metadata_limits_audit_v1_2018_2022.csv", index=False)
    anomaly_audit = numeric_audit.loc[numeric_audit.anomaly_flags.ne("")].copy()
    anomaly_audit.to_csv(output / "numeric_anomalies_vs_limits_v1_2018_2022.csv", index=False)
    duplicate_evidence = duplicate_quarantine_evidence(args.metadata_path)
    (output / "quarantine_evidence_v1.json").write_text(
        json.dumps(json_safe(duplicate_evidence), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    source_review = get_source_field_review(args.metadata_path)
    (output / "source_field_review_v1.json").write_text(
        json.dumps(json_safe(source_review), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    source_hashes = {
        name: sha256(base / name)
        for name in [SOURCE_VALID_ALL, NONNEGATIVE_ALL, SOURCE_VALID_ELIGIBLE, NONNEGATIVE_ELIGIBLE]
    }
    policies = {}
    for name, frame in [("primary", products["primary"]), ("secondary", products["secondary"])]:
        policies[name] = {
            "policy_id": PRIMARY_POLICY if name == "primary" else SECONDARY_POLICY,
            "input_candidate_artifact": NONNEGATIVE_ALL if name == "primary" else SOURCE_VALID_ALL,
            "eligible_months": int(len(frame)),
            "eligible_station_ids": int(frame.site_id.nunique()),
            "years": {str(int(year)): int(count) for year, count in frame.groupby("year").size().items()},
            "negative_monthly_targets": int(frame.pm25.lt(0).sum()),
            "gt500_positive_target_months_retained_and_flagged": int(frame.pm25.gt(500).sum()),
            "gt1000_positive_target_months_retained_and_flagged": int(frame.pm25.gt(1000).sum()),
            "missing_or_invalid_coordinates": int((~frame.coordinate_pair_valid).sum()),
            "minimum_ndays": int(frame.ndays.min()),
        }
    metadata_months = numeric_audit.limits_v_relationship.value_counts().to_dict()
    numeric_summary = {
        "source_valid_candidate_months": int(len(source_valid_candidates)),
        "source_valid_eligible_months_before_operational_exclusions": int(source_valid_eligible_reference.shape[0]),
        "source_valid_negative_candidate_months": int(numeric_audit.source_valid_negative_month.sum()),
        "source_valid_negative_eligible_months": int(source_valid_eligible_reference.pm25.lt(0).sum()),
        "source_valid_gt500_candidate_months": int(numeric_audit.source_valid_gt500_flag.sum()),
        "source_valid_gt1000_candidate_months": int(numeric_audit.source_valid_gt1000_flag.sum()),
        "source_valid_months_outside_limits_v": int(numeric_audit.outside_limits_v.sum()),
        "all_candidate_months_by_limits_v_relationship": {str(k): int(v) for k, v in metadata_months.items()},
        "source_valid_high_or_negative_months_outside_limits_v": int((
            (numeric_audit.source_valid_negative_month | numeric_audit.source_valid_gt500_flag)
            & numeric_audit.outside_limits_v
        ).sum()),
        "high_positive_outcomes_retained_without_numeric_cap": True,
        "upper_numeric_threshold_applied": False,
    }
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional operational cohort; broad source QC is not certification",
        "fixture_output_rows": fixture_counts,
        "source_input_sha256": source_hashes,
        "source_metadata_path": str(args.metadata_path),
        "source_metadata_sha256": sha256(args.metadata_path),
        "policies": policies,
        "numeric_and_metadata_limits_audit": numeric_summary,
        "quarantine": duplicate_evidence,
        "source_field_review": source_review,
        "paired_policy_comparison": {
            "primary_months": int(len(primary)), "secondary_months": int(len(secondary)),
            "common_key_months": int(len(products["common_resampling_manifest"])),
            "policy_union_months": int(len(products["comparison"])),
            "primary_only_months": int(products["comparison"].policy_presence.eq("primary_only").sum()),
            "secondary_only_months": int(products["comparison"].policy_presence.eq("secondary_only").sum()),
            "common_key_coordinate_mismatches": int((~products["common_resampling_manifest"].same_station_month_coordinate).sum()),
            "common_key_targets_changed_by_policy": int(products["common_resampling_manifest"].target_delta_primary_minus_secondary.abs().gt(1e-12).sum()),
            "same_feature_and_fold_rule": "Downstream policy comparison must use common_key_resampling_manifest_v1_2018_2022.csv for identical station-month features and folds; preserve separate targets as columns.",
        },
        "outputs": {
            "primary": primary_name,
            "secondary": secondary_name,
            "primary_rejections": "primary_policy_rejections_v1_2018_2022.csv",
            "secondary_rejections": "secondary_policy_rejections_v1_2018_2022.csv",
            "policy_comparison": "policy_month_comparison_v1_2018_2022.csv",
            "common_key_resampling_manifest": "common_key_resampling_manifest_v1_2018_2022.csv",
            "eligibility_flow": "eligibility_flow_v1_2018_2022.csv",
            "numeric_metadata_limits_audit": "numeric_anomaly_metadata_limits_audit_v1_2018_2022.csv",
            "numeric_anomalies_vs_limits": "numeric_anomalies_vs_limits_v1_2018_2022.csv",
            "quarantine_evidence": "quarantine_evidence_v1.json",
            "source_field_review": "source_field_review_v1.json",
        },
        "no_original_artifacts_overwritten": True,
        "no_model_fit_or_score": True,
    }
    summary_path = output / "eligibility_summary_v1.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    report = f"""# Mexico SINAICA operational station-month cohort v1

Generated {summary['generated_utc']}. This is a provisional eligibility product; no model was fit or scored.

## Policy

- **Primary:** {len(primary):,} station-months from the nonnegative-hourly source-valid sensitivity, applying at least 20 supported days and valid finite source-snapshot coordinates, then quarantining station IDs 503 and 504 for source year 2022.
- **Secondary:** {len(secondary):,} station-months from the source-valid monthly table, applying the same support and coordinate checks, excluding negative monthly targets, and applying the same 503/504 2022 quarantine. Negative source rows remain in this secondary source aggregation; negative monthly outcomes are excluded from the model-eligible artifact.
- Both policies preserve positive extreme values and flag values above 500 and 1,000 µg/m³. No upper cap, winsorization, or factor-1,000 conversion is applied.

## Source anomaly quarantine

The 2022 hourly source CSVs for station IDs 503 (Tecate) and 504 (Ensenada) are byte-identical (SHA-256 {duplicate_evidence['raw_station_year_files'][0]['sha256']}) and each contains 1,281 finite observations from {duplicate_evidence['raw_station_year_files'][0]['first_date']} through {duplicate_evidence['raw_station_year_files'][0]['last_date']}. The current official API also returns identical sampled values for the two distinct station IDs. The 2022 Nov/Dec monthly values are therefore quarantined for both IDs pending source resolution; the root anomaly review contains the raw probe references and hashes.

The SINAICA saved `limits_v` snapshot places station 503 in 2023, while station 504 has no lower bound; this mismatch is recorded in the numeric/metadata limits audit and does not by itself invalidate other values. Station 504's Ensenada name and coordinates may conflict geographically; that is unresolved and documented as a caveat.

## Source conventions and limits

The saved downloader requests `soloVal=1` and maps API `validoAct` to CSV `valid`. The API response schema includes `band`, but the saved hourly CSV writer drops it; the sampled API responses contain blank or null `band`, so no historical band is inferred. Official SINAICA JavaScript labels PM2.5 as µg/m³. SINAICA describes validation as specific to each monitoring system and QA plan; its historical QC reporting also describes anomaly identification followed by source-agency validation or invalidation. High positive values are flagged for review and retained unless independent invalidity evidence is provided.

## Counts and paired comparison

- Primary: {len(primary):,} monthly rows; {primary.site_id.nunique()} station IDs; {int(primary.pm25.gt(500).sum())} positive targets >500; {int(primary.pm25.gt(1000).sum())} >1,000.
- Secondary: {len(secondary):,} monthly rows; {secondary.site_id.nunique()} station IDs; {int(secondary.pm25.gt(500).sum())} positive targets >500; {int(secondary.pm25.gt(1000).sum())} >1,000.
- Shared station-month keys: {len(products['common_resampling_manifest']):,}. For policy comparisons, use `common_key_resampling_manifest_v1_2018_2022.csv` to align identical features and folds.
- Policy differences on shared keys: {int(products['common_resampling_manifest'].target_delta_primary_minus_secondary.abs().gt(1e-12).sum()):,} target values differ between source policies; common-key coordinates are identical. There are {int(products['comparison'].policy_presence.eq('primary_only').sum())} primary-only and {int(products['comparison'].policy_presence.eq('secondary_only').sum())} secondary-only months.
- Published `limits_v` comparisons are descriptive; they are not used as a blanket inclusion rule. See `numeric_anomaly_metadata_limits_audit_v1_2018_2022.csv` for negative/high monthly values and metadata date-window status.

## Outputs

- `{primary_name}` and `{secondary_name}`: new canonical provisional cohorts; originals remain unchanged.
- `primary_policy_rejections_v1_2018_2022.csv` and `secondary_policy_rejections_v1_2018_2022.csv`: every rejected candidate and explicit reasons.
- `eligibility_flow_v1_2018_2022.csv`, `policy_month_comparison_v1_2018_2022.csv`, and `common_key_resampling_manifest_v1_2018_2022.csv`.
- `numeric_anomaly_metadata_limits_audit_v1_2018_2022.csv` and its `numeric_anomalies_vs_limits_v1_2018_2022.csv` flagged-row subset, `quarantine_evidence_v1.json`, `source_field_review_v1.json`, `eligibility_summary_v1.json`, and `provenance.json`.

These artifacts do not certify measurement validity, replace SINAICA source flags, repair station metadata, or contain feature values/model scores.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    names = [primary_name, secondary_name, "primary_policy_rejections_v1_2018_2022.csv",
             "secondary_policy_rejections_v1_2018_2022.csv", "policy_month_comparison_v1_2018_2022.csv",
             "common_key_resampling_manifest_v1_2018_2022.csv", "eligibility_flow_v1_2018_2022.csv",
             "numeric_anomaly_metadata_limits_audit_v1_2018_2022.csv", "numeric_anomalies_vs_limits_v1_2018_2022.csv",
             "quarantine_evidence_v1.json",
             "source_field_review_v1.json", "eligibility_summary_v1.json", "README.md"]
    outputs = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in names}
    provenance = {
        "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "inputs": source_hashes, "metadata_sha256": sha256(args.metadata_path),
        "output_hashes": outputs, "generated_utc": summary["generated_utc"],
        "no_model_fit_or_score": True,
    }
    (output / "provenance.json").write_text(json.dumps(json_safe(provenance), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "complete", "out": str(output), "primary_months": len(primary),
        "secondary_months": len(secondary), "common_key_months": len(products["common_resampling_manifest"]),
        "source_valid_negative_eligible_months": numeric_summary["source_valid_negative_eligible_months"],
        "quarantined_station_year_months": int(numeric_audit.quarantined_station_year.sum()),
        "primary_rejection_rows": len(products["primary_rejections"]),
        "secondary_rejection_rows": len(products["secondary_rejections"]),
        "no_model_fit_or_score": True,
        "summary": str(summary_path), "provenance": str(output / "provenance.json"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
