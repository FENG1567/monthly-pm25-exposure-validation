"""Build corrected OpenAQ physical-station daily/monthly PM2.5 tables."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from download_openaq import YEARS_FOR, cache_path, coverage_pct, get_local_date, unit_conversion
from inventory_openaq import build_sensor_table

ROOT = Path(r"F:\cand1_20261105")
RAW = Path(r"F:\cand1_repair_v2_data\openaq")
OUT = Path(__file__).resolve().parents[3] / "outputs" / "repair_v2" / "openaq"
MIN_MONTH_DAYS = 20
MIN_COVERAGE_PERCENT = 75.0

# OpenAQ defines a location's `monitor` query flag as whether the location is
# considered a reference monitor and `mobile` as whether it is considered a
# mobile location. These are source flags, not a claim that a site holds a
# local regulatory certification.
OPENAQ_LOCATION_FLAGS_DOC = "https://docs.openaq.org/api/operations/locations_get_v3_locations_get"
OPENAQ_INSTRUMENTS_DOC = "https://docs.openaq.org/resources/instruments"

# These families are explicitly marketed as low-cost air sensors by their
# manufacturers/owners. This is a diagnostic label only; primary eligibility is
# determined by OpenAQ's location flags below, and conflicts are retained.
KNOWN_LOW_COST_INSTRUMENT_PATTERNS = {
    "airgradient": "AirGradient family",
    "clarity sensor": "Clarity sensor",
    "clarity node": "Clarity Node",
    "habitatmap sensor": "HabitatMap sensor",
    "airbeam": "AirBeam",
}
LOW_COST_SOURCE_DOCS = [
    "https://www.airgradient.com/outdoor/",
    "https://www.clarity.io/",
    "https://www.habitatmap.org/about",
]


def optional_bool(value: object) -> bool | None:
    """Parse an optional boolean without treating the string 'False' as true."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def low_cost_instrument_evidence(instruments: object) -> str:
    text = str(instruments or "").strip().lower()
    matches = [label for pattern, label in KNOWN_LOW_COST_INSTRUMENT_PATTERNS.items()
               if pattern in text]
    return ";".join(matches)


def classify_source(mobile: bool | None, monitor: bool | None,
                    low_cost_evidence: str) -> tuple[str, bool, str]:
    """Return source stratum, reference-primary eligibility, and conflict note."""
    has_low_cost_evidence = bool(low_cost_evidence)
    primary = mobile is False and monitor is True
    if mobile is True:
        stratum = "mobile_low_cost" if has_low_cost_evidence else "mobile"
    elif mobile is None:
        if monitor is True:
            stratum = "reference_flag_mobility_unknown"
        elif monitor is False and has_low_cost_evidence:
            stratum = "low_cost_mobility_unknown"
        else:
            stratum = "source_flags_unknown"
    elif monitor is True and has_low_cost_evidence:
        stratum = "reference_flag_low_cost_instrument_conflict"
    elif monitor is True:
        stratum = "openaq_reference_monitor"
    elif monitor is False and has_low_cost_evidence:
        stratum = "openaq_nonreference_known_low_cost"
    elif monitor is False:
        stratum = "openaq_nonreference_grade_unverified"
    else:
        stratum = "reference_status_unknown"

    conflicts = []
    if mobile is True and monitor is True:
        conflicts.append("mobile_and_reference_flags_true")
    if monitor is True and has_low_cost_evidence:
        conflicts.append("reference_flag_with_named_low_cost_instrument")
    return stratum, primary, ";".join(conflicts)


def read_location_enrichment() -> pd.DataFrame:
    p = OUT / "canonical_location_provider_metadata.csv"
    if not p.exists():
        return pd.DataFrame(columns=["location_id", "provider_name", "instrument_names", "timezone",
                                     "is_mobile", "is_monitor", "license_names", "owner_name"])
    d = pd.read_csv(p, dtype={"location_id": "int64"}, keep_default_na=False)
    if d.empty:
        return pd.DataFrame(columns=["location_id", "provider_name", "instrument_names", "timezone",
                                     "is_mobile", "is_monitor", "license_names", "owner_name"])
    for col in ["provider_name", "instrument_names", "timezone", "is_mobile", "is_monitor",
                "license_names", "owner_name"]:
        if col not in d:
            d[col] = ""
    return d.drop_duplicates("location_id")


def read_cached_payload(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def rows_from_cache(sensors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    locmeta = read_location_enrichment().set_index("location_id").to_dict("index")
    rows = []
    sensor_audit = []
    for _, m in sensors.iterrows():
        sid = int(m.sensor_id)
        years = YEARS_FOR[str(m.region)]
        path = cache_path(sid, min(years), max(years))
        if not path.exists():
            continue
        payload = read_cached_payload(path)
        if payload.get("source") != "OpenAQ API v3 official":
            raise ValueError(f"Unexpected cache provenance: {path}")
        results = payload.get("results", [])
        lm = locmeta.get(int(m.location_id), {})
        mobile = optional_bool(lm.get("is_mobile"))
        monitor = optional_bool(lm.get("is_monitor"))
        provider_name = str(lm.get("provider_name") or "").strip()
        instrument_names = str(lm.get("instrument_names") or "").strip()
        low_cost_evidence = low_cost_instrument_evidence(instrument_names)
        source_stratum, primary_source_eligible, source_flag_conflict = classify_source(
            mobile, monitor, low_cost_evidence)
        sensor_audit.append({
            "sensor_id": sid, "location_id": int(m.location_id), "iso": str(m.iso),
            "region": str(m.region), "site_id": str(m.site_id),
            "api_result_rows": len(results), "cache_path": str(path),
            "provider_name": provider_name, "owner_name": str(lm.get("owner_name") or "").strip(),
            "instrument_names": instrument_names, "license_names": str(lm.get("license_names") or "").strip(),
            "timezone": str(lm.get("timezone") or "").strip(),
            "source_is_mobile": mobile, "source_is_monitor": monitor,
            "known_low_cost_instrument_evidence": low_cost_evidence,
            "source_grade_stratum": source_stratum,
            "primary_source_eligible": primary_source_eligible,
            "source_flag_conflict": source_flag_conflict,
            "provider_metadata_present": bool(provider_name),
            "instrument_metadata_present": bool(instrument_names),
            "timezone_metadata_present": bool(str(lm.get("timezone") or "").strip()),
            "location_metadata_present": bool(lm),
        })
        for r in results:
            p = r.get("period") or {}
            dt = get_local_date(r)
            param = r.get("parameter") or {}
            raw_unit = param.get("units")
            factor = unit_conversion(raw_unit)
            raw_value = pd.to_numeric(r.get("value"), errors="coerce")
            normalized_value = float(raw_value) * factor if pd.notna(raw_value) and factor is not None else np.nan
            cov = r.get("coverage") or {}
            pct = coverage_pct(r)
            expected = pd.to_numeric(cov.get("expectedCount"), errors="coerce")
            observed = pd.to_numeric(cov.get("observedCount"), errors="coerce")
            within_scope = dt.year in years
            pm_name = str(param.get("name", "")).lower().replace(".", "")
            is_pm25 = pm_name in {"pm25", "pm2.5"}
            good_interval = str(p.get("interval", "")).strip().upper() in {"24:00:00", "P1D", "PT24H", "24H", "24 HOURS"}
            good_value = pd.notna(normalized_value) and 0 <= normalized_value < 2000
            coverage_ok = pct is None or pct >= MIN_COVERAGE_PERCENT
            rows.append({
                "iso": str(m.iso), "region": str(m.region), "site_id": str(m.site_id),
                "sensor_id": sid, "location_id": int(m.location_id), "lat": float(m.lat), "lon": float(m.lon),
                "date": dt, "year": dt.year, "month": dt.month,
                "pm25": normalized_value if good_value else np.nan,
                "raw_value": float(raw_value) if pd.notna(raw_value) else np.nan,
                "units_original": str(raw_unit or ""), "units_normalized": "µg/m³" if factor is not None else "",
                "conversion_factor": factor, "parameter_name": str(param.get("name", "")),
                "parameter_display_name": str(param.get("displayName", "")),
                "period_interval": str(p.get("interval", "")),
                "period_start_local": (p.get("datetimeFrom") or {}).get("local", ""),
                "period_end_local": (p.get("datetimeTo") or {}).get("local", ""),
                "coverage_percent": pct,
                "coverage_expected_count": float(expected) if pd.notna(expected) else np.nan,
                "coverage_observed_count": float(observed) if pd.notna(observed) else np.nan,
                "coverage_expected_interval": str(cov.get("expectedInterval", "")),
                "coverage_observed_interval": str(cov.get("observedInterval", "")),
                "coverage_status": "missing" if pct is None else ("ge75" if pct >= MIN_COVERAGE_PERCENT else "lt75"),
                "flagged": bool((r.get("flagInfo") or {}).get("hasFlags", False)),
                "provider_name": provider_name,
                "owner_name": str(lm.get("owner_name") or "").strip(),
                "instrument_names": instrument_names,
                "license_names": str(lm.get("license_names") or "").strip(),
                "timezone": str(lm.get("timezone") or "").strip(),
                "source_is_mobile": mobile,
                "source_is_monitor": monitor,
                "known_low_cost_instrument_evidence": low_cost_evidence,
                "source_grade_stratum": source_stratum,
                "primary_source_eligible": primary_source_eligible,
                "source_flag_conflict": source_flag_conflict,
                "location_metadata_present": bool(lm),
                "within_scope": within_scope, "is_pm25": is_pm25,
                "valid_daily_interval": good_interval, "valid_value_unit": bool(good_value),
                "coverage_primary_eligible": coverage_ok,
            })
    return pd.DataFrame(rows), pd.DataFrame(sensor_audit)


def build_daily_monthly(rows: pd.DataFrame, sensors: pd.DataFrame,
                        source_stratified: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        cols = ["iso", "site_id", "lat", "lon", "date", "year", "month", "pm25"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols + ["ndays"]), pd.DataFrame()
    source_group_cols = ["source_grade_stratum"] if source_stratified else []
    steps = [
        ("raw_api_records", rows),
        ("requested_local_years", rows.loc[rows.within_scope]),
        ("parameter_pm25", rows.loc[rows.within_scope & rows.is_pm25]),
        ("24h_daily_interval", rows.loc[rows.within_scope & rows.is_pm25 & rows.valid_daily_interval]),
        ("recognized_units_and_valid_value", rows.loc[rows.within_scope & rows.is_pm25 & rows.valid_daily_interval & rows.valid_value_unit]),
        ("coverage_ge75_or_unreported", rows.loc[rows.within_scope & rows.is_pm25 & rows.valid_daily_interval & rows.valid_value_unit & rows.coverage_primary_eligible]),
    ]
    waterfall = []
    grouping = ["iso", "year"] + source_group_cols
    for name, d in steps:
        if d.empty:
            continue
        for key, g in d.groupby(grouping, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            iso, year = key[:2]
            stratum = key[2] if source_stratified else "all_included"
            waterfall.append({"iso": iso, "year": int(year), "source_grade_stratum": stratum, "stage": name,
                              "sensor_date_rows": len(g), "sensors": int(g.sensor_id.nunique()),
                              "physical_sites": int(g.site_id.nunique()),
                              "distinct_sensor_dates": int(g[["sensor_id", "date"]].drop_duplicates().shape[0]),
                              "distinct_physical_site_dates": int(g[["site_id", "date"]].drop_duplicates().shape[0])})
    good = rows.loc[rows.within_scope & rows.is_pm25 & rows.valid_daily_interval &
                    rows.valid_value_unit & rows.coverage_primary_eligible].copy()
    if good.empty:
        cols = ["iso", "site_id", "lat", "lon", "date", "year", "month", "pm25"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols + ["ndays"]), pd.DataFrame(waterfall)
    # One API daily row per sensor/local date. Deduplicate repeated observations
    # within a sensor before collapsing physically co-located sensor aliases.
    sensor_group_cols = ["iso", "region", "site_id", "sensor_id", "date", "year", "month"] + source_group_cols
    sensor_day = (good.groupby(sensor_group_cols, as_index=False, dropna=False)
                  .agg(pm25=("pm25", "mean"), raw_n=("pm25", "size"),
                       lat=("lat", "first"), lon=("lon", "first"),
                       coverage_percent=("coverage_percent", "max"),
                       provider_name=("provider_name", "first"), instrument_names=("instrument_names", "first"),
                       timezone=("timezone", "first"), flagged=("flagged", "max"),
                       source_is_mobile=("source_is_mobile", "first"),
                       source_is_monitor=("source_is_monitor", "first"),
                       known_low_cost_instrument_evidence=("known_low_cost_instrument_evidence", "first"),
                       primary_source_eligible=("primary_source_eligible", "all"),
                       source_grade_stratum=("source_grade_stratum", "first"),
                       source_flag_conflict=("source_flag_conflict", "first"),
                       units_original=("units_original", "first")))
    # Multiple OpenAQ PM2.5 sensors at the exact same country/coordinate are
    # aliases of one physical station. Average sensor-day values equally.
    daily_group_cols = ["iso", "region", "site_id", "date", "year", "month"] + source_group_cols
    daily = (sensor_day.groupby(daily_group_cols, as_index=False, dropna=False)
             .agg(pm25=("pm25", "mean"), lat=("lat", "first"), lon=("lon", "first"),
                  n_sensors=("sensor_id", "nunique"), n_sensor_days=("sensor_id", "size"),
                  coverage_min=("coverage_percent", "min"), coverage_mean=("coverage_percent", "mean"),
                  coverage_max=("coverage_percent", "max"), flagged_sensor_days=("flagged", "sum"),
                  primary_source_eligible=("primary_source_eligible", "all"),
                  reference_sensor_days=("source_is_monitor", lambda v: sum(optional_bool(x) is True for x in v)),
                  nonreference_sensor_days=("source_is_monitor", lambda v: sum(optional_bool(x) is False for x in v)),
                  known_low_cost_sensor_days=("known_low_cost_instrument_evidence", lambda v: sum(bool(x) for x in v)),
                  mobile_sensor_days=("source_is_mobile", lambda v: sum(optional_bool(x) is True for x in v)),
                  source_grade_strata=("source_grade_stratum", lambda v: ";".join(sorted({str(x) for x in v if str(x)}))),
                  source_flag_conflicts=("source_flag_conflict", lambda v: ";".join(sorted({str(x) for x in v if str(x)}))),
                  provider_names=("provider_name", lambda v: ";".join(sorted({str(x) for x in v if str(x)}))),
                  instrument_names=("instrument_names", lambda v: ";".join(sorted({str(x) for x in v if str(x)}))),
                  units_original=("units_original", lambda v: ";".join(sorted({str(x) for x in v if str(x)})))))
    # Keep a stable, machine-readable contributor list for audit/reproducibility.
    contributor_group_cols = ["iso", "site_id", "date"] + source_group_cols
    contributor = (sensor_day.groupby(contributor_group_cols, as_index=False)
                   .agg(sensor_ids=("sensor_id", lambda v: ";".join(map(str, sorted(set(map(int, v))))))))
    daily = daily.merge(contributor, on=contributor_group_cols, how="left")
    eligible = sensors[["sensor_id", "main_model_eligible"]].drop_duplicates("sensor_id")
    sensor_eligible = sensor_day[["iso", "site_id", "sensor_id", "date"] + source_group_cols].merge(
        eligible, on="sensor_id", how="left")
    site_elig_cols = ["iso", "site_id", "date"] + source_group_cols
    site_elig = sensor_eligible.groupby(site_elig_cols, as_index=False)["main_model_eligible"].all()
    daily = daily.merge(site_elig, on=site_elig_cols, how="left")
    monthly_group_cols = ["iso", "region", "site_id", "year", "month"] + source_group_cols
    monthly = (daily.groupby(monthly_group_cols, as_index=False, dropna=False)
               .agg(pm25=("pm25", "mean"), ndays=("date", "nunique"), lat=("lat", "first"), lon=("lon", "first"),
                    n_sensor_days=("n_sensor_days", "sum"), n_sensors_max=("n_sensors", "max"),
                    coverage_mean=("coverage_mean", "mean"), coverage_min=("coverage_min", "min"),
                    coverage_max=("coverage_max", "max"), flagged_sensor_days=("flagged_sensor_days", "sum"),
                    primary_source_eligible=("primary_source_eligible", "all"),
                    reference_sensor_days=("reference_sensor_days", "sum"),
                    nonreference_sensor_days=("nonreference_sensor_days", "sum"),
                    known_low_cost_sensor_days=("known_low_cost_sensor_days", "sum"),
                    mobile_sensor_days=("mobile_sensor_days", "sum"),
                    source_grade_strata=("source_grade_strata", lambda v: ";".join(sorted({x for s in v for x in str(s).split(";") if x}))),
                    source_flag_conflicts=("source_flag_conflicts", lambda v: ";".join(sorted({x for s in v for x in str(s).split(";") if x}))),
                    provider_names=("provider_names", lambda v: ";".join(sorted({x for s in v for x in str(s).split(";") if x}))),
                    instrument_names=("instrument_names", lambda v: ";".join(sorted({x for s in v for x in str(s).split(";") if x}))),
                    main_model_eligible=("main_model_eligible", "all")))
    monthly = monthly.loc[monthly.ndays.ge(MIN_MONTH_DAYS)].copy()
    monthly["ndays"] = monthly.ndays.astype("int16")
    daily = daily.sort_values(["iso", "date", "site_id"] + source_group_cols).reset_index(drop=True)
    monthly = monthly.sort_values(["iso", "year", "month", "site_id"] + source_group_cols).reset_index(drop=True)
    return daily, monthly, pd.DataFrame(waterfall)


def legacy_denominators() -> pd.DataFrame:
    region_files = {"asia": "stations_monthly_asia.parquet", "eur": "stations_monthly_eur.parquet",
                    "chl": "stations_monthly_chl.parquet", "zaf": "stations_monthly_zaf.parquet",
                    "idn": "stations_monthly_idn.parquet"}
    frames = []
    for region, filename in region_files.items():
        path = ROOT / "work" / "features" / filename
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        if not {"iso", "site_id", "year", "month"}.issubset(d.columns):
            continue
        d["region"] = region
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["iso", "year", "legacy_station_months", "legacy_sensor_sites"])
    old = pd.concat(frames, ignore_index=True)
    if "ndays" not in old:
        old["ndays"] = np.nan
    return (old.groupby(["iso", "year"], as_index=False)
            .agg(legacy_station_months=("site_id", "size"), legacy_sensor_sites=("site_id", "nunique"),
                 legacy_mean_ndays=("ndays", "mean"), legacy_min_ndays=("ndays", "min"),
                 legacy_max_ndays=("ndays", "max")))


def write_checkpoint_request_proof(audit: pd.DataFrame) -> None:
    """Write key-free API request evidence for every sensor in this checkpoint."""
    ledger_path = OUT / "corrected_api_ledger.csv"
    requested_ids = set(audit.sensor_id.astype(int)) if not audit.empty else set()
    queries = []
    if ledger_path.exists() and requested_ids:
        ledger = pd.read_csv(ledger_path, dtype={"sensor_id": "Int64"})
        ledger = ledger.loc[ledger.sensor_id.isin(requested_ids)].drop_duplicates("sensor_id", keep="last")
        for row in ledger.to_dict("records"):
            try:
                params = json.loads(str(row.get("query_params") or "{}"))
            except (TypeError, ValueError):
                params = {}
            statuses = [int(x) for x in str(row.get("http_statuses") or "").split(";") if x]
            queries.append({
                "sensor_id": int(row["sensor_id"]),
                "location_id": int(row["location_id"]) if pd.notna(row.get("location_id")) else None,
                "iso": str(row.get("iso") or ""),
                "region": str(row.get("region") or ""),
                "status": str(row.get("status") or ""),
                "endpoint": str(row.get("endpoint") or ""),
                "params": params,
                "requested_date_from": str(row.get("requested_date_from") or ""),
                "requested_date_to": str(row.get("requested_date_to") or ""),
                "http_statuses": statuses,
                "api_pages": int(row["api_pages"]) if pd.notna(row.get("api_pages")) else 0,
                "meta_found": int(row["meta_found"]) if pd.notna(row.get("meta_found")) else None,
                "returned_records": int(row["returned_records"]) if pd.notna(row.get("returned_records")) else 0,
                "local_dates": int(row["local_dates"]) if pd.notna(row.get("local_dates")) else 0,
                "local_date_min": str(row.get("local_date_min") or ""),
                "local_date_max": str(row.get("local_date_max") or ""),
                "outside_interval_records": int(row["outside_interval_records"]) if pd.notna(row.get("outside_interval_records")) else 0,
                "invalid_interval_records": int(row["invalid_interval_records"]) if pd.notna(row.get("invalid_interval_records")) else 0,
                "response_count_reconciled": str(row.get("response_count_reconciled", "")).strip().lower() == "true",
                "raw_path": str(row.get("raw_path") or ""),
                "raw_bytes": int(row["raw_bytes"]) if pd.notna(row.get("raw_bytes")) else 0,
                "raw_sha256": str(row.get("raw_sha256") or ""),
                "error": str(row.get("error") or ""),
            })
    proof = {
        "source": "OpenAQ API v3 official sensor daily endpoint",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "authentication": "X-API-Key header; credential withheld",
        "query_evidence_count": len(queries),
        "queries": queries,
    }
    (OUT / "request_proof_checkpoint.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")


def source_metadata_outputs(sensors: pd.DataFrame, audit: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write sensor-level source flags and their country/alias summaries."""
    source = sensors[["sensor_id", "location_id", "iso", "region", "site_id", "lat", "lon"]].merge(
        audit.drop(columns=["iso", "region", "site_id", "location_id"], errors="ignore"),
        on="sensor_id", how="left")
    source.to_csv(OUT / f"canonical_sensor_source_audit_{prefix}.csv", index=False)
    d = source.copy()
    d["monitor_bool"] = d.source_is_monitor.map(optional_bool)
    d["mobile_bool"] = d.source_is_mobile.map(optional_bool)
    d["primary_bool"] = d.primary_source_eligible.fillna(False).astype(bool)
    d["low_cost_bool"] = d.known_low_cost_instrument_evidence.fillna("").astype(str).str.len().gt(0)
    d["conflict_bool"] = d.source_flag_conflict.fillna("").astype(str).str.len().gt(0)
    d["provider_known"] = (d.provider_name.fillna("").astype(str).str.strip().str.lower()
                            .ne("") & ~d.provider_name.fillna("").astype(str).str.strip().str.lower().isin({"n/a", "unknown", "none"}))
    d["instrument_named"] = (d.instrument_names.fillna("").astype(str).str.strip().str.lower()
                              .ne("") & ~d.instrument_names.fillna("").astype(str).str.strip().str.lower().isin({"n/a", "unknown", "none"}))
    country = (d.groupby("iso", as_index=False)
               .agg(sensors=("sensor_id", "nunique"), locations=("location_id", "nunique"),
                    reference_flag_true_sensors=("monitor_bool", lambda v: sum(x is True for x in v)),
                    reference_flag_false_sensors=("monitor_bool", lambda v: sum(x is False for x in v)),
                    reference_flag_unknown_sensors=("monitor_bool", lambda v: sum(x is None for x in v)),
                    mobile_flag_true_sensors=("mobile_bool", lambda v: sum(x is True for x in v)),
                    stationary_flag_sensors=("mobile_bool", lambda v: sum(x is False for x in v)),
                    mobile_flag_unknown_sensors=("mobile_bool", lambda v: sum(x is None for x in v)),
                    primary_source_eligible_sensors=("primary_bool", "sum"),
                    known_low_cost_instrument_sensors=("low_cost_bool", "sum"),
                    source_flag_conflict_sensors=("conflict_bool", "sum"),
                    provider_named_sensors=("provider_known", "sum"),
                    instrument_named_sensors=("instrument_named", "sum"),
                    timezone_named_sensors=("timezone_metadata_present", "sum")))
    country.to_csv(OUT / f"source_metadata_country_summary_{prefix}.csv", index=False)
    conflicts = d.loc[d.conflict_bool].copy()
    conflicts.to_csv(OUT / f"source_metadata_flag_conflicts_{prefix}.csv", index=False)
    alias = (d.groupby(["iso", "site_id"], as_index=False, dropna=False)
             .agg(n_sensors=("sensor_id", "nunique"), n_locations=("location_id", "nunique"),
                  source_strata=("source_grade_stratum", lambda v: ";".join(sorted({str(x) for x in v if str(x)}))),
                  reference_flag_values=("source_is_monitor", lambda v: ";".join(sorted({str(x) for x in v if str(x)}))),
                  primary_eligible_sensors=("primary_bool", "sum"),
                  nonreference_sensors=("monitor_bool", lambda v: sum(x is False for x in v)),
                  known_low_cost_sensors=("low_cost_bool", "sum")))
    alias["n_source_strata"] = alias.source_strata.map(lambda x: len([v for v in str(x).split(";") if v]))
    alias = alias.loc[alias.n_source_strata.gt(1) | alias.reference_flag_values.str.contains("True;False|False;True", regex=True)]
    alias.to_csv(OUT / f"source_metadata_alias_conflicts_{prefix}.csv", index=False)
    return source, country


def source_country_year_funnel(rows: pd.DataFrame, source: pd.DataFrame, sensors: pd.DataFrame,
                               prefix: str) -> pd.DataFrame:
    """Count target sensors and API daily records by country/year and source flag."""
    target_parts = []
    source_by_sensor = source.set_index("sensor_id")[["source_is_mobile", "source_is_monitor",
                                                       "primary_source_eligible", "source_grade_stratum",
                                                       "known_low_cost_instrument_evidence", "source_flag_conflict"]]
    for s in sensors.itertuples(index=False):
        for year in YEARS_FOR[str(s.region)]:
            target_parts.append({"iso": str(s.iso), "year": int(year), "sensor_id": int(s.sensor_id)})
    target = pd.DataFrame(target_parts).merge(source_by_sensor, on="sensor_id", how="left")
    target["monitor_bool"] = target.source_is_monitor.map(optional_bool)
    target["mobile_bool"] = target.source_is_mobile.map(optional_bool)
    target["primary_bool"] = target.primary_source_eligible.fillna(False).astype(bool)
    target["low_cost_bool"] = target.known_low_cost_instrument_evidence.fillna("").astype(str).str.len().gt(0)
    target["conflict_bool"] = target.source_flag_conflict.fillna("").astype(str).str.len().gt(0)
    expected = (target.groupby(["iso", "year"], as_index=False)
                .agg(target_sensors=("sensor_id", "nunique"),
                     reference_flag_true_sensors=("monitor_bool", lambda v: sum(x is True for x in v)),
                     reference_flag_false_sensors=("monitor_bool", lambda v: sum(x is False for x in v)),
                     reference_flag_unknown_sensors=("monitor_bool", lambda v: sum(x is None for x in v)),
                     mobile_flag_true_sensors=("mobile_bool", lambda v: sum(x is True for x in v)),
                     stationary_flag_sensors=("mobile_bool", lambda v: sum(x is False for x in v)),
                     mobile_flag_unknown_sensors=("mobile_bool", lambda v: sum(x is None for x in v)),
                     primary_source_eligible_sensors=("primary_bool", "sum"),
                     known_low_cost_instrument_sensors=("low_cost_bool", "sum"),
                     source_flag_conflict_sensors=("conflict_bool", "sum")))
    r = rows.loc[rows.within_scope & rows.is_pm25].copy()
    if r.empty:
        result = expected
        for col in ["api_pm25_daily_records", "sensors_with_api_records", "reference_flag_true_records",
                    "reference_flag_false_records", "reference_flag_unknown_records", "primary_eligible_records",
                    "stationary_records", "mobile_records", "known_low_cost_records", "coverage_ge75_records",
                    "coverage_lt75_records", "coverage_unreported_records"]:
            result[col] = 0
    else:
        r["monitor_bool"] = r.source_is_monitor.map(optional_bool)
        r["mobile_bool"] = r.source_is_mobile.map(optional_bool)
        r["primary_bool"] = r.primary_source_eligible.fillna(False).astype(bool)
        r["low_cost_bool"] = r.known_low_cost_instrument_evidence.fillna("").astype(str).str.len().gt(0)
        r["coverage_ge75"] = r.coverage_percent.ge(MIN_COVERAGE_PERCENT)
        r["coverage_lt75"] = r.coverage_percent.notna() & r.coverage_percent.lt(MIN_COVERAGE_PERCENT)
        result_rows = []
        for (iso, year), g in r.groupby(["iso", "year"], dropna=False):
            result_rows.append({
                "iso": iso, "year": int(year), "api_pm25_daily_records": len(g),
                "sensors_with_api_records": int(g.sensor_id.nunique()),
                "reference_flag_true_records": int(g.monitor_bool.map(lambda x: x is True).sum()),
                "reference_flag_false_records": int(g.monitor_bool.map(lambda x: x is False).sum()),
                "reference_flag_unknown_records": int(g.monitor_bool.map(lambda x: x is None).sum()),
                "primary_eligible_records": int(g.primary_bool.sum()),
                "stationary_records": int(g.mobile_bool.map(lambda x: x is False).sum()),
                "mobile_records": int(g.mobile_bool.map(lambda x: x is True).sum()),
                "known_low_cost_records": int(g.low_cost_bool.sum()),
                "coverage_ge75_records": int(g.coverage_ge75.sum()),
                "coverage_lt75_records": int(g.coverage_lt75.sum()),
                "coverage_unreported_records": int(g.coverage_percent.isna().sum()),
            })
        record_counts = pd.DataFrame(result_rows)
        result = expected.merge(record_counts, on=["iso", "year"], how="left")
        record_cols = ["api_pm25_daily_records", "sensors_with_api_records", "reference_flag_true_records",
                       "reference_flag_false_records", "reference_flag_unknown_records", "primary_eligible_records",
                       "stationary_records", "mobile_records", "known_low_cost_records", "coverage_ge75_records",
                       "coverage_lt75_records", "coverage_unreported_records"]
        result[record_cols] = result[record_cols].fillna(0).astype("int64")
    result = result.sort_values(["iso", "year"]).reset_index(drop=True)
    result.to_csv(OUT / f"country_year_source_funnel_{prefix}.csv", index=False)
    return result


def write_outputs(checkpoint: bool = False) -> None:
    sensors = build_sensor_table()
    rows, audit = rows_from_cache(sensors)
    daily, monthly, waterfall = build_daily_monthly(rows, sensors)
    prefix = "checkpoint" if checkpoint else "2018_2025"
    daily.to_parquet(OUT / f"station_day_{prefix}.parquet", index=False)
    monthly.to_parquet(OUT / f"station_month_{prefix}.parquet", index=False)
    pd.DataFrame(waterfall).to_csv(OUT / f"qc_waterfall_{prefix}.csv", index=False)
    audit.to_csv(OUT / f"cache_sensor_audit_{prefix}.csv", index=False)
    if checkpoint:
        write_checkpoint_request_proof(audit)
    if not checkpoint:
        for region in ["asia", "eur", "chl", "zaf", "idn"]:
            region_month = monthly.loc[monthly.region.eq(region)]
            region_day = daily.loc[daily.region.eq(region)]
            if region not in YEARS_FOR:
                continue
            y0, y1 = min(YEARS_FOR[region]), max(YEARS_FOR[region])
            region_month.to_parquet(OUT / f"station_month_{region}_{y0}_{y1}.parquet", index=False)
            region_day.to_parquet(OUT / f"station_day_{region}_{y0}_{y1}.parquet", index=False)
    old = legacy_denominators()
    new = (monthly.groupby(["iso", "year"], as_index=False)
           .agg(new_station_months=("site_id", "size"), new_physical_sites=("site_id", "nunique"),
                new_mean_ndays=("ndays", "mean"), new_min_ndays=("ndays", "min"),
                new_max_ndays=("ndays", "max")))
    daily_counts = (daily.groupby(["iso", "year"], as_index=False)
                    .agg(new_station_days=("site_id", "size"), daily_physical_sites=("site_id", "nunique")))
    den = old.merge(new, on=["iso", "year"], how="outer").merge(daily_counts, on=["iso", "year"], how="outer")
    den.to_csv(OUT / f"country_year_denominators_{prefix}.csv", index=False)
    if not checkpoint:
        expected = len(sensors)
        downloaded = int(audit.sensor_id.nunique()) if not audit.empty else 0
        complete = downloaded == expected
    else:
        complete = False
    summary = {"checkpoint": checkpoint, "complete_sensor_cache": complete,
               "metadata_sensor_count": int(sensors.sensor_id.nunique()),
               "cached_sensor_count": int(audit.sensor_id.nunique()) if not audit.empty else 0,
               "clean_daily_rows": int(len(daily)), "clean_month_rows": int(len(monthly)),
               "distinct_sites": int(monthly.site_id.nunique()) if not monthly.empty else 0,
               "years": sorted(map(int, daily.year.unique())) if not daily.empty else [],
               "main_model_excluded_sites": int(monthly.loc[~monthly.main_model_eligible, "site_id"].nunique()) if not monthly.empty else 0,
               "qc": {"minimum_distinct_days_per_month": MIN_MONTH_DAYS,
                      "coverage_rule": "retain reported daily rows with coverage.percentComplete >=75%; retain rows when coverage is unreported and record this state",
                      "physical_grouping": "equal-weight sensor daily averages within exact country+latitude+longitude station group"}}
    (OUT / f"qc_summary_{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def write_outputs_v2(checkpoint: bool = False) -> None:
    sensors = build_sensor_table()
    rows, cache_audit = rows_from_cache(sensors)
    prefix = "checkpoint" if checkpoint else "2018_2025"
    source_audit, source_country = source_metadata_outputs(sensors, cache_audit, prefix)
    source_funnel = source_country_year_funnel(rows, source_audit, sensors, prefix)
    cache_audit.to_csv(OUT / f"cache_sensor_audit_{prefix}.csv", index=False)

    # Primary reference monitor cohort is filtered at sensor/location level
    # before any co-located sensor alias averaging.
    primary_rows = rows.loc[rows.primary_source_eligible.fillna(False)].copy()
    if not primary_rows.empty:
        assert primary_rows.source_is_mobile.map(optional_bool).eq(False).all()
        assert primary_rows.source_is_monitor.map(optional_bool).eq(True).all()
    stationary_rows = rows.loc[rows.source_is_mobile.map(optional_bool).eq(False)].copy()

    primary_day, primary_month, primary_waterfall = build_daily_monthly(primary_rows, sensors)
    primary_day.to_parquet(OUT / f"station_day_{prefix}.parquet", index=False)
    primary_month.to_parquet(OUT / f"station_month_{prefix}.parquet", index=False)
    primary_waterfall.to_csv(OUT / f"qc_waterfall_{prefix}.csv", index=False)
    primary_daily_counts = (primary_day.groupby(["iso", "year"], as_index=False)
                            .agg(primary_station_days=("site_id", "size"),
                                 primary_daily_sites=("site_id", "nunique")))
    if checkpoint:
        write_checkpoint_request_proof(cache_audit)
    else:
        for region in ["asia", "eur", "chl", "zaf", "idn"]:
            if region not in YEARS_FOR:
                continue
            y0, y1 = min(YEARS_FOR[region]), max(YEARS_FOR[region])
            primary_month.loc[primary_month.region.eq(region)].to_parquet(
                OUT / f"station_month_{region}_{y0}_{y1}.parquet", index=False)
            primary_day.loc[primary_day.region.eq(region)].to_parquet(
                OUT / f"station_day_{region}_{y0}_{y1}.parquet", index=False)
    del primary_day

    stationary_day, stationary_month, stationary_waterfall = build_daily_monthly(stationary_rows, sensors)
    stationary_day.to_parquet(OUT / f"station_day_all_stationary_{prefix}.parquet", index=False)
    stationary_month.to_parquet(OUT / f"station_month_all_stationary_{prefix}.parquet", index=False)
    stationary_waterfall.to_csv(OUT / f"qc_waterfall_all_stationary_{prefix}.csv", index=False)
    stationary_daily_counts = (stationary_day.groupby(["iso", "year"], as_index=False)
                               .agg(all_stationary_station_days=("site_id", "size"),
                                    all_stationary_daily_sites=("site_id", "nunique")))
    del stationary_day, stationary_rows

    stratified_day, stratified_month, stratified_waterfall = build_daily_monthly(
        rows, sensors, source_stratified=True)
    stratified_day.to_parquet(OUT / f"station_day_source_stratified_{prefix}.parquet", index=False)
    stratified_month.to_parquet(OUT / f"station_month_source_stratified_{prefix}.parquet", index=False)
    stratified_waterfall.to_csv(OUT / f"qc_waterfall_source_stratified_{prefix}.csv", index=False)
    (stratified_month.groupby(["iso", "year", "source_grade_stratum"], as_index=False)
     .agg(source_stratum_station_months=("site_id", "size"),
          source_stratum_sites=("site_id", "nunique"),
          mean_ndays=("ndays", "mean"), min_ndays=("ndays", "min"), max_ndays=("ndays", "max"))
     .to_csv(OUT / f"country_year_source_stratum_denominators_{prefix}.csv", index=False))
    stratified_month_rows = int(len(stratified_month))
    del stratified_day, stratified_month

    primary_month_counts = (primary_month.groupby(["iso", "year"], as_index=False)
                            .agg(primary_station_months=("site_id", "size"),
                                 primary_physical_sites=("site_id", "nunique"),
                                 primary_mean_ndays=("ndays", "mean"),
                                 primary_min_ndays=("ndays", "min"),
                                 primary_max_ndays=("ndays", "max")))
    stationary_month_counts = (stationary_month.groupby(["iso", "year"], as_index=False)
                               .agg(all_stationary_station_months=("site_id", "size"),
                                    all_stationary_physical_sites=("site_id", "nunique"),
                                    all_stationary_mean_ndays=("ndays", "mean"),
                                    all_stationary_min_ndays=("ndays", "min"),
                                    all_stationary_max_ndays=("ndays", "max")))
    old = legacy_denominators()
    den = (old.merge(primary_month_counts, on=["iso", "year"], how="outer")
           .merge(primary_daily_counts, on=["iso", "year"], how="outer")
           .merge(stationary_month_counts, on=["iso", "year"], how="outer")
           .merge(stationary_daily_counts, on=["iso", "year"], how="outer"))
    den.to_csv(OUT / f"country_year_denominators_{prefix}.csv", index=False)

    expected = int(sensors.sensor_id.nunique())
    downloaded = int(cache_audit.sensor_id.nunique()) if not cache_audit.empty else 0
    complete = not checkpoint and downloaded == expected
    monitor = source_audit.source_is_monitor.map(optional_bool)
    mobile = source_audit.source_is_mobile.map(optional_bool)
    summary = {
        "checkpoint": checkpoint, "complete_sensor_cache": complete,
        "metadata_sensor_count": expected, "cached_sensor_count": downloaded,
        "location_metadata_rows": int(len(pd.read_csv(OUT / "canonical_location_provider_metadata.csv")))
        if (OUT / "canonical_location_provider_metadata.csv").exists() else 0,
        "location_metadata_unique_locations": int(pd.read_csv(OUT / "canonical_location_provider_metadata.csv").location_id.nunique())
        if (OUT / "canonical_location_provider_metadata.csv").exists() else 0,
        "reference_flag_true_sensors": int(monitor.map(lambda x: x is True).sum()),
        "reference_flag_false_sensors": int(monitor.map(lambda x: x is False).sum()),
        "reference_flag_unknown_sensors": int(monitor.map(lambda x: x is None).sum()),
        "mobile_flag_true_sensors": int(mobile.map(lambda x: x is True).sum()),
        "stationary_flag_sensors": int(mobile.map(lambda x: x is False).sum()),
        "mobile_flag_unknown_sensors": int(mobile.map(lambda x: x is None).sum()),
        "primary_source_eligible_sensors": int(source_audit.primary_source_eligible.fillna(False).astype(bool).sum()),
        "known_low_cost_instrument_sensors": int(source_audit.known_low_cost_instrument_evidence.fillna("").astype(str).str.len().gt(0).sum()),
        "source_flag_conflict_sensors": int(source_audit.source_flag_conflict.fillna("").astype(str).str.len().gt(0).sum()),
        "provider_metadata_present_sensors": int(source_audit.provider_metadata_present.fillna(False).astype(bool).sum()),
        "instrument_metadata_present_sensors": int(source_audit.instrument_metadata_present.fillna(False).astype(bool).sum()),
        "timezone_metadata_present_sensors": int(source_audit.timezone_metadata_present.fillna(False).astype(bool).sum()),
        "primary_clean_daily_rows": int(len(primary_daily_counts)) if primary_daily_counts.empty else int(primary_daily_counts.primary_station_days.sum()),
        "primary_clean_month_rows": int(len(primary_month)),
        "primary_distinct_sites": int(primary_month.site_id.nunique()) if not primary_month.empty else 0,
        "primary_years": sorted(map(int, primary_month.year.unique())) if not primary_month.empty else [],
        "all_stationary_clean_daily_rows": int(stationary_daily_counts.all_stationary_station_days.sum()) if not stationary_daily_counts.empty else 0,
        "all_stationary_clean_month_rows": int(len(stationary_month)),
        "all_stationary_distinct_sites": int(stationary_month.site_id.nunique()) if not stationary_month.empty else 0,
        "all_source_stratified_month_rows": stratified_month_rows,
        "main_model_excluded_primary_sites": int(primary_month.loc[~primary_month.main_model_eligible, "site_id"].nunique()) if not primary_month.empty else 0,
        "source_flag_semantics": {
            "location_monitor": "OpenAQ considers location a reference monitor; not a local regulatory certification",
            "location_mobile": "OpenAQ considers location mobile; primary cohort requires false",
            "primary_rule": "source_is_monitor=true and source_is_mobile=false, filtered before sensor-alias averaging",
            "named_low_cost_devices": "diagnostic instrument-name matches only; retained and do not override OpenAQ location flags",
            "docs": [OPENAQ_LOCATION_FLAGS_DOC, OPENAQ_INSTRUMENTS_DOC, *LOW_COST_SOURCE_DOCS],
        },
        "qc": {
            "years_requested": {r: [min(y), max(y)] for r, y in YEARS_FOR.items()},
            "minimum_distinct_days_per_month": MIN_MONTH_DAYS,
            "coverage_rule": "retain rows with reported coverage >=75%; also retain unreported coverage and report it separately",
            "time_rule": "use period.datetimeFrom.local actual local calendar date; no UTC date substitution",
            "unit_rule": "convert recognized µg/m3, mg/m3, or ng/m3 to µg/m3; unknown units and negative or >=2000 values are excluded",
            "physical_grouping": "equal-weight sensor-day averages within exact country+latitude+longitude station group",
            "primary_alias_rule": "reference-eligible sensor records are filtered before any exact-coordinate sensor alias averaging",
            "secondary_outputs": "all-stationary pooled and source-stratified daily/monthly tables are supplied for cohort/bias sensitivity",
        },
    }
    (OUT / f"qc_summary_{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_outputs_v2(checkpoint=a.checkpoint)


if __name__ == "__main__":
    main()
