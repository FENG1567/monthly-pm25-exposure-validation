"""Rebuild EPA AQS PM2.5 station-day/month data for repair-v2.

Reads original 2018-2023 AirData files without modifying them. Downloaded 2024+
files are expected under F:\\cand1_repair_v2_data\\epa.  The primary daily
series keeps the two full-day AQS sample durations in LC units, requires at
least 75% observation completeness, removes methods 236/238 per the supplied
EPA advisory, and collapses event-status, duration/method, then POC duplicates
before calculating one value per site-date. Excluded-event-only rows remain in
the primary data so smoke-event observations are not erased.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[3]
ORIGINAL = Path(r"F:\cand1_20261105")
RAW_ROOT = Path(r"F:\cand1_repair_v2_data\epa")
OUT = WORKSPACE / "outputs" / "repair_v2" / "epa"
MIN_DAYS_PER_MONTH = 20
MIN_OBSERVATION_PERCENT = 75.0
GOOD_DURATIONS = {"24 HOUR", "24-HR BLK AVG"}
ADVISORY_EXCLUDE_METHODS = {"236", "238"}
USECOLS = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC",
    "Latitude", "Longitude", "Sample Duration", "Pollutant Standard",
    "Date Local", "Units of Measure", "Event Type", "Observation Count",
    "Observation Percent", "Arithmetic Mean", "Method Code", "Method Name",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(year: int) -> Path:
    if year <= 2022:
        return ORIGINAL / "incoming" / "aqs" / str(year) / f"daily_88101_{year}.zip"
    if year == 2023:
        return ORIGINAL / "incoming" / "epa_daily" / f"daily_88101_{year}.zip"
    return RAW_ROOT / "aqs" / "daily" / f"daily_88101_{year}.zip"


def get_csv_member(path: Path) -> str:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure in {path}: {bad}")
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {csvs}")
        return csvs[0]


def load_year(path: Path, year: int) -> tuple[pd.DataFrame, dict, list[pd.DataFrame], pd.DataFrame]:
    """Read one year in chunks; retain only sufficient fields and rows."""
    member = get_csv_member(path)
    stage = Counter()
    legacy_parts: list[pd.DataFrame] = []
    eligible_parts: list[pd.DataFrame] = []
    method_raw: list[pd.DataFrame] = []
    required = set(USECOLS)
    with zipfile.ZipFile(path) as zf, zf.open(member) as raw:
        for chunk in pd.read_csv(raw, usecols=lambda c: c.strip() in required,
                                 dtype=str, chunksize=250_000, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"{path} missing AQS columns: {sorted(missing)}")
            stage["raw_rows"] += len(chunk)
            param = chunk["Parameter Code"].astype(str).str.strip().eq("88101")
            stage["parameter_88101"] += int(param.sum())
            chunk = chunk.loc[param].copy()
            units = chunk["Units of Measure"].astype(str).str.strip().eq("Micrograms/cubic meter (LC)")
            stage["units_lc"] += int(units.sum())
            chunk = chunk.loc[units].copy()

            # Capture the legacy prep_stations.py path before QC filters: it
            # groups all durations/methods straight to site-date, then applies
            # the distinct-date 20-day/month rule.
            old = chunk.copy()
            old["sid"] = (old["State Code"].str.strip().str.zfill(2) +
                          old["County Code"].str.strip().str.zfill(3) +
                          old["Site Num"].str.strip().str.zfill(4))
            old["date"] = pd.to_datetime(old["Date Local"], errors="coerce")
            old["pm25"] = pd.to_numeric(old["Arithmetic Mean"], errors="coerce")
            old["lat"] = pd.to_numeric(old["Latitude"], errors="coerce")
            old["lon"] = pd.to_numeric(old["Longitude"], errors="coerce")
            old["pm25"] = old["pm25"].where((old["pm25"] >= 0) & (old["pm25"] < 2000))
            old = old.dropna(subset=["date", "lat", "lon"])
            old = old.loc[old["date"].dt.year.eq(year)]
            if not old.empty:
                g = old.groupby(["sid", "date"], as_index=False).agg(
                    value_sum=("pm25", "sum"), value_n=("pm25", "count"),
                    lat=("lat", "median"), lon=("lon", "median"))
                legacy_parts.append(g)

            stage["full_day_duration"] += int(chunk["Sample Duration"].astype(str).str.strip().isin(GOOD_DURATIONS).sum())
            chunk = chunk.loc[chunk["Sample Duration"].astype(str).str.strip().isin(GOOD_DURATIONS)].copy()
            chunk["obs_count_n"] = pd.to_numeric(chunk["Observation Count"], errors="coerce")
            chunk["obs_pct_n"] = pd.to_numeric(chunk["Observation Percent"], errors="coerce")
            complete = chunk["obs_count_n"].ge(1) & chunk["obs_pct_n"].ge(MIN_OBSERVATION_PERCENT)
            stage["completeness_ge_75pct"] += int(complete.sum())
            chunk = chunk.loc[complete].copy()
            method = chunk["Method Code"].astype(str).str.strip()
            method_ok = ~method.isin(ADVISORY_EXCLUDE_METHODS)
            stage["removed_method_236_238"] += int((~method_ok).sum())
            stage["after_method_exclusions"] += int(method_ok.sum())
            method_raw.append(chunk.assign(_method_included=method_ok))
            chunk = chunk.loc[method_ok].copy()

            chunk["sid"] = (chunk["State Code"].str.strip().str.zfill(2) +
                            chunk["County Code"].str.strip().str.zfill(3) +
                            chunk["Site Num"].str.strip().str.zfill(4))
            chunk["date"] = pd.to_datetime(chunk["Date Local"], errors="coerce")
            chunk["pm25"] = pd.to_numeric(chunk["Arithmetic Mean"], errors="coerce")
            chunk["lat"] = pd.to_numeric(chunk["Latitude"], errors="coerce")
            chunk["lon"] = pd.to_numeric(chunk["Longitude"], errors="coerce")
            valid = (chunk["date"].notna() & chunk["date"].dt.year.eq(year) &
                     chunk["pm25"].ge(0) & chunk["pm25"].lt(2000) &
                     chunk["lat"].between(-90, 90) & chunk["lon"].between(-180, 180))
            stage["valid_date_value_coordinates"] += int(valid.sum())
            chunk = chunk.loc[valid].copy()
            if not chunk.empty:
                chunk["poc"] = chunk["POC"].astype(str).str.strip()
                chunk["duration"] = chunk["Sample Duration"].astype(str).str.strip()
                chunk["method_code"] = chunk["Method Code"].astype(str).str.strip()
                chunk["method_name"] = chunk["Method Name"].astype(str).str.strip()
                chunk["standard"] = chunk["Pollutant Standard"].fillna("").astype(str).str.strip()
                chunk["event"] = chunk["Event Type"].fillna("").astype(str).str.strip()
                chunk.loc[chunk["event"].str.lower().isin(["nan", "none"]), "event"] = ""
                eligible_parts.append(chunk[["sid", "date", "poc", "duration", "method_code",
                                              "method_name", "standard", "event", "pm25", "lat", "lon"]])
    if not method_raw:
        method_counts = pd.DataFrame(columns=["method_code", "method_name", "n_after_completeness", "n_excluded_advisory"])
    else:
        mr = pd.concat(method_raw, ignore_index=True)
        method_counts = (mr.groupby(["Method Code", "Method Name", "_method_included"], dropna=False)
                           .size().rename("n").reset_index())
        method_counts = method_counts.rename(columns={"Method Code": "method_code", "Method Name": "method_name",
                                                       "_method_included": "kept_after_advisory"})
    eligible = pd.concat(eligible_parts, ignore_index=True) if eligible_parts else pd.DataFrame(
        columns=["sid", "date", "poc", "duration", "method_code", "method_name", "standard", "event", "pm25", "lat", "lon"])
    stage["eligible_source_rows"] = len(eligible)
    return eligible, dict(stage), legacy_parts, method_counts


def new_daily(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        empty = pd.DataFrame(columns=["country", "site_id", "date", "year", "month", "lat", "lon", "pm25", "n_poc"])
        return empty, pd.DataFrame(), pd.DataFrame(), empty.copy()
    # One source record per monitor POC/date/duration/method/standard. If EPA
    # publishes both Included and Excluded versions for that key, Included is
    # preferred; an Excluded-only observation is retained to keep smoke days.
    key = ["sid", "date", "poc", "duration", "method_code", "standard"]
    rows = rows.copy()
    rows["event_rank"] = np.select([rows["event"].eq("Included"), rows["event"].eq("Excluded")], [0, 1], default=2)
    rows["event_norm"] = np.select([rows["event_rank"].eq(0), rows["event_rank"].eq(1)], ["Included", "Excluded"], default="Unspecified")
    event_stats = (rows.groupby("event_norm", dropna=False).agg(rows=("pm25", "size"),
                   mean_pm25=("pm25", "mean")).reset_index())
    per_event = (rows.groupby(key + ["event_rank", "event_norm"], as_index=False, dropna=False)
                   .agg(pm25=("pm25", "mean"), lat=("lat", "median"), lon=("lon", "median"),
                        n_raw_rows=("pm25", "size"), method_name=("method_name", "first")))
    best_rank = per_event.groupby(key, as_index=False, dropna=False)["event_rank"].min()
    chosen = per_event.merge(best_rank, on=key + ["event_rank"], how="inner")
    chosen = chosen.sort_values(key + ["event_rank", "event_norm"]).drop_duplicates(key, keep="first")
    def aggregate(selected: pd.DataFrame) -> pd.DataFrame:
        # First reduce multiple full-day durations and methods within a POC/day;
        # then average the independent monitor POCs to one station-date value.
        poc_day = (selected.groupby(["sid", "date", "poc"], as_index=False, dropna=False)
                   .agg(pm25=("pm25", "mean"), lat=("lat", "median"), lon=("lon", "median"),
                        n_measurements=("pm25", "size")))
        daily = (poc_day.groupby(["sid", "date"], as_index=False, dropna=False)
                 .agg(pm25=("pm25", "mean"), lat=("lat", "median"), lon=("lon", "median"),
                      n_poc=("poc", "nunique"), n_measurements=("n_measurements", "sum")))
        daily["country"] = "USA"
        daily["site_id"] = "US_" + daily["sid"].astype(str)
        daily["year"] = daily["date"].dt.year.astype("int16")
        daily["month"] = daily["date"].dt.month.astype("int8")
        return daily[["country", "site_id", "date", "year", "month", "lat", "lon", "pm25", "n_poc", "n_measurements"]]

    daily = aggregate(chosen)
    # Prespecified sensitivity: retain records with EPA's Included designation
    # only. Primary includes every unique observation, retaining Excluded-only
    # event records to avoid deleting smoke-related high concentrations.
    included_daily = aggregate(chosen.loc[chosen["event_norm"].eq("Included")])
    method_counts = (chosen.groupby(["method_code", "method_name"], dropna=False)
                     .size().rename("n_selected_source_keys").reset_index())
    return daily, event_stats, method_counts, included_daily


def monthly_from_daily(daily: pd.DataFrame, min_days: int = MIN_DAYS_PER_MONTH) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["country", "site_id", "lat", "lon", "year", "month", "pm25", "ndays", "n_poc_days"])
    m = (daily.groupby(["country", "site_id", "year", "month"], as_index=False)
         .agg(pm25=("pm25", "mean"), ndays=("date", "nunique"), lat=("lat", "median"), lon=("lon", "median"),
              n_poc_days=("n_poc", "sum")))
    m = m.loc[m["ndays"].ge(min_days)].copy()
    m["ndays"] = m["ndays"].astype("int16")
    return m[["country", "site_id", "lat", "lon", "year", "month", "pm25", "ndays", "n_poc_days"]].sort_values(["year", "month", "site_id"])


def legacy_monthly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["country", "site_id", "year", "month", "old_pm25", "old_ndays"])
    x = pd.concat(parts, ignore_index=True)
    # Chunk aggregates are combined with sums/counts, preserving the original
    # row-weighted mean before the legacy per-site-date/month collapse.
    d = (x.groupby(["sid", "date"], as_index=False)
         .agg(value_sum=("value_sum", "sum"), value_n=("value_n", "sum")))
    d["pm25"] = d["value_sum"] / d["value_n"].replace(0, np.nan)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    m = (d.dropna(subset=["pm25"]).groupby(["sid", "year", "month"], as_index=False)
         .agg(old_pm25=("pm25", "mean"), old_ndays=("date", "nunique")))
    m = m.loc[m["old_ndays"].ge(MIN_DAYS_PER_MONTH)].copy()
    m["country"] = "USA"
    m["site_id"] = "US_" + m["sid"].astype(str)
    return m[["country", "site_id", "year", "month", "old_pm25", "old_ndays"]]


def build(years: range) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    legacy_parts: list[pd.DataFrame] = []
    waterfall = []
    methods = []
    source_ledger = []
    for year in years:
        path = source_path(year)
        if not path.exists():
            raise FileNotFoundError(f"Missing AQS daily file for {year}: {path}")
        print(f"Reading {year}: {path}", flush=True)
        rows, counts, legacy, method_raw = load_year(path, year)
        all_rows.append(rows)
        legacy_parts.extend(legacy)
        source_ledger.append({"year": year, "file": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256(path), "zip_crc_ok": True,
                              "zip_member": get_csv_member(path),
                              "url": f"https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip",
                              "source": "original_read_only" if year <= 2023 else "repair_v2_download"})
        for stage, n in counts.items():
            waterfall.append({"year": year, "stage": stage, "records": n})
        if not method_raw.empty:
            method_raw.insert(0, "year", year)
            methods.append(method_raw)
        print(f"  filtered source rows={len(rows):,}; legacy groups={sum(len(p) for p in legacy):,}", flush=True)
    all_qc = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    by_year = []
    daily_frames = []
    event_summaries = []
    selected_method_summaries = []
    included_only_frames = []
    if not all_qc.empty:
        for year, yrows in all_qc.groupby(all_qc["date"].dt.year):
            dy, ev, mc, included_dy = new_daily(yrows)
            daily_frames.append(dy)
            included_only_frames.append(included_dy)
            ev.insert(0, "year", year)
            event_summaries.append(ev)
            mc.insert(0, "year", year)
            selected_method_summaries.append(mc)
            source_key = ["sid", "date", "poc", "duration", "method_code", "standard"]
            waterfall.extend([
                {"year": int(year), "stage": "event_status_candidates_by_source_key", "records": int(yrows.groupby(source_key + ["event"], dropna=False).ngroups)},
                {"year": int(year), "stage": "event_status_resolved_source_keys", "records": int(yrows.groupby(source_key, dropna=False).ngroups)},
                {"year": int(year), "stage": "monitor_poc_days", "records": int(yrows.groupby(["sid", "date", "poc"], dropna=False).ngroups)},
                {"year": int(year), "stage": "station_days", "records": len(dy)},
            ])
            by_year.append({"year": int(year), "eligible_source_rows": len(yrows),
                            "event_status_candidates": int(yrows.groupby(source_key + ["event"], dropna=False).ngroups),
                            "event_status_resolved_source_keys": int(yrows.groupby(source_key, dropna=False).ngroups),
                            "station_days": len(dy), "stations": dy["site_id"].nunique()})
            print(f"  {year}: station-days={len(dy):,}; stations={dy.site_id.nunique():,}", flush=True)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    monthly = monthly_from_daily(daily)
    join_keys = ["country", "site_id", "year", "month"]
    included_daily = pd.concat(included_only_frames, ignore_index=True) if included_only_frames else pd.DataFrame()
    included_monthly = monthly_from_daily(included_daily)
    event_sensitivity = monthly.merge(included_monthly, on=join_keys, how="outer", suffixes=("_primary", "_included_only"), indicator=True)
    event_sensitivity["support"] = event_sensitivity["_merge"].map({"both": "both", "left_only": "primary_only", "right_only": "included_only"})
    event_sensitivity["delta_included_minus_primary"] = event_sensitivity["pm25_included_only"] - event_sensitivity["pm25_primary"]
    event_sensitivity = event_sensitivity.drop(columns="_merge")
    oldm = legacy_monthly(legacy_parts)
    comparison = oldm.merge(monthly, on=join_keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    comparison["old_vs_new"] = comparison["_merge"].map({"both": "both", "left_only": "old_only", "right_only": "new_only"})
    comparison["delta_pm25_new_minus_old"] = comparison["pm25"] - comparison["old_pm25"]
    comparison["abs_delta_pm25"] = comparison["delta_pm25_new_minus_old"].abs()
    comparison = comparison.drop(columns="_merge")
    comp_summary = (comparison.groupby("year", as_index=False)
                    .agg(old_station_months=("old_pm25", "count"), new_station_months=("pm25", "count"),
                         common_station_months=("old_vs_new", lambda x: int((x == "both").sum())),
                         old_only=("old_vs_new", lambda x: int((x == "old_only").sum())),
                         new_only=("old_vs_new", lambda x: int((x == "new_only").sum())),
                         mean_delta=("delta_pm25_new_minus_old", "mean"),
                         mean_abs_delta=("abs_delta_pm25", "mean"), max_abs_delta=("abs_delta_pm25", "max")))
    years_s = f"{min(years)}_{max(years)}"
    daily_path = OUT / f"station_day_{years_s}.parquet"
    monthly_path = OUT / f"station_month_{years_s}.parquet"
    daily.sort_values(["date", "site_id"]).to_parquet(daily_path, index=False)
    monthly.to_parquet(monthly_path, index=False)
    pd.DataFrame(waterfall).to_csv(OUT / f"qc_waterfall_{years_s}.csv", index=False)
    pd.DataFrame(by_year).to_csv(OUT / f"qc_station_day_counts_{years_s}.csv", index=False)
    pd.concat(methods, ignore_index=True).to_csv(OUT / f"qc_method_year_pre_advisory_{years_s}.csv", index=False) if methods else None
    pd.concat(selected_method_summaries, ignore_index=True).to_csv(OUT / f"qc_method_year_selected_{years_s}.csv", index=False) if selected_method_summaries else None
    pd.concat(event_summaries, ignore_index=True).to_csv(OUT / f"qc_event_status_{years_s}.csv", index=False) if event_summaries else None
    comparison.to_csv(OUT / f"old_vs_new_station_month_{years_s}.csv", index=False)
    comp_summary.to_csv(OUT / f"old_vs_new_summary_{years_s}.csv", index=False)
    event_sensitivity.to_csv(OUT / f"sensitivity_included_only_{years_s}.csv", index=False)
    included_monthly.to_parquet(OUT / f"sensitivity_included_only_station_month_{years_s}.parquet", index=False)
    pd.DataFrame(source_ledger).to_csv(OUT / f"input_ledger_{years_s}.csv", index=False)
    summary = {"years": [min(years), max(years)], "created_utc": datetime.now(timezone.utc).isoformat(),
               "primary_rules": {"parameter_code": "88101", "units": "Micrograms/cubic meter (LC)",
                                 "sample_duration": sorted(GOOD_DURATIONS),
                                 "min_observation_percent": MIN_OBSERVATION_PERCENT,
                                 "exclude_method_codes": sorted(ADVISORY_EXCLUDE_METHODS),
                                 "event_handling": "Included preferred when the same site/POC/date/duration/method/standard has multiple event statuses; Excluded-only and unspecified observations retained",
                                 "event_sensitivity": "Secondary series removes Excluded-only and unspecified records after event duplicate resolution; not used for primary because it may remove smoke-related observations",
                                 "aggregation": "event-status key -> duration/method within POC-day -> mean of POCs to site-day -> monthly mean of distinct site-days; require >=20 distinct days"},
               "station_days": int(len(daily)), "station_months": int(len(monthly)),
               "stations": int(monthly.site_id.nunique()) if not monthly.empty else 0,
               "parquet": {"daily": str(daily_path), "monthly": str(monthly_path)}}
    (OUT / f"qc_summary_{years_s}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2023)
    a = p.parse_args()
    build(range(a.start_year, a.end_year + 1))


if __name__ == "__main__":
    main()
