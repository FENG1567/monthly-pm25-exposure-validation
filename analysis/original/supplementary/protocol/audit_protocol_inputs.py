#!/usr/bin/env python3
"""Summarize the pinned final station-month inputs and saved spatial folds.

Read-only audit. Uses parquet column projection and bounded Arrow batches; it
does not fit models, alter source files, or emit row-level target data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


V = Path("/external/project/validation")
OUT = V / "stoten_revision_20261116/protocol"
BATCH_ROWS = 25_000

DATASETS: list[dict[str, str]] = []
GLOBAL_INPUT_ROOT_BY_COUNTRY = {
    "AUS": "global_union_v2_remaining_models_retry1_20261114",
    "BD": "global_union_v2_remaining_models_retry1_20261114",
    "CL": "ready_global_models_20261113_080649Z",
    "ES": "ready_global_additional_models_20261113",
    "GR": "ready_global_models_20261113_080649Z",
    "IT": "ready_global_models_20261113_080649Z",
    "MEX": "ready_global_models_20261113_080649Z",
    "NP": "global_union_v2_remaining_models_retry1_20261114",
    "PK": "global_union_v2_remaining_models_retry1_20261114",
    "PT": "ready_global_additional_models_20261113",
    "TH": "global_union_v2_remaining_models_retry1_20261114",
    "VN": "global_union_v2_remaining_models_retry1_20261114",
    "ZA": "ready_global_additional_models_20261113",
}
GLOBAL_RESULT_ROOT_BY_COUNTRY = GLOBAL_INPUT_ROOT_BY_COUNTRY
for country in ("AUS", "BD", "CL", "ES", "GR", "IT", "MEX", "NP", "PK", "PT", "TH", "VN", "ZA"):
    DATASETS.append({
        "country": country,
        "source": "Queensland monitoring" if country == "AUS" else (
            "SINAICA" if country == "MEX" else "OpenAQ"),
        "policy": "QLD retain-valid-negative-hourly / coordinate-repaired primary" if country == "AUS" else (
            "SINAICA operational-primary; source policy label nonnegative-hourly/quarantine 503/504/2022"
            if country == "MEX" else "OpenAQ reference-sensor-first"),
        "path": str(V / f"{GLOBAL_INPUT_ROOT_BY_COUNTRY[country]}/inputs/{country}/strict.parquet"),
        "domestic_status": "ineligible in accepted domestic T1 table" if country in {"BD", "NP", "VN"} else "eligible in accepted domestic T1 table",
    })
DATASETS += [
    {
        "country": "BRA", "source": "MonitorAr", "policy": "source-month-coordinate convention; 2022 historical model cohort",
        "path": str(V / "bra_in_final_feature_candidates_v1_20261115/datasets/BRA_historical/BRA_historical_bra_strict.parquet"),
        "domestic_status": "eligible in accepted domestic T1 table",
    },
    {
        "country": "IN", "source": "OpenAQ", "policy": "OpenAQ reference-sensor-first",
        "path": str(V / "bra_in_final_feature_candidates_v1_20261115/datasets/IN_historical/IN_historical_in_strict.parquet"),
        "domestic_status": "eligible in accepted domestic T1 table",
    },
    {
        "country": "USA", "source": "US EPA AQS/CSN", "policy": "primary target excludes method codes 236/238",
        "path": str(V / "uscan_final_feature_candidates_v2_20261115/usa_hist/usa_hist_usa_strict.parquet"),
        "domestic_status": "eligible in accepted domestic T1 table",
    },
    {
        "country": "CAN", "source": "NAPS", "policy": "primary all-published target; common-method definitions separate sensitivities",
        "path": str(V / "uscan_final_feature_candidates_v2_20261115/can_hist_primary/can_hist_primary_can_strict.parquet"),
        "domestic_status": "eligible in accepted domestic T1 table",
    },
]

ALIASES = {
    "target": ("pm25", "pm25_mean", "mean_pm25", "pm25_monthly", "target", "y"),
    "site": ("site_id", "station_id", "physical_site_id", "site", "monitor_id"),
    "year": ("year",),
    "month": ("month",),
    "valid_days": ("ndays", "n_days", "valid_days", "n_valid_days", "valid_date_count"),
    "lat": ("lat", "latitude", "station_latitude", "site_latitude", "lat_f"),
    "lon": ("lon", "longitude", "station_longitude", "site_longitude", "lon_f"),
    "unit": ("unit", "units", "pm25_unit", "concentration_unit"),
}
FEATURES = (
    "aod_raw", "aod_fill", "aod_used", "aod_missing", "aod_raw_missing",
    "nobs", "valid_days", "valid_pixel_count", "valid_pixel_days",
    "blh", "wind10", "tp", "t2m", "rh", "vent", "pop", "lpop",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def resolve_columns(names: list[str]) -> dict[str, str | None]:
    by_norm = {norm(name): name for name in names}
    resolved: dict[str, str | None] = {}
    for key, candidates in ALIASES.items():
        resolved[key] = next((by_norm[norm(c)] for c in candidates if norm(c) in by_norm), None)
    for feature in FEATURES:
        resolved[feature] = by_norm.get(norm(feature))
    return resolved


class StreamStats:
    """Exact moments and extrema with a bounded reservoir for percentiles."""

    def __init__(self, seed: int, sample_capacity: int = 20_000) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.sample: list[float] = []
        self.sample_capacity = sample_capacity
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype="float64")
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            return

        batch_n = int(arr.size)
        batch_mean = float(np.mean(arr))
        batch_m2 = float(np.sum((arr - batch_mean) ** 2))
        if self.n == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            combined_n = self.n + batch_n
            delta = batch_mean - self.mean
            self.m2 += batch_m2 + delta * delta * self.n * batch_n / combined_n
            self.mean += delta * batch_n / combined_n
        self.n += batch_n
        batch_min = float(np.min(arr))
        batch_max = float(np.max(arr))
        self.minimum = batch_min if self.minimum is None else min(self.minimum, batch_min)
        self.maximum = batch_max if self.maximum is None else max(self.maximum, batch_max)

        start = 0
        if len(self.sample) < self.sample_capacity:
            take = min(self.sample_capacity - len(self.sample), batch_n)
            self.sample.extend(float(x) for x in arr[:take])
            start = take
        if start < batch_n:
            rest = arr[start:]
            prior_n = self.n - batch_n + start
            # Vectorized candidate positions; only reservoir hits are applied sequentially.
            high = prior_n + np.arange(1, len(rest) + 1, dtype="int64")
            candidates = (self.rng.random(len(rest)) * high).astype("int64")
            for index in np.flatnonzero(candidates < self.sample_capacity):
                self.sample[int(candidates[index])] = float(rest[index])

    def summary(self, percentiles: tuple[int, ...] = (5, 50, 95)) -> dict[str, Any]:
        if self.n == 0:
            return {"n_finite": 0, "min": None, "max": None,
                    "mean": None, "sd": None, "sample_n": 0,
                    **{f"p{p:02d}" if p < 100 else f"p{p}": None for p in percentiles}}
        sample = np.asarray(self.sample, dtype="float64")
        qs = np.percentile(sample, percentiles) if sample.size else [None] * len(percentiles)
        result: dict[str, Any] = {
            "n_finite": self.n, "min": self.minimum, "max": self.maximum,
            "mean": self.mean, "sd": math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else None,
            "sample_n": int(sample.size),
        }
        for p, value in zip(percentiles, qs):
            key = f"p{p:02d}" if p < 100 else f"p{p}"
            result[key] = float(value) if value is not None else None
        return result


def summarize_dataset(item: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(item["path"])
    if not path.is_file():
        raise FileNotFoundError(f"selected analysis input is missing: {path}")
    sha_before = digest(path)
    parquet = pq.ParquetFile(path)
    names = list(parquet.schema_arrow.names)
    cols = resolve_columns(names)
    for required in ("target", "site", "year", "month"):
        if cols[required] is None:
            raise ValueError(f"{item['country']} lacks {required} field; available={names}")
    fields = dict(cols)
    read_cols = list(dict.fromkeys(str(v) for v in fields.values() if v is not None))

    target_stats_acc = StreamStats(seed=sum(map(ord, item["country"])) + 101, sample_capacity=100_000)
    valid_days_stats_acc = StreamStats(seed=sum(map(ord, item["country"])) + 102)
    site_ids: set[str] = set()
    site_months: set[tuple[str, int, int]] = set()
    site_month_counts: Counter[tuple[str, int, int]] = Counter()
    years: Counter[int] = Counter()
    months: set[int] = set()
    coords_by_site: dict[str, set[tuple[float, float]]] = defaultdict(set)
    units: set[str] = set()
    feature_acc: dict[str, dict[str, Any]] = {
        feature: {"present": cols.get(feature) is not None, "n": 0, "finite": 0,
                  "nonfinite": 0,
                  "stats": StreamStats(seed=sum(map(ord, item["country"])) + 1000 + i)}
        for i, feature in enumerate(FEATURES)
    }
    n_rows = n_nonfinite_target = n_negative = n_log_invalid = n_zero = n_ge_2000 = 0

    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=read_cols, use_threads=False):
        frame = batch.to_pandas()
        n = len(frame)
        n_rows += n
        target = np.asarray(frame[str(cols["target"])], dtype="float64")
        finite = np.isfinite(target)
        n_nonfinite_target += int((~finite).sum())
        vals = target[finite]
        target_stats_acc.update(vals)
        n_negative += int((vals < 0).sum())
        n_log_invalid += int((vals < 0).sum())  # _target() rejects all negative values before log(y+0.1).
        n_zero += int((vals == 0).sum())
        n_ge_2000 += int((vals >= 2000).sum())

        site_col, year_col, month_col = (str(cols[k]) for k in ("site", "year", "month"))
        sites = frame[site_col].astype(str).to_numpy()
        yrs = np.asarray(frame[year_col], dtype="int64")
        mos = np.asarray(frame[month_col], dtype="int64")
        for site, year, month in zip(sites, yrs, mos):
            site_ids.add(site)
            key = (site, int(year), int(month))
            site_months.add(key)
            site_month_counts[key] += 1
            years[int(year)] += 1
            months.add(int(month))

        if cols["valid_days"] is not None:
            day_values = np.asarray(frame[str(cols["valid_days"])], dtype="float64")
            valid_days_stats_acc.update(day_values)
        if cols["unit"] is not None:
            units.update(str(x) for x in frame[str(cols["unit"])].dropna().unique()[:20])
        if cols["lat"] is not None and cols["lon"] is not None:
            lat = np.asarray(frame[str(cols["lat"])], dtype="float64")
            lon = np.asarray(frame[str(cols["lon"])], dtype="float64")
            for site, la, lo in zip(sites, lat, lon):
                if math.isfinite(la) and math.isfinite(lo):
                    coords_by_site[site].add((float(la), float(lo)))

        for feature in FEATURES:
            col = cols.get(feature)
            if col is None:
                continue
            arr = np.asarray(frame[str(col)], dtype="float64")
            ok = np.isfinite(arr)
            record = feature_acc[feature]
            record["n"] += n
            record["finite"] += int(ok.sum())
            record["nonfinite"] += int((~ok).sum())
            record["stats"].update(arr[ok])

    duplicate_rows = sum(v - 1 for v in site_month_counts.values() if v > 1)
    target_stats = target_stats_acc.summary((5, 25, 50, 75, 95))
    day_stats = valid_days_stats_acc.summary((5, 50, 95))
    per_year = ";".join(f"{y}:{years[y]}" for y in sorted(years))
    row = {
        "country": item["country"], "source": item["source"], "source_policy_label": item["policy"],
        "domestic_T1_status": item["domestic_status"], "analysis_input_path": str(path),
        "analysis_input_sha256_observed": sha_before, "target_field": cols["target"],
        "site_field": cols["site"], "valid_day_field": cols["valid_days"],
        "source_unit_field": cols["unit"], "source_unit_values_observed": "|".join(sorted(units)) if units else "not recorded in selected analysis table",
        "target_unit_analysis_convention": "µg m-3 per accepted model/SI target definition; unit conversion not independently replayed for legacy monthly tables",
        "n_station_month_rows": n_rows, "n_finite_targets": target_stats["n_finite"],
        "n_nonfinite_targets": n_nonfinite_target, "n_distinct_sites": len(site_ids),
        "n_distinct_site_months": len(site_months), "duplicate_site_month_rows": duplicate_rows,
        "year_min": min(years) if years else None, "year_max": max(years) if years else None,
        "n_years": len(years), "year_row_counts": per_year, "months_observed": "|".join(map(str, sorted(months))),
        "valid_days_n": day_stats["n_finite"], "valid_days_min": day_stats["min"],
        "valid_days_median": day_stats["p50"], "valid_days_max": day_stats["max"],
        "valid_day_coverage_status": "explicit field in analysis input" if cols["valid_days"] else "not recorded in analysis input; upstream sampling denominator unavailable",
        "pm25_min": target_stats["min"], "pm25_p05": target_stats["p05"],
        "pm25_p25": target_stats["p25"], "pm25_median": target_stats["p50"],
        "pm25_p75": target_stats["p75"], "pm25_p95": target_stats["p95"],
        "pm25_max": target_stats["max"], "pm25_mean": target_stats["mean"],
        "pm25_sd": target_stats["sd"], "pm25_percentile_sample_n": target_stats["sample_n"],
        "pm25_percentile_method": "exact when n_finite<=100000; otherwise deterministic-seed uniform reservoir sample (100000)",
        "n_negative_pm25": n_negative, "n_log_transform_domain_violations": n_log_invalid,
        "n_zero_pm25": n_zero, "n_pm25_ge_2000": n_ge_2000,
        "n_sites_with_multiple_input_coordinates": sum(len(x) > 1 for x in coords_by_site.values()),
        "coordinate_fields": f"{cols['lat']},{cols['lon']}" if cols["lat"] and cols["lon"] else "not present",
        "target_distribution_weighting": "unweighted station-month records; descriptive only",
        "hash_stable_pre_post": None,
    }
    sha_after = digest(path)
    row["hash_stable_pre_post"] = sha_before == sha_after
    row["analysis_input_sha256_post"] = sha_after
    if sha_before != sha_after:
        raise RuntimeError(f"input changed during audit: {path}")

    coverage_rows = []
    for feature in FEATURES:
        rec = feature_acc[feature]
        stats = rec["stats"].summary((5, 50, 95))
        coverage_rows.append({
            "country": item["country"], "source": item["source"], "analysis_input_path": str(path),
            "analysis_input_sha256": sha_before, "feature": feature,
            "column_present": rec["present"], "n_rows": rec["n"], "n_finite": rec["finite"],
            "finite_fraction": rec["finite"] / rec["n"] if rec["n"] else None,
            "n_nonfinite": rec["nonfinite"], "min": stats["min"], "p05": stats["p05"],
            "median": stats["p50"], "p95": stats["p95"], "max": stats["max"],
            "mean": stats["mean"], "sd": stats["sd"],
            "percentile_sample_n": stats["sample_n"],
            "percentile_method": "exact when n_finite<=20000; otherwise deterministic-seed uniform reservoir sample (20000)",
            "unit_or_formula_recorded_in_analysis_input": False if feature in {"blh", "wind10", "tp", "t2m", "rh", "vent", "aod_raw", "aod_fill", "aod_used"} else None,
            "scope_note": "analysis-input coverage, not global raw-source completeness",
        })
    return row, coverage_rows


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def validate_summary_row(row: dict[str, Any]) -> None:
    n_finite = int(row["n_finite_targets"])
    sample_n = int(row["pm25_percentile_sample_n"])
    if n_finite <= 100_000 and sample_n != n_finite:
        raise ValueError(f"expected exact target percentiles for {row['country']}: n={n_finite}, sample={sample_n}")
    if n_finite > 100_000 and sample_n != 100_000:
        raise ValueError(f"unexpected target reservoir size for {row['country']}: {sample_n}")
    quantiles = [float(row[k]) for k in ("pm25_min", "pm25_p05", "pm25_p25", "pm25_median", "pm25_p75", "pm25_p95", "pm25_max")]
    if any(left > right for left, right in zip(quantiles, quantiles[1:])):
        raise ValueError(f"non-monotonic target summary quantiles for {row['country']}: {quantiles}")
    if int(row["n_negative_pm25"]) or int(row["n_log_transform_domain_violations"]):
        raise ValueError(f"negative target violates validation_core._target domain for {row['country']}")


def spatial_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bra_in_root = V / "bra_in_historical_model_pack_v1_20261115/outputs"
    for country in ("AUS", "BD", "CL", "ES", "GR", "IT", "MEX", "NP", "PK", "PT", "TH", "VN", "ZA"):
        result_root = V / f"{GLOBAL_RESULT_ROOT_BY_COUNTRY[country]}/results"
        job = result_root / f"{country}_strict_no_met_hgb"
        add_spatial_manifest_rows(rows, country, job)
    for country in ("BRA", "IN"):
        job = bra_in_root / f"{country}_strict_no_met_hgb"
        add_spatial_manifest_rows(rows, country, job)

    split_path = V / "uscan32_postfit_audit_v3_20261115/split_denominators.csv"
    with split_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for r in reader:
            if r.get("qa_variant") != "strict" or r.get("profile") != "no_met" or r.get("estimator_kind") != "hgb":
                continue
            task = r.get("task", "")
            if task not in {"T2_spatial", "T2_spatial_buffer50km"}:
                continue
            rows.append({
                "country": r.get("country"), "task": task, "fold_id": r.get("fold_id"),
                "split_rule_code": "five fixed KMeans blocks; geographic scale from runner; raw geometry not present in audited denominator export",
                "buffer_km": 50 if task.endswith("buffer50km") else 0,
                "n_train_rows": r.get("train_rows"), "n_test_rows": r.get("test_rows"),
                "n_train_stations": r.get("train_stations"), "n_test_stations": r.get("test_stations"),
                "n_buffer_excluded_stations": None,
                "distance_km_min": None, "distance_km_median": None, "distance_km_p90": None,
                "test_within_50km_train_pct": None,
                "geometry_distance_evidence": "not in accepted split_denominators.csv; do not infer from count fields",
                "source_path": str(split_path), "source_job_id": r.get("job_id"),
                "status": r.get("status"),
            })
    return rows


def add_spatial_manifest_rows(out: list[dict[str, Any]], country: str, job: Path) -> None:
    summary_path = job / "fold_summaries.csv"
    split_path = job / "outer_fold_split_manifest.json"
    if not summary_path.is_file():
        out.append({"country": country, "task": "T2_spatial/T2_spatial_buffer50km", "status": "fold_summary_missing",
                    "source_path": str(job)})
        return
    excluded: dict[tuple[str, str], int] = {}
    if split_path.is_file():
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        for split in manifest.get("splits", []):
            excluded[(str(split.get("task", "")), str(split.get("fold_id", "")))] = len(split.get("buffer_excluded_station_ids", []))
    with summary_path.open(newline="", encoding="utf-8-sig") as stream:
        for r in csv.DictReader(stream):
            task = r.get("task", "")
            if task not in {"T2_spatial", "T2_spatial_buffer50km"}:
                continue
            out.append({
                "country": country, "task": task, "fold_id": r.get("fold_id"),
                "split_rule_code": "fixed KMeans station blocks; five blocks; local kilometre approximation; random_state=0,n_init=20",
                "buffer_km": r.get("buffer_km"), "n_train_rows": r.get("n_train"),
                "n_test_rows": r.get("n_test"), "n_train_stations": r.get("n_train_stations"),
                "n_test_stations": r.get("n_test_stations"),
                "n_buffer_excluded_stations": excluded.get((task, str(r.get("fold_id"))), None),
                "distance_km_min": r.get("distance_km_min"), "distance_km_median": r.get("distance_km_median"),
                "distance_km_p90": r.get("distance_km_p90"),
                "test_within_50km_train_pct": r.get("test_within_50km_train_pct"),
                "geometry_distance_evidence": "saved fold_summaries.csv; exact KMeans input geometry in runner code",
                "source_path": str(summary_path), "source_job_id": job.name,
                "status": "saved split summary",
            })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for item in DATASETS:
        row, cov = summarize_dataset(item)
        validate_summary_row(row)
        targets.append(row)
        coverage.extend(cov)
        print(f"INPUT_SUMMARY country={item['country']} rows={row['n_station_month_rows']} sites={row['n_distinct_sites']} years={row['year_min']}-{row['year_max']} negative_pm25={row['n_negative_pm25']} target_percentiles_exact={row['pm25_percentile_sample_n']==row['n_finite_targets']} hash_stable={row['hash_stable_pre_post']}", flush=True)
    spatial = spatial_rows()
    atomic_csv(OUT / "target_country_summary.csv", targets)
    atomic_csv(OUT / "feature_coverage_summary.csv", coverage)
    atomic_csv(OUT / "spatial_split_geometry_summary.csv", spatial)
    print(json.dumps({
        "status": "complete", "inputs": len(targets), "country_rows": len(targets),
        "feature_coverage_rows": len(coverage), "spatial_rows": len(spatial),
        "target_output": str(OUT / "target_country_summary.csv"),
        "feature_output": str(OUT / "feature_coverage_summary.csv"),
        "spatial_output": str(OUT / "spatial_split_geometry_summary.csv"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
