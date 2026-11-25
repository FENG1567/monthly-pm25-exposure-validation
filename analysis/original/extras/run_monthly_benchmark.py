#!/usr/bin/env python3
"""Match accepted future OOF predictions to true monthly SatPM and raw CAMS."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from netCDF4 import Dataset


ROOT = Path("/external/project/validation/monthly_satpm_v1_20261116")
VAL = Path("/external/project/validation")
PLAN = ROOT / "analysis_plan_v2.json"
FUTURE_ACCEPTANCE = VAL / "future28_actual_acceptance_v1_20261115/independent_acceptance.json"
FUTURE_ACCEPTANCE_SHA = "f69e0440213cd0ec550f955634b7f79aadbc318fae154079c42e8e6af5bf3e61"
COORDINATES = VAL / "satpm_auxiliary_source_v1_20261115/satpm_v6gl03_monthly_coordinate_samples.parquet"
COORDINATES_SHA = "d05cab8eee11f1e0b04aa3f00b1f3ddc7178680e4af2eea5c67b7d6e41eee7ee"
COORDINATE_ACCEPTANCE = VAL / "satpm_independent_acceptance_v2_20261115/independent_acceptance.json"
COORDINATE_ACCEPTANCE_SHA = "b3f432d00b9ac673817c1075bb36bf8c325133b73e23f51531b8f59e738f61e1"
CAMS = Path("/external/project/cams/station_sample_outputs/station_month_2018_2025.parquet")
CAMS_SHA = "1e09a856984e3c92474b697a4894464ebaea93bad3cd39ad83c030d0058c5554"
PRODUCT_MANIFEST = ROOT / "input_manifest/satpm_monthly_download_manifest.json"
QA_PATH = ROOT / "input_manifest/qa.json"
METRICS_PATH = ROOT / "monthly_benchmark_metrics.csv"
DELTAS_PATH = ROOT / "monthly_benchmark_paired_deltas.csv"
COVERAGE_PATH = ROOT / "monthly_match_coverage.csv"
MATCHED_PATH = ROOT / "monthly_matched_station_months.csv"
YEARS = {2023, 2024}
COUNTRIES = {"USA": "usa_primary", "CAN": "can_primary_all_published"}
CELLS = [("fixed", 2023), ("fixed", 2024), ("rolling", 2024)]
PROFILES = {"no_met", "with_met"}
METRICS = ("r2", "rmse", "mae", "bias")
SEED = 20261116
BOOTSTRAPS = 500
MIN_PAIRS = 50
MIN_SITES = 3
TARGET_TOLERANCE = 1e-10
OUT_PATHS = (METRICS_PATH, DELTAS_PATH, COVERAGE_PATH, MATCHED_PATH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: expected={expected} actual={actual}")
    return actual


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def norm_country(value: Any) -> str:
    country = str(value).upper()
    return "USA" if country in {"US", "USA"} else "CAN" if country in {"CA", "CAN"} else country


def row_key(country: Any, site_id: Any, year: Any, month: Any) -> tuple[str, str, int, int]:
    return (norm_country(country), str(site_id), int(year), int(month))


def metric_values(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return {metric: float("nan") for metric in METRICS}
    yy, pp, ww = y[keep], pred[keep], w[keep]
    total = float(ww.sum())
    error = pp - yy
    denominator = float(np.sum(ww * (yy - np.average(yy, weights=ww)) ** 2))
    r2 = float("nan") if denominator <= 0 else float(1.0 - np.sum(ww * error**2) / denominator)
    return {
        "r2": r2,
        "rmse": float(np.sqrt(np.sum(ww * error**2) / total)),
        "mae": float(np.sum(ww * np.abs(error)) / total),
        "bias": float(np.sum(ww * error) / total),
    }


def bootstrap_deltas(
    frame: pd.DataFrame,
    comparator_column: str,
    enabled: bool,
) -> tuple[dict[str, dict[str, float]], int, str]:
    y = frame["obs"].to_numpy(dtype=np.float64)
    model = frame["pred"].to_numpy(dtype=np.float64)
    comparator = frame[comparator_column].to_numpy(dtype=np.float64)
    sites = sorted(frame["site_id"].astype(str).unique().tolist())
    model_stats = metric_values(y, model)
    comparator_stats = metric_values(y, comparator)
    point = {metric: model_stats[metric] - comparator_stats[metric] for metric in METRICS}
    if not enabled:
        return ({m: {"delta": point[m], "ci_low": float("nan"), "ci_high": float("nan")} for m in METRICS}, 0,
                "ci_not_computed_minimum_support")

    site_to_index = {site: index for index, site in enumerate(sites)}
    row_codes = frame["site_id"].astype(str).map(site_to_index).to_numpy(dtype=np.int64)
    rng = np.random.default_rng(SEED)
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(sites), size=len(sites))
        counts = np.bincount(sampled, minlength=len(sites)).astype(np.float64)
        weights = counts[row_codes]
        weighted_model = metric_values(y, model, weights)
        weighted_comparator = metric_values(y, comparator, weights)
        for metric in METRICS:
            delta = weighted_model[metric] - weighted_comparator[metric]
            if math.isfinite(delta):
                values[metric].append(delta)
    valid = min((len(values[m]) for m in METRICS), default=0)
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        if not values[metric]:
            out[metric] = {"delta": point[metric], "ci_low": float("nan"), "ci_high": float("nan")}
        else:
            lo, hi = np.percentile(np.asarray(values[metric], dtype=np.float64), [2.5, 97.5])
            out[metric] = {"delta": point[metric], "ci_low": float(lo), "ci_high": float(hi)}
    status = "ci_computed" if valid == BOOTSTRAPS else "ci_computed_with_nonfinite_bootstrap_draws"
    return out, valid, status


def expected_prediction_paths() -> list[dict[str, Any]]:
    paths = []
    for country, policy in COUNTRIES.items():
        for mode, year in CELLS:
            branch = "fixed" if mode == "fixed" else "rolling"
            path = (VAL / "final_future_fixed_rolling_20261114" / branch / "primary" / policy
                    / str(year) / "strict" / "external_oof_predictions.parquet")
            paths.append({"country": country, "mode": mode, "year": year, "path": path})
    return paths


def read_predictions(pins: dict[str, str], input_hashes: dict[str, str]) -> pd.DataFrame:
    required = ["country", "profile", "task", "arm", "target_strategy", "backtransform",
                "site_id", "year", "month", "obs", "pred"]
    records: list[pd.DataFrame] = []
    expected_counts: dict[tuple[str, str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for metadata in expected_prediction_paths():
        path: Path = metadata["path"]
        path_text = str(path)
        expected = pins.get(path_text)
        if not expected:
            raise RuntimeError(f"future acceptance does not pin required OOF artifact: {path_text}")
        input_hashes[path_text] = require_sha(path, str(expected))
        parquet = pq.ParquetFile(path)
        missing = set(required) - set(parquet.schema_arrow.names)
        if missing:
            raise RuntimeError(f"required OOF columns missing from {path}: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=required, batch_size=8192):
            frame = batch.to_pandas()
            country = frame["country"].astype(str).str.upper().map(norm_country)
            profile = frame["profile"].astype(str).str.lower()
            mask = (
                country.eq(metadata["country"])
                & profile.isin(PROFILES)
                & frame["arm"].astype(str).str.lower().eq("full")
                & frame["target_strategy"].astype(str).str.lower().eq("log_smear")
                & frame["backtransform"].astype(str).str.lower().eq("smearing")
                & pd.to_numeric(frame["year"], errors="coerce").eq(metadata["year"])
            )
            selected = frame.loc[mask].copy()
            if selected.empty:
                continue
            selected["country"] = metadata["country"]
            selected["profile"] = selected["profile"].astype(str).str.lower()
            selected["site_id"] = selected["site_id"].astype(str)
            selected["year"] = pd.to_numeric(selected["year"], errors="raise").astype(int)
            selected["month"] = pd.to_numeric(selected["month"], errors="raise").astype(int)
            selected["obs"] = pd.to_numeric(selected["obs"], errors="coerce")
            selected["pred"] = pd.to_numeric(selected["pred"], errors="coerce")
            selected["validation_mode"] = metadata["mode"]
            selected["external_year"] = metadata["year"]
            selected["model_alias"] = (
                selected["country"].str.lower() + "_" + metadata["mode"] + "_" + str(metadata["year"])
                + "_strict_" + selected["profile"] + "_hgb"
            )
            if not np.isfinite(selected[["obs", "pred"]].to_numpy(dtype=np.float64)).all():
                raise RuntimeError(f"nonfinite accepted observation/prediction in selected OOF rows: {path}")
            if (~selected["month"].between(1, 12)).any():
                raise RuntimeError(f"invalid calendar month in selected OOF rows: {path}")
            for profile_name, part in selected.groupby("profile"):
                expected_counts[(metadata["country"], metadata["mode"], metadata["year"])][profile_name] += len(part)
            records.append(selected)

    if not records:
        raise RuntimeError("no accepted future OOF rows survived the locked strict/HGB/full/log-smear filter")
    predictions = pd.concat(records, ignore_index=True)
    if not predictions["profile"].isin(PROFILES).all():
        raise RuntimeError("unexpected selected profile")
    expected_cells = {(country, mode, year) for country in COUNTRIES for mode, year in CELLS}
    if set((c, m, int(y)) for c, m, y in predictions[["country", "validation_mode", "external_year"]].drop_duplicates().itertuples(index=False, name=None)) != expected_cells:
        raise RuntimeError("selected OOF cell census differs from the locked 2-country x 3-cell design")
    for cell, counts in expected_counts.items():
        if set(counts) != PROFILES:
            raise RuntimeError(f"required no_met/with_met profile missing in {cell}: {dict(counts)}")
    dup_cols = ["country", "site_id", "year", "month", "validation_mode", "external_year", "profile"]
    if predictions.duplicated(dup_cols).any():
        dup = predictions.loc[predictions.duplicated(dup_cols, keep=False), dup_cols].head(10).to_dict("records")
        raise RuntimeError(f"duplicate exact station-month OOF keys after locked filtering: {dup}")
    predictions["key_tuple"] = [row_key(c, s, y, m) for c, s, y, m in predictions[["country", "site_id", "year", "month"]].itertuples(index=False, name=None)]
    print(f"OOF_SELECTED rows={len(predictions)} cells={len(expected_cells)} profiles={sorted(PROFILES)}", flush=True)
    return predictions


def read_coordinates(candidate_keys: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    required = ["country", "site_id", "year", "month", "lat", "lon", "pm25", "ndays"]
    parquet = pq.ParquetFile(COORDINATES)
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise RuntimeError(f"monthly coordinate table is missing columns: {sorted(missing)}")
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for batch in parquet.iter_batches(columns=required, batch_size=8192):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["country"] = frame["country"].astype(str).str.upper().map(norm_country)
        frame["site_id"] = frame["site_id"].astype(str)
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
        frame = frame.loc[frame["country"].isin(COUNTRIES) & frame["year"].isin(YEARS) & frame["month"].between(1, 12)]
        for row in frame.itertuples(index=False):
            key = row_key(row.country, row.site_id, row.year, row.month)
            if key not in candidate_keys:
                continue
            if key in out:
                raise RuntimeError(f"duplicate station-month key in accepted coordinate source: {key}")
            out[key] = {
                "lat": float(row.lat) if pd.notna(row.lat) else float("nan"),
                "lon": float(row.lon) if pd.notna(row.lon) else float("nan"),
                "target": float(row.pm25) if pd.notna(row.pm25) else float("nan"),
                "ndays": int(row.ndays) if pd.notna(row.ndays) else None,
            }
    print(f"COORDINATES_SELECTED keys={len(out)} candidate_keys={len(candidate_keys)}", flush=True)
    return out


def read_cams(candidate_keys: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    required = ["country", "site_id", "year", "month", "pm25_obs_ug_m3", "cams_month_present",
                "cams_pm25_ug_m3", "cams_sample_status", "cams_n_finite_corners", "cams_valid_weight_sum"]
    parquet = pq.ParquetFile(CAMS)
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise RuntimeError(f"accepted CAMS monthly table is missing columns: {sorted(missing)}")
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    exact_duplicates = 0
    for batch in parquet.iter_batches(columns=required, batch_size=8192):
        frame = batch.to_pandas()
        frame["country"] = frame["country"].astype(str).str.upper().map(norm_country)
        frame["site_id"] = frame["site_id"].astype(str)
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
        frame = frame.loc[frame["country"].isin(COUNTRIES) & frame["year"].isin(YEARS) & frame["month"].between(1, 12)]
        for row in frame.itertuples(index=False):
            key = row_key(row.country, row.site_id, row.year, row.month)
            if key not in candidate_keys:
                continue
            record = {
                "target": float(row.pm25_obs_ug_m3) if pd.notna(row.pm25_obs_ug_m3) else float("nan"),
                "value": float(row.cams_pm25_ug_m3) if pd.notna(row.cams_pm25_ug_m3) else float("nan"),
                "month_present": bool(row.cams_month_present) if pd.notna(row.cams_month_present) else False,
                "status": str(row.cams_sample_status),
                "n_finite_corners": int(row.cams_n_finite_corners) if pd.notna(row.cams_n_finite_corners) else 0,
                "valid_weight_sum": float(row.cams_valid_weight_sum) if pd.notna(row.cams_valid_weight_sum) else 0.0,
            }
            if key in out:
                previous = out[key]
                same = (
                    np.isclose(previous["target"], record["target"], rtol=0, atol=TARGET_TOLERANCE, equal_nan=True)
                    and np.isclose(previous["value"], record["value"], rtol=0, atol=TARGET_TOLERANCE, equal_nan=True)
                    and previous["month_present"] == record["month_present"]
                    and previous["status"] == record["status"]
                )
                if not same:
                    raise RuntimeError(f"conflicting duplicate key in CAMS monthly table: {key}")
                exact_duplicates += 1
                continue
            out[key] = record
    print(f"CAMS_SELECTED keys={len(out)} identical_duplicates_deduplicated={exact_duplicates}", flush=True)
    return out


def nearest_axis(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ascending = bool(axis[-1] > axis[0])
    sorted_axis = axis if ascending else axis[::-1]
    positions = np.searchsorted(sorted_axis, values)
    high = np.clip(positions, 0, len(sorted_axis) - 1)
    low = np.clip(positions - 1, 0, len(sorted_axis) - 1)
    use_high = np.abs(sorted_axis[high] - values) < np.abs(sorted_axis[low] - values)
    sorted_index = np.where(use_high, high, low)
    index = sorted_index if ascending else len(axis) - 1 - sorted_index
    distance = np.abs(axis[index] - values)
    return index.astype(np.int64), distance


def nearest_longitude(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.mod(axis, 360.0))
    sorted_axis = np.mod(axis[order], 360.0)
    targets = np.mod(values, 360.0)
    positions = np.searchsorted(sorted_axis, targets)
    high = positions % len(sorted_axis)
    low = (positions - 1) % len(sorted_axis)
    high_distance = np.abs((sorted_axis[high] - targets + 180.0) % 360.0 - 180.0)
    low_distance = np.abs((sorted_axis[low] - targets + 180.0) % 360.0 - 180.0)
    use_high = high_distance < low_distance
    chosen = np.where(use_high, high, low)
    distance = np.where(use_high, high_distance, low_distance)
    return order[chosen].astype(np.int64), distance


def sample_satpm(
    product_files: dict[str, dict[str, Any]],
    coordinates: dict[tuple[str, str, int, int], dict[str, Any]],
    candidate_keys: set[tuple[str, str, int, int]],
) -> tuple[dict[tuple[str, str, int, int], float], dict[str, dict[str, int]]]:
    by_month: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for key in candidate_keys:
        if key in coordinates:
            by_month[f"{key[2]}{key[3]:02d}"].append(key)
    samples: dict[tuple[str, str, int, int], float] = {}
    month_counts: dict[str, dict[str, int]] = {}
    for month, keys in sorted(by_month.items()):
        entry = product_files.get(month)
        if not entry:
            raise RuntimeError(f"no verified monthly V6GL03 file for matched month={month}")
        path = Path(entry["path"])
        with Dataset(path, "r") as dataset:
            if "PM25" not in dataset.variables or "lat" not in dataset.variables or "lon" not in dataset.variables:
                raise RuntimeError(f"unexpected V6GL03 monthly NetCDF variables in {path}")
            lat_axis = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            lon_axis = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            variable = dataset.variables["PM25"]
            units = str(getattr(variable, "units", ""))
            if units.lower().replace(" ", "") not in {"ug/m3", "µg/m3", "ugm-3"}:
                raise RuntimeError(f"unexpected PM25 units in {path}: {units!r}")
            lat_step = float(np.median(np.abs(np.diff(np.sort(lat_axis)))))
            lon_mod = np.sort(np.mod(lon_axis, 360.0))
            lon_steps = np.diff(np.r_[lon_mod, lon_mod[0] + 360.0])
            lon_step = float(np.median(lon_steps))
            if not np.isclose(lat_step, 0.1, atol=1e-4) or not np.isclose(lon_step, 0.1, atol=1e-4):
                raise RuntimeError(f"unexpected native V6GL03 grid spacing in {path}: lat={lat_step}, lon={lon_step}")

            latitudes = np.asarray([coordinates[key]["lat"] for key in keys], dtype=np.float64)
            longitudes = np.asarray([coordinates[key]["lon"] for key in keys], dtype=np.float64)
            coord_finite = np.isfinite(latitudes) & np.isfinite(longitudes)
            lat_indices = np.zeros(len(keys), dtype=np.int64)
            lon_indices = np.zeros(len(keys), dtype=np.int64)
            lat_distance = np.full(len(keys), np.inf, dtype=np.float64)
            lon_distance = np.full(len(keys), np.inf, dtype=np.float64)
            if coord_finite.any():
                lat_indices[coord_finite], lat_distance[coord_finite] = nearest_axis(lat_axis, latitudes[coord_finite])
                lon_indices[coord_finite], lon_distance[coord_finite] = nearest_longitude(lon_axis, longitudes[coord_finite])
            in_grid = coord_finite & (lat_distance <= 0.051) & (lon_distance <= 0.051)
            value_count = 0
            if in_grid.any():
                selected = np.flatnonzero(in_grid)
                row_min, row_max = int(lat_indices[selected].min()), int(lat_indices[selected].max())
                col_min, col_max = int(lon_indices[selected].min()), int(lon_indices[selected].max())
                # Read a bounded North America crop for this single monthly file.
                crop = np.ma.asarray(variable[row_min:row_max + 1, col_min:col_max + 1])
                crop_values = np.asarray(np.ma.filled(crop, np.nan), dtype=np.float64)
                for index in selected:
                    value = float(crop_values[lat_indices[index] - row_min, lon_indices[index] - col_min])
                    if math.isfinite(value):
                        samples[keys[index]] = value
                        value_count += 1
            month_counts[month] = {
                "candidate_coordinate_keys": len(keys),
                "finite_coordinate_keys": int(coord_finite.sum()),
                "native_grid_keys": int(in_grid.sum()),
                "finite_satpm_values": value_count,
            }
        print(f"SAMPLED month={month} candidates={len(keys)} grid={month_counts[month]['native_grid_keys']} value={month_counts[month]['finite_satpm_values']}", flush=True)
    return samples, month_counts


def check_target_matches(
    predictions: pd.DataFrame,
    coordinates: dict[tuple[str, str, int, int], dict[str, Any]],
    cams: dict[tuple[str, str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    coord_checks = 0
    cams_checks = 0
    mismatches: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        key = row.key_tuple
        obs = float(row.obs)
        if key in coordinates:
            target = coordinates[key]["target"]
            coord_checks += 1
            if not math.isfinite(target) or not math.isclose(obs, target, rel_tol=0, abs_tol=TARGET_TOLERANCE):
                mismatches.append({"source": "satpm_monthly_coordinate_source", "key": key, "oof_obs": obs, "source_obs": target})
        if key in cams:
            target = cams[key]["target"]
            cams_checks += 1
            if not math.isfinite(target) or not math.isclose(obs, target, rel_tol=0, abs_tol=TARGET_TOLERANCE):
                mismatches.append({"source": "cams_station_month_source", "key": key, "oof_obs": obs, "source_obs": target})
        if len(mismatches) >= 100:
            break
    if mismatches:
        preview = mismatches[:10]
        raise RuntimeError(f"target mismatch across exact station-month sources; checked_coord={coord_checks}; checked_cams={cams_checks}; examples={preview}")
    return {"coordinate_target_rows_checked": coord_checks, "cams_target_rows_checked": cams_checks, "mismatch_rows": 0,
            "target_tolerance_absolute": TARGET_TOLERANCE, "target_tolerance_relative": 0.0}


def score_cells(
    predictions: pd.DataFrame,
    coordinates: dict[tuple[str, str, int, int], dict[str, Any]],
    cams: dict[tuple[str, str, int, int], dict[str, Any]],
    satpm: dict[tuple[str, str, int, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []

    group_columns = ["country", "validation_mode", "external_year", "profile", "task", "model_alias"]
    for group_key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        country, mode, year, profile, task, alias = group_key
        group = group.copy()
        base: dict[str, Any] = {
            "country": country,
            "validation_mode": mode,
            "external_year": int(year),
            "profile": profile,
            "qa_variant": "strict",
            "estimator_kind": "hgb",
            "arm": "full",
            "target_strategy": "log_smear",
            "backtransform": "smearing",
            "task": task,
            "model_alias": alias,
        }
        group_keys = group["key_tuple"].tolist()
        group["lat"] = [coordinates.get(key, {}).get("lat", float("nan")) for key in group_keys]
        group["lon"] = [coordinates.get(key, {}).get("lon", float("nan")) for key in group_keys]
        group["coordinate_target"] = [coordinates.get(key, {}).get("target", float("nan")) for key in group_keys]
        group["ndays"] = [coordinates.get(key, {}).get("ndays") for key in group_keys]
        group["satpm_v6gl03_monthly"] = [satpm.get(key, float("nan")) for key in group_keys]
        cams_records = [cams.get(key) for key in group_keys]
        group["cams_source_obs"] = [record["target"] if record else float("nan") for record in cams_records]
        group["cams_raw_value"] = [record["value"] if record else float("nan") for record in cams_records]
        group["cams_sample_status"] = [record["status"] if record else "missing_key" for record in cams_records]
        group["cams_raw_supported"] = [
            bool(record and record["month_present"] and record["status"] == "sampled"
                 and math.isfinite(record["value"]) and record["n_finite_corners"] >= 1
                 and record["valid_weight_sum"] > 0)
            for record in cams_records
        ]
        group["coordinate_key_matched"] = [key in coordinates for key in group_keys]
        group["satpm_v6gl03_supported"] = [math.isfinite(float(value)) for value in group["satpm_v6gl03_monthly"]]
        group["satpm_coordinate_in_grid"] = [
            bool(key in coordinates and math.isfinite(coordinates[key]["lat"]) and math.isfinite(coordinates[key]["lon"]))
            for key in group_keys
        ]
        group["target_match_verified"] = [key in coordinates for key in group_keys]

        for comparator, col in (("SatPM_V6GL03_monthly", "satpm_v6gl03_monthly"), ("CAMS_raw", "cams_raw_value")):
            valid_mask = np.isfinite(group[col].to_numpy(dtype=np.float64))
            if comparator == "CAMS_raw":
                valid_mask &= group["cams_raw_supported"].to_numpy(dtype=bool)
            matched = group.loc[valid_mask].copy()
            n_pairs = len(matched)
            n_sites = int(matched["site_id"].astype(str).nunique())
            ci_enabled = n_pairs >= MIN_PAIRS and n_sites >= MIN_SITES
            cov = {
                **base,
                "comparison": comparator,
                "n_prediction_rows": len(group),
                "n_prediction_sites": int(group["site_id"].astype(str).nunique()),
                "n_coordinate_keys": int(group["coordinate_key_matched"].sum()),
                "n_target_matched_rows": int((group["coordinate_key_matched"] & np.isclose(group["obs"], group["coordinate_target"], rtol=0, atol=TARGET_TOLERANCE, equal_nan=False)).sum()),
                "n_finite_site_coordinates": int(group["satpm_coordinate_in_grid"].sum()),
                "n_satpm_supported_rows": int(group["satpm_v6gl03_supported"].sum()),
                "n_cams_supported_rows": int(group["cams_raw_supported"].sum()),
                "n_matched_pairs": n_pairs,
                "n_matched_stations": n_sites,
                "row_loss_from_prediction_support": len(group) - n_pairs,
                "ci_status": "eligible" if ci_enabled else "ci_not_computed_minimum_support",
            }
            if n_pairs == 0:
                cov["comparison_status"] = "no_exact_supported_pairs"
                coverage_rows.append(cov)
                print(f"CELL {country} {mode} {year} {profile} {comparator} n=0", flush=True)
                continue
            cov["comparison_status"] = "eligible" if ci_enabled else "point_estimates_only"
            coverage_rows.append(cov)
            y = matched["obs"].to_numpy(dtype=np.float64)
            full_pred = matched["pred"].to_numpy(dtype=np.float64)
            comparator_pred = matched[col].to_numpy(dtype=np.float64)
            full_stats = metric_values(y, full_pred)
            comparator_stats = metric_values(y, comparator_pred)
            metric_rows.append({
                **base,
                "comparison": comparator,
                "n_matched_pairs": n_pairs,
                "n_matched_stations": n_sites,
                "full_r2": full_stats["r2"],
                "full_rmse_ug_m3": full_stats["rmse"],
                "full_mae_ug_m3": full_stats["mae"],
                "full_bias_ug_m3": full_stats["bias"],
                "comparator_r2": comparator_stats["r2"],
                "comparator_rmse_ug_m3": comparator_stats["rmse"],
                "comparator_mae_ug_m3": comparator_stats["mae"],
                "comparator_bias_ug_m3": comparator_stats["bias"],
                "contrast": "full minus comparator",
                "bootstrap_replicates": BOOTSTRAPS if ci_enabled else 0,
                "bootstrap_seed": SEED if ci_enabled else "",
            })
            deltas, valid_reps, ci_status = bootstrap_deltas(matched, col, ci_enabled)
            for metric in METRICS:
                delta_rows.append({
                    **base,
                    "comparison": comparator,
                    "metric": metric,
                    "contrast": "full minus comparator",
                    "delta": deltas[metric]["delta"],
                    "ci_low": deltas[metric]["ci_low"],
                    "ci_high": deltas[metric]["ci_high"],
                    "n_matched_pairs": n_pairs,
                    "n_matched_stations": n_sites,
                    "bootstrap_replicates": BOOTSTRAPS if ci_enabled else 0,
                    "bootstrap_valid_replicates": valid_reps,
                    "bootstrap_seed": SEED if ci_enabled else "",
                    "ci_status": ci_status,
                })
            print(f"CELL {country} {mode} {year} {profile} {comparator} n={n_pairs} sites={n_sites} ci={ci_status}", flush=True)

        keep_columns = ["country", "site_id", "year", "month", "profile", "validation_mode", "external_year",
                        "task", "model_alias", "obs", "pred", "lat", "lon", "ndays", "coordinate_key_matched",
                        "target_match_verified", "satpm_v6gl03_monthly", "satpm_v6gl03_supported",
                        "cams_raw_value", "cams_raw_supported", "cams_sample_status"]
        matched_rows.extend(group[keep_columns].replace({np.nan: None}).to_dict("records"))
    return metric_rows, delta_rows, coverage_rows, matched_rows


def run() -> dict[str, Any]:
    if any(path.exists() for path in OUT_PATHS):
        raise FileExistsError("refusing to overwrite existing monthly benchmark outputs")
    required_sources = [PLAN, FUTURE_ACCEPTANCE, COORDINATES, COORDINATE_ACCEPTANCE, CAMS, PRODUCT_MANIFEST]
    if not all(path.is_file() for path in required_sources):
        absent = [str(path) for path in required_sources if not path.is_file()]
        raise FileNotFoundError(f"required source files missing: {absent}")

    input_hashes: dict[str, str] = {}
    plan_sha = sha256(PLAN)
    input_hashes[str(PLAN)] = plan_sha
    input_hashes[str(FUTURE_ACCEPTANCE)] = require_sha(FUTURE_ACCEPTANCE, FUTURE_ACCEPTANCE_SHA)
    input_hashes[str(COORDINATES)] = require_sha(COORDINATES, COORDINATES_SHA)
    input_hashes[str(COORDINATE_ACCEPTANCE)] = require_sha(COORDINATE_ACCEPTANCE, COORDINATE_ACCEPTANCE_SHA)
    input_hashes[str(CAMS)] = require_sha(CAMS, CAMS_SHA)

    acceptance = json.loads(FUTURE_ACCEPTANCE.read_text(encoding="utf-8"))
    if acceptance.get("decision") != "accepted" or acceptance.get("independent_postfit_pass") is not True:
        raise RuntimeError("future OOF acceptance receipt is not independently accepted")
    pins = acceptance.get("accepted_artifact_pins")
    if not isinstance(pins, dict):
        raise RuntimeError("future acceptance has no accepted_artifact_pins mapping")
    coords_acceptance = json.loads(COORDINATE_ACCEPTANCE.read_text(encoding="utf-8"))
    if coords_acceptance.get("decision") not in {"accepted", "accepted_bounded"}:
        raise RuntimeError(f"monthly coordinate source QA is not accepted: {coords_acceptance.get('decision')}")

    product_manifest = json.loads(PRODUCT_MANIFEST.read_text(encoding="utf-8"))
    if product_manifest.get("status") != "complete" or int(product_manifest.get("months", 0)) != 24:
        raise RuntimeError("monthly V6GL03 product manifest is not complete for the locked 24-month window")
    expected_months = {f"{year}{month:02d}" for year in YEARS for month in range(1, 13)}
    product_files: dict[str, dict[str, Any]] = {}
    if len(product_manifest.get("files", [])) != 24:
        raise RuntimeError("monthly V6GL03 product manifest has the wrong file count")
    for item in product_manifest["files"]:
        month = str(item["month"])
        if month not in expected_months or month in product_files:
            raise RuntimeError(f"unexpected or duplicate monthly V6GL03 product entry: {month}")
        path = Path(item["path"])
        actual_sha = require_sha(path, str(item["sha256"]))
        if path.stat().st_size != int(item["expected_bytes_from_s3_listing"]):
            raise RuntimeError(f"product file size disagrees with S3 source listing: {path}")
        input_hashes[str(path)] = actual_sha
        product_files[month] = item
    if set(product_files) != expected_months:
        raise RuntimeError(f"V6GL03 month coverage is incomplete: {sorted(expected_months - set(product_files))}")
    input_hashes[str(PRODUCT_MANIFEST)] = sha256(PRODUCT_MANIFEST)

    predictions = read_predictions(pins, input_hashes)
    candidate_keys = set(predictions["key_tuple"].tolist())
    coordinates = read_coordinates(candidate_keys)
    cams = read_cams(candidate_keys)
    target_qa = check_target_matches(predictions, coordinates, cams)
    satpm_values, month_counts = sample_satpm(product_files, coordinates, candidate_keys)
    metrics, deltas, coverage, matched = score_cells(predictions, coordinates, cams, satpm_values)

    if not metrics or not deltas or not coverage:
        raise RuntimeError(f"benchmark produced empty summary outputs: metrics={len(metrics)} deltas={len(deltas)} coverage={len(coverage)}")
    write_csv_atomic(METRICS_PATH, metrics)
    write_csv_atomic(DELTAS_PATH, deltas)
    write_csv_atomic(COVERAGE_PATH, coverage)
    write_csv_atomic(MATCHED_PATH, matched)

    output_hashes = {str(path): sha256(path) for path in (*OUT_PATHS, PRODUCT_MANIFEST)}
    output_rows = {
        str(METRICS_PATH): len(metrics),
        str(DELTAS_PATH): len(deltas),
        str(COVERAGE_PATH): len(coverage),
        str(MATCHED_PATH): len(matched),
    }
    qa = {
        "format": "stoten-m07-monthly-satpm-qa-v1",
        "status": "passed",
        "analysis_plan": str(PLAN),
        "analysis_plan_sha256": plan_sha,
        "server_output_root": str(ROOT),
        "execution": {"pid": os.getpid(), "threads": 1},
        "product": {"version": "V6GL03", "region": "GL", "resolution_degrees": 0.1,
                    "temporal_resolution": "monthly", "months": 24,
                    "first_month": "202301", "last_month": "202412",
                    "total_bytes": int(product_manifest["total_bytes"]),
                    "monthly_support_counts": month_counts},
        "locked_model_design": {"countries": sorted(COUNTRIES), "cells": [f"{m}_{y}" for m, y in CELLS],
                                 "profiles": sorted(PROFILES), "qa_variant": "strict", "estimator_kind": "hgb",
                                 "arm": "full", "target_strategy": "log_smear", "backtransform": "smearing"},
        "target_match_qa": target_qa,
        "key_definition": ["country", "site_id", "year", "month"],
        "comparisons": {"SatPM_V6GL03_monthly": "exact same target/site-month; nearest native grid cell, no interpolation",
                        "CAMS_raw": "exact same target/site-month; sampled status, source month present, finite value and valid spatial weight"},
        "bootstrap": {"unit": "site_id", "replicates": BOOTSTRAPS, "seed": SEED,
                      "minimum_pairs": MIN_PAIRS, "minimum_stations": MIN_SITES,
                      "status_counts": {str(key): int(value) for key, value in pd.Series([r["ci_status"] for r in deltas]).value_counts().items()}},
        "input_sha256": input_hashes,
        "outputs": {"rows": output_rows, "sha256": output_hashes},
        "limitations": [
            "SatPM V6GL03 is an external product comparator, not an independent measurement truth; monitoring-calibration/input overlap remains possible and unquantified.",
            "Metrics are conditional on the accepted OOF predictions and the exact monthly matched support; no model was refit.",
            "The new monthly product comparison is limited to fixed 2023, fixed 2024, and rolling 2024 for USA/CAN. Historical comparisons remain the existing annual summaries.",
            "2025 is not analyzed because the locked source-support window ends in 2024.",
        ],
    }
    write_json_atomic(QA_PATH, qa)
    print(f"COMPLETE metrics={len(metrics)} deltas={len(deltas)} coverage={len(coverage)} matched_rows={len(matched)} qa={QA_PATH}", flush=True)
    return qa


def main() -> int:
    try:
        run()
        return 0
    except Exception as exc:
        failure = {
            "format": "stoten-m07-monthly-satpm-qa-v1",
            "status": "failed",
            "analysis_plan": str(PLAN),
            "server_output_root": str(ROOT),
            "execution": {"pid": os.getpid(), "threads": 1},
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc().splitlines()[-12:],
            "note": "No target, model, or comparator values were silently changed. Inspect the locked inputs and report the discrepancy before any rerun.",
        }
        try:
            write_json_atomic(QA_PATH, failure)
        except Exception:
            pass
        print(f"FAILED {failure['error']}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
