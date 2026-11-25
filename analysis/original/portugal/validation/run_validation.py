#!/usr/bin/env python
"""Run repair-v2 station, buffered-spatial, and year-specific validation.

Examples:
  py -3.12 -X utf8 -B run_validation.py --countries USA,CAN,TH,IN,ES
  py -3.12 -X utf8 -B run_validation.py --feature-override USA=...parquet
  py -3.12 -X utf8 -B run_validation.py --target-override USA=...csv

The USA target override is an exact, one-to-one (site_id, year, month) join.
When supplied, all pre-existing USA targets are first cleared so rows rejected
by QC cannot fall back to the provisional target. Precomputed feature tables
are explicitly marked provisional until corrected targets are supplied.
"""
from __future__ import annotations

import os

# Avoid BLAS/OpenMP oversubscription when running many independent folds.
for _name, _value in {
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}.items():
    os.environ.setdefault(_name, _value)

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from validation_core import (
    ESTIMATOR_CLASS_NAMES, ModelConfig, SCHEMA_VERSION, feature_matrix, fit_model,
    predict_model, schema_for,
)
from support_diagnostics import (
    SUPPORT_DISTANCE_KM, SUPPORT_LOWER_QUANTILE, SUPPORT_UPPER_QUANTILE,
    add_distance_support, feature_groups_for_arm, training_feature_support,
    write_support_outputs,
)
from conformal_sensitivity import (
    CONFORMAL_COVERAGE, FEASIBILITY_MIN_CALIBRATION_STATIONS,
    FEASIBILITY_MIN_PROPER_STATIONS, PROPER_TRAIN_FRACTION,
    feasibility_calibration_station_n, station_split_conformal,
)

ROOT_DEFAULT = Path(r"F:\cand1_20261105")
GEO = ["lat_f", "lon_f", "lpop", "month_sin", "month_cos", "month_sin2", "month_cos2"]
MET = ["blh", "wind10", "tp", "t2m", "rh", "vent"]
AOD = ["aod_used"]
FIRE = ["lgfas", "lfrp", "lfn", "lgfas_r5", "lgfas_r15", "lfrp_r5"]
COUNTRIES = [
    "BRA", "MEX", "USA", "CAN", "AUS", "ES", "PT", "GR", "IT", "CL",
    "ZA", "IN", "PK", "NP", "BD", "TH", "VN", "PH",
]

TABLES = {
    "amer": ["BRA", "MEX", "USA", "CAN"],
    "aus": ["AUS"],
    "asia": ["IN", "PK", "NP", "BD", "TH", "VN", "PH"],
    "eur": ["ES", "PT", "GR", "IT"],
    "chl": ["CL"],
    "idn": ["ID"],
    "zaf": ["ZA"],
}
RELATIVE_TABLES = {
    "amer": Path("work/features/stations_features_all.parquet"),
    "aus": Path("work/features/aus/stations_features_aus.parquet"),
    "asia": Path("work/features/asia/stations_features_asia.parquet"),
    "eur": Path("work/features/eur/stations_features_eur_met.parquet"),
    "chl": Path("work/features/chl/stations_features_chl.parquet"),
    "idn": Path("work/features/idn/stations_features_idn.parquet"),
    "zaf": Path("work/features/zaf/stations_features_zaf.parquet"),
}
MET_PROFILE = {
    "amer": list(MET),
    "aus": [],
    "asia": [],
    "eur": list(MET),
    "chl": [],
    "idn": [],
    "zaf": [],
}
SOURCE_MET = {
    "amer": "ERA5 features available in the source table",
    "aus": "no ERA5 features in the source table",
    "asia": "no ERA5 features in the source table",
    "eur": "ERA5 features available in the source table",
    "chl": "no ERA5 features in the source table",
    "idn": "no ERA5 features in the source table",
    "zaf": "no ERA5 features in the source table",
}
ARMS = ("B", "B+AOD", "B+fire", "full")
MIN_STATIONS_T1_T2 = 5
MIN_T3_TRAIN_STATIONS = 10
MIN_T3_TEST_STATIONS = 3
T3_MIN_TRAIN_ROWS = 100
T3_MIN_TEST_ROWS = 50
SPATIAL_BUFFER_KM = 50.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_support_fingerprint(
    frame: pd.DataFrame, features: list[str], target_col: str = "pm25"
) -> str:
    """Hash the exact ordered rows and values supplied to one fitted fold."""
    columns = list(dict.fromkeys([
        "country", "site_id", "year", "month", "lat", "lon", target_col,
        *features, *(["aod_raw"] if "aod_used" in features else []),
    ]))
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(f"training fingerprint columns are absent: {missing}")
    selected = frame[columns].copy()
    if "_row_id" in frame:
        selected.insert(0, "_source_row_id", pd.to_numeric(frame["_row_id"], errors="raise").to_numpy())
    else:
        selected.insert(0, "_source_row_id", np.arange(len(selected), dtype="int64"))
    payload = selected.to_csv(
        index=False, na_rep="<NA>", float_format="%.17g", lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}".encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_row_key_sha256(frame: pd.DataFrame) -> str:
    """Hash unique station-month keys in a stable canonical order."""
    required = ["country", "site_id", "year", "month"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"row-key hash lacks columns: {missing}")
    keys = [
        (str(country).upper(), str(site_id), int(year), int(month))
        for country, site_id, year, month in frame[required].itertuples(index=False, name=None)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("row-key hash requires unique country/site/year/month keys")
    return _json_sha256(sorted(keys))


def _year_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.year.astype(int).value_counts().sort_index()
    return {str(int(year)): int(count) for year, count in counts.items()}


def outer_fold_split_record(
    *, country: str, cohort_label: str, task: str, fold_id: str,
    buffer_km: float, train: pd.DataFrame, test: pd.DataFrame,
    all_rows: pd.DataFrame, full_features: list[str],
    feature_source_sha256: list[str], target_source_sha256: str | None,
    analysis_row_key_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind one outer split to exact source keys and model training support."""
    train_rows = train["_row_id"].astype(int).tolist()
    test_rows = test["_row_id"].astype(int).tolist()
    is_year_holdout = task.startswith("T3_year_")
    train_sites = sorted(train.site_id.astype(str).unique().tolist())
    test_sites = sorted(test.site_id.astype(str).unique().tolist())
    train_ids = sorted(train_rows) if is_year_holdout else train_sites
    test_ids = sorted(test_rows) if is_year_holdout else test_sites
    id_unit = "row_id" if is_year_holdout else "site_id"
    all_sites = set(all_rows.site_id.astype(str))
    excluded_sites = sorted(all_sites - set(train_sites) - set(test_sites))
    if task.startswith("T1_"):
        split_rule = "GroupKFold grouped by site_id; train and test station IDs are disjoint"
    elif task.startswith("T2_spatial_buffer"):
        split_rule = f"fixed station spatial blocks with training stations at least {buffer_km:g} km from test stations"
    elif task == "T2_spatial":
        split_rule = "fixed KMeans station blocks; station blocks are held out"
    elif is_year_holdout:
        split_rule = f"train rows have year < {fold_id}; test rows have year == {fold_id}"
    else:
        split_rule = "explicit train/test row IDs recorded"
    return {
        "country": country,
        "cohort_label": cohort_label,
        "task": task,
        "fold_id": str(fold_id),
        "split_rule": split_rule,
        "buffer_km": float(buffer_km),
        "train_ids_unit": id_unit,
        "train_ids": train_ids,
        "test_ids_unit": id_unit,
        "test_ids": test_ids,
        "train_ids_sha256": _json_sha256(train_ids),
        "test_ids_sha256": _json_sha256(test_ids),
        "train_row_ids_sha256": _array_sha256(np.sort(np.asarray(train_rows, dtype=np.int64))),
        "test_row_ids_sha256": _array_sha256(np.sort(np.asarray(test_rows, dtype=np.int64))),
        "train_row_key_sha256": canonical_row_key_sha256(train),
        "test_row_key_sha256": canonical_row_key_sha256(test),
        "analysis_row_key_sha256": analysis_row_key_sha256 or canonical_row_key_sha256(all_rows),
        "feature_source_sha256": list(feature_source_sha256),
        "target_source_sha256": target_source_sha256,
        "train_row_count": int(len(train)),
        "test_row_count": int(len(test)),
        "train_station_count": int(len(train_sites)),
        "test_station_count": int(len(test_sites)),
        "train_year_counts": _year_counts(train),
        "test_year_counts": _year_counts(test),
        "buffer_excluded_station_ids": excluded_sites,
        "training_support_fingerprint": training_support_fingerprint(
            train, full_features, target_col="pm25"
        ),
    }


def bind_outer_model(
    record: dict[str, Any], *, bundle: dict[str, Any], model_path: Path,
    train: pd.DataFrame, runtime: dict[str, Any], smear_strategy: str,
) -> dict[str, Any]:
    """Fingerprint the saved full model against its matrix, y, groups and runtime."""
    artifact_sha = sha256_file(model_path)
    matrix = feature_matrix(train, bundle["schema"])
    raw_target = pd.to_numeric(train[bundle["target_col"]], errors="coerce").to_numpy(dtype="float64")
    model_target = np.log(raw_target + 0.1) if bundle["target_strategy"] == "log_smear" else raw_target
    binding = {
        "model_artifact_sha256": artifact_sha,
        "ordered_feature_matrix_sha256": _array_sha256(matrix),
        "feature_schema": bundle["schema"],
        "raw_target_y_sha256": _array_sha256(raw_target),
        "model_target_y_sha256": _array_sha256(model_target),
        "ordered_group_ids_sha256": _json_sha256(train.site_id.astype(str).tolist()),
        "training_support_fingerprint": record["training_support_fingerprint"],
        "training_row_key_sha256": record["train_row_key_sha256"],
        "source_hashes": {
            "features": record["feature_source_sha256"],
            "targets": record["target_source_sha256"],
        },
        "model_config": bundle["config"],
        "runtime": runtime,
        "target_strategy": bundle["target_strategy"],
        "smear_strategy": smear_strategy,
        "smear": bundle["smear"],
        "smear_info": bundle["smear_info"],
        "estimator_kind": bundle["estimator_kind"],
        "estimator_class": bundle["estimator_class"],
    }
    model_record = {
        "path": str(model_path.resolve()),
        "sha256": artifact_sha,
        "estimator_kind": bundle["estimator_kind"],
        "estimator_class": bundle["estimator_class"],
        "target_strategy": bundle["target_strategy"],
        "smear_strategy": smear_strategy,
        "training_support_fingerprint": record["training_support_fingerprint"],
        "binding_sha256": _json_sha256(binding),
        "binding": binding,
    }
    record["full_model"] = model_record
    return model_record


BOOTSTRAP_ENGINES = ("reference", "cluster_stats")
_FAST_BOOTSTRAP_FUNCTION = None
_FAST_BOOTSTRAP_PATH: Path | None = None
_FAST_BOOTSTRAP_SHA256: str | None = None


def _load_fast_bootstrap():
    """Load the opt-in fast implementation from the source tree or run bundle."""
    global _FAST_BOOTSTRAP_FUNCTION, _FAST_BOOTSTRAP_PATH, _FAST_BOOTSTRAP_SHA256
    if _FAST_BOOTSTRAP_FUNCTION is not None:
        if _FAST_BOOTSTRAP_PATH is None or _FAST_BOOTSTRAP_SHA256 is None:
            raise RuntimeError("fast bootstrap cache is incomplete")
        if sha256_file(_FAST_BOOTSTRAP_PATH) != _FAST_BOOTSTRAP_SHA256:
            raise RuntimeError("fast bootstrap helper changed during this run")
        return _FAST_BOOTSTRAP_FUNCTION, _FAST_BOOTSTRAP_PATH, _FAST_BOOTSTRAP_SHA256

    here = Path(__file__).resolve().parent
    candidates = (here / "fast_cluster_bootstrap.py", here.parent / "fast_cluster_bootstrap.py")
    helper_path = next((path for path in candidates if path.is_file()), None)
    if helper_path is None:
        raise FileNotFoundError("cluster_stats bootstrap requires fast_cluster_bootstrap.py in the run bundle")
    spec = importlib.util.spec_from_file_location("repair_v2_fast_cluster_bootstrap", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fast bootstrap helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "paired_bootstrap", None)
    if not callable(function):
        raise ImportError("fast bootstrap helper has no callable paired_bootstrap")
    helper_hash = sha256_file(helper_path)
    _FAST_BOOTSTRAP_FUNCTION = function
    _FAST_BOOTSTRAP_PATH = helper_path.resolve()
    _FAST_BOOTSTRAP_SHA256 = helper_hash
    return function, _FAST_BOOTSTRAP_PATH, helper_hash


def bootstrap_provenance(engine: str) -> dict[str, Any]:
    if engine not in BOOTSTRAP_ENGINES:
        raise ValueError(f"unsupported bootstrap engine: {engine}")
    if engine == "reference":
        return {"engine": engine, "helper_path": None, "helper_sha256": None}
    _, helper_path, helper_hash = _load_fast_bootstrap()
    return {
        "engine": engine,
        "helper_path": str(helper_path),
        "helper_sha256": helper_hash,
    }


def runtime_provenance(script_path: Path | str, bootstrap_engine: str = "reference") -> dict[str, Any]:
    """Record executable and model-code versions for a validation run."""
    here = Path(__file__).resolve().parent
    files = {
        "runner": Path(script_path).resolve(),
        "run_validation_helpers": here / "run_validation.py",
        "validation_core": here / "validation_core.py",
        "support_diagnostics": here / "support_diagnostics.py",
        "conformal_sensitivity": here / "conformal_sensitivity.py",
    }
    bootstrap = bootstrap_provenance(bootstrap_engine)
    if bootstrap["helper_path"] is not None:
        files["fast_cluster_bootstrap"] = Path(bootstrap["helper_path"])
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "bootstrap": bootstrap,
        "code_sha256": {name: sha256_file(path) for name, path in files.items()},
    }


def load_input_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "manifest": json.loads(resolved.read_text(encoding="utf-8")),
    }


def parse_overrides(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"override must be COUNTRY=PATH, got {item!r}")
        country, path = item.split("=", 1)
        country = country.strip().upper()
        if not country or not path:
            raise ValueError(f"invalid override {item!r}")
        result[country] = Path(path)
    return result


def parse_feature_overrides(values: list[str] | None) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"feature override must be COUNTRY=PATH, got {item!r}")
        country, path = item.split("=", 1)
        country = country.strip().upper()
        if not country or not path:
            raise ValueError(f"invalid feature override {item!r}")
        result.setdefault(country, []).append(Path(path))
    return result


def load_input_tables(
    root: Path,
    feature_overrides: dict[str, list[Path]],
    requested_countries: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[Path]]]:
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, list[Path]] = {}
    source_paths = {source: root / rel for source, rel in RELATIVE_TABLES.items()}
    source_countries = {c: source for source, values in TABLES.items() for c in values}
    for source, path in source_paths.items():
        selected = [
            c for c in TABLES[source]
            if c not in feature_overrides
            and (requested_countries is None or c in requested_countries)
        ]
        if selected:
            if not path.exists():
                raise FileNotFoundError(f"feature table for {source} is absent: {path}")
            data = pd.read_parquet(path)
            if "country" not in data:
                raise ValueError(f"{path} lacks required country column")
            for country in selected:
                if country in set(data.country.astype(str)):
                    frames[country] = data.loc[data.country.astype(str) == country].copy()
                    paths[country] = [path]
    for country, override_paths in feature_overrides.items():
        selected_parts = []
        for path in override_paths:
            if not path.exists():
                raise FileNotFoundError(f"feature override is absent: {path}")
            data = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            if "country" not in data:
                data["country"] = country
            part = data.loc[data.country.astype(str).str.upper() == country].copy()
            if part.empty:
                raise ValueError(f"feature override {path} has no rows for {country}")
            selected_parts.append(part)
        frames[country] = pd.concat(selected_parts, ignore_index=True, sort=False)
        paths[country] = override_paths
    unknown = sorted(set(feature_overrides) - set(source_countries) - set(COUNTRIES))
    if unknown:
        raise ValueError(f"unsupported feature override country codes: {unknown}")
    return frames, paths


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["site_id", "year", "month"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"station-month keys missing: {missing}")
    out = frame[required].copy()
    out["site_id"] = out.site_id.astype(str)
    out["year"] = pd.to_numeric(out.year, errors="raise").astype(int)
    out["month"] = pd.to_numeric(out.month, errors="raise").astype(int)
    return out


def replace_targets(
    frame: pd.DataFrame,
    target_path: Path,
    *,
    country: str,
    target_column: str = "pm25",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace all country targets by an exact keyed QC table; no old-target fallback."""
    target = pd.read_csv(target_path) if target_path.suffix.lower() != ".parquet" else pd.read_parquet(target_path)
    if "country" in target:
        target = target.loc[target.country.astype(str).str.upper() == country].copy()
    if target.empty:
        raise ValueError(f"target override contains no rows for {country}")
    if target_column not in target:
        raise ValueError(f"target override lacks target column {target_column!r}")
    keys = _key_frame(target)
    if keys.duplicated().any():
        sample = keys.loc[keys.duplicated(keep=False)].head(5).to_dict("records")
        raise ValueError(f"target override has duplicate keys, examples: {sample}")
    values = pd.to_numeric(target[target_column], errors="coerce").to_numpy(dtype="float64")
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("target override has infinite or negative target values")
    source = frame.copy()
    source_keys = _key_frame(source)
    if source_keys.duplicated().any():
        raise ValueError(f"feature table has duplicate station-months for {country}")
    source_years = set(source_keys.year.astype(int))
    in_scope = keys.year.isin(source_years).to_numpy()
    target_scope = keys.loc[in_scope].copy()
    value_scope = values[in_scope]
    source["_repair_key"] = list(map(tuple, source_keys.to_numpy()))
    target["_repair_key"] = list(map(tuple, keys.to_numpy()))
    target_scope["_repair_key"] = list(map(tuple, target_scope.to_numpy()))
    lookup = dict(zip(target_scope["_repair_key"], value_scope))
    unmatched = sorted(set(lookup) - set(source["_repair_key"]))
    matched_count = int(len(set(lookup) & set(source["_repair_key"])))
    # Deliberately clear every old country target before applying the QC result.
    source["pm25"] = [lookup.get(k, np.nan) for k in source["_repair_key"]]
    source.drop(columns="_repair_key", inplace=True)
    applied = int(np.isfinite(source.pm25.to_numpy(dtype="float64")).sum())
    return source, {
        "status": "epa_qc_replaced",
        "path": str(target_path.resolve()),
        "sha256": sha256_file(target_path),
        "target_column": target_column,
        "keys_in_override": int(len(target)),
        "keys_in_feature_years": int(in_scope.sum()),
        "keys_outside_feature_years": int((~in_scope).sum()),
        "keys_matched_to_features": matched_count,
        "keys_without_feature_row_in_scope": int(len(unmatched)),
        "finite_targets_applied": applied,
        "feature_rows_rejected_or_missing_from_qc": int(len(source) - applied),
    }


def collapse_exact_duplicates(frame: pd.DataFrame, model_features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse exact repeated station-month rows, reject conflicting duplicates."""
    keys = ["country", "site_id", "year", "month"]
    missing = [c for c in keys + ["pm25"] + model_features if c not in frame]
    if missing:
        raise ValueError(f"duplicate audit lacks required columns: {missing}")
    d = frame.copy()
    key_data = d[keys]
    duplicate_mask = key_data.duplicated(keep=False)
    if not duplicate_mask.any():
        return d, {"duplicate_keys": 0, "exact_rows_collapsed": 0, "conflicting_keys": 0}
    check = keys + ["pm25"] + model_features
    dup = d.loc[duplicate_mask, check]
    conflicts = []
    for key, group in dup.groupby(keys, dropna=False, sort=False):
        if len(group.drop_duplicates().index) > 1:
            conflicts.append(key if isinstance(key, tuple) else (key,))
    if conflicts:
        raise ValueError(
            f"conflicting duplicate station-month targets/features ({len(conflicts)} keys); "
            f"examples={conflicts[:5]}"
        )
    keep_cols = list(d.columns)
    before = len(d)
    d = d.drop_duplicates(subset=keys, keep="first").copy()
    return d[keep_cols], {
        "duplicate_keys": int(len(conflicts)),
        "exact_rows_collapsed": int(before - len(d)),
        "conflicting_keys": 0,
    }


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "r2": float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "bias": float(np.mean(p - y)),
    }


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    a1, a2 = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lat2, dtype=float))
    dlat = a2 - a1
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    x = np.sin(dlat / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(x)))


def _site_table(d: pd.DataFrame) -> pd.DataFrame:
    return d.drop_duplicates("site_id")[["site_id", "lat", "lon"]].copy()


def _nearest_distances(test: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    train_sites = _site_table(train)
    if train_sites.empty:
        return np.full(len(test), np.nan)
    test_sites = test[["site_id", "lat", "lon"]].drop_duplicates("site_id")
    tlats = train_sites.lat.to_numpy(float)
    tlons = train_sites.lon.to_numpy(float)
    distances: dict[str, float] = {}
    for row in test_sites.itertuples(index=False):
        vals = _haversine(tlats, tlons, row.lat, row.lon)
        distances[str(row.site_id)] = float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan
    return test.site_id.astype(str).map(distances).to_numpy(dtype=float)


def _station_blocks(d: pd.DataFrame, n_splits: int = 5) -> dict[str, Any]:
    sites = _site_table(d)
    if len(sites) < n_splits:
        raise ValueError(f"needs at least {n_splits} unique stations, found {len(sites)}")
    mean_lat = float(sites.lat.mean())
    coords = np.column_stack([
        sites.lon.to_numpy(float) * math.cos(math.radians(mean_lat)) * 111.32,
        sites.lat.to_numpy(float) * 111.32,
    ])
    km = KMeans(n_clusters=n_splits, random_state=0, n_init=20)
    sites["block"] = km.fit_predict(coords)
    mapping = dict(zip(sites.site_id.astype(str), sites.block.astype(int)))
    return mapping


def task_splits(d: pd.DataFrame, *, buffer_km: float = 0.0):
    """Yield pre-specified outer splits with group/spatial/temporal independence."""
    n_sites = d.site_id.nunique()
    if n_sites >= MIN_STATIONS_T1_T2:
        cv = GroupKFold(n_splits=5)
        for fold, (tr, te) in enumerate(cv.split(d, groups=d.site_id.astype(str))):
            yield "T1_station", str(fold), np.asarray(tr), np.asarray(te), 0.0
        sites = _site_table(d)
        if sites[["lat", "lon"]].drop_duplicates().shape[0] >= 5:
            mapping = _station_blocks(d, 5)
            blocks = d.site_id.astype(str).map(mapping).to_numpy()
            for fold in range(5):
                te = np.flatnonzero(blocks == fold)
                tr = np.flatnonzero(blocks != fold)
                yield "T2_spatial", str(fold), tr, te, 0.0
                if buffer_km > 0 and len(te):
                    test_sites = _site_table(d.iloc[te])
                    train_sites = _site_table(d.iloc[tr])
                    keep = np.ones(len(train_sites), dtype=bool)
                    for row in test_sites.itertuples(index=False):
                        dist = _haversine(train_sites.lat, train_sites.lon, row.lat, row.lon)
                        keep &= dist >= buffer_km
                    kept_sites = set(train_sites.loc[keep, "site_id"].astype(str))
                    tr_buf = np.flatnonzero(
                        np.isin(d.site_id.astype(str).to_numpy(), list(kept_sites))
                    )
                    yield f"T2_spatial_buffer{int(buffer_km)}km", str(fold), tr_buf, te, float(buffer_km)
        else:
            yield "T2_spatial", "ineligible", np.array([], dtype=int), np.array([], dtype=int), 0.0
            if buffer_km > 0:
                yield f"T2_spatial_buffer{int(buffer_km)}km", "ineligible", np.array([], dtype=int), np.array([], dtype=int), float(buffer_km)
    elif n_sites >= 3:
        cv = GroupKFold(n_splits=n_sites)
        for fold, (tr, te) in enumerate(cv.split(d, groups=d.site_id.astype(str))):
            yield "T1_station_exploratory", str(fold), np.asarray(tr), np.asarray(te), 0.0
        yield "T2_spatial", "ineligible", np.array([], dtype=int), np.array([], dtype=int), 0.0
        if buffer_km > 0:
            yield f"T2_spatial_buffer{int(buffer_km)}km", "ineligible", np.array([], dtype=int), np.array([], dtype=int), float(buffer_km)
    else:
        yield "T1_station", "ineligible", np.array([], dtype=int), np.array([], dtype=int), 0.0
        yield "T2_spatial", "ineligible", np.array([], dtype=int), np.array([], dtype=int), 0.0
        if buffer_km > 0:
            yield f"T2_spatial_buffer{int(buffer_km)}km", "ineligible", np.array([], dtype=int), np.array([], dtype=int), float(buffer_km)
    years = sorted(int(x) for x in d.year.unique() if int(x) >= 2022)
    if not years:
        yield "T3_year", "no_year_ge_2022", np.array([], dtype=int), np.array([], dtype=int), 0.0
    for year in years:
        tr = np.flatnonzero(d.year.to_numpy(int) < year)
        te = np.flatnonzero(d.year.to_numpy(int) == year)
        yield "T3_year", str(year), tr, te, 0.0


def _screen_baseline(train: pd.DataFrame, base_features: list[str],
                     config: ModelConfig, target_col: str) -> dict[str, Any]:
    """Inner grouped-CV baseline skill using only one outer-training fold."""
    n_groups = train.site_id.nunique()
    if n_groups < 2:
        return {"inner_r2": np.nan, "inner_n": 0, "inner_reason": "fewer_than_2_train_stations"}
    n_splits = min(5, n_groups)
    pred = np.full(len(train), np.nan)
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(train, groups=train.site_id.astype(str)):
        bundle = fit_model(
            train.iloc[tr], base_features, target_col=target_col,
            config=config, smear_strategy="in_sample",
        )
        pred[te] = predict_model(bundle, train.iloc[te], backtransform="smearing")
    y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(float)
    m = _metric(y, pred)
    return {
        "inner_r2": m["r2"], "inner_rmse": m["rmse"], "inner_mae": m["mae"],
        "inner_bias": m["bias"], "inner_n": m["n"], "inner_splits": n_splits,
        "inner_smear": "in_sample_train_only",
    }


def _paired_bootstrap(group: np.ndarray, y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                      replicates: int, seed: int, engine: str = "reference") -> dict[str, Any]:
    if engine == "cluster_stats":
        function, _, _ = _load_fast_bootstrap()
        return function(group, y, pa, pb, replicates, seed)
    if engine != "reference":
        raise ValueError(f"unsupported bootstrap engine: {engine}")
    groups = np.asarray(group).astype(str)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {k: [] for k in ("rmse", "mae", "bias", "r2")}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        ma, mb = _metric(y[idx], pa[idx]), _metric(y[idx], pb[idx])
        for metric in deltas:
            if np.isfinite(ma[metric]) and np.isfinite(mb[metric]):
                deltas[metric].append(float(ma[metric] - mb[metric]))
    out: dict[str, Any] = {
        "bootstrap_replicates": replicates, "bootstrap_unit": "station",
        "bootstrap_independent_units": int(len(unique)),
    }
    for metric, values in deltas.items():
        out[f"delta_{metric}_lo"] = float(np.percentile(values, 2.5)) if values else np.nan
        out[f"delta_{metric}_hi"] = float(np.percentile(values, 97.5)) if values else np.nan
    return out


def _uncertainty_scope(task: str, outer_folds: int) -> dict[str, Any]:
    if task.startswith("T2_spatial"):
        return {
            "station_interval_scope": "conditional on observed spatial folds; resamples test stations",
            "spatial_block_interval_status": "unsupported_only_5_spatial_blocks",
            "n_spatial_test_blocks": int(outer_folds),
            "temporal_interval_status": "not_applicable",
        }
    if task.startswith("T3_year_"):
        return {
            "station_interval_scope": "conditional on one held-out year; resamples test stations",
            "spatial_block_interval_status": "not_applicable",
            "n_spatial_test_blocks": 0,
            "temporal_interval_status": "unsupported_single_heldout_year",
            "n_temporal_test_blocks": 1,
        }
    return {
        "station_interval_scope": "station-cluster bootstrap conditional on observed T1 folds",
        "spatial_block_interval_status": "not_applicable",
        "n_spatial_test_blocks": 0,
        "temporal_interval_status": "not_applicable",
    }


def _duplicate_audit(d: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["country", "site_id", "year", "month"]
    mask = d.duplicated(keys, keep=False)
    if not mask.any():
        return d, {"duplicate_keys": 0, "exact_rows_collapsed": 0, "conflicting_keys": 0}
    conflicts = []
    compared = [c for c in d.columns if c != "_row_id"]
    for key, group in d.loc[mask].groupby(keys, dropna=False, sort=False):
        if len(group[compared].drop_duplicates()) > 1:
            conflicts.append(key)
    if conflicts:
        raise ValueError(
            f"conflicting station-month duplicates ({len(set(conflicts))} keys); "
            f"examples={list(dict.fromkeys(conflicts))[:5]}"
        )
    before = len(d)
    duplicate_keys = int(d.loc[mask, keys].drop_duplicates().shape[0])
    return d.drop_duplicates(keys, keep="first").copy(), {
        "duplicate_keys": duplicate_keys,
        "exact_rows_collapsed": int(before - int(d.drop_duplicates(keys).shape[0])),
        "conflicting_keys": 0,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=True), encoding="utf-8")


def run(args) -> int:
    if args.bootstrap_engine == "cluster_stats":
        _load_fast_bootstrap()
    root = Path(args.source_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    feature_overrides = parse_feature_overrides(args.feature_override)
    target_overrides = parse_overrides(args.target_override)
    target_status_overrides = parse_overrides(args.target_status)
    input_provenance_overrides = parse_overrides(args.input_provenance)
    countries = [c.strip().upper() for c in args.countries.split(",")] if args.countries else COUNTRIES
    unknown = sorted(set(countries) - set(COUNTRIES))
    if unknown:
        raise ValueError(f"unsupported country codes: {unknown}")
    frames, source_paths = load_input_tables(root, feature_overrides, countries)
    target_status: dict[str, dict[str, Any]] = {}
    for country in countries:
        if country not in frames:
            target_status[country] = {"status": "ineligible", "reason": "no_precomputed_feature_table"}
            continue
        if country in target_overrides:
            frames[country], target_status[country] = replace_targets(
                frames[country], target_overrides[country], country=country,
                target_column=args.override_target_column,
            )
        else:
            target_status[country] = {
                "status": (
                    "feature_override_target_unverified" if country in feature_overrides
                    else "provisional_unqc_station_target" if country == "USA"
                    else "precomputed_station_target"
                ),
                "source": [str(p.resolve()) for p in source_paths.get(country, [])],
            }
        if country in target_status_overrides:
            target_status[country]["status"] = str(target_status_overrides[country])

    config = ModelConfig(
        max_iter=args.max_iter, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=25, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=0,
        estimator_kind=args.estimator_kind, ridge_alpha=1.0,
    )
    runtime_record = runtime_provenance(Path(__file__).resolve(), args.bootstrap_engine)
    metrics_rows, delta_rows, fold_delta_rows, oof_rows, fold_rows, screen_rows, eligibility = [], [], [], [], [], [], []
    training_range_frames, support_row_frames = [], []
    conformal_interval_frames, conformal_summary_rows = [], []
    conformal_assignment_frames, conformal_score_frames = [], []
    conformal_feasibility_interval_frames, conformal_feasibility_summary_rows = [], []
    conformal_feasibility_assignment_frames, conformal_feasibility_score_frames = [], []
    conformal_model_manifest: dict[str, Any] = {}
    conformal_feasibility_model_manifest: dict[str, Any] = {}
    schema_manifest: dict[str, Any] = {}
    duplicate_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}
    outer_fold_split_records: list[dict[str, Any]] = []
    outer_fold_assignment_frames: list[pd.DataFrame] = []
    split_record_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for country in countries:
        if country not in frames:
            eligibility.append({"country": country, "task": "all", "status": "ineligible",
                                "reason": "no_precomputed_feature_table"})
            continue
        source = next(s for s, values in TABLES.items() if country in values)
        d = frames[country].copy()
        d["country"] = country
        cohort_label = target_status.get(country, {}).get("status", "precomputed_station_target")
        met_features = [] if args.no_met else list(MET_PROFILE[source])
        base = GEO + met_features
        arm_features = {
            "B": base,
            "B+AOD": base + AOD,
            "B+fire": base + FIRE,
            "full": base + AOD + FIRE,
        }
        required = sorted(set(itertools.chain.from_iterable(arm_features.values())) | {
            "country", "site_id", "lat", "lon", "year", "month", "pm25", "aod_raw"
        })
        absent = [c for c in required if c not in d.columns]
        if absent:
            reason = f"missing_fixed_schema_columns:{','.join(absent)}"
            for task in ("T1_station", "T2_spatial", f"T2_spatial_buffer{int(SPATIAL_BUFFER_KM)}km", "T3_year"):
                eligibility.append({"country": country, "task": task, "status": "ineligible", "reason": reason})
            continue
        d["year"] = pd.to_numeric(d.year, errors="coerce")
        d["month"] = pd.to_numeric(d.month, errors="coerce")
        d["pm25"] = pd.to_numeric(d.pm25, errors="coerce")
        coord_ok = np.isfinite(pd.to_numeric(d.lat, errors="coerce")) & np.isfinite(pd.to_numeric(d.lon, errors="coerce"))
        target_ok = np.isfinite(d.pm25) & (d.pm25 >= 0)
        d = d.loc[coord_ok & target_ok].copy()
        if d.empty:
            eligibility.append({"country": country, "task": "all", "status": "ineligible",
                                "reason": "no_finite_nonnegative_targets_with_coordinates"})
            continue
        d["year"] = d.year.astype(int)
        d["month"] = d.month.astype(int)
        d["site_id"] = d.site_id.astype(str)
        d.reset_index(drop=True, inplace=True)
        d["_row_id"] = np.arange(len(d), dtype=np.int64)
        print(f"BEGIN country={country} rows={len(d)} stations={d.site_id.nunique()} "
              f"cohort={cohort_label}", flush=True)
        try:
            d, dup = _duplicate_audit(d, sorted(set(itertools.chain.from_iterable(arm_features.values()))))
        except ValueError as exc:
            eligibility.append({"country": country, "task": "all", "status": "provisional_ineligible",
                                "reason": str(exc)})
            duplicate_rows.append({"country": country, "status": "conflicting_duplicates", "detail": str(exc)})
            continue
        duplicate_rows.append({"country": country, **dup})
        d.reset_index(drop=True, inplace=True)
        d["_row_id"] = np.arange(len(d), dtype=np.int64)
        for arm, features in arm_features.items():
            # AOD missingness is based on the raw, pre-fill retrieval variable.
            schema = schema_for(features, {"aod_used": "aod_raw"} if "aod_used" in features else None)
            schema_manifest[f"{country}:{arm}"] = {
                **schema, "source": source, "met_features": met_features,
                "met_availability": SOURCE_MET[source],
            }
        feature_paths = source_paths[country]
        feature_hashes = [sha256_file(p) for p in feature_paths]
        target_source_hash = target_status.get(country, {}).get("sha256")
        analysis_row_key_hash = canonical_row_key_sha256(d)
        source_manifest[country] = {
            "feature_tables": [str(p.resolve()) for p in feature_paths],
            "feature_sha256": feature_hashes,
            "feature_bytes": [int(p.stat().st_size) for p in feature_paths],
            "feature_rows_source": int(len(frames[country])),
            "analysis_rows": int(len(d)),
            "analysis_row_key_sha256": analysis_row_key_hash,
            "target_source_sha256": target_source_hash,
            "target_status": target_status.get(country, {}),
            "cohort_label": cohort_label,
            "country_met_profile": met_features,
            "country_met_profile_name": "no_met" if not met_features else "with_met",
            "country_met_profile_selection": "no_met_override" if args.no_met else "source_declared",
            "duplicate_audit": dup,
        }
        if country in input_provenance_overrides:
            source_manifest[country]["feature_input_provenance"] = load_input_provenance(
                input_provenance_overrides[country]
            )

        # Pre-build all splits once; these indices are shared by every arm and inverse.
        splits = list(task_splits(d, buffer_km=SPATIAL_BUFFER_KM))
        valid_task_folds: dict[str, list[tuple[str, np.ndarray, np.ndarray, float]]] = {}
        for task, fold_id, train_idx, test_idx, buffer in splits:
            if fold_id == "ineligible":
                eligibility.append({"country": country, "task": task, "fold_id": fold_id,
                                    "status": "ineligible", "reason": f"requires>={MIN_STATIONS_T1_T2}_stations"})
                continue
            if fold_id == "no_year_ge_2022":
                eligibility.append({"country": country, "task": task, "fold_id": fold_id,
                                    "status": "undefined", "reason": "no held-out year >=2022 in table"})
                continue
            task_key = f"T3_year_{fold_id}" if task == "T3_year" else task
            is_t3 = task == "T3_year"
            tr, te = d.iloc[train_idx], d.iloc[test_idx]
            min_train = T3_MIN_TRAIN_ROWS if is_t3 else 100
            min_test = T3_MIN_TEST_ROWS if is_t3 else 1
            min_train_sites = (
                MIN_T3_TRAIN_STATIONS if is_t3
                else 2 if task_key == "T1_station_exploratory"
                else MIN_STATIONS_T1_T2 - 1
            )
            min_test_sites = MIN_T3_TEST_STATIONS if is_t3 else 1
            reason = ""
            if len(tr) < min_train:
                reason = f"train_rows<{min_train}"
            elif len(te) < min_test:
                reason = f"test_rows<{min_test}"
            elif tr.site_id.nunique() < min_train_sites:
                reason = f"train_stations<{min_train_sites}"
            elif te.site_id.nunique() < min_test_sites:
                reason = f"test_stations<{min_test_sites}"
            if reason:
                eligibility.append({"country": country, "task": task_key, "fold_id": fold_id,
                                    "status": "ineligible", "reason": reason})
                continue
            valid_task_folds.setdefault(task_key, []).append((fold_id, train_idx, test_idx, buffer))
            eligibility.append({"country": country, "task": task_key, "fold_id": fold_id,
                                "status": "eligible", "reason": ""})
        if not valid_task_folds:
            continue

        for task, fold_specs in valid_task_folds.items():
            all_task_rows = []
            for fold_id, train_idx, test_idx, buffer in fold_specs:
                tr = d.iloc[train_idx].copy()
                te = d.iloc[test_idx].copy()
                split_record = outer_fold_split_record(
                    country=country, cohort_label=cohort_label, task=task,
                    fold_id=str(fold_id), buffer_km=buffer, train=tr, test=te,
                    all_rows=d, full_features=arm_features["full"],
                    feature_source_sha256=feature_hashes,
                    target_source_sha256=target_source_hash,
                    analysis_row_key_sha256=analysis_row_key_hash,
                )
                outer_fold_split_records.append(split_record)
                split_record_by_key[(country, task, str(fold_id))] = split_record
                assignment = te[["country", "site_id", "year", "month", "_row_id"]].copy()
                assignment.rename(columns={"_row_id": "row_id"}, inplace=True)
                assignment["cohort_label"] = cohort_label
                assignment["task"] = task
                assignment["fold_id"] = str(fold_id)
                assignment["role"] = "test"
                outer_fold_assignment_frames.append(assignment)
                print(f"FOLD {country} {task} fold={fold_id} train={len(tr)} "
                      f"test={len(te)} train_stations={tr.site_id.nunique()} "
                      f"test_stations={te.site_id.nunique()}", flush=True)
                # Screening is explicitly run inside this outer-training fold.
                screen = _screen_baseline(tr, base, config, "pm25")
                screen_row = {
                    "country": country, "cohort_label": cohort_label,
                    "task": task, "fold_id": fold_id,
                    "n_train": int(len(tr)), "n_train_stations": int(tr.site_id.nunique()),
                    "n_test": int(len(te)), "n_test_stations": int(te.site_id.nunique()),
                    **screen,
                }
                screen_rows.append(screen_row)
                nearest = _nearest_distances(te, tr)
                fold_summary = {
                    "country": country, "cohort_label": cohort_label,
                    "task": task, "fold_id": fold_id,
                    "buffer_km": buffer, "n_train": int(len(tr)), "n_test": int(len(te)),
                    "n_train_stations": int(tr.site_id.nunique()),
                    "n_test_stations": int(te.site_id.nunique()),
                    "train_years": ",".join(map(str, sorted(tr.year.unique()))),
                    "test_years": ",".join(map(str, sorted(te.year.unique()))),
                    "test_year": int(te.year.iloc[0]) if te.year.nunique() == 1 else np.nan,
                    "distance_km_min": float(np.nanmin(nearest)),
                    "distance_km_median": float(np.nanmedian(nearest)),
                    "distance_km_p90": float(np.nanpercentile(nearest, 90)),
                    "test_within_50km_train_pct": float(np.mean(nearest < 50.0)),
                    "aod_raw_missing_pct": float(te.aod_raw.isna().mean()),
                    "aod_used_missing_pct": float(te.aod_used.isna().mean()),
                    "screen_baseline_inner_r2": screen.get("inner_r2", np.nan),
                }
                fold_rows.append(fold_summary)
                for arm, features in arm_features.items():
                    feature_groups = feature_groups_for_arm(
                        features, geography=GEO, meteorology=MET,
                        aod=AOD, fire=FIRE,
                    )
                    support_ranges, support_flags = training_feature_support(
                        tr, te, features, feature_groups
                    )
                    support_ranges["country"] = country
                    support_ranges["cohort_label"] = cohort_label
                    support_ranges["task"] = task
                    support_ranges["fold_id"] = str(fold_id)
                    support_ranges["arm"] = arm
                    support_flags["country"] = country
                    support_flags["cohort_label"] = cohort_label
                    support_flags["task"] = task
                    support_flags["fold_id"] = str(fold_id)
                    support_flags["arm"] = arm
                    site_by_row = dict(zip(te["_row_id"].astype(int), te.site_id.astype(str)))
                    support_flags["site_id"] = support_flags.row_id.map(site_by_row)
                    support_flags = add_distance_support(
                        support_flags, nearest, te["_row_id"].to_numpy(dtype="int64")
                    )
                    training_range_frames.append(support_ranges)
                    support_row_frames.append(support_flags)
                    fit_specs: list[tuple[str, str]] = []
                    if "log_smear" in args.target_strategies:
                        if arm in args.crossfit_smear_arms and "crossfit_station" in args.smear_strategies:
                            fit_specs.append(("log_smear", "crossfit_station"))
                        if arm in args.in_sample_smear_arms and "in_sample" in args.smear_strategies:
                            fit_specs.append(("log_smear", "in_sample"))
                    if "original_scale" in args.target_strategies and arm in args.original_scale_arms:
                        fit_specs.append(("original_scale", "not_applicable"))
                    for target_strategy, smear_strategy in fit_specs:
                        bundle = fit_model(
                            tr, features, config=config,
                            smear_strategy=("in_sample" if smear_strategy == "not_applicable" else smear_strategy),
                            smear_splits=args.smear_splits,
                            target_strategy=target_strategy,
                        )
                        if arm == "full" and target_strategy == "log_smear" and smear_strategy == "crossfit_station":
                            fold_model_dir = out / "models" / "folds" / country
                            fold_model_dir.mkdir(parents=True, exist_ok=True)
                            fold_model_path = fold_model_dir / f"{task}_{fold_id}_{arm}_{smear_strategy}.joblib"
                            joblib.dump(bundle, fold_model_path, compress=3)
                            split_record = split_record_by_key[(country, task, str(fold_id))]
                            bind_outer_model(
                                split_record, bundle=bundle, model_path=fold_model_path,
                                train=tr, runtime=runtime_record, smear_strategy=smear_strategy,
                            )
                            source_manifest[country].setdefault("outer_fold_models", []).append({
                                "task": task, "fold_id": fold_id,
                                "path": str(fold_model_path.resolve()),
                                "sha256": sha256_file(fold_model_path),
                                "schema": bundle["schema"],
                                "config": bundle["config"],
                                "estimator_kind": bundle["estimator_kind"],
                                "estimator_class": bundle["estimator_class"],
                                "target_strategy": target_strategy,
                                "smear_strategy": smear_strategy,
                                "smear": bundle["smear"],
                                "smear_info": bundle["smear_info"],
                                "fit_n": bundle["fit_n"],
                                "fit_stations": bundle["fit_stations"],
                                "training_support_fingerprint": split_record["training_support_fingerprint"],
                                "binding_sha256": split_record["full_model"]["binding_sha256"],
                            })
                        if target_strategy == "log_smear":
                            predictions = [
                                ("smearing", predict_model(bundle, te, backtransform="smearing")),
                                ("original", predict_model(bundle, te, backtransform="original")),
                            ]
                        else:
                            predictions = [
                                ("model_native", predict_model(bundle, te, backtransform="model_native")),
                            ]
                        for backtransform, prediction in predictions:
                            for pos, (_, row) in enumerate(te.iterrows()):
                                rec = {
                                    "country": country, "task": task, "fold_id": fold_id,
                                    "cohort_label": cohort_label,
                                    "arm": arm, "target_strategy": target_strategy,
                                    "smear_strategy": smear_strategy,
                                    "backtransform": backtransform, "row_id": int(row["_row_id"]),
                                    "site_id": str(row.site_id), "year": int(row.year), "month": int(row.month),
                                    "obs": float(row.pm25), "pred": float(prediction[pos]),
                                    "nearest_train_station_km": float(nearest[pos]),
                                    "n_train": int(len(tr)), "n_test": int(len(te)),
                                    "n_train_stations": int(tr.site_id.nunique()),
                                    "n_test_stations": int(te.site_id.nunique()),
                                    "test_year": int(te.year.iloc[0]) if te.year.nunique() == 1 else np.nan,
                                    "smear": float(bundle["smear"]),
                                    "smear_method": bundle["smear_info"]["method"],
                                    "target_status": target_status.get(country, {}).get("status", "precomputed_station_target"),
                                }
                                all_task_rows.append(rec)
                                oof_rows.append(rec)
                conformal = station_split_conformal(
                    tr, te, arm_features["full"], config=config,
                    seed_parts=(country, task, str(fold_id)),
                    target_col="pm25", group_col="site_id",
                    smear_splits=args.smear_splits,
                    coverage=CONFORMAL_COVERAGE,
                    proper_fraction=PROPER_TRAIN_FRACTION,
                )
                conformal_context = {
                    "country": country, "cohort_label": cohort_label,
                    "task": task, "fold_id": str(fold_id), "arm": "full",
                }
                for table_name in ("intervals", "station_assignments", "calibration_scores"):
                    conformal[table_name]["country"] = country
                    conformal[table_name]["cohort_label"] = cohort_label
                    conformal[table_name]["task"] = task
                    conformal[table_name]["fold_id"] = str(fold_id)
                    conformal[table_name]["arm"] = "full"
                conformal["summary"].update(conformal_context)
                conformal["summary"]["split_seed"] = conformal["seed"]
                conformal_interval_frames.append(conformal["intervals"])
                conformal_summary_rows.append(conformal["summary"])
                conformal_assignment_frames.append(conformal["station_assignments"])
                conformal_score_frames.append(conformal["calibration_scores"])
                conformal_model_dir = out / "models" / "conformal" / country
                conformal_model_dir.mkdir(parents=True, exist_ok=True)
                conformal_model_path = conformal_model_dir / f"{task}_{fold_id}_full_split_conformal.joblib"
                joblib.dump(conformal["bundle"], conformal_model_path, compress=3)
                conformal_model_manifest[f"{task}:{fold_id}"] = {
                    **conformal_context,
                    "path": str(conformal_model_path.resolve()),
                    "sha256": sha256_file(conformal_model_path),
                    "q90": conformal["summary"]["q90"],
                    "calibration_station_n": conformal["summary"]["calibration_station_n"],
                    "proper_train_station_n": conformal["summary"]["proper_train_station_n"],
                    "fit_n": conformal["bundle"]["fit_n"],
                }

                # Keep the finite-rank feasibility allocation in its own
                # sensitivity outputs. The primary 80/20 split above remains
                # unchanged and keeps its own intervals, summaries, and model.
                outer_train_station_n = int(tr.site_id.astype(str).nunique())
                feasibility_calibration_n = feasibility_calibration_station_n(
                    outer_train_station_n
                )
                feasibility_key = f"{task}:{fold_id}"
                feasibility_record = {
                    **conformal_context,
                    "split_label": "finite_q90_feasibility_min9_calibration",
                    "outer_train_station_n": outer_train_station_n,
                    "minimum_calibration_station_n": FEASIBILITY_MIN_CALIBRATION_STATIONS,
                    "minimum_proper_train_station_n": FEASIBILITY_MIN_PROPER_STATIONS,
                    "requested_calibration_station_n": (
                        int(feasibility_calibration_n)
                        if feasibility_calibration_n is not None else np.nan
                    ),
                }
                if feasibility_calibration_n is None:
                    feasibility_record.update({
                        "feasibility_status": "ineligible",
                        "eligibility_reason": (
                            "requires at least "
                            f"{FEASIBILITY_MIN_CALIBRATION_STATIONS + FEASIBILITY_MIN_PROPER_STATIONS} "
                            "outer-training stations for 9 calibration and 2 proper-training stations"
                        ),
                        "nominal_coverage": CONFORMAL_COVERAGE,
                        "q90": np.nan,
                        "q90_is_infinite": np.nan,
                    })
                    conformal_feasibility_summary_rows.append(feasibility_record)
                    conformal_feasibility_model_manifest[feasibility_key] = feasibility_record.copy()
                else:
                    feasibility = station_split_conformal(
                        tr, te, arm_features["full"], config=config,
                        seed_parts=(country, task, str(fold_id)),
                        target_col="pm25", group_col="site_id",
                        smear_splits=args.smear_splits,
                        coverage=CONFORMAL_COVERAGE,
                        proper_fraction=PROPER_TRAIN_FRACTION,
                        calibration_station_n=int(feasibility_calibration_n),
                        split_label="finite_q90_feasibility_min9_calibration",
                    )
                    feasibility_context = {
                        **conformal_context,
                        "split_label": "finite_q90_feasibility_min9_calibration",
                    }
                    for table_name in ("intervals", "station_assignments", "calibration_scores"):
                        feasibility[table_name]["country"] = country
                        feasibility[table_name]["cohort_label"] = cohort_label
                        feasibility[table_name]["task"] = task
                        feasibility[table_name]["fold_id"] = str(fold_id)
                        feasibility[table_name]["arm"] = "full"
                        feasibility[table_name]["split_label"] = feasibility_context["split_label"]
                    feasibility["summary"].update(feasibility_context)
                    feasibility["summary"].update({
                        "feasibility_status": "eligible",
                        "eligibility_reason": "",
                        "minimum_calibration_station_n": FEASIBILITY_MIN_CALIBRATION_STATIONS,
                        "minimum_proper_train_station_n": FEASIBILITY_MIN_PROPER_STATIONS,
                        "requested_calibration_station_n": int(feasibility_calibration_n),
                        "split_seed": feasibility["seed"],
                    })
                    conformal_feasibility_interval_frames.append(feasibility["intervals"])
                    conformal_feasibility_summary_rows.append(feasibility["summary"])
                    conformal_feasibility_assignment_frames.append(feasibility["station_assignments"])
                    conformal_feasibility_score_frames.append(feasibility["calibration_scores"])
                    feasibility_model_dir = out / "models" / "conformal_feasibility" / country
                    feasibility_model_dir.mkdir(parents=True, exist_ok=True)
                    feasibility_model_path = feasibility_model_dir / (
                        f"{task}_{fold_id}_full_finite_q90_feasibility.joblib"
                    )
                    joblib.dump(feasibility["bundle"], feasibility_model_path, compress=3)
                    feasibility_proper_ids = set(
                        feasibility["station_assignments"].loc[
                            feasibility["station_assignments"].split_role == "proper_train",
                            "station_id",
                        ].astype(str)
                    )
                    feasibility_proper_train = tr.loc[
                        tr.site_id.astype(str).isin(feasibility_proper_ids)
                    ]
                    conformal_feasibility_model_manifest[feasibility_key] = {
                        **feasibility_context,
                        "feasibility_status": "eligible",
                        "path": str(feasibility_model_path.resolve()),
                        "sha256": sha256_file(feasibility_model_path),
                        "q90": feasibility["summary"]["q90"],
                        "calibration_station_n": feasibility["summary"]["calibration_station_n"],
                        "proper_train_station_n": feasibility["summary"]["proper_train_station_n"],
                        "fit_n": feasibility["bundle"]["fit_n"],
                        "training_support_fingerprint": training_support_fingerprint(
                            feasibility_proper_train, arm_features["full"], target_col="pm25"
                        ),
                    }
            if not all_task_rows:
                continue
            task_oof = pd.DataFrame(all_task_rows)
            variant_specs = []
            if "log_smear" in args.target_strategies:
                for smear_strategy in args.smear_strategies:
                    if smear_strategy == "crossfit_station" and not args.crossfit_smear_arms:
                        continue
                    if smear_strategy == "in_sample" and not args.in_sample_smear_arms:
                        continue
                    for backtransform in ("smearing", "original"):
                        variant_specs.append(("log_smear", smear_strategy, backtransform))
            if "original_scale" in args.target_strategies and args.original_scale_arms:
                variant_specs.append(("original_scale", "not_applicable", "model_native"))
            for target_strategy, smear_strategy, backtransform in variant_specs:
                variant = task_oof.loc[
                    (task_oof.target_strategy == target_strategy)
                    & (task_oof.smear_strategy == smear_strategy)
                    & (task_oof.backtransform == backtransform)
                ]
                if variant.empty:
                    continue
                wide = variant.pivot(index="row_id", columns="arm", values="pred")
                aligned = d.set_index("_row_id").loc[wide.index]
                y = aligned.pm25.to_numpy(float)
                groups = aligned.site_id.astype(str).to_numpy()
                available_arms = [arm for arm in ARMS if arm in wide.columns]
                for fold_id, fold_part in variant.groupby("fold_id", sort=True):
                    fold_wide = fold_part.pivot(index="row_id", columns="arm", values="pred")
                    fold_rows_aligned = d.set_index("_row_id").loc[fold_wide.index]
                    fold_y = fold_rows_aligned.pm25.to_numpy(float)
                    for arm_a, arm_b in itertools.combinations(
                        [arm for arm in ARMS if arm in fold_wide.columns], 2
                    ):
                        fm_a = _metric(fold_y, fold_wide[arm_a].to_numpy(float))
                        fm_b = _metric(fold_y, fold_wide[arm_b].to_numpy(float))
                        fold_delta_rows.append({
                            "country": country, "cohort_label": cohort_label,
                            "task": task, "fold_id": str(fold_id),
                            "test_years": ",".join(map(str, sorted(fold_rows_aligned.year.unique()))),
                            "target_strategy": target_strategy, "smear_strategy": smear_strategy,
                            "backtransform": backtransform, "arm_a": arm_a, "arm_b": arm_b,
                            "n": int(len(fold_y)),
                            "n_stations": int(fold_rows_aligned.site_id.nunique()),
                            "delta_rmse": float(fm_a["rmse"] - fm_b["rmse"]),
                            "delta_mae": float(fm_a["mae"] - fm_b["mae"]),
                            "delta_bias": float(fm_a["bias"] - fm_b["bias"]),
                            "delta_r2": float(fm_a["r2"] - fm_b["r2"])
                            if np.isfinite(fm_a["r2"]) and np.isfinite(fm_b["r2"]) else np.nan,
                        })
                for arm in available_arms:
                    m = _metric(y, wide[arm].to_numpy(float))
                    metrics_rows.append({
                        "country": country, "cohort_label": cohort_label, "task": task,
                        "target_strategy": target_strategy, "smear_strategy": smear_strategy,
                        "backtransform": backtransform,
                        "arm": arm, "n": m["n"], "n_stations": int(aligned.site_id.nunique()),
                        "test_years": ",".join(map(str, sorted(aligned.year.unique()))),
                        **{k: v for k, v in m.items() if k != "n"},
                    })
                for arm_a, arm_b in itertools.combinations(available_arms, 2):
                    pa, pb = wide[arm_a].to_numpy(float), wide[arm_b].to_numpy(float)
                    ma, mb = _metric(y, pa), _metric(y, pb)
                    delta = {
                        "country": country, "cohort_label": cohort_label, "task": task,
                        "target_strategy": target_strategy, "smear_strategy": smear_strategy,
                        "backtransform": backtransform,
                        "arm_a": arm_a, "arm_b": arm_b,
                        "delta_direction": "arm_a minus arm_b",
                        "n": int(len(y)), "n_stations": int(aligned.site_id.nunique()),
                        "delta_rmse": float(ma["rmse"] - mb["rmse"]),
                        "delta_mae": float(ma["mae"] - mb["mae"]),
                        "delta_bias": float(ma["bias"] - mb["bias"]),
                        "delta_r2": float(ma["r2"] - mb["r2"]) if np.isfinite(ma["r2"]) and np.isfinite(mb["r2"]) else np.nan,
                    }
                    delta.update(_paired_bootstrap(
                        groups, y, pa, pb, args.bootstrap_replicates,
                        seed=0 + len(delta_rows), engine=args.bootstrap_engine,
                    ))
                    block_rows = [
                        row for row in fold_delta_rows
                        if row["country"] == country and row["task"] == task
                        and row["target_strategy"] == target_strategy
                        and row["smear_strategy"] == smear_strategy
                        and row["backtransform"] == backtransform
                        and row["arm_a"] == arm_a and row["arm_b"] == arm_b
                    ]
                    if block_rows:
                        delta["outer_fold_count"] = int(len(block_rows))
                        delta["fold_delta_rmse_min"] = float(min(r["delta_rmse"] for r in block_rows))
                        delta["fold_delta_rmse_max"] = float(max(r["delta_rmse"] for r in block_rows))
                    delta.update(_uncertainty_scope(task, int(task_oof.fold_id.nunique())))
                    delta_rows.append(delta)
            if {"crossfit_station", "in_sample"}.issubset(args.smear_strategies):
                log_rows = task_oof.loc[
                    (task_oof.target_strategy == "log_smear")
                    & (task_oof.backtransform == "smearing")
                    & (task_oof.arm == "full")
                    & (task_oof.smear_strategy == "crossfit_station")
                ]
                scale_rows = task_oof.loc[
                    (task_oof.target_strategy == "log_smear")
                    & (task_oof.backtransform == "smearing")
                    & (task_oof.arm == "full")
                    & (task_oof.smear_strategy == "in_sample")
                ]
                if not log_rows.empty and not scale_rows.empty:
                    compare = log_rows[["row_id", "obs", "site_id", "pred"]].merge(
                        scale_rows[["row_id", "pred"]], on="row_id", suffixes=("_crossfit", "_insample"),
                        validate="one_to_one",
                    )
                    y = compare.obs.to_numpy(float)
                    pa = compare.pred_crossfit.to_numpy(float)
                    pb = compare.pred_insample.to_numpy(float)
                    ma, mb = _metric(y, pa), _metric(y, pb)
                    delta = {
                        "country": country, "cohort_label": cohort_label, "task": task,
                        "target_strategy": "log_smear_crossfit_vs_insample",
                        "smear_strategy": "crossfit_station_vs_in_sample",
                        "backtransform": "corrected_primary",
                        "arm_a": "log_smear:full:crossfit_station",
                        "arm_b": "log_smear:full:in_sample",
                        "delta_direction": "arm_a minus arm_b",
                        "n": int(len(y)), "n_stations": int(compare.site_id.nunique()),
                        "delta_rmse": float(ma["rmse"] - mb["rmse"]),
                        "delta_mae": float(ma["mae"] - mb["mae"]),
                        "delta_bias": float(ma["bias"] - mb["bias"]),
                        "delta_r2": float(ma["r2"] - mb["r2"]) if np.isfinite(ma["r2"]) and np.isfinite(mb["r2"]) else np.nan,
                    }
                    delta.update(_paired_bootstrap(
                        compare.site_id.to_numpy(), y, pa, pb, args.bootstrap_replicates,
                        seed=0 + len(delta_rows), engine=args.bootstrap_engine,
                    ))
                    delta.update(_uncertainty_scope(task, int(task_oof.fold_id.nunique())))
                    delta_rows.append(delta)
            if "original_scale" in args.target_strategies:
                log_rows = task_oof.loc[
                    (task_oof.target_strategy == "log_smear")
                    & (task_oof.backtransform == "smearing")
                    & (task_oof.arm == "full")
                    & (task_oof.smear_strategy == "crossfit_station")
                ]
                scale_rows = task_oof.loc[
                    (task_oof.target_strategy == "original_scale")
                    & (task_oof.backtransform == "model_native")
                    & (task_oof.arm == "full")
                ]
                if not log_rows.empty and not scale_rows.empty:
                    compare = log_rows[["row_id", "obs", "site_id", "pred"]].merge(
                        scale_rows[["row_id", "pred"]], on="row_id", suffixes=("_log_smear", "_original_scale"),
                        validate="one_to_one",
                    )
                    y = compare.obs.to_numpy(float)
                    pa = compare.pred_log_smear.to_numpy(float)
                    pb = compare.pred_original_scale.to_numpy(float)
                    ma, mb = _metric(y, pa), _metric(y, pb)
                    delta = {
                        "country": country, "cohort_label": cohort_label, "task": task,
                        "target_strategy": "log_smear_vs_original_scale",
                        "smear_strategy": "crossfit_station_vs_not_applicable",
                        "backtransform": "primary_vs_model_native",
                        "arm_a": "log_smear:full:crossfit_station",
                        "arm_b": "original_scale:full",
                        "delta_direction": "arm_a minus arm_b",
                        "n": int(len(y)), "n_stations": int(compare.site_id.nunique()),
                        "delta_rmse": float(ma["rmse"] - mb["rmse"]),
                        "delta_mae": float(ma["mae"] - mb["mae"]),
                        "delta_bias": float(ma["bias"] - mb["bias"]),
                        "delta_r2": float(ma["r2"] - mb["r2"]) if np.isfinite(ma["r2"]) and np.isfinite(mb["r2"]) else np.nan,
                    }
                    delta.update(_paired_bootstrap(
                        compare.site_id.to_numpy(), y, pa, pb, args.bootstrap_replicates,
                        seed=0 + len(delta_rows), engine=args.bootstrap_engine,
                    ))
                    delta.update(_uncertainty_scope(task, int(task_oof.fold_id.nunique())))
                    delta_rows.append(delta)

        # Final full-data model artifact for downstream grid/component use.
        source_manifest[country]["full_models"] = {}
        final_specs: list[tuple[str, str]] = []
        if "log_smear" in args.target_strategies:
            if "crossfit_station" in args.smear_strategies and "full" in args.crossfit_smear_arms:
                final_specs.append(("log_smear", "crossfit_station"))
            if "in_sample" in args.smear_strategies and "full" in args.in_sample_smear_arms:
                final_specs.append(("log_smear", "in_sample"))
        if "original_scale" in args.target_strategies and "full" in args.original_scale_arms:
            final_specs.append(("original_scale", "not_applicable"))
        for target_strategy, smear_strategy in final_specs:
            bundle = fit_model(
                d, arm_features["full"], config=config,
                smear_strategy=("in_sample" if smear_strategy == "not_applicable" else smear_strategy),
                smear_splits=args.smear_splits,
                target_strategy=target_strategy,
            )
            model_path = out / "models" / f"{country}_full_{target_strategy}_{smear_strategy}.joblib"
            joblib.dump(bundle, model_path, compress=3)
            source_manifest[country]["full_models"][f"{target_strategy}:{smear_strategy}"] = {
                "path": str(model_path.resolve()), "sha256": sha256_file(model_path),
                "schema": bundle["schema"], "config": bundle["config"],
                "estimator_kind": bundle["estimator_kind"],
                "estimator_class": bundle["estimator_class"],
                "target_strategy": bundle["target_strategy"],
                "smear_strategy": smear_strategy,
                "smear": bundle["smear"], "smear_info": bundle["smear_info"],
                "fit_n": bundle["fit_n"], "fit_stations": bundle["fit_stations"],
                "training_support_fingerprint": training_support_fingerprint(
                    d, arm_features["full"], target_col="pm25"
                ),
            }

    pd.DataFrame(metrics_rows).to_csv(out / "task_metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out / "paired_deltas.csv", index=False)
    pd.DataFrame(fold_delta_rows).to_csv(out / "fold_pair_deltas.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_summaries.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(out / "screening_inner_cv.csv", index=False)
    eligibility_df = pd.DataFrame(eligibility)
    if not eligibility_df.empty:
        eligibility_df["cohort_label"] = eligibility_df.country.map(
            lambda c: target_status.get(c, {}).get("status", "no_feature_table")
        )
    eligibility_df.to_csv(out / "eligibility.csv", index=False)
    duplicate_df = pd.DataFrame(duplicate_rows)
    if not duplicate_df.empty:
        duplicate_df["cohort_label"] = duplicate_df.country.map(
            lambda c: target_status.get(c, {}).get("status", "no_feature_table")
        )
    duplicate_df.to_csv(out / "duplicate_audit.csv", index=False)
    oof = pd.DataFrame(oof_rows)
    if not oof.empty:
        oof.to_parquet(out / "oof_predictions.parquet", index=False)
    write_support_outputs(out, training_range_frames, support_row_frames, oof)
    pd.concat(conformal_interval_frames, ignore_index=True).to_parquet(
        out / "conformal_intervals.parquet", index=False
    ) if conformal_interval_frames else None
    pd.DataFrame(conformal_summary_rows).to_csv(out / "conformal_summary.csv", index=False)
    pd.concat(conformal_assignment_frames, ignore_index=True).to_csv(
        out / "conformal_station_splits.csv", index=False
    ) if conformal_assignment_frames else None
    pd.concat(conformal_score_frames, ignore_index=True).to_csv(
        out / "conformal_calibration_scores.csv", index=False
    ) if conformal_score_frames else None
    pd.concat(conformal_feasibility_interval_frames, ignore_index=True).to_parquet(
        out / "conformal_feasibility_intervals.parquet", index=False
    ) if conformal_feasibility_interval_frames else None
    pd.DataFrame(conformal_feasibility_summary_rows).to_csv(
        out / "conformal_feasibility_summary.csv", index=False
    )
    pd.concat(conformal_feasibility_assignment_frames, ignore_index=True).to_csv(
        out / "conformal_feasibility_station_splits.csv", index=False
    ) if conformal_feasibility_assignment_frames else None
    pd.concat(conformal_feasibility_score_frames, ignore_index=True).to_csv(
        out / "conformal_feasibility_calibration_scores.csv", index=False
    ) if conformal_feasibility_score_frames else None
    _write_json(out / "conformal_manifest.json", {
        "nominal_coverage": CONFORMAL_COVERAGE,
        "proper_train_fraction": PROPER_TRAIN_FRACTION,
        "split_unit": "physical station ID, deterministic within each outer-training set",
        "score": "maximum absolute original-scale residual across calibration months per station",
        "quantile": "ceil((n_calibration_stations + 1) * 0.90) order statistic; infinity if rank exceeds calibration count",
        "point_model": "separate 80% proper-training full-arm log-smear model with station-cross-fitted smearing",
        "interval": "[max(0, point_prediction - q90), point_prediction + q90]",
        "coverage_limits": "T1 station exchangeability is approximate under spatial dependence; T2/T3 report observed coverage under shift, with no new-year guarantee.",
        "models": conformal_model_manifest,
    })
    _write_json(out / "conformal_feasibility_manifest.json", {
        "nominal_coverage": CONFORMAL_COVERAGE,
        "split_label": "finite_q90_feasibility_min9_calibration",
        "minimum_calibration_stations": FEASIBILITY_MIN_CALIBRATION_STATIONS,
        "minimum_proper_training_stations": FEASIBILITY_MIN_PROPER_STATIONS,
        "eligibility_rule": (
            "eligible when outer training has at least the sum of the minimum calibration "
            "and proper-training station counts"
        ),
        "calibration_allocation": "max(ceil(0.20 * outer_train_station_n), 9)",
        "proper_training_allocation": "remaining outer-training stations",
        "score": "maximum absolute original-scale residual across calibration months per station",
        "point_model": "separate proper-training full-arm log-smear model with station-cross-fitted smearing",
        "coverage_limits": "Sensitivity allocation chosen for finite-sample q90 rank feasibility; observed held-out coverage has the same split-specific limitations as the primary analysis.",
        "models": conformal_feasibility_model_manifest,
    })
    assignment_columns = [
        "country", "cohort_label", "task", "fold_id", "role",
        "row_id", "site_id", "year", "month",
    ]
    assignments = (
        pd.concat(outer_fold_assignment_frames, ignore_index=True)[assignment_columns]
        if outer_fold_assignment_frames else pd.DataFrame(columns=assignment_columns)
    )
    if not assignments.empty:
        assignments.sort_values(
            ["country", "task", "fold_id", "row_id"], inplace=True, kind="mergesort"
        )
    assignment_path = out / "outer_fold_assignments.parquet"
    assignments.to_parquet(assignment_path, index=False)
    split_manifest_path = out / "outer_fold_split_manifest.json"
    split_manifest = {
        "format": "repair-v2-outer-fold-split-manifest-v1",
        "key_columns": ["country", "site_id", "year", "month"],
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": sha256_file(assignment_path),
        "assignment_rows": int(len(assignments)),
        "model_config": config.__dict__,
        "runtime": runtime_record,
        "splits": outer_fold_split_records,
    }
    split_manifest_path.write_text(
        json.dumps(split_manifest, separators=(",", ":"), default=str, allow_nan=False),
        encoding="utf-8",
    )
    _write_json(out / "schema_manifest.json", schema_manifest)
    _write_json(out / "source_manifest.json", source_manifest)
    _write_json(out / "run_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(root.resolve()),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "countries_requested": countries,
        "countries_with_oof": sorted(oof.country.unique().tolist()) if not oof.empty else [],
        "model_config": config.__dict__,
        "estimator_kind": args.estimator_kind,
        "estimator_class": ESTIMATOR_CLASS_NAMES[args.estimator_kind],
        "runtime": runtime_record,
        "smear_strategies": args.smear_strategies,
        "crossfit_smear_arms": args.crossfit_smear_arms,
        "in_sample_smear_arms": args.in_sample_smear_arms,
        "target_strategies": args.target_strategies,
        "original_scale_arms": args.original_scale_arms,
        "smear_splits": args.smear_splits,
        "primary_backtransform": "smear*exp(z)-0.1",
        "sensitivity_backtransform": "exp(z)-0.1",
        "feature_arms": list(ARMS),
        "country_meteorology_profile_selection": (
            "no_met_override" if args.no_met else "source_declared_fixed_profile"
        ),
        "minimum_stations_T1_T2": MIN_STATIONS_T1_T2,
        "minimum_train_test_rows_T3": [T3_MIN_TRAIN_ROWS, T3_MIN_TEST_ROWS],
        "minimum_train_test_stations_T3": [MIN_T3_TRAIN_STATIONS, MIN_T3_TEST_STATIONS],
        "T2_buffer_km": SPATIAL_BUFFER_KM,
        "training_support_diagnostics": {
            "distance_cutoff_km": SUPPORT_DISTANCE_KM,
            "range_quantiles": [SUPPORT_LOWER_QUANTILE, SUPPORT_UPPER_QUANTILE],
            "range_reference": "each outer-training fold only",
            "geography_reported_separately_from_environmental_features": True,
            "stratum_metrics_have_no_separate_intervals": True,
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap": bootstrap_provenance(args.bootstrap_engine),
        "outer_fold_split_manifest": {
            "path": str(split_manifest_path.resolve()),
            "sha256": sha256_file(split_manifest_path),
            "assignment_path": str(assignment_path.resolve()),
            "assignment_sha256": sha256_file(assignment_path),
            "assignment_rows": int(len(assignments)),
        },
        "limitations": [
            "USA targets remain provisional and uncorrected unless --target-override USA=... is supplied.",
            "Source station feature tables are precomputed; upstream target/QC provenance is inherited.",
            "Cross-country metric differences are descriptive and not a pooled transportability estimate.",
        ],
    })
    print(f"WROTE {out.resolve()}")
    print(f"OOF rows={len(oof):,}; countries={sorted(oof.country.unique().tolist()) if not oof.empty else []}")
    if not oof.empty:
        print(pd.DataFrame(metrics_rows).query("backtransform == 'smearing' and arm == 'full'")[
            ["country", "task", "n", "rmse", "mae", "bias", "r2"]
        ].to_string(index=False))
    return 0


def self_check() -> int:
    from validation_core import feature_matrix, predict_model, predict_grid_rows, counterfactual_frame

    schema = schema_for(["x", "aod_used"], {"aod_used": "aod_raw"})
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "aod_used": [0.5, 0.2, np.nan],
                       "aod_raw": [0.5, np.nan, np.nan]})
    X = feature_matrix(df, schema)
    assert X.shape == (3, 4) and np.array_equal(X[:, 2:], np.array([[0, 0], [1, 1], [0, 1]], dtype=np.float32))
    try:
        feature_matrix(df.drop(columns="x"), schema)
    except ValueError:
        pass
    else:
        raise AssertionError("strict schema accepted a missing feature")
    try:
        feature_matrix(df.drop(columns="aod_raw"), schema)
    except ValueError:
        pass
    else:
        raise AssertionError("AOD schema accepted missing raw-coverage source")
    class Fixed:
        def predict(self, matrix):
            return np.full(len(matrix), np.log(2.1))
    bundle = {"bundle_version": SCHEMA_VERSION, "schema": schema, "estimator": Fixed(), "smear": 1.5}
    expected_smear = 1.5 * np.exp(np.log(2.1)) - 0.1
    expected_orig = np.exp(np.log(2.1)) - 0.1
    assert np.allclose(predict_model(bundle, df, backtransform="smearing"), expected_smear)
    assert np.allclose(predict_model(bundle, df, backtransform="original"), expected_orig)
    assert np.array_equal(
        predict_grid_rows(bundle, df, chunk_size=1),
        predict_model(bundle, df),
    )
    cf = counterfactual_frame(pd.DataFrame({"local": [1], "smoothed": [2]}), ["local", "smoothed"])
    assert cf.iloc[0].tolist() == [0.0, 0.0]
    native_bundle = dict(bundle, target_strategy="original_scale", estimator=type(
        "FixedOriginal", (), {"predict": lambda self, matrix: np.full(len(matrix), 3.5)}
    )())
    assert np.allclose(
        predict_model(native_bundle, df, backtransform="model_native"),
        3.5,
    )
    # Outer fold checks: station groups cannot overlap; T1/T2 tests partition once.
    tiny = pd.DataFrame({
        "site_id": np.repeat([f"s{i}" for i in range(10)], 2),
        "lat": np.repeat(np.arange(10), 2) * 1.0,
        "lon": np.repeat(np.arange(10), 2) * 1.0,
        "year": np.repeat([2021] * 10, 2), "month": np.tile([1, 2], 10),
    })
    folds = list(GroupKFold(5).split(tiny, groups=tiny.site_id))
    tested = []
    for tr, te in folds:
        assert set(tiny.site_id.iloc[tr]).isdisjoint(set(tiny.site_id.iloc[te]))
        tested.extend(te.tolist())
    assert sorted(tested) == list(range(len(tiny)))
    spatial = list(task_splits(tiny, buffer_km=SPATIAL_BUFFER_KM))
    for task_name in ("T1_station", "T2_spatial"):
        task_tested = [i for task, _, _, te, _ in spatial if task == task_name for i in te]
        assert sorted(task_tested) == list(range(len(tiny)))
        for task, _, tr, te, _ in spatial:
            if task == task_name:
                assert set(tiny.site_id.iloc[tr]).isdisjoint(set(tiny.site_id.iloc[te]))
    for task, _, tr, te, buffer in spatial:
        if task != "T2_spatial_buffer50km" or len(te) == 0:
            continue
        train_sites = _site_table(tiny.iloc[tr])
        test_sites = _site_table(tiny.iloc[te])
        for row in test_sites.itertuples(index=False):
            assert np.all(_haversine(train_sites.lat, train_sites.lon, row.lat, row.lon) >= buffer)
    print("self-check passed: inverse, strict schema, indicators, grid chunk parity, counterfactual, grouped folds")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(ROOT_DEFAULT))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--countries", default=",".join(COUNTRIES))
    parser.add_argument("--feature-override", action="append", default=[], metavar="COUNTRY=PATH",
                        help="replace one country with a table; repeat to concatenate years")
    parser.add_argument("--target-override", action="append", default=[], metavar="COUNTRY=PATH")
    parser.add_argument("--target-status", action="append", default=[], metavar="COUNTRY=STATUS")
    parser.add_argument(
        "--input-provenance", action="append", default=[], metavar="COUNTRY=JSON",
        help="attach a QA/feature-builder provenance manifest to this country's source record",
    )
    parser.add_argument("--override-target-column", default="pm25")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument(
        "--estimator-kind", choices=tuple(ESTIMATOR_CLASS_NAMES), default="hgb",
        help="model-class sensitivity; Ridge uses median imputation, standardization, alpha=1.0",
    )
    parser.add_argument("--smear-strategies", default="crossfit_station,in_sample",
                        help="train-only log-smear methods; crossfit is primary, in_sample sensitivity")
    parser.add_argument("--crossfit-smear-arms", default="B,B+AOD,B+fire,full")
    parser.add_argument("--in-sample-smear-arms", default="full")
    parser.add_argument("--smear-splits", type=int, default=3)
    parser.add_argument("--target-strategies", default="log_smear,original_scale")
    parser.add_argument("--original-scale-arms", default="full",
                        help="comma-separated arms to fit directly on PM2.5; default full only")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument(
        "--bootstrap-engine", choices=BOOTSTRAP_ENGINES, default="reference",
        help="station-cluster interval implementation; cluster_stats is an equivalent fast aggregation",
    )
    parser.add_argument(
        "--no-met", action="store_true",
        help="exclude source meteorology columns from the base profile (sensitivity run)",
    )
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_engine == "cluster_stats":
        _load_fast_bootstrap()
    args.target_strategies = [s.strip() for s in args.target_strategies.split(",") if s.strip()]
    args.smear_strategies = [s.strip() for s in args.smear_strategies.split(",") if s.strip()]
    args.crossfit_smear_arms = [s.strip() for s in args.crossfit_smear_arms.split(",") if s.strip()]
    args.in_sample_smear_arms = [s.strip() for s in args.in_sample_smear_arms.split(",") if s.strip()]
    args.original_scale_arms = [s.strip() for s in args.original_scale_arms.split(",") if s.strip()]
    invalid = sorted(set(args.target_strategies) - {"log_smear", "original_scale"})
    if invalid or not args.target_strategies:
        parser.error(f"invalid --target-strategies values: {invalid}")
    invalid_smear = sorted(set(args.smear_strategies) - {"crossfit_station", "in_sample"})
    if invalid_smear or not args.smear_strategies:
        parser.error(f"invalid --smear-strategies values: {invalid_smear}")
    for label, selected in (
        ("--crossfit-smear-arms", args.crossfit_smear_arms),
        ("--in-sample-smear-arms", args.in_sample_smear_arms),
        ("--original-scale-arms", args.original_scale_arms),
    ):
        invalid_selected = sorted(set(selected) - set(ARMS))
        if invalid_selected:
            parser.error(f"invalid {label} values: {invalid_selected}")
    invalid_arms = sorted(set(args.original_scale_arms) - set(ARMS))
    if invalid_arms:
        parser.error(f"invalid --original-scale-arms values: {invalid_arms}")
    if args.self_check_only:
        return self_check()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

