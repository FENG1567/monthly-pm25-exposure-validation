#!/usr/bin/env python
"""Locked external-year evaluation: train through cutoff, test one later year.

The external-year table is never appended to station/spatial folds. Profiles
with and without meteorology are independently prespecified and scored on the
same station-month keys, allowing the historical common-no-met product inputs
to be compared without changing the test sample.
"""
from __future__ import annotations

import os

for _key, _value in {
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(_key, _value)

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from validation_core import ModelConfig, fit_model, predict_model, schema_for
from validation_core import ESTIMATOR_CLASS_NAMES
from support_diagnostics import (
    SUPPORT_DISTANCE_KM, SUPPORT_LOWER_QUANTILE, SUPPORT_UPPER_QUANTILE,
    add_distance_support, feature_groups_for_arm, training_feature_support,
    write_support_outputs,
)
from conformal_sensitivity import (
    CONFORMAL_COVERAGE, CONFORMAL_SCORE,
    FEASIBILITY_MIN_CALIBRATION_STATIONS, FEASIBILITY_MIN_PROPER_STATIONS,
    PROPER_TRAIN_FRACTION, feasibility_calibration_station_n,
    station_split_conformal,
)
from run_validation import (
    AOD, ARMS, BOOTSTRAP_ENGINES, FIRE, GEO, MET, _load_fast_bootstrap,
    _metric, _paired_bootstrap, _nearest_distances, _write_json,
    _screen_baseline, _uncertainty_scope, bootstrap_provenance,
    load_input_provenance, replace_targets, runtime_provenance, sha256_file,
    training_support_fingerprint,
)


FINAL_EXPECTED_KEYS_FORMAT = "repair-v2-final-multiplicity-expected-test-keys-v1"
FINAL_EXPECTED_KEYS_FIELD = "final_multiplicity_expected_test_keys"
FINAL_EXPECTED_KEY_COLUMNS = (
    "country", "cohort_label", "profile", "task", "fold_id",
    "row_id", "site_id", "year", "month",
)
FINAL_EXPECTED_KEY_ROW_SELECTION = (
    "external_features_with_assigned_target_and_finite_nonnegative_pm25_and_finite_"
    "coordinates_and_year_equal_external_year"
)


def _ordered_target_values_sha256(values) -> str:
    """Hash artifact-row-ordered float64 values using the final loader's array convention."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}".encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load(path: Path, country: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    required = ["country", "site_id", "lat", "lon", "year", "month", "pm25"]
    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError(f"{path} lacks required fields: {missing}")
    d = d.loc[d.country.astype(str).str.upper() == country].copy()
    if d.empty:
        raise ValueError(f"{path} contains no {country} rows")
    d["country"] = country
    d["site_id"] = d.site_id.astype(str)
    d["year"] = pd.to_numeric(d.year, errors="raise").astype(int)
    d["month"] = pd.to_numeric(d.month, errors="raise").astype(int)
    d["pm25"] = pd.to_numeric(d.pm25, errors="coerce")
    if d.duplicated(["site_id", "year", "month"]).any():
        raise ValueError(f"{path} has duplicate station-month keys")
    return d


def _build_final_expected_test_keys(
    test: pd.DataFrame,
    *,
    country: str,
    cohort_label: str,
    profiles: list[str],
    external_year: int,
) -> pd.DataFrame:
    """Build final expected keys from the validated external test frame itself."""
    country = str(country).upper()
    profiles = list(profiles)
    if not profiles or set(profiles) - {"with_met", "no_met"}:
        raise ValueError("profiles must contain with_met and/or no_met")
    if len(set(profiles)) != len(profiles):
        raise ValueError("duplicate profile in final expected-test-key request")
    if not isinstance(cohort_label, str) or not cohort_label.strip():
        raise ValueError("final expected-test keys require an explicit target cohort label")
    if test.empty:
        raise ValueError("cannot commit empty external test keys")
    required = {"country", "site_id", "year", "month", "pm25", "lat", "lon", "_row_id"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError(f"validated external test frame lacks key/acceptance fields: {missing}")

    source_country = test["country"].astype(str).str.upper()
    if not source_country.eq(country).all():
        raise ValueError("validated external test frame contains a different country")
    if test["site_id"].isna().any():
        raise ValueError("validated external test frame contains null site IDs")
    site_ids = test["site_id"].astype(str)
    if site_ids.str.len().eq(0).any():
        raise ValueError("validated external test frame contains empty site IDs")

    years = pd.to_numeric(test["year"], errors="raise").to_numpy(dtype="int64")
    if not np.equal(years, int(external_year)).all():
        raise ValueError("final expected-test keys require exact external-year rows")
    months = pd.to_numeric(test["month"], errors="raise").to_numpy(dtype="int64")
    row_ids = pd.to_numeric(test["_row_id"], errors="raise").to_numpy(dtype="int64")
    if not np.array_equal(row_ids, np.arange(len(test), dtype="int64")):
        raise ValueError("external test _row_id must preserve zero-based source order")
    targets = pd.to_numeric(test["pm25"], errors="coerce").to_numpy(dtype="float64")
    lat = pd.to_numeric(test["lat"], errors="coerce").to_numpy(dtype="float64")
    lon = pd.to_numeric(test["lon"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("final expected-test keys require finite nonnegative assigned pm25")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("final expected-test keys require finite coordinates")
    base = pd.DataFrame({
        "row_id": row_ids,
        "site_id": site_ids.to_numpy(dtype=object),
        "year": years,
        "month": months,
        "obs": targets.astype("float64", copy=False),
    })
    if base.duplicated(["site_id", "year", "month"]).any():
        raise ValueError("final expected-test keys contain duplicate station-month rows")

    key_frames = []
    for profile in profiles:
        key_frames.append(pd.DataFrame({
            "country": country,
            "cohort_label": cohort_label,
            "profile": profile,
            "task": f"T4_external_{int(external_year)}_{profile}",
            "fold_id": str(int(external_year)),
            "row_id": base["row_id"].to_numpy(dtype="int64"),
            "site_id": base["site_id"].to_numpy(dtype=object),
            "year": base["year"].to_numpy(dtype="int64"),
            "month": base["month"].to_numpy(dtype="int64"),
            "obs": base["obs"].to_numpy(dtype="float64"),
        }))
    keys = pd.concat(key_frames, ignore_index=True)[[*FINAL_EXPECTED_KEY_COLUMNS, "obs"]]
    if keys.duplicated(list(FINAL_EXPECTED_KEY_COLUMNS)).any():
        raise ValueError("final expected-test keys contain duplicate profile/task identities")
    for profile in profiles[1:]:
        observed = keys.loc[
            keys["profile"].eq(profile), ["row_id", "site_id", "year", "month", "obs"]
        ]
        if not observed.reset_index(drop=True).equals(base.reset_index(drop=True)):
            raise ValueError("requested meteorology profiles do not share the same external test cohort")
    return keys


def _write_final_expected_test_keys(
    out: Path,
    keys: pd.DataFrame,
    *,
    external_feature_sha256: str,
    external_target_source_sha256: str,
) -> dict:
    """Persist the independently derived exact-key artifact before any model fit."""
    path = Path(out) / "expected_test_keys.parquet"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stale expected-test-key artifact: {path}")
    artifact_columns = [*FINAL_EXPECTED_KEY_COLUMNS, "obs"]
    keys.loc[:, artifact_columns].to_parquet(path, index=False)
    digest = sha256_file(path)
    return {
        "format": FINAL_EXPECTED_KEYS_FORMAT,
        "path": str(path.resolve()),
        "sha256": digest,
        "rows": int(len(keys)),
        "key_columns": list(FINAL_EXPECTED_KEY_COLUMNS),
        "row_selection": FINAL_EXPECTED_KEY_ROW_SELECTION,
        "target_column": "pm25",
        "observed_column": "obs",
        "target_values_sha256": _ordered_target_values_sha256(keys["obs"].to_numpy(dtype="float64")),
        "derived_from": {
            "external_features_sha256": str(external_feature_sha256).lower(),
            "external_target_source_sha256": str(external_target_source_sha256).lower(),
        },
    }


def _bind_target_source(
    source: dict,
    feature_source: dict,
    *,
    role: str,
    target_column: str,
    used_override: bool,
    input_provenance: dict | None,
) -> dict:
    """Make the target source re-hashable without self-certifying embedded targets."""
    bound = dict(source)
    if used_override:
        raw_path, raw_sha = bound.get("path"), bound.get("sha256")
        if not raw_path or not raw_sha:
            raise ValueError(f"{role} target override lacks a path/SHA-256 source record")
        path = Path(raw_path).resolve()
        digest = str(raw_sha).lower()
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{role} target override does not match its recorded source SHA-256")
        bound.update({
            "path": str(path), "sha256": digest,
            "role": f"{role}_target_override_table",
            "source_kind": "standalone_target_override_table",
            "target_column": target_column,
        })
    else:
        bound.update({
            "path": feature_source["path"],
            "sha256": feature_source["sha256"],
            "role": f"{role}_target_embedded_in_feature_file",
            "source_kind": "embedded_feature_target_column",
            "target_column": "pm25",
            "independent_target_acceptance_asserted": False,
        })
    if input_provenance is not None:
        bound["input_provenance_reference"] = {
            "path": input_provenance["path"],
            "sha256": input_provenance["sha256"],
        }
        manifest = input_provenance.get("manifest")
        if isinstance(manifest, dict):
            declared = {
                name: manifest[name]
                for name in (
                    "target_source", "target_provenance", "target_status",
                    "target_verified", "target_provenance_path",
                    "target_provenance_sha256", "target_column",
                )
                if name in manifest
            }
            if declared:
                bound["declared_target_cleaning_provenance"] = declared
    return bound


def _assert_source_unchanged(record: dict, label: str) -> None:
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != str(record["sha256"]).lower():
        raise ValueError(f"{label} source changed during the external run: {path}")


def _profile_features(profile: str) -> list[str]:
    if profile == "with_met":
        return GEO + MET
    if profile == "no_met":
        return GEO.copy()
    raise ValueError(f"unsupported feature profile: {profile}")


def _paired_rows(rows: pd.DataFrame, *, country: str, cohort: str,
                 profile: str, task: str, arm_a: str, arm_b: str,
                 prediction_a: np.ndarray, prediction_b: np.ndarray,
                 y: np.ndarray, groups: np.ndarray, reps: int, seed: int,
                 bootstrap_engine: str = "reference") -> dict:
    ma, mb = _metric(y, prediction_a), _metric(y, prediction_b)
    out = {
        "country": country, "cohort_label": cohort, "profile": profile, "task": task,
        "arm_a": arm_a, "arm_b": arm_b, "delta_direction": "arm_a minus arm_b",
        "n": int(len(y)), "n_stations": int(len(np.unique(groups))),
        "delta_rmse": float(ma["rmse"] - mb["rmse"]),
        "delta_mae": float(ma["mae"] - mb["mae"]),
        "delta_bias": float(ma["bias"] - mb["bias"]),
        "delta_r2": float(ma["r2"] - mb["r2"])
        if np.isfinite(ma["r2"]) and np.isfinite(mb["r2"]) else np.nan,
    }
    out.update(_paired_bootstrap(
        groups, y, prediction_a, prediction_b, reps, seed, engine=bootstrap_engine
    ))
    out.update(_uncertainty_scope(f"T3_year_{task.rsplit('_', 1)[-1]}", 1))
    return out


def run(args) -> int:
    if args.bootstrap_engine == "cluster_stats":
        _load_fast_bootstrap()
    country = args.country.upper()
    train_path = Path(args.training_features).resolve()
    test_path = Path(args.external_features).resolve()
    training_features_source = {
        "path": str(train_path), "sha256": sha256_file(train_path),
    }
    external_features_source = {
        "path": str(test_path), "sha256": sha256_file(test_path),
    }
    train_all = _load(train_path, country)
    test_all = _load(test_path, country)
    if args.training_target_override:
        training_target_override = Path(args.training_target_override).resolve()
        training_target_override_sha256 = sha256_file(training_target_override)
        train_all, train_target_status = replace_targets(
            train_all, training_target_override, country=country,
            target_column=args.override_target_column,
        )
        if train_target_status.get("sha256") != training_target_override_sha256:
            raise ValueError("training target override changed while it was being assigned")
        if args.training_target_status != "precomputed_training_target":
            train_target_status["target_join_status"] = train_target_status.get("status")
            train_target_status["status"] = str(args.training_target_status)
    else:
        train_target_status = {"status": args.training_target_status}
    if args.external_target_override:
        external_target_override = Path(args.external_target_override).resolve()
        external_target_override_sha256 = sha256_file(external_target_override)
        test_all, test_target_status = replace_targets(
            test_all, external_target_override, country=country,
            target_column=args.override_target_column,
        )
        if test_target_status.get("sha256") != external_target_override_sha256:
            raise ValueError("external target override changed while it was being assigned")
        if args.external_target_status != "external_target_unverified":
            test_target_status["target_join_status"] = test_target_status.get("status")
            test_target_status["status"] = str(args.external_target_status)
    else:
        test_target_status = {"status": args.external_target_status}
    train = train_all.loc[train_all.year <= args.train_through].copy()
    if (train.year > args.train_through).any():
        raise AssertionError("external-year training frame crossed the frozen cutoff")
    test = test_all.loc[test_all.year == args.external_year].copy()
    if test.empty:
        raise ValueError(f"no test rows for requested year {args.external_year}")
    if (set(train.year) & set(test.year)) or test.year.nunique() != 1:
        raise ValueError("train and external year must be disjoint and test must contain exactly one year")
    for name, frame in (("training", train), ("external", test)):
        finite = np.isfinite(frame.pm25) & (frame.pm25 >= 0)
        coord = np.isfinite(pd.to_numeric(frame.lat, errors="coerce")) & np.isfinite(
            pd.to_numeric(frame.lon, errors="coerce")
        )
        if not (finite & coord).all():
            raise ValueError(f"{name} has invalid target or coordinates; repair inputs before scoring")
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train["_row_id"] = np.arange(len(train), dtype=np.int64)
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    if train.site_id.nunique() < args.min_train_stations:
        raise ValueError(f"training has fewer than {args.min_train_stations} independent stations")
    if test.site_id.nunique() < args.min_test_stations:
        raise ValueError(f"test has fewer than {args.min_test_stations} independent stations")

    training_input_provenance = load_input_provenance(
        Path(args.training_input_provenance)
    ) if args.training_input_provenance else None
    external_input_provenance = load_input_provenance(
        Path(args.external_input_provenance)
    ) if args.external_input_provenance else None
    train_target_source = _bind_target_source(
        train_target_status, training_features_source,
        role="training", target_column=args.override_target_column,
        used_override=bool(args.training_target_override),
        input_provenance=training_input_provenance,
    )
    external_target_source = _bind_target_source(
        test_target_status, external_features_source,
        role="external", target_column=args.override_target_column,
        used_override=bool(args.external_target_override),
        input_provenance=external_input_provenance,
    )
    _assert_source_unchanged(training_features_source, "training feature")
    _assert_source_unchanged(external_features_source, "external feature")
    _assert_source_unchanged(train_target_source, "training target")
    _assert_source_unchanged(external_target_source, "external target")
    if training_input_provenance is not None:
        _assert_source_unchanged(training_input_provenance, "training input provenance")
    if external_input_provenance is not None:
        _assert_source_unchanged(external_input_provenance, "external input provenance")

    profiles = list(args.profiles)
    if not profiles or set(profiles) - {"with_met", "no_met"}:
        raise ValueError("--profiles must contain with_met and/or no_met")
    if len(set(profiles)) != len(profiles):
        raise ValueError("--profiles must not contain duplicate profiles")
    expected_test_keys = _build_final_expected_test_keys(
        test, country=country, cohort_label=str(external_target_source["status"]),
        profiles=profiles, external_year=args.external_year,
    )
    out = Path(args.out).resolve()
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing external output: {out}")
    else:
        out.mkdir(parents=True)
    expected_test_key_commitment = _write_final_expected_test_keys(
        out, expected_test_keys,
        external_feature_sha256=external_features_source["sha256"],
        external_target_source_sha256=external_target_source["sha256"],
    )

    config = ModelConfig(
        max_iter=args.max_iter, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=25, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=0,
        estimator_kind=args.estimator_kind, ridge_alpha=1.0,
    )
    (out / "models").mkdir(exist_ok=True)
    metrics_rows, delta_rows, oof_rows, fold_rows, screen_rows = [], [], [], [], []
    training_range_frames, support_row_frames = [], []
    conformal_interval_frames, conformal_summary_rows = [], []
    conformal_assignment_frames, conformal_score_frames = [], []
    conformal_feasibility_interval_frames, conformal_feasibility_summary_rows = [], []
    conformal_feasibility_assignment_frames, conformal_feasibility_score_frames = [], []
    conformal_model_manifest = {}
    conformal_feasibility_model_manifest = {}
    schema_manifest = {}
    model_manifest = {}
    profile_sizes = {}
    for profile in profiles:
        base = _profile_features(profile)
        arms = {
            "B": base,
            "B+AOD": base + AOD,
            "B+fire": base + FIRE,
            "full": base + AOD + FIRE,
        }
        required = sorted(set(itertools.chain.from_iterable(arms.values())) | {"aod_raw"})
        missing = [c for c in required if c not in train.columns or c not in test.columns]
        if missing:
            raise ValueError(f"{profile} exact feature schema missing columns in train/test: {missing}")
        # Keep the same test station-months in both declared profiles.
        profile_sizes[profile] = {"n_train": int(len(train)), "n_test": int(len(test))}
        for arm, features in arms.items():
            schema_manifest[f"{profile}:{arm}"] = schema_for(
                features, {"aod_used": "aod_raw"} if "aod_used" in features else None
            )
        task = f"T4_external_{args.external_year}_{profile}"
        training_fingerprint = training_support_fingerprint(
            train, arms["full"], target_col="pm25"
        )
        screen = _screen_baseline(train, base, config, "pm25")
        screen_rows.append({
            "country": country, "cohort_label": test_target_status["status"],
            "profile": profile, "task": task,
            "n_train": len(train), "n_train_stations": train.site_id.nunique(),
            "n_test": len(test), "n_test_stations": test.site_id.nunique(),
            **screen,
        })
        nearest = _nearest_distances(test, train)
        fold_rows.append({
            "country": country, "cohort_label": test_target_status["status"],
            "profile": profile, "task": task, "fold_id": str(args.external_year),
            "n_train": int(len(train)), "n_test": int(len(test)),
            "n_train_stations": int(train.site_id.nunique()),
            "n_test_stations": int(test.site_id.nunique()),
            "train_years": ",".join(map(str, sorted(train.year.unique()))),
            "test_years": str(args.external_year),
            "distance_km_min": float(np.nanmin(nearest)),
            "distance_km_median": float(np.nanmedian(nearest)),
            "distance_km_p90": float(np.nanpercentile(nearest, 90)),
            "test_within_50km_train_pct": float(np.mean(nearest < 50)),
            "aod_raw_missing_pct": float(test.aod_raw.isna().mean()),
            "aod_used_missing_pct": float(test.aod_used.isna().mean()),
            "screen_baseline_inner_r2": screen.get("inner_r2", np.nan),
        })
        profile_rows = []
        for arm, features in arms.items():
            feature_groups = feature_groups_for_arm(
                features, geography=GEO, meteorology=MET,
                aod=AOD, fire=FIRE,
            )
            support_ranges, support_flags = training_feature_support(
                train, test, features, feature_groups
            )
            support_ranges["country"] = country
            support_ranges["cohort_label"] = test_target_status["status"]
            support_ranges["profile"] = profile
            support_ranges["task"] = task
            support_ranges["fold_id"] = str(args.external_year)
            support_ranges["arm"] = arm
            support_flags["country"] = country
            support_flags["cohort_label"] = test_target_status["status"]
            support_flags["profile"] = profile
            support_flags["task"] = task
            support_flags["fold_id"] = str(args.external_year)
            support_flags["arm"] = arm
            site_by_row = dict(zip(test["_row_id"].astype(int), test.site_id.astype(str)))
            support_flags["site_id"] = support_flags.row_id.map(site_by_row)
            support_flags = add_distance_support(
                support_flags, nearest, test["_row_id"].to_numpy(dtype="int64")
            )
            training_range_frames.append(support_ranges)
            support_row_frames.append(support_flags)
            fit_specs = []
            if arm in args.crossfit_smear_arms and "crossfit_station" in args.smear_strategies:
                fit_specs.append(("log_smear", "crossfit_station"))
            if arm in args.in_sample_smear_arms and "in_sample" in args.smear_strategies:
                fit_specs.append(("log_smear", "in_sample"))
            if arm in args.original_scale_arms:
                fit_specs.append(("original_scale", "not_applicable"))
            for target_strategy, smear_strategy in fit_specs:
                bundle = fit_model(
                    train, features, config=config,
                    smear_strategy=("in_sample" if smear_strategy == "not_applicable" else smear_strategy),
                    smear_splits=args.smear_splits, target_strategy=target_strategy,
                )
                if target_strategy == "log_smear":
                    predictions = [
                        ("smearing", predict_model(bundle, test, backtransform="smearing")),
                        ("original", predict_model(bundle, test, backtransform="original")),
                    ]
                else:
                    predictions = [("model_native", predict_model(bundle, test, backtransform="model_native"))]
                for backtransform, pred in predictions:
                    for i, row in test.iterrows():
                        record = {
                            "country": country, "cohort_label": test_target_status["status"],
                            "profile": profile, "task": task, "fold_id": str(args.external_year),
                            "target_strategy": target_strategy, "smear_strategy": smear_strategy,
                            "backtransform": backtransform, "arm": arm,
                            "row_id": int(row["_row_id"]), "site_id": str(row.site_id),
                            "year": int(row.year), "month": int(row.month),
                            "obs": float(row.pm25), "pred": float(pred[i]),
                            "nearest_train_station_km": float(nearest[i]),
                            "n_train": int(len(train)), "n_test": int(len(test)),
                            "n_train_stations": int(train.site_id.nunique()),
                            "n_test_stations": int(test.site_id.nunique()),
                            "smear": float(bundle["smear"]),
                            "smear_method": bundle["smear_info"]["method"],
                        }
                        profile_rows.append(record)
                        oof_rows.append(record)
                if arm == "full":
                    name = f"{country}_{task}_full_{target_strategy}_{smear_strategy}.joblib"
                    model_path = out / "models" / name
                    joblib.dump(bundle, model_path, compress=3)
                    model_manifest[f"{profile}:{target_strategy}:{smear_strategy}"] = {
                        "path": str(model_path.resolve()), "sha256": sha256_file(model_path),
                        "schema": bundle["schema"], "config": bundle["config"],
                        "estimator_kind": bundle["estimator_kind"],
                        "estimator_class": bundle["estimator_class"],
                        "target_strategy": bundle["target_strategy"],
                        "smear_strategy": smear_strategy,
                        "smear": bundle["smear"], "smear_info": bundle["smear_info"],
                        "fit_n": bundle["fit_n"],
                        "fit_stations": bundle["fit_stations"],
                        "training_support_fingerprint": training_fingerprint,
                    }
        conformal = station_split_conformal(
            train, test, arms["full"], config=config,
            seed_parts=(country, "T4_external", str(args.external_year)),
            target_col="pm25", group_col="site_id",
            smear_splits=args.smear_splits,
            coverage=CONFORMAL_COVERAGE,
            proper_fraction=PROPER_TRAIN_FRACTION,
        )
        conformal_context = {
            "country": country, "cohort_label": test_target_status["status"],
            "profile": profile, "task": task, "fold_id": str(args.external_year),
            "arm": "full",
        }
        for table_name in ("intervals", "station_assignments", "calibration_scores"):
            conformal[table_name]["country"] = country
            conformal[table_name]["cohort_label"] = test_target_status["status"]
            conformal[table_name]["profile"] = profile
            conformal[table_name]["task"] = task
            conformal[table_name]["fold_id"] = str(args.external_year)
            conformal[table_name]["arm"] = "full"
        conformal["summary"].update(conformal_context)
        conformal["summary"]["split_seed"] = conformal["seed"]
        conformal_interval_frames.append(conformal["intervals"])
        conformal_summary_rows.append(conformal["summary"])
        conformal_assignment_frames.append(conformal["station_assignments"])
        conformal_score_frames.append(conformal["calibration_scores"])
        conformal_model_dir = out / "models" / "conformal"
        conformal_model_dir.mkdir(parents=True, exist_ok=True)
        conformal_model_path = conformal_model_dir / f"{country}_{task}_full_split_conformal.joblib"
        joblib.dump(conformal["bundle"], conformal_model_path, compress=3)
        conformal_model_manifest[f"{profile}"] = {
            **conformal_context,
            "path": str(conformal_model_path.resolve()),
            "sha256": sha256_file(conformal_model_path),
            "q90": conformal["summary"]["q90"],
            "calibration_station_n": conformal["summary"]["calibration_station_n"],
            "proper_train_station_n": conformal["summary"]["proper_train_station_n"],
            "fit_n": conformal["bundle"]["fit_n"],
        }
        # Keep the larger calibration allocation in separate outputs so the
        # primary 80/20 external-year result remains independently traceable.
        outer_train_station_n = int(train.site_id.astype(str).nunique())
        feasibility_calibration_n = feasibility_calibration_station_n(
            outer_train_station_n
        )
        feasibility_key = str(profile)
        feasibility_context = {
            **conformal_context,
            "split_label": "finite_q90_feasibility_min9_calibration",
        }
        if feasibility_calibration_n is None:
            feasibility_record = {
                **feasibility_context,
                "feasibility_status": "ineligible",
                "eligibility_reason": (
                    "requires at least "
                    f"{FEASIBILITY_MIN_CALIBRATION_STATIONS + FEASIBILITY_MIN_PROPER_STATIONS} "
                    "outer-training stations for 9 calibration and 2 proper-training stations"
                ),
                "outer_train_station_n": outer_train_station_n,
                "minimum_calibration_station_n": FEASIBILITY_MIN_CALIBRATION_STATIONS,
                "minimum_proper_train_station_n": FEASIBILITY_MIN_PROPER_STATIONS,
                "requested_calibration_station_n": np.nan,
                "nominal_coverage": CONFORMAL_COVERAGE,
                "q90": np.nan,
                "q90_is_infinite": np.nan,
            }
            conformal_feasibility_summary_rows.append(feasibility_record)
            conformal_feasibility_model_manifest[feasibility_key] = feasibility_record.copy()
        else:
            feasibility = station_split_conformal(
                train, test, arms["full"], config=config,
                seed_parts=(country, "T4_external", str(args.external_year)),
                target_col="pm25", group_col="site_id",
                smear_splits=args.smear_splits,
                coverage=CONFORMAL_COVERAGE,
                proper_fraction=PROPER_TRAIN_FRACTION,
                calibration_station_n=int(feasibility_calibration_n),
                split_label="finite_q90_feasibility_min9_calibration",
            )
            for table_name in ("intervals", "station_assignments", "calibration_scores"):
                feasibility[table_name]["country"] = country
                feasibility[table_name]["cohort_label"] = test_target_status["status"]
                feasibility[table_name]["profile"] = profile
                feasibility[table_name]["task"] = task
                feasibility[table_name]["fold_id"] = str(args.external_year)
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
            feasibility_model_dir = out / "models" / "conformal_feasibility"
            feasibility_model_dir.mkdir(parents=True, exist_ok=True)
            feasibility_model_path = feasibility_model_dir / (
                f"{country}_{task}_full_finite_q90_feasibility.joblib"
            )
            joblib.dump(feasibility["bundle"], feasibility_model_path, compress=3)
            feasibility_proper_ids = set(
                feasibility["station_assignments"].loc[
                    feasibility["station_assignments"].split_role == "proper_train",
                    "station_id",
                ].astype(str)
            )
            feasibility_proper_train = train.loc[
                train.site_id.astype(str).isin(feasibility_proper_ids)
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
                    feasibility_proper_train, arms["full"], target_col="pm25"
                ),
            }
        profile_oof = pd.DataFrame(profile_rows)
        for (target_strategy, smear_strategy, backtransform), variant in profile_oof.groupby(
            ["target_strategy", "smear_strategy", "backtransform"], sort=True
        ):
            wide = variant.pivot(index="row_id", columns="arm", values="pred")
            aligned = test.set_index("_row_id").loc[wide.index]
            y = aligned.pm25.to_numpy(float)
            groups = aligned.site_id.astype(str).to_numpy()
            available = [arm for arm in ARMS if arm in wide.columns]
            for arm in available:
                m = _metric(y, wide[arm].to_numpy(float))
                metrics_rows.append({
                    "country": country, "cohort_label": test_target_status["status"],
                    "profile": profile, "task": task,
                    "target_strategy": target_strategy, "smear_strategy": smear_strategy,
                    "backtransform": backtransform, "arm": arm,
                    "n": m["n"], "n_stations": int(aligned.site_id.nunique()),
                    "test_years": str(args.external_year),
                    **{k: v for k, v in m.items() if k != "n"},
                })
            for arm_a, arm_b in itertools.combinations(available, 2):
                delta = _paired_rows(
                    variant, country=country, cohort=test_target_status["status"],
                    profile=profile, task=task, arm_a=arm_a, arm_b=arm_b,
                    prediction_a=wide[arm_a].to_numpy(float),
                    prediction_b=wide[arm_b].to_numpy(float), y=y, groups=groups,
                    reps=args.bootstrap_replicates, seed=len(delta_rows),
                    bootstrap_engine=args.bootstrap_engine,
                )
                delta.update({
                    "target_strategy": target_strategy,
                    "smear_strategy": smear_strategy,
                    "backtransform": backtransform,
                })
                delta_rows.append(delta)
    pd.DataFrame(metrics_rows).to_csv(out / "external_metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out / "paired_deltas.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_summaries.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(out / "screening_inner_cv.csv", index=False)
    oof = pd.DataFrame(oof_rows)
    oof.to_parquet(out / "external_oof_predictions.parquet", index=False)
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
        "split_unit": "physical station ID, deterministic within the frozen training period",
        "score": "maximum absolute original-scale residual across calibration months per station",
        "quantile": "ceil((n_calibration_stations + 1) * 0.90) order statistic; infinity if rank exceeds calibration count",
        "point_model": "separate 80% proper-training full-arm log-smear model with station-cross-fitted smearing",
        "interval": "[max(0, point_prediction - q90), point_prediction + q90]",
        "coverage_limits": "T4 reports observed coverage for the inspected external year; no new-year guarantee.",
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
        "coverage_limits": "Sensitivity allocation chosen for finite-sample q90 rank feasibility; observed held-out coverage is limited to this external year.",
        "models": conformal_feasibility_model_manifest,
    })
    manifest = {
        "protocol": "T4 external year, train <= cutoff, test exactly one later year",
        "country": country, "train_through": args.train_through,
        "external_year": args.external_year, "profiles": profiles,
        "profile_sample_sizes": profile_sizes, "model_config": config.__dict__,
        "estimator_kind": args.estimator_kind,
        "estimator_class": ESTIMATOR_CLASS_NAMES[args.estimator_kind],
        "runtime": runtime_provenance(Path(__file__).resolve(), args.bootstrap_engine),
        "bootstrap": bootstrap_provenance(args.bootstrap_engine),
        "smear_strategies": args.smear_strategies,
        "target_strategies": args.target_strategies,
        "train_target_source": train_target_source,
        "external_target_source": external_target_source,
        "training_features": {
            **training_features_source, "rows_source": int(len(train_all)),
        },
        "external_features": {
            **external_features_source, "rows_source": int(len(test_all)),
        },
        "training_input_provenance": training_input_provenance,
        "external_input_provenance": external_input_provenance,
        FINAL_EXPECTED_KEYS_FIELD: expected_test_key_commitment,
        "schema": schema_manifest, "models": model_manifest,
        "conformal_models": conformal_model_manifest,
        "split_conformal_sensitivity": {
            "nominal_coverage": CONFORMAL_COVERAGE,
            "proper_train_fraction": PROPER_TRAIN_FRACTION,
            "score": CONFORMAL_SCORE,
            "coverage_is_reported_for_observed_external_year_only": True,
        },
        "split_conformal_feasibility_sensitivity": {
            "split_label": "finite_q90_feasibility_min9_calibration",
            "minimum_calibration_stations": FEASIBILITY_MIN_CALIBRATION_STATIONS,
            "minimum_proper_training_stations": FEASIBILITY_MIN_PROPER_STATIONS,
            "calibration_allocation": "max(ceil(0.20 * outer_train_station_n), 9)",
            "eligibility_rule": "outer training must contain at least 11 stations",
            "coverage_is_reported_for_observed_external_year_only": True,
        },
        "training_support_diagnostics": {
            "distance_cutoff_km": SUPPORT_DISTANCE_KM,
            "range_quantiles": [SUPPORT_LOWER_QUANTILE, SUPPORT_UPPER_QUANTILE],
            "range_reference": "training rows through the frozen cutoff only",
            "geography_reported_separately_from_environmental_features": True,
            "stratum_metrics_have_no_separate_intervals": True,
        },
        "external_year_interval_limit": (
            "Station bootstrap intervals are conditional on this one external year; "
            "year-level temporal uncertainty cannot be estimated from one holdout year."
        ),
    }
    _assert_source_unchanged(training_features_source, "training feature")
    _assert_source_unchanged(external_features_source, "external feature")
    _assert_source_unchanged(train_target_source, "training target")
    _assert_source_unchanged(external_target_source, "external target")
    if training_input_provenance is not None:
        _assert_source_unchanged(training_input_provenance, "training input provenance")
    if external_input_provenance is not None:
        _assert_source_unchanged(external_input_provenance, "external input provenance")
    expected_key_path = Path(expected_test_key_commitment["path"])
    if sha256_file(expected_key_path) != expected_test_key_commitment["sha256"]:
        raise ValueError("final expected-test-key artifact changed during the external run")
    (out / "external_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"WROTE {out.resolve()}")
    print(pd.DataFrame(metrics_rows).query(
        "target_strategy == 'log_smear' and smear_strategy == 'crossfit_station' and "
        "backtransform == 'smearing' and arm == 'full'"
    )[["country", "profile", "n", "rmse", "mae", "bias", "r2"]].to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True)
    parser.add_argument("--training-features", required=True)
    parser.add_argument("--external-features", required=True)
    parser.add_argument("--training-target-override")
    parser.add_argument("--external-target-override")
    parser.add_argument("--training-target-status", default="precomputed_training_target")
    parser.add_argument("--external-target-status", default="external_target_unverified")
    parser.add_argument("--training-input-provenance")
    parser.add_argument("--external-input-provenance")
    parser.add_argument("--override-target-column", default="pm25")
    parser.add_argument("--train-through", type=int, default=2022)
    parser.add_argument("--external-year", type=int, default=2023)
    parser.add_argument("--profiles", default="with_met,no_met")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument(
        "--estimator-kind", choices=tuple(ESTIMATOR_CLASS_NAMES), default="hgb",
        help="model-class sensitivity; Ridge uses median imputation, standardization, alpha=1.0",
    )
    parser.add_argument("--smear-strategies", default="crossfit_station,in_sample")
    parser.add_argument("--smear-splits", type=int, default=3)
    parser.add_argument("--target-strategies", default="log_smear,original_scale")
    parser.add_argument("--crossfit-smear-arms", default="B,B+AOD,B+fire,full")
    parser.add_argument("--in-sample-smear-arms", default="full")
    parser.add_argument("--original-scale-arms", default="full")
    parser.add_argument("--min-train-stations", type=int, default=10)
    parser.add_argument("--min-test-stations", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument(
        "--bootstrap-engine", choices=BOOTSTRAP_ENGINES, default="reference",
        help="station-cluster interval implementation; cluster_stats is an equivalent fast aggregation",
    )
    args = parser.parse_args()
    args.profiles = [s.strip() for s in args.profiles.split(",") if s.strip()]
    args.smear_strategies = [s.strip() for s in args.smear_strategies.split(",") if s.strip()]
    args.target_strategies = [s.strip() for s in args.target_strategies.split(",") if s.strip()]
    args.crossfit_smear_arms = [s.strip() for s in args.crossfit_smear_arms.split(",") if s.strip()]
    args.in_sample_smear_arms = [s.strip() for s in args.in_sample_smear_arms.split(",") if s.strip()]
    args.original_scale_arms = [s.strip() for s in args.original_scale_arms.split(",") if s.strip()]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

