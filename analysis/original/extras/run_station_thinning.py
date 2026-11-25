"""Fail-closed station-thinning sensitivity for the final USA/CAN T3 split.

No split is generated here. Inputs must bind the accepted strict-QA feature and
target sources to the exact T3/2022 split, assignments, and OOF targets from the
final core validation run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if HERE.parent.name == "repair_v2":
    WORKSPACE = HERE.parents[2]
elif HERE.parent.name == "work":
    WORKSPACE = HERE.parent.parent
else:
    WORKSPACE = HERE.parent
VALIDATION_DIR = Path(os.environ.get(
    "CAND1_FINAL_VALIDATION_DIR",
    str(WORKSPACE / "work" / "repair_v2" / "validation"),
)).expanduser().resolve()
DEFAULT_OUTPUT_ROOT = WORKSPACE / "outputs" / "repair_v2" / "station_thinning"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import run_validation as core_runner  # noqa: E402
import validation_core as model_core  # noqa: E402


FORMAT = "repair-v2-station-thinning-input-manifest-v1"
OUTPUT_FORMAT = "repair-v2-station-thinning-run-manifest-v1"
FINAL_CANDIDATE_FORMAT = "repair-v2-final-historical-source-candidate-v1"
FINAL_ACCEPTANCE_FORMAT = "repair-v2-final-historical-independent-qa-acceptance-v1"
SPLIT_FORMAT = "repair-v2-outer-fold-split-manifest-v1"
CANONICAL_GEOMETRY_SHA256 = "f1ec94d31019a2776bcd19e92e224773c972ed0b68069756e848b0a361444a5e"
CORE_TASK = "T3_year_2022"
CORE_FOLD = "2022"
TRAIN_YEARS = (2018, 2019, 2020, 2021)
TEST_YEAR = 2022
SAMPLE_SIZES = (5, 10, 25, 50, 100)
SEEDS = (20261113, 20261114, 20261115)
PROFILES = ("with_met", "no_met")
ARMS = ("B", "full")
DATASETS = {
    "usa_hist": ("USA", "EPAclean"),
    "can_hist_primary": ("CAN", "NAPS_all_published_primary"),
}
QA_FIELDS = {
    "overlay_status": "geometry_verified_source_coverage_audited",
    "geometry_complete": True,
    "source_coverage_audited": True,
    "raw_source_gap_audit_pending": False,
}
MODEL_CONFIG = asdict(model_core.ModelConfig())
SPLIT_CHECK_FIELDS = (
    "train_row_count", "test_row_count", "train_station_count", "test_station_count",
    "train_row_key_sha256", "test_row_key_sha256", "analysis_row_key_sha256",
    "train_row_ids_sha256", "test_row_ids_sha256", "train_year_counts", "test_year_counts",
    "training_support_fingerprint",
)


class PreflightError(ValueError):
    """A missing or contradictory input binding; execution must stop."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} JSON must contain an object")
    return value


def resolve_path(raw: str | Path, base: Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def verify_ref(
    ref: Mapping[str, Any], *, label: str, base: Path,
    accepted_hashes: set[str] | None = None,
) -> tuple[Path, str]:
    if not isinstance(ref, Mapping) or not ref.get("path") or not ref.get("sha256"):
        raise PreflightError(f"{label} must include path and sha256")
    path = resolve_path(str(ref["path"]), base)
    expected = str(ref["sha256"]).lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise PreflightError(f"{label} has an invalid SHA-256")
    if not path.is_file():
        raise PreflightError(f"{label} is absent: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise PreflightError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    if accepted_hashes is not None and actual not in accepted_hashes:
        raise PreflightError(f"{label} is not covered by the QA acceptance artifact")
    return path, actual


def load_json_ref(
    ref: Mapping[str, Any], *, label: str, base: Path,
) -> tuple[dict[str, Any], Path, str]:
    path, digest = verify_ref(ref, label=label, base=base)
    return read_json(path, label), path, digest


def read_table(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    raise PreflightError(f"unsupported tabular input extension: {path}")


def assert_historical_year_domain(path: Path, *, label: str) -> set[int]:
    """Inspect only the year column before any PM2.5 values are loaded."""
    try:
        years = pd.to_numeric(read_table(path, columns=["year"])["year"], errors="coerce")
    except Exception as exc:
        raise PreflightError(f"cannot read only the year column from {label}: {exc}") from exc
    if years.empty or years.isna().any() or not np.isfinite(years.to_numpy(dtype=float)).all():
        raise PreflightError(f"{label} has an empty or invalid year column")
    if not np.equal(years.to_numpy(dtype=float), np.floor(years.to_numpy(dtype=float))).all():
        raise PreflightError(f"{label} contains non-integer years")
    values = set(years.astype(int).tolist())
    outside = sorted(values - set(range(2018, 2023)))
    if outside:
        raise PreflightError(
            f"{label} has years outside the locked 2018-2022 period: {outside}; "
            "PM2.5 values were not loaded"
        )
    return values


def expected_features(profile: str) -> dict[str, list[str]]:
    if profile not in PROFILES:
        raise PreflightError(f"unsupported meteorology profile: {profile}")
    met = list(core_runner.MET) if profile == "with_met" else []
    base = list(core_runner.GEO) + met
    return {"B": base, "full": base + list(core_runner.AOD) + list(core_runner.FIRE)}


def nested_station_samples(
    train_site_ids: Sequence[str], *, seeds: Sequence[int] = SEEDS,
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
) -> dict[int, dict[int, list[str]]]:
    """Return nested samples from site IDs only; no row values or outcomes enter."""
    sites = sorted({str(site) for site in train_site_ids})
    if not sites:
        raise PreflightError("cannot sample an empty training-site network")
    result: dict[int, dict[int, list[str]]] = {}
    for seed in seeds:
        permutation = np.random.default_rng(int(seed)).permutation(np.asarray(sites, dtype=object))
        ordered = [str(site) for site in permutation.tolist()]
        per_seed: dict[int, list[str]] = {}
        previous: set[str] = set()
        for size in sample_sizes:
            if int(size) > len(ordered):
                continue
            selected = ordered[: int(size)]
            selected_set = set(selected)
            if len(selected_set) != int(size) or not previous.issubset(selected_set):
                raise AssertionError("nested physical-site sampling invariant failed")
            previous = selected_set
            per_seed[int(size)] = selected
        result[int(seed)] = per_seed
    return result


def select_site_rows(train: pd.DataFrame, sites: Sequence[str]) -> pd.DataFrame:
    requested = {str(site) for site in sites}
    selected = train.loc[train.site_id.astype(str).isin(requested)].copy()
    observed = set(selected.site_id.astype(str))
    if observed != requested:
        raise PreflightError(f"sampled site rows differ from requested site IDs: {sorted(requested ^ observed)[:5]}")
    return selected.reset_index(drop=True)


def metric_values(y: Sequence[float], prediction: Sequence[float]) -> dict[str, float | int]:
    return core_runner._metric(np.asarray(y, dtype=float), np.asarray(prediction, dtype=float))


def fit_train_only(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], *,
    fit_model: Callable[..., Mapping[str, Any]] | None = None,
    predict_model: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Fit and crossfit-smear from the supplied training rows only."""
    fit_func = fit_model or model_core.fit_model
    predict_func = predict_model or model_core.predict_model
    bundle = fit_func(
        train, list(features), target_col="pm25", group_col="site_id",
        config=model_core.ModelConfig(), smear_strategy="crossfit_station",
        smear_splits=3, target_strategy="log_smear",
    )
    smear_info = bundle.get("smear_info", {})
    if smear_info.get("method") != "crossfit_station" or int(smear_info.get("n_splits", 0)) < 2:
        raise PreflightError("model did not use training-only station crossfit smearing")
    prediction = np.asarray(predict_func(bundle, test, backtransform="smearing"), dtype=float)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise PreflightError("model returned invalid or incomplete held-out predictions")
    return prediction, bundle


def _validate_candidate_acceptance(
    manifest: Mapping[str, Any], manifest_base: Path,
) -> tuple[dict[str, Any], Path, set[str], dict[str, Any]]:
    candidate, candidate_path, candidate_sha = load_json_ref(
        manifest.get("candidate_manifest", {}), label="final source candidate manifest", base=manifest_base,
    )
    acceptance, acceptance_path, _ = load_json_ref(
        manifest.get("acceptance_manifest", {}), label="QA acceptance manifest", base=manifest_base,
    )
    if candidate.get("format") != FINAL_CANDIDATE_FORMAT or candidate.get("candidate_state") != "final":
        raise PreflightError("final source candidate is absent or not explicitly final")
    if candidate.get("period") != "2018-2022":
        raise PreflightError("final source candidate period must be exactly 2018-2022")
    if acceptance.get("format") != FINAL_ACCEPTANCE_FORMAT:
        raise PreflightError("unsupported QA acceptance format")
    if acceptance.get("decision") != "accepted" or acceptance.get("independent_qa_pass") is not True:
        raise PreflightError("QA acceptance is not an explicit pass")
    if str(acceptance.get("candidate_manifest_sha256", "")).lower() != candidate_sha:
        raise PreflightError("QA acceptance does not bind the final candidate manifest")
    if not str(acceptance.get("QA signatory", "")).strip():
        raise PreflightError("QA acceptance lacks QA signatory identity")
    accepted_artifacts = acceptance.get("accepted_artifacts")
    if not isinstance(accepted_artifacts, list) or not accepted_artifacts:
        raise PreflightError("QA acceptance does not hash-bind accepted artifacts")
    accepted_hashes: set[str] = set()
    for index, ref in enumerate(accepted_artifacts):
        _, digest = verify_ref(ref, label=f"accepted_artifacts[{index}]", base=acceptance_path.parent)
        accepted_hashes.add(digest)
    evidence = acceptance.get("review_evidence")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise PreflightError("QA acceptance requires at least two QA evidence references")
    for index, ref in enumerate(evidence):
        verify_ref(
            ref, label=f"QA evidence[{index}]", base=acceptance_path.parent,
            accepted_hashes=accepted_hashes,
        )
    datasets = candidate.get("datasets")
    expected_candidate_keys = {
        "usa_hist", "can_hist_primary", "can_common_all_published", "can_common_exclude_236_238",
    }
    if not isinstance(datasets, dict) or set(datasets) != expected_candidate_keys:
        raise PreflightError("final candidate dataset registry differs from the locked historical candidate contract")
    geometry = candidate.get("canonical_geometry_manifest")
    _, geometry_hash = verify_ref(
        geometry, label="canonical geometry manifest", base=candidate_path.parent,
        accepted_hashes=accepted_hashes,
    )
    if geometry_hash != CANONICAL_GEOMETRY_SHA256:
        raise PreflightError("canonical geometry manifest hash differs from the locked historical universe")
    return candidate, candidate_path, accepted_hashes, acceptance


def _validate_cohort_candidate(
    candidate: Mapping[str, Any], candidate_path: Path, accepted_hashes: set[str],
    dataset_key: str,
) -> tuple[dict[str, Any], Path, Path | None, dict[str, Any]]:
    country, target_policy = DATASETS[dataset_key]
    record = candidate["datasets"][dataset_key]
    if record.get("country") != country or record.get("target_policy") != target_policy:
        raise PreflightError(f"{dataset_key} country/target policy differs from the locked primary cohort")
    if not str(record.get("target_status", "")).strip():
        raise PreflightError(f"{dataset_key} lacks a final target-status declaration")
    target_source_ref = record.get("target_source")
    target_path: Path | None = None
    if country == "USA" and not isinstance(target_source_ref, dict):
        raise PreflightError("USA EPAclean cohort requires its hash-bound final target override")
    if target_source_ref is not None:
        target_path, _ = verify_ref(
            target_source_ref, label=f"{dataset_key} target source", base=candidate_path.parent,
            accepted_hashes=accepted_hashes,
        )
        assert_historical_year_domain(target_path, label=f"{dataset_key} target source")
    variants = record.get("variants")
    if not isinstance(variants, dict) or "strict" not in variants:
        raise PreflightError(f"{dataset_key} final candidate lacks its strict-QA variant")
    variant = variants["strict"]
    feature_path, feature_sha = verify_ref(
        variant.get("feature_table", {}), label=f"{dataset_key} strict feature table",
        base=candidate_path.parent, accepted_hashes=accepted_hashes,
    )
    assert_historical_year_domain(feature_path, label=f"{dataset_key} strict feature table")
    provenance_path, provenance_sha = verify_ref(
        variant.get("qa_provenance", {}), label=f"{dataset_key} strict QA provenance",
        base=candidate_path.parent, accepted_hashes=accepted_hashes,
    )
    merge_path, merge_sha = verify_ref(
        variant.get("qa_merge_manifest", {}), label=f"{dataset_key} strict QA merge manifest",
        base=candidate_path.parent, accepted_hashes=accepted_hashes,
    )
    provenance = read_json(provenance_path, f"{dataset_key} strict QA provenance")
    if provenance.get("aod_variant") != "strict":
        raise PreflightError(f"{dataset_key} QA provenance does not declare strict AOD QA")
    for field, expected in QA_FIELDS.items():
        if provenance.get(field) != expected:
            raise PreflightError(f"{dataset_key} QA provenance fails {field}={expected!r}")
    p_feature = provenance.get("feature_table", {})
    p_merge = provenance.get("qa_merge_manifest", {})
    if str(p_feature.get("sha256", "")).lower() != feature_sha:
        raise PreflightError(f"{dataset_key} QA provenance does not bind its strict feature table")
    if str(p_merge.get("sha256", "")).lower() != merge_sha:
        raise PreflightError(f"{dataset_key} candidate/provenance QA merge hashes differ")
    return record, feature_path, target_path, {
        "feature_sha256": feature_sha, "target_source_sha256": (
            str(target_source_ref["sha256"]).lower() if target_source_ref is not None else None
        ),
        "qa_provenance_sha256": provenance_sha, "qa_merge_manifest_sha256": merge_sha,
        "feature_path": str(feature_path), "target_source_path": str(target_path) if target_path else None,
    }


def _load_core_rows(
    *, country: str, dataset_record: Mapping[str, Any], feature_path: Path,
    target_path: Path | None, profile_source_records: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Recreate run_validation's input cleaning and row IDs from final inputs."""
    feature_hashes = []
    first_record = profile_source_records[0]
    expected_paths = first_record.get("feature_tables", [])
    expected_hashes = first_record.get("feature_sha256", [])
    if len(expected_paths) != 1 or len(expected_hashes) != 1:
        raise PreflightError("station-thinning requires exactly one core feature table for this cohort")
    feature_sha = sha256_file(feature_path)
    if str(expected_hashes[0]).lower() != feature_sha:
        raise PreflightError("final candidate feature table hash differs from the core source manifest")
    feature_hashes.append(feature_sha)
    for source_record in profile_source_records[1:]:
        if source_record.get("feature_sha256") != first_record.get("feature_sha256"):
            raise PreflightError("with_met and no_met core runs use different feature-table hashes")
    target_status_record = first_record.get("target_status", {})
    target_source_sha = first_record.get("target_source_sha256")
    candidate_target_sha = sha256_file(target_path) if target_path else None
    if target_source_sha != candidate_target_sha:
        raise PreflightError("final candidate target-source hash differs from the core source manifest")
    if target_status_record.get("status") != dataset_record.get("target_status"):
        raise PreflightError("final candidate target-status differs from the core source manifest")
    if target_path and str(target_status_record.get("sha256", "")).lower() != candidate_target_sha:
        raise PreflightError("core target provenance hash differs from the final target source")
    for source_record in profile_source_records[1:]:
        if source_record.get("target_source_sha256") != target_source_sha:
            raise PreflightError("with_met and no_met core runs use different target-source hashes")

    frame, paths = core_runner.load_input_tables(
        Path("."), {country: [feature_path]}, [country],
    )
    if country not in frame or paths.get(country) != [feature_path]:
        raise PreflightError("core runner did not load the accepted final feature table")
    d = frame[country].copy()
    d["country"] = country
    if target_path is not None:
        d, _ = core_runner.replace_targets(d, target_path, country=country, target_column="pm25")

    features = expected_features("with_met")["full"]
    all_model_features = sorted(set(
        expected_features("with_met")["B"] + expected_features("with_met")["full"]
    ))
    required = ["country", "site_id", "lat", "lon", "year", "month", "pm25", "aod_raw", *all_model_features]
    missing = sorted(set(required) - set(d.columns))
    if missing:
        raise PreflightError(f"final feature table lacks fixed model columns: {missing}")
    d["year"] = pd.to_numeric(d.year, errors="coerce")
    d["month"] = pd.to_numeric(d.month, errors="coerce")
    d["pm25"] = pd.to_numeric(d.pm25, errors="coerce")
    coord_ok = np.isfinite(pd.to_numeric(d.lat, errors="coerce")) & np.isfinite(
        pd.to_numeric(d.lon, errors="coerce")
    )
    target_ok = np.isfinite(d.pm25) & (d.pm25 >= 0)
    d = d.loc[coord_ok & target_ok].copy()
    if d.empty:
        raise PreflightError("final core input has no finite nonnegative target rows")
    d["year"] = d.year.astype(int)
    d["month"] = d.month.astype(int)
    d["site_id"] = d.site_id.astype(str)
    if not set(d.year.astype(int)).issubset(set(range(2018, 2023))):
        raise PreflightError("cleaned core rows extend outside 2018-2022")
    d.reset_index(drop=True, inplace=True)
    d["_row_id"] = np.arange(len(d), dtype=np.int64)
    d, _ = core_runner._duplicate_audit(d, all_model_features)
    d.reset_index(drop=True, inplace=True)
    d["_row_id"] = np.arange(len(d), dtype=np.int64)
    if d.duplicated(["country", "site_id", "year", "month"]).any():
        raise PreflightError("core preprocessing left duplicate station-month keys")
    return d, feature_hashes, target_source_sha


def _validate_model_config(run_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], profile: str) -> None:
    for label, value in (("run", run_manifest.get("model_config")), ("split", split_manifest.get("model_config"))):
        if not isinstance(value, Mapping) or set(MODEL_CONFIG) - set(value):
            raise PreflightError(f"core {label} manifest omits fixed ModelConfig fields")
        try:
            effective = asdict(model_core.ModelConfig(**dict(value or {})))
        except (TypeError, ValueError) as exc:
            raise PreflightError(f"invalid {label} ModelConfig in core manifest: {exc}") from exc
        if effective != MODEL_CONFIG:
            raise PreflightError(f"core {label} ModelConfig differs from the locked HGB configuration")
    if run_manifest.get("estimator_kind") != "hgb":
        raise PreflightError("core T3 run is not the fixed HGB primary estimator")
    if int(run_manifest.get("smear_splits", -1)) != 3:
        raise PreflightError("core runner did not use three station crossfit smearing groups")
    for field, needed in (("crossfit_smear_arms", set(ARMS)), ("target_strategies", {"log_smear"})):
        if not needed.issubset(set(run_manifest.get(field, []))):
            raise PreflightError(f"core run manifest lacks required {field}: {sorted(needed)}")
    source_selection = run_manifest.get("country_meteorology_profile_selection")
    if profile == "no_met":
        if source_selection != "no_met_override":
            raise PreflightError("core run does not match the declared no_met meteorology profile")
        request = run_manifest.get("country_meteorology_profile_request")
        if request is not None:
            effective = request.get("effective_features_by_country") if isinstance(request, Mapping) else None
            oof = set(run_manifest.get("countries_with_oof", []))
            if (not isinstance(request, Mapping)
                or request.get("requested_profile") != "source_declared"
                or request.get("no_met") is not True
                or not isinstance(effective, Mapping) or not effective
                or not oof.issubset(effective)
                or any(country not in core_runner.COUNTRIES or features != []
                       for country, features in effective.items())):
                raise PreflightError("no_met core run has a conflicting meteorology profile request")
        return

    if profile != "with_met":
        raise PreflightError(f"unsupported station-thinning meteorology profile: {profile}")
    if source_selection == "source_declared_fixed_profile":
        # Older accepted core runs predate the explicit request record.
        request = run_manifest.get("country_meteorology_profile_request")
        if request is not None and (
            not isinstance(request, Mapping)
            or request.get("requested_profile") != "source_declared"
            or request.get("no_met") is not False
            or not isinstance(request.get("effective_features_by_country"), Mapping)
            or not request.get("effective_features_by_country")
            or not set(run_manifest.get("countries_with_oof", [])).issubset(request["effective_features_by_country"])
            or any(country not in core_runner.COUNTRIES or features != list(core_runner.MET)
                   for country, features in request["effective_features_by_country"].items())
        ):
            raise PreflightError("source-declared core run has a conflicting meteorology profile request")
        return
    if source_selection != "explicit_era5_six":
        raise PreflightError("core run does not match the declared with_met meteorology profile")

    request = run_manifest.get("country_meteorology_profile_request")
    if not isinstance(request, Mapping):
        raise PreflightError("explicit ERA5 six-field core run lacks its meteorology profile request")
    if request.get("requested_profile") != "era5_six" or request.get("no_met") is not False:
        raise PreflightError("explicit ERA5 six-field core run has a conflicting meteorology profile request")
    effective_by_country = request.get("effective_features_by_country")
    fixed_met = list(core_runner.MET)
    oof_countries = {
        str(country).strip().upper()
        for country in run_manifest.get("countries_with_oof", [])
    }
    if (
        not isinstance(effective_by_country, Mapping)
        or not effective_by_country
        or any(
            not isinstance(country, str)
            or country != country.strip().upper()
            or country not in core_runner.COUNTRIES
            or features != fixed_met
            for country, features in effective_by_country.items()
        )
        or not oof_countries.issubset(effective_by_country)
    ):
        raise PreflightError("explicit ERA5 six-field request does not declare the fixed meteorology profile for every country")


def _validate_runtime(run_manifest: Mapping[str, Any]) -> None:
    recorded = run_manifest.get("runtime", {}).get("code_sha256", {})
    expected = {
        "runner": sha256_file(VALIDATION_DIR / "run_validation.py"),
        "run_validation_helpers": sha256_file(VALIDATION_DIR / "run_validation.py"),
        "validation_core": sha256_file(VALIDATION_DIR / "validation_core.py"),
    }
    for name, digest in expected.items():
        if recorded.get(name) != digest:
            raise PreflightError(f"core runtime hash for {name} differs from the audited local implementation")


def _validate_schema(
    schema_manifest: Mapping[str, Any], country: str, profile: str,
) -> dict[str, list[str]]:
    features = expected_features(profile)
    for arm in ARMS:
        record = schema_manifest.get(f"{country}:{arm}")
        expected = model_core.schema_for(
            features[arm], {"aod_used": "aod_raw"} if "aod_used" in features[arm] else None,
        )
        if not isinstance(record, Mapping):
            raise PreflightError(f"core schema manifest lacks {country}:{arm}")
        for field in ("schema_version", "base_features", "indicator_source_by_feature", "required_source_columns", "indicator_columns", "model_columns"):
            if record.get(field) != expected.get(field):
                raise PreflightError(f"core {country}:{arm} schema differs from the locked {profile} schema ({field})")
        expected_met = list(core_runner.MET) if profile == "with_met" else []
        if record.get("met_features") != expected_met or record.get("source") != "amer":
            raise PreflightError(f"core {country}:{arm} schema declares an unexpected source/met profile")
    return features


def _split_record(split_manifest: Mapping[str, Any], country: str) -> dict[str, Any]:
    matches = [record for record in split_manifest.get("splits", []) if (
        str(record.get("country", "")).upper() == country
        and str(record.get("task")) == CORE_TASK
        and str(record.get("fold_id")) == CORE_FOLD
    )]
    if len(matches) != 1:
        raise PreflightError(f"core split has {len(matches)} exact {country}/{CORE_TASK}/{CORE_FOLD} records")
    return dict(matches[0])


def validate_assignment_collisions(assignments: pd.DataFrame) -> None:
    """Validate the run_validation outer-test assignment uniqueness contract."""
    required = {"country", "task", "fold_id", "row_id", "role"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise PreflightError(f"core outer-fold assignment table lacks fields: {missing}")
    test = assignments.loc[assignments.role.astype(str).eq("test")]
    if test.duplicated(["country", "task", "row_id"], keep=False).any():
        duplicate_rows = test.loc[
            test.duplicated(["country", "task", "row_id"], keep=False),
            ["country", "task", "fold_id", "row_id"],
        ].head(6)
        raise PreflightError(
            "core test row is assigned to multiple folds or duplicated within a fold: "
            f"{duplicate_rows.to_dict('records')}"
        )


def assert_split_record_matches(
    record: Mapping[str, Any], *, train: pd.DataFrame, test: pd.DataFrame,
    all_rows: pd.DataFrame, full_features: list[str], feature_hashes: list[str],
    target_sha: str | None,
) -> None:
    """Recompute the audited core split fingerprint from exact keyed rows."""
    try:
        recomputed = core_runner.outer_fold_split_record(
            country=str(record.get("country", "")).upper(),
            cohort_label=str(record.get("cohort_label", "")),
            task=str(record.get("task", "")), fold_id=str(record.get("fold_id", "")),
            buffer_km=float(record.get("buffer_km", 0.0)), train=train, test=test,
            all_rows=all_rows, full_features=full_features,
            feature_source_sha256=feature_hashes, target_source_sha256=target_sha,
        )
    except Exception as exc:
        raise PreflightError(f"could not recompute the exact core T3 split fingerprint: {exc}") from exc
    for field in SPLIT_CHECK_FIELDS:
        if record.get(field) != recomputed.get(field):
            raise PreflightError(f"core split {field} differs from the accepted final input")


def assert_oof_targets_match(test: pd.DataFrame, predictions: pd.DataFrame, *, arm: str) -> None:
    """Require the bound core OOF keys and observed targets to match the final rows."""
    selected = predictions.copy()
    selected["row_id"] = pd.to_numeric(selected.row_id, errors="raise").astype(int)
    if selected.row_id.duplicated().any():
        raise PreflightError(f"core OOF has duplicated test row IDs for arm={arm}")
    expected = test.sort_values("_row_id", kind="mergesort").reset_index(drop=True)
    selected.sort_values("row_id", inplace=True, kind="mergesort")
    if not np.array_equal(selected.row_id.to_numpy(int), expected._row_id.to_numpy(int)):
        raise PreflightError(f"core OOF test row IDs differ for arm={arm}")
    for field in ("site_id", "year", "month"):
        if not np.array_equal(selected[field].astype(str).to_numpy(), expected[field].astype(str).to_numpy()):
            raise PreflightError(f"core OOF station-month keys differ for arm={arm}/{field}")
    obs = pd.to_numeric(selected.obs, errors="coerce").to_numpy(float)
    if not np.array_equal(obs, expected.pm25.to_numpy(float)):
        raise PreflightError(f"core OOF targets differ from the accepted final input for arm={arm}")


def _validate_model_binding(
    record: Mapping[str, Any], source_record: Mapping[str, Any], schema: Mapping[str, Any],
    feature_hashes: list[str], target_sha: str | None, run_runtime: Mapping[str, Any],
    train: pd.DataFrame, model_artifact_sha256: str,
) -> None:
    model_record = record.get("full_model")
    if not isinstance(model_record, Mapping):
        raise PreflightError("core T3/2022 split lacks its full-arm model binding")
    binding = model_record.get("binding")
    if not isinstance(binding, Mapping) or json_sha256(binding) != model_record.get("binding_sha256"):
        raise PreflightError("core full-model binding is absent or its fingerprint is invalid")
    expected_sources = {"features": feature_hashes, "targets": target_sha}
    raw_y = pd.to_numeric(train.pm25, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(raw_y).all() or np.any(raw_y < 0):
        raise PreflightError("core model binding training targets are invalid")
    model_y = np.log(raw_y + 0.1)
    matrix_sha = core_runner._array_sha256(model_core.feature_matrix(train, schema))
    raw_y_sha = core_runner._array_sha256(raw_y)
    model_y_sha = core_runner._array_sha256(model_y)
    groups_sha = json_sha256(train.site_id.astype(str).tolist())
    expected = {
        "feature_schema": dict(schema), "model_config": MODEL_CONFIG,
        "target_strategy": "log_smear", "smear_strategy": "crossfit_station",
        "estimator_kind": "hgb", "estimator_class": "HistGradientBoostingRegressor",
        "training_support_fingerprint": record.get("training_support_fingerprint"),
        "training_row_key_sha256": record.get("train_row_key_sha256"),
        "source_hashes": expected_sources, "runtime": dict(run_runtime),
        "ordered_feature_matrix_sha256": matrix_sha,
        "raw_target_y_sha256": raw_y_sha,
        "model_target_y_sha256": model_y_sha,
        "ordered_group_ids_sha256": groups_sha,
        "model_artifact_sha256": model_artifact_sha256,
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise PreflightError(f"core full-model binding differs for {field}")
    smear_info = binding.get("smear_info", {})
    if not isinstance(smear_info, Mapping):
        raise PreflightError("core full-model smearing evidence is not an object")
    smear = pd.to_numeric(pd.Series([binding.get("smear")]), errors="coerce").iloc[0]
    if (smear_info.get("method") != "crossfit_station"
            or int(smear_info.get("n_splits", 0)) != 3
            or int(smear_info.get("n_rows", -1)) != len(train)
            or int(smear_info.get("n_stations", -1)) != int(train.site_id.nunique())
            or not np.isfinite(smear) or float(smear) <= 0):
        raise PreflightError("core full model does not bind independent three-fold training-only smearing")
    candidates = [item for item in source_record.get("outer_fold_models", []) if (
        str(item.get("task")) == CORE_TASK and str(item.get("fold_id")) == CORE_FOLD
        and item.get("target_strategy") == "log_smear"
        and item.get("smear_strategy") == "crossfit_station"
    )]
    if len(candidates) != 1:
        raise PreflightError("core source manifest lacks one unambiguous registered full T3 model")
    model = candidates[0]
    if (model.get("sha256") != model_record.get("sha256")
            or model.get("sha256") != model_artifact_sha256
            or str(model.get("path")) != str(model_record.get("path"))
            or model.get("binding_sha256") != model_record.get("binding_sha256")
            or model.get("training_support_fingerprint") != record.get("training_support_fingerprint")
            or model.get("schema") != dict(schema)
            or model.get("config") != MODEL_CONFIG
            or model.get("estimator_kind") != "hgb"
            or model.get("estimator_class") != "HistGradientBoostingRegressor"
            or model.get("smear") != binding.get("smear")
            or model.get("smear_info") != smear_info
            or int(model.get("fit_n", -1)) != len(train)
            or int(model.get("fit_stations", -1)) != int(train.site_id.nunique())):
        raise PreflightError("core source manifest model record differs from the split-bound model")


def _validate_core_oof(
    oof_path: Path, *, country: str, test: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    assert_historical_year_domain(oof_path, label=f"{country} core OOF predictions")
    columns = [
        "country", "task", "fold_id", "arm", "target_strategy", "smear_strategy",
        "backtransform", "row_id", "site_id", "year", "month", "obs", "pred",
    ]
    try:
        oof = read_table(oof_path, columns=columns)
    except Exception as exc:
        raise PreflightError(f"cannot read bound core OOF predictions: {exc}") from exc
    outputs: dict[str, pd.DataFrame] = {}
    test_ids = set(test._row_id.astype(int))
    for arm in ARMS:
        selected = oof.loc[
            oof.country.astype(str).str.upper().eq(country)
            & oof.task.astype(str).eq(CORE_TASK)
            & oof.fold_id.astype(str).eq(CORE_FOLD)
            & oof.arm.astype(str).eq(arm)
            & oof.target_strategy.astype(str).eq("log_smear")
            & oof.smear_strategy.astype(str).eq("crossfit_station")
            & oof.backtransform.astype(str).eq("smearing")
        ].copy()
        if selected.empty:
            raise PreflightError(f"core OOF is missing {country}/{CORE_TASK}/{CORE_FOLD}/{arm} predictions")
        selected["row_id"] = pd.to_numeric(selected.row_id, errors="raise").astype(int)
        if set(selected.row_id) != test_ids:
            raise PreflightError(f"core OOF test row IDs differ for arm={arm}")
        assert_oof_targets_match(test, selected, arm=arm)
        outputs[arm] = selected.reset_index(drop=True)
    return outputs


def _load_profile(
    profile_record: Mapping[str, Any], *, profile: str, country: str,
    core_rows: pd.DataFrame, feature_hashes: list[str], target_sha: str | None,
    source_record: Mapping[str, Any], expected_qa_provenance_sha: str, base: Path,
) -> dict[str, Any]:
    refs = {}
    for field in (
        "run_manifest", "source_manifest", "schema_manifest", "split_manifest",
        "assignments", "oof_predictions", "full_model_artifact",
    ):
        if not isinstance(profile_record.get(field), Mapping):
            raise PreflightError(f"{profile} profile input manifest lacks {field} reference")
        refs[field], _ = verify_ref(profile_record[field], label=f"{profile} core {field}", base=base)
    run_manifest = read_json(refs["run_manifest"], f"{profile} core run manifest")
    source_manifest = read_json(refs["source_manifest"], f"{profile} core source manifest")
    schema_manifest = read_json(refs["schema_manifest"], f"{profile} core schema manifest")
    split_manifest = read_json(refs["split_manifest"], f"{profile} core split manifest")
    if split_manifest.get("format") != SPLIT_FORMAT:
        raise PreflightError("unsupported core outer-fold split manifest format")
    run_split = run_manifest.get("outer_fold_split_manifest", {})
    if (run_split.get("sha256") != sha256_file(refs["split_manifest"])
            or run_split.get("assignment_sha256") != sha256_file(refs["assignments"])):
        raise PreflightError("core run manifest does not bind the supplied split and assignment artifacts")
    if split_manifest.get("assignment_sha256") != sha256_file(refs["assignments"]):
        raise PreflightError("core split manifest does not bind the supplied assignment table")
    if int(split_manifest.get("assignment_rows", -1)) != len(read_table(refs["assignments"])):
        raise PreflightError("core assignment row count differs from the split manifest")
    if run_manifest.get("schema_version") != model_core.SCHEMA_VERSION:
        raise PreflightError("core run schema version differs from the audited prediction interface")
    if country not in {str(item).upper() for item in run_manifest.get("countries_with_oof", [])}:
        raise PreflightError(f"core {profile} run does not include {country} OOF rows")
    _validate_model_config(run_manifest, split_manifest, profile)
    _validate_runtime(run_manifest)
    if country not in source_manifest or not isinstance(source_manifest[country], Mapping):
        raise PreflightError(f"core {profile} source manifest lacks {country}")
    if source_manifest[country].get("feature_sha256") != feature_hashes:
        raise PreflightError(f"core {profile} source manifest feature hashes differ from final inputs")
    if source_manifest[country].get("target_source_sha256") != target_sha:
        raise PreflightError(f"core {profile} source manifest target hash differs from final inputs")
    qa_provenance = source_manifest[country].get("feature_input_provenance", {})
    if str(qa_provenance.get("sha256", "")).lower() != expected_qa_provenance_sha:
        raise PreflightError(f"core {profile} source manifest lacks the accepted strict QA provenance binding")
    expected_profiles = source_manifest[country].get("country_met_profile_name")
    if expected_profiles != profile:
        raise PreflightError(f"core {profile} source manifest declares met profile {expected_profiles!r}")
    features = _validate_schema(schema_manifest, country, profile)
    record = _split_record(split_manifest, country)
    if record.get("train_ids_unit") != "row_id" or record.get("test_ids_unit") != "row_id":
        raise PreflightError("T3/2022 core split must bind explicit row IDs")
    if record.get("feature_source_sha256") != feature_hashes or record.get("target_source_sha256") != target_sha:
        raise PreflightError("T3/2022 core split source hashes differ from the accepted final input")
    if str(record.get("split_rule", "")).find("year < 2022") < 0:
        raise PreflightError("core T3/2022 record does not declare the year holdout rule")
    assignments = read_table(refs["assignments"])
    validate_assignment_collisions(assignments)
    required_assignment = {"country", "site_id", "year", "month", "row_id", "task", "fold_id", "role"}
    if not required_assignment.issubset(assignments.columns):
        raise PreflightError("core outer-fold assignment table lacks required columns")
    assignment = assignments.loc[
        assignments.country.astype(str).str.upper().eq(country)
        & assignments.task.astype(str).eq(CORE_TASK)
        & assignments.fold_id.astype(str).eq(CORE_FOLD)
        & assignments.role.astype(str).eq("test")
    ].copy()
    if assignment.empty or assignment.row_id.duplicated().any():
        raise PreflightError("core T3/2022 assignment is empty or has duplicate test row IDs")
    assignment["row_id"] = pd.to_numeric(assignment.row_id, errors="raise").astype("int64")
    train_ids = [int(item) for item in record.get("train_ids", [])]
    if len(train_ids) != len(set(train_ids)) or len(train_ids) != len(record.get("train_ids", [])):
        raise PreflightError("core T3 training row IDs are duplicated")
    test_ids = assignment.row_id.astype(int).tolist()
    if sorted(test_ids) != sorted(int(item) for item in record.get("test_ids", [])):
        raise PreflightError("core split test row IDs differ from actual core assignments")
    if record.get("train_ids_sha256") != json_sha256(sorted(train_ids)):
        raise PreflightError("core T3 training row-ID hash is invalid")
    if record.get("test_ids_sha256") != json_sha256(sorted(test_ids)):
        raise PreflightError("core T3 test row-ID hash is invalid")
    if set(train_ids) & set(test_ids):
        raise PreflightError("core T3 training and test row IDs overlap")
    row_lookup = core_rows.set_index("_row_id", drop=False)
    if not set(train_ids + test_ids).issubset(row_lookup.index):
        raise PreflightError("core T3 assignments refer to unavailable final source rows")
    train = core_rows.loc[core_rows._row_id.isin(train_ids)].copy().reset_index(drop=True)
    test = core_rows.loc[core_rows._row_id.isin(test_ids)].copy().reset_index(drop=True)
    if not train.year.isin(TRAIN_YEARS).all() or not test.year.eq(TEST_YEAR).all():
        raise PreflightError("core T3 partition does not use 2018-2021 train and all eligible 2022 test rows")
    assignment_keys = assignment[["row_id", "country", "site_id", "year", "month"]].copy()
    assigned_source = row_lookup.loc[assignment_keys.row_id].reset_index(drop=True)
    for field in ("country", "site_id", "year", "month"):
        if not np.array_equal(
            assignment_keys[field].astype(str).to_numpy(), assigned_source[field].astype(str).to_numpy(),
        ):
            raise PreflightError(f"core T3 assignment keys differ from source rows for {field}")
    if len(test) != len(assignment) or len(train) != int(record.get("train_row_count", -1)):
        raise PreflightError("core T3 train/test row counts differ from actual partition")
    if set(train._row_id.astype(int)) | set(test._row_id.astype(int)) != set(core_rows._row_id.astype(int)):
        raise PreflightError("core T3 train/test partition does not cover the final 2018-2022 analysis rows")
    full_schema = model_core.schema_for(features["full"], {"aod_used": "aod_raw"})
    model_artifact_sha = sha256_file(refs["full_model_artifact"])
    _validate_model_binding(
        record, source_manifest[country], full_schema, feature_hashes, target_sha,
        run_manifest.get("runtime", {}), train, model_artifact_sha,
    )
    assert_split_record_matches(
        record, train=train, test=test, all_rows=core_rows,
        full_features=features["full"], feature_hashes=feature_hashes,
        target_sha=target_sha,
    )
    oof = _validate_core_oof(refs["oof_predictions"], country=country, test=test)
    return {
        "profile": profile, "train": train, "test": test, "features": features,
        "run_manifest": run_manifest, "source_manifest": source_manifest,
        "split_manifest": split_manifest, "split_record": record,
        "assignments_path": str(refs["assignments"]), "oof": oof,
        "core_manifest_hashes": {name: sha256_file(path) for name, path in refs.items()},
    }


def validate_input_manifest(path: Path) -> dict[str, Any]:
    """Validate accepted final inputs and reconstruct only the locked T3 rows."""
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise PreflightError(f"station-thinning input manifest is absent: {manifest_path}")
    manifest = read_json(manifest_path, "station-thinning input manifest")
    if manifest.get("format") != FORMAT:
        raise PreflightError(f"unsupported station-thinning input manifest: {manifest.get('format')!r}")
    if manifest.get("analysis_class") != "execution_phase_exploratory_sensitivity":
        raise PreflightError("input manifest must preserve the execution-phase exploratory sensitivity label")
    candidate, candidate_path, accepted_hashes, acceptance = _validate_candidate_acceptance(
        manifest, manifest_path.parent,
    )
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != 2:
        raise PreflightError("manifest must bind both USA and Canada primary strict-QA cohorts")
    prepared = []
    seen = set()
    for cohort in cohorts:
        dataset_key = str(cohort.get("dataset_key", ""))
        if dataset_key not in DATASETS or dataset_key in seen:
            raise PreflightError(f"unsupported or duplicate station-thinning dataset: {dataset_key!r}")
        seen.add(dataset_key)
        country, target_policy = DATASETS[dataset_key]
        if (str(cohort.get("country", "")).upper() != country
                or cohort.get("target_policy") != target_policy
                or cohort.get("qa_variant") != "strict"):
            raise PreflightError(f"{dataset_key} is not the locked primary strict-QA cohort")
        dataset_record, feature_path, target_path, data_refs = _validate_cohort_candidate(
            candidate, candidate_path, accepted_hashes, dataset_key,
        )
        profiles = cohort.get("profiles")
        if not isinstance(profiles, dict) or set(profiles) != set(PROFILES):
            raise PreflightError(f"{dataset_key} must bind both with_met and no_met core runs")
        source_records = []
        for profile in PROFILES:
            record = profiles[profile]
            source_path, _ = verify_ref(record.get("source_manifest", {}), label=f"{dataset_key}/{profile} source manifest", base=manifest_path.parent)
            source_manifest = read_json(source_path, f"{dataset_key}/{profile} source manifest")
            source_records.append(source_manifest.get(country, {}))
        core_rows, feature_hashes, target_sha = _load_core_rows(
            country=country, dataset_record=dataset_record, feature_path=feature_path,
            target_path=target_path, profile_source_records=source_records,
        )
        profile_results = {}
        for profile in PROFILES:
            profile_results[profile] = _load_profile(
                profiles[profile], profile=profile, country=country, core_rows=core_rows,
                feature_hashes=feature_hashes, target_sha=target_sha,
                source_record=source_records[0],
                expected_qa_provenance_sha=data_refs["qa_provenance_sha256"],
                base=manifest_path.parent,
            )
        baseline = profile_results[PROFILES[0]]
        for profile in PROFILES[1:]:
            current = profile_results[profile]
            if core_runner.canonical_row_key_sha256(current["test"]) != core_runner.canonical_row_key_sha256(baseline["test"]):
                raise PreflightError(f"{dataset_key} held-out keys differ across meteorology profiles")
            if not np.array_equal(
                current["test"]._row_id.to_numpy(dtype=np.int64),
                baseline["test"]._row_id.to_numpy(dtype=np.int64),
            ):
                raise PreflightError(f"{dataset_key} held-out core row IDs differ across meteorology profiles")
            if not np.array_equal(current["test"].pm25.to_numpy(float), baseline["test"].pm25.to_numpy(float)):
                raise PreflightError(f"{dataset_key} held-out targets differ across meteorology profiles")
        heldout = baseline["test"]
        prepared.append({
            "dataset_key": dataset_key, "country": country, "target_policy": target_policy,
            "qa_variant": "strict", "core_rows": core_rows, "profiles": profile_results,
            "test_key_sha256": core_runner.canonical_row_key_sha256(heldout),
            "test_target_sha256": core_runner._array_sha256(heldout.pm25.to_numpy(dtype="float64")),
            "n_test": int(len(heldout)), "n_test_sites": int(heldout.site_id.nunique()),
            "data_refs": data_refs,
        })
    if seen != set(DATASETS):
        raise PreflightError("manifest omits one or more required USA/CAN primary cohorts")
    return {
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "candidate_manifest_path": str(candidate_path), "candidate_manifest_sha256": sha256_file(candidate_path),
        "acceptance_manifest_sha256": sha256_file(resolve_path(manifest["acceptance_manifest"]["path"], manifest_path.parent)),
        "accepted_artifact_hashes": sorted(accepted_hashes), "cohorts": prepared,
    }


def _distance_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    distances = np.asarray(core_runner._nearest_distances(test, train), dtype=float)
    finite = distances[np.isfinite(distances)]
    if len(finite) != len(distances) or len(finite) == 0:
        raise PreflightError("nearest training-station distances are incomplete")
    return {
        "distance_km_min": float(np.min(finite)),
        "distance_km_median": float(np.median(finite)),
        "distance_km_p90": float(np.percentile(finite, 90)),
        "distance_km_mean": float(np.mean(finite)),
    }


def _prediction_rows(
    *, cohort: Mapping[str, Any], profile: str, arm: str, sample_label: str,
    requested_n: int | None, seed: int | str, fit_id: str, fit_reused: bool,
    test: pd.DataFrame, prediction: np.ndarray,
) -> list[dict[str, Any]]:
    if len(test) != len(prediction):
        raise AssertionError("prediction length differs from the fixed held-out test rows")
    records = []
    for pos, (_, row) in enumerate(test.iterrows()):
        records.append({
            "dataset_key": cohort["dataset_key"], "country": cohort["country"],
            "profile": profile, "arm": arm, "sample_size": sample_label,
            "requested_station_n": requested_n, "seed": str(seed), "fit_id": fit_id,
            "fit_reused": bool(fit_reused), "row_id": int(row["_row_id"]),
            "site_id": str(row.site_id), "year": int(row.year), "month": int(row.month),
            "obs": float(row.pm25), "pred": float(prediction[pos]),
        })
    return records


def _metric_record(
    *, cohort: Mapping[str, Any], profile: str, arm: str, sample_label: str,
    requested_n: int | None, seed: int | str, fit_id: str, fit_reused: bool,
    train: pd.DataFrame, test: pd.DataFrame, prediction: np.ndarray,
) -> dict[str, Any]:
    values = metric_values(test.pm25.to_numpy(float), prediction)
    overlap = set(test.site_id.astype(str)) & set(train.site_id.astype(str))
    n_test_sites = int(test.site_id.nunique())
    return {
        "dataset_key": cohort["dataset_key"], "country": cohort["country"],
        "target_policy": cohort["target_policy"], "qa_variant": "strict",
        "profile": profile, "arm": arm, "sample_size": sample_label,
        "requested_station_n": requested_n, "seed": str(seed), "fit_id": fit_id,
        "fit_reused": bool(fit_reused), "n_train": int(len(train)),
        "n_train_stations": int(train.site_id.nunique()), "n_test": int(len(test)),
        "n_test_stations": n_test_sites, "overlap_test_sites_n": int(len(overlap)),
        "overlap_test_sites_pct": float(100.0 * len(overlap) / n_test_sites) if n_test_sites else np.nan,
        "training_site_set_sha256": json_sha256(sorted(train.site_id.astype(str).unique().tolist())),
        "test_key_sha256": cohort["test_key_sha256"],
        "test_target_sha256": cohort["test_target_sha256"],
        **_distance_summary(train, test), **values,
    }


def _paired_delta(
    *, cohort: Mapping[str, Any], profile: str, sample_label: str,
    requested_n: int | None, seed: int | str, train: pd.DataFrame, test: pd.DataFrame,
    pred_b: np.ndarray, pred_full: np.ndarray, fit_reused: bool = False,
) -> dict[str, Any]:
    b, full = metric_values(test.pm25.to_numpy(float), pred_b), metric_values(test.pm25.to_numpy(float), pred_full)
    distance = _distance_summary(train, test)
    return {
        "dataset_key": cohort["dataset_key"], "country": cohort["country"],
        "profile": profile, "sample_size": sample_label, "requested_station_n": requested_n,
        "seed": str(seed), "delta_direction": "B minus full", "fit_reused": bool(fit_reused),
        "n_train": int(len(train)), "n_train_stations": int(train.site_id.nunique()),
        "n_test": int(len(test)), "n_test_stations": int(test.site_id.nunique()),
        "overlap_test_sites_n": int(len(set(test.site_id.astype(str)) & set(train.site_id.astype(str)))),
        "training_site_set_sha256": json_sha256(sorted(train.site_id.astype(str).unique().tolist())),
        "test_key_sha256": cohort["test_key_sha256"],
        "test_target_sha256": cohort["test_target_sha256"], **distance,
        "delta_rmse": float(b["rmse"] - full["rmse"]),
        "delta_mae": float(b["mae"] - full["mae"]),
        "delta_bias": float(b["bias"] - full["bias"]),
        "delta_r2": float(b["r2"] - full["r2"]) if np.isfinite(b["r2"]) and np.isfinite(full["r2"]) else np.nan,
    }


def _record_fit(
    *, cohort: Mapping[str, Any], profile: str, arm: str, sample_label: str,
    requested_n: int | None, seed: int | str, fit_id: str, fit_reused: bool,
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str],
    prediction_cache: dict[str, np.ndarray], core_oof: pd.DataFrame | None = None,
    fit_func: Callable[..., tuple[np.ndarray, Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if fit_reused:
        prediction = prediction_cache[fit_id]
    else:
        prediction, bundle = (fit_func or fit_train_only)(train, test, features)
        prediction_cache[fit_id] = prediction
        if sample_label == "full":
            if int(bundle.get("fit_n", -1)) != len(train) or int(bundle.get("fit_stations", -1)) != train.site_id.nunique():
                raise PreflightError("full reference model training counts do not match the core training network")
            if core_oof is None:
                raise PreflightError("full reference model lacks bound core OOF predictions")
            expected = core_oof.sort_values("row_id", kind="mergesort").pred.to_numpy(float)
            if not np.allclose(prediction, expected, rtol=0, atol=1e-10):
                raise PreflightError(
                    f"refitted full network predictions differ from the bound core T3/2022 reference ({arm})"
                )
    metric = _metric_record(
        cohort=cohort, profile=profile, arm=arm, sample_label=sample_label,
        requested_n=requested_n, seed=seed, fit_id=fit_id, fit_reused=fit_reused,
        train=train, test=test, prediction=prediction,
    )
    predictions = _prediction_rows(
        cohort=cohort, profile=profile, arm=arm, sample_label=sample_label,
        requested_n=requested_n, seed=seed, fit_id=fit_id, fit_reused=fit_reused,
        test=test, prediction=prediction,
    )
    return metric, predictions


def evaluate_station_thinning(
    prepared: Mapping[str, Any], *, fit_predict: Callable[..., tuple[np.ndarray, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Fit all permitted nested samples; full networks are fitted once per arm/profile."""
    fit_func = fit_predict or fit_train_only
    metrics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    station_lists: list[dict[str, Any]] = []
    for cohort in prepared["cohorts"]:
        for profile in PROFILES:
            profile_data = cohort["profiles"][profile]
            train = profile_data["train"]
            test = profile_data["test"]
            features = profile_data["features"]
            sites = sorted(train.site_id.astype(str).unique().tolist())
            designs = nested_station_samples(sites)
            full_predictions: dict[str, np.ndarray] = {}
            prediction_cache: dict[str, np.ndarray] = {}
            full_train = train.copy().reset_index(drop=True)
            station_lists.append({
                "dataset_key": cohort["dataset_key"], "profile": profile,
                "seed": "shared_full", "requested_station_n": None,
                "station_ids_ordered": sites,
            })
            for arm in ARMS:
                fit_id = f"{cohort['dataset_key']}:{profile}:{arm}:full"
                metric, rows = _record_fit(
                    cohort=cohort, profile=profile, arm=arm, sample_label="full",
                    requested_n=None, seed="shared_full", fit_id=fit_id, fit_reused=False,
                    train=full_train, test=test, features=features[arm],
                    prediction_cache=prediction_cache, core_oof=profile_data["oof"][arm],
                    fit_func=fit_func,
                )
                metrics.append(metric)
                predictions.extend(rows)
                full_predictions[arm] = prediction_cache[fit_id]
            deltas.append(_paired_delta(
                cohort=cohort, profile=profile, sample_label="full", requested_n=None,
                seed="shared_full", train=full_train, test=test,
                pred_b=full_predictions["B"], pred_full=full_predictions["full"],
            ))
            for size in SAMPLE_SIZES:
                if size > len(sites):
                    statuses.append({
                        "dataset_key": cohort["dataset_key"], "country": cohort["country"],
                        "profile": profile, "sample_size": str(size), "requested_station_n": size,
                        "seed": "all", "status": "not_implementable",
                        "available_training_stations": len(sites),
                    })
                    continue
                for seed in SEEDS:
                    selected_sites = designs[int(seed)][int(size)]
                    station_lists.append({
                        "dataset_key": cohort["dataset_key"], "profile": profile,
                        "seed": int(seed), "requested_station_n": int(size),
                        "station_ids_ordered": selected_sites,
                    })
                    sample_train = select_site_rows(train, selected_sites)
                    if sample_train.site_id.nunique() != int(size):
                        raise AssertionError("station sample did not retain the full physical-site groups")
                    sample_predictions: dict[str, np.ndarray] = {}
                    is_full_network = set(selected_sites) == set(sites)
                    for arm in ARMS:
                        fit_id = (
                            f"{cohort['dataset_key']}:{profile}:{arm}:full" if is_full_network
                            else f"{cohort['dataset_key']}:{profile}:{arm}:n{size}:s{seed}"
                        )
                        metric, rows = _record_fit(
                            cohort=cohort, profile=profile, arm=arm, sample_label=str(size),
                            requested_n=int(size), seed=int(seed), fit_id=fit_id,
                            fit_reused=is_full_network, train=sample_train, test=test,
                            features=features[arm], prediction_cache=prediction_cache,
                            fit_func=fit_func,
                        )
                        metrics.append(metric)
                        predictions.extend(rows)
                        sample_predictions[arm] = prediction_cache[fit_id]
                    deltas.append(_paired_delta(
                        cohort=cohort, profile=profile, sample_label=str(size),
                        requested_n=int(size), seed=int(seed), train=sample_train, test=test,
                        pred_b=sample_predictions["B"], pred_full=sample_predictions["full"],
                        fit_reused=is_full_network,
                    ))
    return {
        "metrics": pd.DataFrame(metrics), "paired_deltas": pd.DataFrame(deltas),
        "predictions": pd.DataFrame(predictions), "statuses": pd.DataFrame(statuses),
        "sampled_station_lists": station_lists,
    }


def descriptive_repeat_ranges(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Report min/max/range across the three seed draws; never a CI."""
    records: list[dict[str, Any]] = []
    metric_names = ("rmse", "mae", "bias", "r2")
    sampled = metrics.loc[metrics.sample_size.isin([str(n) for n in SAMPLE_SIZES])]
    for keys, group in sampled.groupby(["dataset_key", "profile", "arm", "sample_size"], sort=True):
        if group.seed.astype(str).nunique() != len(SEEDS) or group.fit_reused.astype(bool).any():
            continue
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            records.append({
                "analysis": "model_metric", "dataset_key": keys[0], "profile": keys[1],
                "arm": keys[2], "sample_size": keys[3], "metric": metric,
                "n_seed_draws": len(SEEDS), "minimum": float(values.min()),
                "maximum": float(values.max()), "range": float(values.max() - values.min()),
                "interpretation": "descriptive repeat range; not a confidence interval or independent experiments",
            })
    paired_sampled = paired.loc[paired.sample_size.isin([str(n) for n in SAMPLE_SIZES])]
    for keys, group in paired_sampled.groupby(["dataset_key", "profile", "sample_size"], sort=True):
        if group.seed.astype(str).nunique() != len(SEEDS) or group.fit_reused.astype(bool).any():
            continue
        for metric in ("delta_rmse", "delta_mae", "delta_bias", "delta_r2"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            records.append({
                "analysis": "paired_B_minus_full", "dataset_key": keys[0],
                "profile": keys[1], "arm": "B minus full", "sample_size": keys[2],
                "metric": metric, "n_seed_draws": len(SEEDS),
                "minimum": float(values.min()), "maximum": float(values.max()),
                "range": float(values.max() - values.min()),
                "interpretation": "descriptive repeat range; not a confidence interval or independent experiments",
            })
    return pd.DataFrame(records)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_preflight_status(output_root: Path, payload: Mapping[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    result = dict(payload)
    result.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    path = output_root / "preflight_status.json"
    _write_json(path, result)
    return path


def write_run_outputs(prepared: Mapping[str, Any], evaluated: Mapping[str, Any], output_root: Path) -> Path:
    run_id = f"run_{_utc_stamp()}_{prepared['manifest_sha256'][:10]}"
    run_dir = output_root / run_id
    if run_dir.exists():
        raise PreflightError(f"refusing to overwrite existing station-thinning run: {run_dir}")
    run_dir.mkdir(parents=True)
    metrics = evaluated["metrics"]
    paired = evaluated["paired_deltas"]
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    paired.to_csv(run_dir / "paired_deltas.csv", index=False)
    descriptive_repeat_ranges(metrics, paired).to_csv(run_dir / "repeat_ranges.csv", index=False)
    evaluated["statuses"].to_csv(run_dir / "level_status.csv", index=False)
    evaluated["predictions"].to_parquet(run_dir / "heldout_predictions.parquet", index=False)
    _write_json(run_dir / "sampled_station_lists.json", {
        "sampling_unit": "physical site_id; retain every eligible 2018-2021 training month",
        "site_order": "sorted site IDs permuted separately by each fixed seed; each n is a permutation prefix",
        "seeds": list(SEEDS), "sample_sizes": list(SAMPLE_SIZES),
        "records": evaluated["sampled_station_lists"],
    })
    output_manifest = {
        "format": OUTPUT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_exploratory_sensitivity",
        "analysis_class": "execution_phase_exploratory_sensitivity",
        "protocol_addendum": "work/repair_v2/station_thinning_protocol_addendum_20261113.md",
        "candidate_manifest_sha256": prepared["candidate_manifest_sha256"],
        "acceptance_manifest_sha256": prepared["acceptance_manifest_sha256"],
        "input_manifest_sha256": prepared["manifest_sha256"],
        "fixed_split": {"task": CORE_TASK, "fold_id": CORE_FOLD, "train_years": list(TRAIN_YEARS), "test_year": TEST_YEAR},
        "sample_sizes": list(SAMPLE_SIZES), "seeds": list(SEEDS), "profiles": list(PROFILES),
        "arms": list(ARMS), "model_config": MODEL_CONFIG,
        "target": "log(pm25 + 0.1); smearing*exp(z)-0.1; clip at zero per audited core",
        "smearing": "three station-group folds fit independently inside each sampled training set",
        "full_network_fit_rule": "one fit per country/profile/arm; reused only when a requested sample contains every training site",
        "heldout_binding": "all profiles, arms, levels and seeds preserve the exact core T3/2022 key and target hashes; row predictions retained",
        "replicate_interpretation": "report all three random draws; repeat ranges are descriptive, not confidence intervals or independent experiments",
        "selection_rule": "no test PM, fire, outcome, performance, or future-year score is used for station sampling or level selection",
        "multiplicity_scope": "supplementary sensitivity; excluded from the 26-unit primary simultaneous interval family",
        "limitations": "nested thinning of two existing networks does not establish a universal station minimum, new-network transportability, or smoke-source attribution reliability",
        "cohorts": [{
            "dataset_key": c["dataset_key"], "country": c["country"],
            "test_key_sha256": c["test_key_sha256"], "test_target_sha256": c["test_target_sha256"],
            "n_test": c["n_test"], "n_test_sites": c["n_test_sites"],
            "accepted_data_refs": c["data_refs"],
            "profiles": {p: {
                "core_manifest_hashes": c["profiles"][p]["core_manifest_hashes"],
                "n_train_sites": int(c["profiles"][p]["train"].site_id.nunique()),
            } for p in PROFILES},
        } for c in prepared["cohorts"]],
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(run_dir / name)}
            for name in (
                "metrics.csv", "paired_deltas.csv", "repeat_ranges.csv", "level_status.csv",
                "heldout_predictions.parquet", "sampled_station_lists.json",
            )
        },
    }
    _write_json(run_dir / "run_manifest.json", output_manifest)
    return run_dir


def _preflight_summary(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready_awaiting_root_go",
        "analysis_class": "execution_phase_exploratory_sensitivity",
        "input_manifest_path": prepared["manifest_path"],
        "input_manifest_sha256": prepared["manifest_sha256"],
        "candidate_manifest_sha256": prepared["candidate_manifest_sha256"],
        "acceptance_manifest_sha256": prepared["acceptance_manifest_sha256"],
        "cohorts": [{
            "dataset_key": c["dataset_key"], "country": c["country"],
            "n_test": c["n_test"], "n_test_sites": c["n_test_sites"],
            "test_key_sha256": c["test_key_sha256"], "test_target_sha256": c["test_target_sha256"],
            "accepted_data_refs": c["data_refs"],
            "train_stations_by_profile": {
                profile: int(c["profiles"][profile]["train"].site_id.nunique()) for profile in PROFILES
            },
        } for c in prepared["cohorts"]],
        "execution": "not_started; --execute plus ROOT_GO_STATION_THINNING=YES is required",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="hash-bound final station-thinning input manifest")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="fit only after final-input preflight and explicit release approval")
    args = parser.parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    if args.manifest is None:
        path = write_preflight_status(output_root, {
            "status": "pending_final_inputs",
            "analysis_class": "execution_phase_exploratory_sensitivity",
            "required": [
                "accepted final 2018-2022 source candidate and QA acceptance",
                "USA/CAN strict-QA primary feature/target sources",
                "with_met and no_met final core T3/2022 run/source/schema/split/assignment/OOF manifests",
                "explicit station-thinning input manifest with SHA-256 references",
            ],
            "execution": "not_started; no source tables, future PM2.5 scores, or models were read/fitted",
        })
        print(f"PREFLIGHT_PENDING {path}")
        return 0
    try:
        prepared = validate_input_manifest(args.manifest)
    except (PreflightError, OSError, ValueError, KeyError, TypeError) as exc:
        path = write_preflight_status(output_root, {
            "status": "blocked_input_contract", "manifest_path": str(args.manifest.resolve()),
            "error": str(exc), "execution": "not_started; no model fitting performed",
        })
        print(f"PREFLIGHT_BLOCKED {path}: {exc}", file=sys.stderr)
        return 2
    if not args.execute:
        path = write_preflight_status(output_root, _preflight_summary(prepared))
        print(f"PREFLIGHT_READY_ROOT_GO_REQUIRED {path}")
        return 0
    if os.environ.get("ROOT_GO_STATION_THINNING") != "YES":
        path = write_preflight_status(output_root, _preflight_summary(prepared))
        print(f"EXECUTION_NOT_STARTED_ROOT_GO_REQUIRED {path}", file=sys.stderr)
        return 2
    evaluated = evaluate_station_thinning(prepared)
    run_dir = write_run_outputs(prepared, evaluated, output_root)
    print(f"COMPLETED {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
