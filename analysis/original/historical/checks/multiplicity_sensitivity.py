"""Exploratory max-|t| multiplicity sensitivity for primary strict HGB OOFs.

This rehearsal leaves existing descriptive pointwise-CI tables unchanged. It
uses a shared station multiplier bootstrap over the fixed three RMSE contrasts
and does not refit any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from assess_deployment_support import input_context  # noqa: E402


STRICT_QA = "strict"
TARGET = "log_smear"
SMEAR = "crossfit_station"
BACKTRANSFORM = "smearing"
ARMS = ("B", "B+AOD", "full")
CONTRASTS = (
    ("B+AOD - B", "B+AOD", "B"),
    ("full - B+AOD", "full", "B+AOD"),
    ("full - B", "full", "B"),
)
N_REPLICATES = 10_000
SEED = 20261113
CHUNK_SIZE = 256
MIN_TEST_STATIONS = 30
SE_TOLERANCE = 1e-12
PRIMARY_HISTORICAL_TASKS = {"T1_station", "T2_spatial", "T2_spatial_buffer50km", "T3_year_2022"}
T4_TASKS = {"T4_external_2023_with_met", "T4_external_2023_no_met"}
PROFILES = ("with_met", "no_met")
TRUSTED_VALIDATION_CORE_SHA256 = "a37bf3a381d007d5d541cdf74cf1a8ba81777b126189d1eb65bc6d5deafb562c"
TRUSTED_VALIDATION_CORE_ARCHIVE = HERE.parent / "validation" / "qa_feature_inputs" / "qa_server_bundle_v3.zip"
TRUSTED_VALIDATION_CORE_MEMBER = "validation/validation_core.py"
ROW_IDENTITY = [
    "country", "cohort_label", "profile", "task", "fold_id", "row_id",
    "site_id", "year", "month",
]
OOF_COLUMNS = [*ROW_IDENTITY, "target_strategy", "smear_strategy", "backtransform", "arm", "obs", "pred"]

RUN_REGISTRY = [
    {"country": "USA", "kind": "historical", "profile": "with_met", "run": "usa_hist_strict_with_met", "cohort": "EPA_QC_2018_2022_base_support_pending_augmentation"},
    {"country": "USA", "kind": "historical", "profile": "no_met", "run": "usa_hist_strict_no_met", "cohort": "EPA_QC_2018_2022_base_support_pending_augmentation"},
    {"country": "CAN", "kind": "historical", "profile": "with_met", "run": "can_hist_primary_strict_with_met", "cohort": "NAPS_all_published_2018_2022_primary"},
    {"country": "CAN", "kind": "historical", "profile": "no_met", "run": "can_hist_primary_strict_no_met", "cohort": "NAPS_all_published_2018_2022_primary"},
    {"country": "USA", "kind": "external", "profile": None, "run": "usa_t4_2023_strict", "cohort": "EPA_QC_2023_verified_old_support_pending_expansion"},
    {"country": "CAN", "kind": "external", "profile": None, "run": "can_t4_2023_strict", "cohort": "NAPS_all_published_2023_verified_old_support_pending_expansion"},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path, role: str) -> dict:
    path = Path(path).resolve()
    try:
        relative = str(path.relative_to(root.resolve()))
    except ValueError:
        relative = None
    return {"role": role, "path": str(path), "workspace_relative_path": relative,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def trusted_validation_core_record(root: Path) -> dict:
    """Verify and record the frozen validation core used by legacy QA runs."""
    if not TRUSTED_VALIDATION_CORE_ARCHIVE.is_file():
        raise FileNotFoundError(TRUSTED_VALIDATION_CORE_ARCHIVE)
    try:
        with zipfile.ZipFile(TRUSTED_VALIDATION_CORE_ARCHIVE) as archive:
            core_bytes = archive.read(TRUSTED_VALIDATION_CORE_MEMBER)
    except KeyError as exc:
        raise ValueError(
            f"Trusted source archive lacks {TRUSTED_VALIDATION_CORE_MEMBER}"
        ) from exc
    core_sha256 = hashlib.sha256(core_bytes).hexdigest()
    if core_sha256 != TRUSTED_VALIDATION_CORE_SHA256:
        raise ValueError(
            "Trusted validation_core.py hash mismatch: "
            f"expected {TRUSTED_VALIDATION_CORE_SHA256}, got {core_sha256}"
        )
    record = file_record(TRUSTED_VALIDATION_CORE_ARCHIVE, root,
                         "trusted_validation_core_source_archive")
    record["archive_member"] = TRUSTED_VALIDATION_CORE_MEMBER
    record["archive_member_sha256"] = core_sha256
    return record


def _model_declarations(value, path: str = "$", in_model_record: bool = False):
    """Yield explicitly declared model kind/class values from model records."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).casefold()
            child_path = f"{path}.{key}"
            explicit_label = normalized_key in {
                "model_class", "model_kind", "model_type",
                "estimator_class", "estimator_kind", "estimator_type",
            }
            if explicit_label or (in_model_record and normalized_key in {
                "class", "class_name", "kind", "type",
            }):
                yield child_path, child
            child_is_model_record = (
                in_model_record
                or normalized_key in {
                    "model", "models", "model_record", "model_records",
                    "model_config", "estimator", "estimators",
                }
                or normalized_key.startswith(("model_", "estimator_"))
                or "model" in normalized_key
                or "estimator" in normalized_key
            )
            yield from _model_declarations(child, child_path, child_is_model_record)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _model_declarations(child, f"{path}[{index}]", in_model_record)


def _flatten_model_labels(value):
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_model_labels(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_model_labels(child)
    elif value is not None and str(value).strip():
        yield str(value).strip()


def _is_hgb_label(label: str) -> bool:
    normalized = "".join(character for character in label.casefold() if character.isalnum())
    return (
        normalized in {"histgradientboosting", "hgb", "hgbregressor"}
        or normalized.endswith("histgradientboostingregressor")
    )


def validate_model_provenance(manifests: list[dict], run_name: str) -> str:
    """Reject non-HGB declarations and bind unlabeled legacy models to trusted code."""
    declarations = [
        (path, label)
        for manifest in manifests
        for path, raw_value in _model_declarations(manifest)
        for label in _flatten_model_labels(raw_value)
    ]
    non_hgb = [(path, label) for path, label in declarations if not _is_hgb_label(label)]
    if non_hgb:
        path, label = non_hgb[0]
        raise ValueError(
            f"Non-HGB model declaration in {run_name} at {path}: {label}"
        )
    if declarations:
        return "explicit_hgb_model_declaration"

    core_hashes = [
        manifest.get("runtime", {}).get("code_sha256", {}).get("validation_core")
        for manifest in manifests
        if isinstance(manifest.get("runtime"), dict)
        and isinstance(manifest.get("runtime", {}).get("code_sha256"), dict)
    ]
    if not core_hashes or any(value != TRUSTED_VALIDATION_CORE_SHA256 for value in core_hashes):
        observed = [value for value in core_hashes if value]
        raise ValueError(
            f"Unlabeled legacy model records require validation_core.py SHA-256 "
            f"{TRUSTED_VALIDATION_CORE_SHA256} in {run_name}; found {observed}"
        )
    return f"trusted_legacy_validation_core:{TRUSTED_VALIDATION_CORE_SHA256}"


def validate_registered_tasks(tasks, spec: dict) -> None:
    """Require complete registered task coverage and reject unexpected tasks."""
    observed = set(pd.Series(tasks, dtype=str).dropna().unique())
    expected = PRIMARY_HISTORICAL_TASKS if spec["kind"] == "historical" else T4_TASKS
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            f"Task registry mismatch in {spec['run']}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _fold_token(value) -> str:
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def validate_historical_row_counts(selected: pd.DataFrame, fold_summary: pd.DataFrame,
                                   spec: dict) -> None:
    """Bind selected unique test rows to the registered per-task fold counts."""
    required = {"task", "fold_id", "n_test"}
    if not required.issubset(fold_summary.columns):
        raise ValueError(
            f"Historical fold summary lacks required columns in {spec['run']}: "
            f"{sorted(required - set(fold_summary.columns))}"
        )
    validate_registered_tasks(fold_summary["task"].astype(str).unique(), spec)
    expected = fold_summary[["task", "fold_id", "n_test"]].copy()
    expected["task"] = expected["task"].astype(str)
    expected["fold_key"] = expected["fold_id"].map(_fold_token)
    counts = pd.to_numeric(expected["n_test"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts <= 0).any()
            or not np.equal(counts, np.floor(counts)).all()):
        raise ValueError(f"Invalid historical n_test fold counts in {spec['run']}")
    expected["n_test"] = counts.astype(int)
    expected = expected[["task", "fold_key", "n_test"]]
    if expected.duplicated(["task", "fold_key"]).any():
        raise ValueError(f"Duplicate task/fold rows in historical fold summary for {spec['run']}")

    actual_rows = selected.drop_duplicates(ROW_IDENTITY).copy()
    actual_rows["task"] = actual_rows["task"].astype(str)
    actual_rows["fold_key"] = actual_rows["fold_id"].map(_fold_token)
    actual = actual_rows.groupby(["task", "fold_key"], dropna=False).size().rename("n_rows")
    expected_counts = expected.set_index(["task", "fold_key"])["n_test"]
    expected_map = expected_counts.to_dict()
    actual_map = actual.to_dict()
    if expected_map != actual_map:
        mismatches = {
            str(key): {"expected": expected_map.get(key), "observed": actual_map.get(key)}
            for key in sorted(set(expected_map) | set(actual_map))
            if expected_map.get(key) != actual_map.get(key)
        }
        raise ValueError(f"Historical test-row/fold count mismatch in {spec['run']}: {mismatches}")


def validate_external_row_counts(selected: pd.DataFrame, manifest: dict, spec: dict) -> None:
    """Bind selected T4 rows per profile to the external manifest's test counts."""
    profile_sizes = manifest.get("profile_sample_sizes")
    if not isinstance(profile_sizes, dict) or not set(PROFILES).issubset(profile_sizes):
        raise ValueError(f"T4 manifest lacks fixed per-profile sample sizes: {spec['run']}")
    actual_rows = selected.drop_duplicates(ROW_IDENTITY)
    actual = actual_rows.groupby(["task", "profile"], dropna=False).size().to_dict()
    mismatches = {}
    for profile in PROFILES:
        task = f"T4_external_2023_{profile}"
        profile_record = profile_sizes[profile]
        if not isinstance(profile_record, dict) or "n_test" not in profile_record:
            raise ValueError(f"T4 manifest lacks n_test for {profile}: {spec['run']}")
        expected = pd.to_numeric(pd.Series([profile_record["n_test"]]), errors="raise").iloc[0]
        if not np.isfinite(expected) or expected <= 0 or expected != np.floor(expected):
            raise ValueError(f"Invalid T4 n_test for {profile}: {spec['run']}")
        key = (task, profile)
        observed = actual.get(key)
        if observed != int(expected):
            mismatches[str(key)] = {"expected": int(expected), "observed": observed}
    unexpected = set(actual) - {(f"T4_external_2023_{p}", p) for p in PROFILES}
    if unexpected:
        for key in sorted(unexpected):
            mismatches[str(key)] = {"expected": None, "observed": actual[key]}
    if mismatches:
        raise ValueError(f"T4 test-row/profile count mismatch in {spec['run']}: {mismatches}")


def validate_primary_target(frame: pd.DataFrame) -> None:
    """Fail closed unless the selected OOF rows use the required primary transform."""
    for column, expected in [("target_strategy", TARGET), ("smear_strategy", SMEAR),
                             ("backtransform", BACKTRANSFORM)]:
        if column not in frame or not frame[column].astype(str).eq(expected).all():
            raise ValueError(f"OOF rows must use {column}={expected}")


def weighted_rmse_from_station_sums(
    weights: np.ndarray,
    station_counts: np.ndarray,
    station_sse: np.ndarray,
) -> np.ndarray:
    """Compute arm RMSEs from station-level sufficient statistics.

    Each row of ``weights`` is one multiplier draw. The same weighted station
    row-count denominator is used for every arm in a cell.
    """
    weights = np.asarray(weights, dtype=float)
    station_counts = np.asarray(station_counts, dtype=float)
    station_sse = np.asarray(station_sse, dtype=float)
    if weights.ndim != 2 or station_counts.ndim != 1 or station_sse.ndim != 2:
        raise ValueError("Expected weights[draw,station], counts[station], SSE[station,arm]")
    if weights.shape[1] != station_counts.size or station_sse.shape[0] != station_counts.size:
        raise ValueError("Station dimensions do not align")
    if not (np.isfinite(weights).all() and np.isfinite(station_counts).all()
            and np.isfinite(station_sse).all()):
        raise ValueError("Nonfinite multiplier or station sufficient statistic")
    if (weights <= 0).any() or (station_counts < 0).any() or station_counts.sum() <= 0 or (station_sse < 0).any():
        raise ValueError("Multipliers must be positive, counts nonnegative with positive total, and SSE nonnegative")
    denominator = weights @ station_counts
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("Invalid weighted row-count denominator")
    numerators = weights @ station_sse
    if not np.isfinite(numerators).all() or (numerators < 0).any():
        raise ValueError("Invalid weighted SSE")
    return np.sqrt(numerators / denominator[:, None])


def contrast_draws(weights: np.ndarray, station_counts: np.ndarray, station_sse: np.ndarray) -> np.ndarray:
    rmse = weighted_rmse_from_station_sums(weights, station_counts, station_sse)
    b_idx, aod_idx, full_idx = 0, 1, 2
    return np.column_stack((
        rmse[:, aod_idx] - rmse[:, b_idx],
        rmse[:, full_idx] - rmse[:, aod_idx],
        rmse[:, full_idx] - rmse[:, b_idx],
    ))


def row_level_weighted_rmse(
    y: np.ndarray, predictions: np.ndarray, stations: np.ndarray, station_weights: dict[str, float]
) -> np.ndarray:
    """Small reference implementation used by the aggregation equivalence test."""
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    stations = np.asarray(stations).astype(str)
    weights = np.asarray([station_weights[x] for x in stations], dtype=float)
    denominator = weights.sum()
    return np.sqrt((weights[:, None] * (predictions - y[:, None]) ** 2).sum(axis=0) / denominator)


def cell_eligibility(n_stations: int, min_stations: int = MIN_TEST_STATIONS) -> tuple[bool, str]:
    if n_stations < min_stations:
        return False, f"sparse_test_cell: {n_stations} physical stations < {min_stations}"
    return True, "candidate_pending_bootstrap_SE"


def stable_se(se: float, estimate: float, tolerance: float = SE_TOLERANCE) -> bool:
    if not np.isfinite(se) or not np.isfinite(estimate):
        return False
    return se > tolerance * max(1.0, abs(estimate))


def _manifest_for(spec: dict, run_dir: Path) -> tuple[Path, dict, dict]:
    manifest_path = run_dir / ("source_manifest.json" if spec["kind"] == "historical" else "external_run_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = input_context(manifest)
    country = spec["country"]
    if country not in context:
        raise ValueError(f"Manifest lacks input-stage context for {country}: {manifest_path}")
    stage = context[country]
    if stage.get("qa_variant") != STRICT_QA or stage.get("input_acceptance_stage") != "preliminary":
        raise ValueError(f"Expected preliminary strict QA context in {manifest_path}: {stage}")
    if spec["kind"] == "historical":
        record = manifest.get(country, {})
        declared_profile = record.get("country_met_profile_name")
        selection = record.get("country_met_profile_selection")
        if declared_profile != spec["profile"] or selection not in {"source_declared", "no_met_override"}:
            raise ValueError(f"Historical profile mapping is not supported by the explicit registry for {spec['run']}: {declared_profile}/{selection}")
        if record.get("cohort_label") != spec["cohort"]:
            raise ValueError(f"Historical target cohort mismatch for {spec['run']}")
        if spec["country"] == "CAN" and record.get("target_status", {}).get("status") != "NAPS_all_published_2018_2022_primary":
            raise ValueError(f"CAN history is not the primary all-published cohort: {spec['run']}")
    else:
        if manifest.get("country") != country or int(manifest.get("train_through", -1)) != 2022 or int(manifest.get("external_year", -1)) != 2023:
            raise ValueError(f"T4 run is not fixed train<=2022/test=2023: {spec['run']}")
        if not set(manifest.get("profiles", [])) >= set(PROFILES):
            raise ValueError(f"T4 run lacks fixed MET profiles: {spec['run']}")
    return manifest_path, manifest, stage


def _read_spec(root: Path, spec: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    validation_root = root / "outputs/repair_v2/validation/qa_results_v3_pending_augmentation_20261113"
    run_dir = validation_root / spec["run"]
    pred_path = run_dir / ("oof_predictions.parquet" if spec["kind"] == "historical" else "external_oof_predictions.parquet")
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    manifest_path, manifest, stage = _manifest_for(spec, run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    if spec["kind"] == "historical":
        if not run_manifest_path.is_file():
            raise FileNotFoundError(run_manifest_path)
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        model_manifests = [manifest, run_manifest]
    else:
        run_manifest = None
        model_manifests = [manifest]
    model_provenance = validate_model_provenance(model_manifests, spec["run"])
    columns = OOF_COLUMNS if spec["kind"] == "external" else [c for c in OOF_COLUMNS if c != "profile"]
    prediction = pd.read_parquet(pred_path, columns=columns)
    if spec["kind"] == "historical":
        if "profile" in prediction.columns:
            raise ValueError(f"Historical OOF unexpectedly contains profile; expected source-manifest mapping: {spec['run']}")
        prediction["profile"] = spec["profile"]
    else:
        if not set(prediction.profile.astype(str).unique()).issubset(set(PROFILES)):
            raise ValueError(f"Unexpected T4 profile in {spec['run']}")
        profile_from_task = prediction.task.astype(str).str.extract(r"_(with_met|no_met)$", expand=False)
        if not profile_from_task.equals(prediction.profile.astype(str)):
            raise ValueError(f"T4 task/profile mismatch in {spec['run']}")
    if prediction.country.astype(str).unique().tolist() != [spec["country"]]:
        raise ValueError(f"Country mismatch in {spec['run']}")
    if prediction.cohort_label.astype(str).unique().tolist() != [spec["cohort"]]:
        raise ValueError(f"Target-policy/cohort mismatch in {spec['run']}")
    if spec["kind"] == "historical":
        validate_registered_tasks(prediction.task.astype(str).unique(), spec)
        if prediction.year.astype(int).max() > 2022:
            raise ValueError(f"Historical OOF includes a post-2022 row: {spec['run']}")
    else:
        validate_registered_tasks(prediction.task.astype(str).unique(), spec)
        if set(prediction.year.astype(int).unique()) != {2023}:
            raise ValueError(f"T4 source is not fixed 2023: {spec['run']}")
    selected = prediction.loc[
        prediction.target_strategy.astype(str).eq(TARGET)
        & prediction.smear_strategy.astype(str).eq(SMEAR)
        & prediction.backtransform.astype(str).eq(BACKTRANSFORM)
        & prediction.arm.astype(str).isin(ARMS)
    ].copy()
    if selected.empty:
        raise ValueError(f"No primary three-arm predictions in {spec['run']}")
    validate_primary_target(selected)
    if selected[ROW_IDENTITY].isna().any().any() or selected[["obs", "pred"]].isna().any().any():
        raise ValueError(f"Null key, observation, or prediction in {spec['run']}")
    if not np.isfinite(selected[["obs", "pred"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite prediction rows in {spec['run']}")
    if selected.duplicated(ROW_IDENTITY + ["arm"]).any():
        raise ValueError(f"Duplicate row/arm identity in {spec['run']}")
    fold_summary_path = None
    if spec["kind"] == "historical":
        fold_summary_path = run_dir / "fold_summaries.csv"
        if not fold_summary_path.is_file():
            raise FileNotFoundError(fold_summary_path)
        validate_historical_row_counts(selected, pd.read_csv(fold_summary_path), spec)
    else:
        validate_external_row_counts(selected, manifest, spec)
    sources = [
        file_record(pred_path, root, "strict_primary_oof_predictions"),
        file_record(manifest_path, root, "source_context_manifest"),
    ]
    if run_manifest is not None:
        sources.append(file_record(run_manifest_path, root, "model_run_manifest"))
    if fold_summary_path is not None:
        sources.append(file_record(fold_summary_path, root, "strict_fold_summary"))
    return selected, sources, {**stage, "run": spec["run"], "country": spec["country"],
                               "profile_mapping": (f"source_manifest:{spec['profile']}:{manifest[spec['country']].get('country_met_profile_selection')}"
                                                   if spec["kind"] == "historical" else "profile column + task suffix"),
                               "cohort_label": spec["cohort"], "qa_variant": STRICT_QA,
                               "model_provenance_validation": model_provenance}


def _build_wide_cells(selected: pd.DataFrame, scope_by_run: dict[str, str]) -> list[dict]:
    cells = []
    grouping = ["country", "cohort_label", "task", "profile"]
    for key, frame in selected.groupby(grouping, sort=True, dropna=False):
        country, cohort, task, profile = key
        for field, expected in [("target_strategy", TARGET), ("smear_strategy", SMEAR), ("backtransform", BACKTRANSFORM)]:
            if not frame[field].astype(str).eq(expected).all():
                raise ValueError(f"Mixed target/transform in cell {key}: {field}")
        by_arm = {}
        for arm in ARMS:
            part = frame.loc[frame.arm.astype(str).eq(arm)].sort_values(ROW_IDENTITY).reset_index(drop=True)
            if part.empty:
                raise ValueError(f"Missing {arm} arm in cell {key}")
            by_arm[arm] = part
        reference = by_arm["B"]
        for arm in ("B+AOD", "full"):
            other = by_arm[arm]
            if len(reference) != len(other) or not reference[ROW_IDENTITY].reset_index(drop=True).equals(other[ROW_IDENTITY].reset_index(drop=True)):
                raise ValueError(f"Three-arm row keys do not match exactly in {key}")
            if not np.array_equal(reference.obs.to_numpy(dtype=float), other.obs.to_numpy(dtype=float)):
                raise ValueError(f"Three-arm observations differ in {key}")
        wide = reference[ROW_IDENTITY + ["obs"]].copy()
        for arm in ("B", "B+AOD", "full"):
            wide["pred_" + arm.replace("+", "_")] = by_arm[arm].pred.to_numpy(dtype=float)
        wide["site_id"] = wide.site_id.astype(str)
        # At least one source row for this cell is retained in the explicit registry.
        source_kind = "T4_2023" if str(task).startswith("T4_external") else "historical_2018_2022"
        site_stats = wide.assign(
            sse_B=(wide["pred_B"] - wide.obs) ** 2,
            sse_B_AOD=(wide["pred_B_AOD"] - wide.obs) ** 2,
            sse_full=(wide["pred_full"] - wide.obs) ** 2,
        ).groupby("site_id", sort=True).agg(
            n_rows=("obs", "size"), sse_B=("sse_B", "sum"),
            sse_B_AOD=("sse_B_AOD", "sum"), sse_full=("sse_full", "sum"),
        )
        stations = site_stats.index.to_numpy(dtype=str)
        counts = site_stats.n_rows.to_numpy(dtype=float)
        sse = site_stats[["sse_B", "sse_B_AOD", "sse_full"]].to_numpy(dtype=float)
        if not (np.isfinite(counts).all() and np.isfinite(sse).all()):
            raise ValueError(f"Nonfinite station sufficient statistics in {key}")
        n_rows = int(counts.sum())
        n_stations = int(len(stations))
        point_rmse = np.sqrt(sse.sum(axis=0) / n_rows)
        deltas = np.array([point_rmse[1] - point_rmse[0], point_rmse[2] - point_rmse[1], point_rmse[2] - point_rmse[0]])
        cells.append({
            "country": country, "cohort_label": cohort, "task": task, "profile": profile,
            "scope": source_kind, "n_rows": n_rows, "n_test_stations": n_stations,
            "station_ids": stations, "station_counts": counts, "station_sse": sse,
            "point_rmse": point_rmse, "point_contrasts": deltas,
            "point_contrast_by_name": {contrast[0]: float(delta) for contrast, delta in zip(CONTRASTS, deltas)},
        })
    return cells


def _bootstrap_country_cells(
    cells: list[dict],
    *,
    country: str,
    child_seed: np.random.SeedSequence,
    n_replicates: int,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict]:
    country_cells = [cell for cell in cells if cell["country"] == country]
    union_stations = np.array(sorted({
        station for cell in country_cells for station in cell["station_ids"].tolist()
    }), dtype=str)
    station_lookup = {station: i for i, station in enumerate(union_stations)}
    prepared = []
    for cell_index, cell in enumerate(country_cells):
        counts = np.zeros(len(union_stations), dtype=float)
        sse = np.zeros((len(union_stations), len(ARMS)), dtype=float)
        indices = np.array([station_lookup[s] for s in cell["station_ids"]], dtype=int)
        counts[indices] = cell["station_counts"]
        sse[indices] = cell["station_sse"]
        cell["country_station_index"] = {s: station_lookup[s] for s in cell["station_ids"]}
        is_eligible, reason = cell_eligibility(cell["n_test_stations"])
        cell["eligibility_reason"] = reason
        cell["sparse_eligible"] = is_eligible
        prepared.append((cell, counts, sse))

    rng = np.random.default_rng(child_seed)
    draws = {id(cell): np.full((n_replicates, len(CONTRASTS)), np.nan, dtype=float)
             for cell, _, _ in prepared}
    invalid_weight_draws = 0
    invalid_draw_indices = []
    for start in range(0, n_replicates, chunk_size):
        stop = min(start + chunk_size, n_replicates)
        weights = rng.exponential(scale=1.0, size=(stop - start, len(union_stations)))
        weight_valid = np.isfinite(weights).all(axis=1) & (weights > 0).all(axis=1)
        invalid_weight_draws += int((~weight_valid).sum())
        invalid_draw_indices.extend((start + np.flatnonzero(~weight_valid)).tolist())
        # Invalid multipliers are not redrawn. Their outputs remain invalid so
        # any affected cell is excluded explicitly rather than silently dropped.
        valid_local = np.flatnonzero(weight_valid)
        for cell, counts, sse in prepared:
            if not cell["sparse_eligible"]:
                continue
            if len(valid_local):
                draws[id(cell)][start + valid_local] = contrast_draws(weights[valid_local], counts, sse)
    for cell, _, _ in prepared:
        cell["bootstrap_contrasts"] = draws[id(cell)]
    diagnostics = {
        "country": country,
        "n_country_stations": int(len(union_stations)),
        "same_station_weights_shared_across_cells": True,
        "independent_country_stream": True,
        "invalid_weight_draws": invalid_weight_draws,
        "invalid_draw_indices": invalid_draw_indices,
        "replicates_requested": n_replicates,
        "no_redraws": True,
        "seed_sequence_spawn_key": list(child_seed.spawn_key),
    }
    return draws, diagnostics


def _assemble_results(cells: list[dict], n_replicates: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exclusion_reasons = {}
    for cell in cells:
        if not cell.get("sparse_eligible", True):
            exclusion_reasons[id(cell)] = cell.get("eligibility_reason", "sparse")
            cell["family_eligible"] = False
            continue
        boot = cell.get("bootstrap_contrasts")
        if boot is None or boot.shape != (n_replicates, len(CONTRASTS)):
            exclusion_reasons[id(cell)] = "missing_or_wrong_replicate_count"
            cell["family_eligible"] = False
            continue
        if not np.isfinite(boot).all():
            n_invalid = int((~np.isfinite(boot)).any(axis=1).sum())
            exclusion_reasons[id(cell)] = f"invalid_bootstrap_draws:{n_invalid}; no redraw/drop"
            cell["family_eligible"] = False
            continue
        se = boot.std(axis=0, ddof=1)
        unstable = [
            contrast[0] for contrast, value, estimate in zip(CONTRASTS, se, cell["point_contrasts"])
            if not stable_se(float(value), float(estimate))
        ]
        if unstable:
            exclusion_reasons[id(cell)] = "zero_or_unstable_SE:" + ";".join(unstable)
            cell["family_eligible"] = False
            continue
        cell["bootstrap_se"] = se
        cell["family_eligible"] = True
        cell["eligibility_reason"] = "included in fixed max-|t| family"
    eligible = [cell for cell in cells if cell.get("family_eligible", False)]
    if not eligible:
        raise ValueError("No estimable cells remain for the max-|t| family")
    standardized = []
    for cell in eligible:
        standardized.append((cell["bootstrap_contrasts"] - cell["point_contrasts"]) / cell["bootstrap_se"])
    joint = np.concatenate(standardized, axis=1)
    if joint.shape != (n_replicates, len(eligible) * len(CONTRASTS)):
        raise AssertionError("Max-|t| family matrix shape mismatch")
    max_abs = np.max(np.abs(joint), axis=1)
    if not np.isfinite(max_abs).all():
        raise ValueError("Invalid max-|t| draw; do not silently drop or redraw")
    critical = float(np.quantile(max_abs, 0.95, method="linear"))
    contrast_rows, cell_rows, excluded_rows = [], [], []
    for cell in cells:
        eligible_flag = bool(cell.get("family_eligible", False))
        reason = exclusion_reasons.get(id(cell), cell.get("eligibility_reason", "not included"))
        cell["eligibility_reason"] = reason
        rmse = cell["point_rmse"]
        nsites = cell["n_test_stations"]
        base_row = {
            "country": cell["country"], "cohort_label": cell["cohort_label"],
            "scope": cell["scope"], "task": cell["task"], "profile": cell["profile"],
            "n_rows": cell["n_rows"], "n_test_stations": nsites,
            "rmse_B": float(rmse[0]), "rmse_B_AOD": float(rmse[1]), "rmse_full": float(rmse[2]),
            "family_eligible": eligible_flag, "eligibility_reason": reason,
        }
        cell_rows.append(base_row)
        if not eligible_flag:
            excluded_rows.append({**base_row, "exclusion_reason": exclusion_reasons.get(id(cell), reason)})
        for index, (name, numerator_arm, denominator_arm) in enumerate(CONTRASTS):
            estimate = float(cell["point_contrasts"][index])
            se = float(cell["bootstrap_se"][index]) if eligible_flag else np.nan
            contrast_rows.append({
                **base_row,
                "contrast": name,
                "numerator_arm": numerator_arm,
                "denominator_arm": denominator_arm,
                "delta_rmse": estimate,
                "station_multiplier_se": se,
                "max_abs_t_critical_95": critical if eligible_flag else np.nan,
                "simultaneous_95_low": estimate - critical * se if eligible_flag else np.nan,
                "simultaneous_95_high": estimate + critical * se if eligible_flag else np.nan,
                "family_size_contrasts": len(eligible) * len(CONTRASTS),
                "replicates": n_replicates if eligible_flag else 0,
                "interval_scope": "single fixed max-|t| family; conditional on fitted predictions, observed folds and years",
            })
    summary = {
        "max_abs_t_critical_95": critical,
        "n_cells_total": len(cells),
        "n_cells_eligible": len(eligible),
        "n_cells_excluded": len(cells) - len(eligible),
        "n_contrasts_in_family": len(eligible) * len(CONTRASTS),
        "n_draws": n_replicates,
        "max_abs_t_quantile_method": "numpy.quantile(method='linear')",
        "max_abs_t_by_draw": max_abs,
    }
    return pd.DataFrame(contrast_rows), pd.DataFrame(cell_rows), {**summary, "exclusions": pd.DataFrame(excluded_rows)}


def _metadata_rows(cells_frame: pd.DataFrame, context_by_cell: dict, code_hash: str, source_fingerprint: str) -> pd.DataFrame:
    rows = []
    for row in cells_frame.to_dict("records"):
        key = (row["country"], row["scope"], row["profile"])
        context = context_by_cell[key]
        rows.append({
            **row,
            **context,
            "main_training_overlay_status": context.get("training_overlay_status"),
            "main_evaluation_overlay_status": context.get("evaluation_overlay_status"),
            "diagnostic_code_sha256": code_hash,
            "input_sources_sha256": source_fingerprint,
        })
    return pd.DataFrame(rows)


def run(root: Path, out: Path, n_replicates: int = N_REPLICATES,
        seed: int = SEED, chunk_size: int = CHUNK_SIZE) -> dict:
    if n_replicates != 10_000:
        raise ValueError("The fixed rehearsal requires exactly 10,000 draws")
    if seed != SEED:
        raise ValueError(f"The fixed rehearsal seed is {SEED}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    helper_path = HERE.parent / "assess_deployment_support.py"
    helper_sha_at_start = sha256(helper_path)
    validation_root = root / "outputs/repair_v2/validation/qa_results_v3_pending_augmentation_20261113"
    source_records, stage_registry, selected_frames, context_by_cell = [
        trusted_validation_core_record(root)
    ], [], [], {}
    for spec in RUN_REGISTRY:
        frame, records, stage = _read_spec(root, spec)
        source_records += records
        stage_registry.append(stage)
        if spec["kind"] == "historical":
            context_by_cell[(spec["country"], "historical_2018_2022", spec["profile"])] = stage
        else:
            for profile in PROFILES:
                context_by_cell[(spec["country"], "T4_2023", profile)] = stage
        selected_frames.append(frame)
    selected = pd.concat(selected_frames, ignore_index=True)
    cells = _build_wide_cells(selected, {})
    if not cells:
        raise ValueError("No complete primary cells found")
    # A cell must map back to one and only one expected source context.
    for cell in cells:
        cell["scope"] = "T4_2023" if cell["scope"] == "T4_2023" else "historical_2018_2022"
        if (cell["country"], cell["scope"], cell["profile"]) not in context_by_cell:
            raise ValueError(f"Cell lacks a registered source context: {cell}")
    code_path = HERE
    test_path = HERE.parent / "test_multiplicity_sensitivity.py"
    contract_path = HERE.parent / "analysis_contract.md"
    code_hash = sha256(code_path)
    source_records += [
        file_record(helper_path, root, "input_context_helper"),
        file_record(contract_path, root, "analysis_contract"),
        file_record(test_path, root, "focused_statistical_guard_tests"),
    ]
    if source_records[-3]["sha256"] != helper_sha_at_start:
        raise RuntimeError("Support helper changed during this run; aborting provenance binding")
    source_fingerprint = hashlib.sha256(
        json.dumps(source_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seed_children = np.random.SeedSequence(seed).spawn(2)
    country_diagnostics = []
    for country, child in zip(["CAN", "USA"], seed_children):
        _, diagnostics = _bootstrap_country_cells(
            cells, country=country, child_seed=child, n_replicates=n_replicates,
            chunk_size=chunk_size,
        )
        country_diagnostics.append(diagnostics)
    contrasts, cells_frame, assembled = _assemble_results(cells, n_replicates)
    max_abs = assembled.pop("max_abs_t_by_draw")
    exclusions = assembled.pop("exclusions")
    cells_frame = _metadata_rows(cells_frame, context_by_cell, code_hash, source_fingerprint)
    if not exclusions.empty:
        exclusions = _metadata_rows(exclusions, context_by_cell, code_hash, source_fingerprint)
    else:
        exclusions = cells_frame.iloc[0:0].copy()
        exclusions["exclusion_reason"] = pd.Series(dtype=str)
    # Per-contrast rows inherit the explicit stage registry from their cell.
    contrasts = contrasts.merge(
        cells_frame[["country", "cohort_label", "task", "scope", "profile", "main_training_overlay_status",
                     "main_evaluation_overlay_status", "input_acceptance_stage", "qa_variant",
                     "diagnostic_code_sha256", "input_sources_sha256"]],
        on=["country", "cohort_label", "task", "scope", "profile"], how="left", validate="many_to_one",
    )
    out.mkdir(parents=True, exist_ok=True)
    contrast_path = out / "simultaneous_rmse_contrasts.csv"
    cell_path = out / "cell_rmse_and_eligibility.csv"
    exclusion_path = out / "excluded_cells.csv"
    max_path = out / "max_abs_t_draws.csv"
    contrasts.to_csv(contrast_path, index=False)
    cells_frame.to_csv(cell_path, index=False)
    exclusions.to_csv(exclusion_path, index=False)
    pd.DataFrame({
        "replicate": np.arange(1, n_replicates + 1),
        "max_abs_t": max_abs,
        "qa_variant": STRICT_QA,
        "input_acceptance_stage": "preliminary",
        "diagnostic_code_sha256": code_hash,
        "input_sources_sha256": source_fingerprint,
    }).to_csv(max_path, index=False)

    output_records = [file_record(path, root, "diagnostic_output") for path in
                      [contrast_path, cell_path, exclusion_path, max_path]]
    summary = {
        "status": "exploratory multiplicity sensitivity rehearsal on preliminary strict QA OOF predictions",
        "not_final_inference": True,
        "qa_variant": STRICT_QA,
        "target_strategy": TARGET,
        "smear_strategy": SMEAR,
        "backtransform": BACKTRANSFORM,
        "cohorts": {
            "USA_historical": "EPA_QC_2018_2022_base_support_pending_augmentation",
            "CAN_historical": "NAPS_all_published_2018_2022_primary",
            "USA_T4_2023": "EPA_QC_2023_verified_old_support_pending_expansion",
            "CAN_T4_2023": "NAPS_all_published_2023_verified_old_support_pending_expansion",
        },
        "task_family": sorted(PRIMARY_HISTORICAL_TASKS) + ["T4_external_2023_with_met", "T4_external_2023_no_met"],
        "profiles": list(PROFILES),
        "contrasts": [
            {"name": name, "numerator_arm": numerator, "denominator_arm": denominator}
            for name, numerator, denominator in CONTRASTS
        ],
        "eligibility_rule": f"at least {MIN_TEST_STATIONS} unique held-out physical station IDs per country/task/profile cell; fixed without using effect estimates",
        "bootstrap": {
            "type": "positive station multiplier bootstrap, not ordinary task-wise multinomial bootstrap and not a model-refit bootstrap",
            "distribution": "Exp(1), mean=1, variance=1",
            "replicates": n_replicates,
            "seed": seed,
            "chunk_size": chunk_size,
            "same_country_station_weights_shared_across_tasks_profiles_contrasts": True,
            "country_streams_independent": True,
            "task_specific_weighted_row_count_normalizer": True,
            "invalid_replicates_redrawn_or_dropped": False,
            "invalid_weight_draws": sum(x["invalid_weight_draws"] for x in country_diagnostics),
            "invalid_weight_draw_indices_by_country": {x["country"]: x["invalid_draw_indices"] for x in country_diagnostics},
            "centered_statistics": "Delta* - Delta_hat",
            "contrastwise_SE": "sample SD of the 10,000 joint multiplier draws, ddof=1",
            "family_statistic": "per-draw maximum absolute standardized centered deviation across all estimable cell contrasts",
            "critical_value": assembled["max_abs_t_critical_95"],
            "confidence_level": 0.95,
            "critical_quantile": "linear empirical 95th percentile",
            "unstable_SE_rule": f"exclude the entire cell if any contrast SE <= {SE_TOLERANCE} * max(1, |Delta_hat|) or nonfinite",
        },
        "family": {
            "n_cells_total": assembled["n_cells_total"],
            "n_cells_eligible": assembled["n_cells_eligible"],
            "n_cells_excluded": assembled["n_cells_excluded"],
            "n_contrasts": assembled["n_contrasts_in_family"],
            "exclusions_included_in_separate_table": True,
            "pointwise_ci_tables_changed": False,
            "pointwise_ci_interpretation": "existing descriptive pointwise intervals do not control family-wise error",
        },
        "country_bootstrap_diagnostics": country_diagnostics,
        "source_context_registry": stage_registry,
        "sources": source_records,
        "code": file_record(code_path, root, "multiplicity_sensitivity_code"),
        "outputs": output_records,
        "interpretation_limitations": [
            "Preliminary inputs only; workflow rehearsal, not final paper inference.",
            "Conditional on fitted predictions and observed folds/years; assumes exchangeable station clusters.",
            "No spatial, fire-event, new-year, or whole-matrix coverage guarantee.",
            "No model refitting; no clinical confirmatory or preregistered inference claim.",
            "No future 2024/2025 or rolling scores were included.",
        ],
    }
    summary_path = out / "family_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    findings_path = out / "findings.md"
    _write_findings(findings_path, summary, cells_frame, contrasts, exclusions)
    provenance_path = out / "provenance.json"
    provenance = {
        "family_summary": summary,
        "source_registry_hash": source_fingerprint,
        "source_count": len(source_records),
        "source_files": source_records,
        "code_sha256": code_hash,
        "trusted_validation_core_sha256": TRUSTED_VALIDATION_CORE_SHA256,
        "bootstrap_max_abs_t_draws_sha256": sha256(max_path),
        "outputs": [file_record(path, root, "final_output") for path in
                    [contrast_path, cell_path, exclusion_path, max_path, summary_path, findings_path]],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    checksums_path = out / "checksums.json"
    checksums = [file_record(path, root, "final_output") for path in
                 [contrast_path, cell_path, exclusion_path, max_path, summary_path, findings_path, provenance_path]]
    checksums_path.write_text(json.dumps({"outputs": checksums}, indent=2), encoding="utf-8")
    return {**summary, "output_dir": str(out), "checksums_path": str(checksums_path),
            "findings_path": str(findings_path)}


def _write_findings(path: Path, summary: dict, cells: pd.DataFrame,
                    contrasts: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    lines = [
        "# Multiplicity sensitivity rehearsal",
        "",
        "This is a preliminary workflow rehearsal on strict QA HGB OOF predictions. It does not change the existing descriptive pointwise-CI tables; those intervals do not control family-wise error.",
        "",
        "## Method",
        "",
        "The fixed family has three RMSE contrasts in each country × task × MET-profile cell: `B+AOD − B`, `full − B+AOD`, and `full − B`. Cells require at least 30 physical test stations. Rows from each of the three arms were matched on country, primary cohort, profile, task, fold, row ID, site, year, and month, with identical observations required.",
        "",
        f"We used {summary['bootstrap']['replicates']:,} positive Exp(1) station multipliers (mean and variance 1; seed {summary['bootstrap']['seed']}). The same multiplier was reused for each station across tasks, profiles, and contrasts within a country; countries used independent streams. Each task cell used its own weighted row-count denominator. Contrasts were centered at their observed estimates, studentized by the SD of the joint draws, and calibrated by the 95th percentile of the maximum absolute statistic over the estimable family. This is not a task-wise multinomial or model-refit bootstrap.",
        "",
        "## Results",
        "",
        f"The fixed family contains {summary['family']['n_cells_eligible']} eligible cells ({summary['family']['n_contrasts']} contrasts); {summary['family']['n_cells_excluded']} cells are excluded and retained in `excluded_cells.csv`. The common 95% max-|t| critical value is {summary['bootstrap']['critical_value']:.3f}.",
        "",
        "`simultaneous_rmse_contrasts.csv` gives each RMSE difference and its max-|t| interval. `cell_rmse_and_eligibility.csv` gives the arm RMSEs and station counts. The 10,000 per-replicate family maxima are retained in `max_abs_t_draws.csv` for audit.",
        "",
        "## Limits",
        "",
        "Inputs are preliminary and this is a rehearsal, not final paper inference. Inference is conditional on fitted predictions and observed folds/years and assumes exchangeable station clusters. It does not guarantee spatial, event, new-year, or whole-matrix coverage. No models were refit; no 2024/2025 or rolling scores were included. Do not describe this as clinical confirmatory or preregistered inference.",
        "",
        "Historical selected-row counts were checked per task and fold against the hash-bound `fold_summaries.csv`; T4 profile counts were checked against `profile_sample_sizes.n_test` in the source manifest. These counts can detect rows omitted from all three arms, but they cannot identify a missing row key if another key keeps the count unchanged. The OOF files themselves are recorded by checksum in `provenance.json`.",
        "",
    ]
    if not exclusions.empty:
        lines += ["Excluded cells:", "", "| Country | Task | Profile | Stations | Reason |", "|---|---|---|---:|---|"]
        for row in exclusions.itertuples(index=False):
            lines.append(f"| {row.country} | {row.task} | {row.profile} | {row.n_test_stations} | {row.exclusion_reason} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicates", type=int, default=N_REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    root = args.workspace.resolve()
    output = args.out.resolve() if args.out else root / "outputs/repair_v2/diagnostics/multiplicity_rehearsal"
    result = run(root, output, args.replicates, args.seed, args.chunk_size)
    print(json.dumps({
        "status": result["status"],
        "n_cells_eligible": result["family"]["n_cells_eligible"],
        "n_cells_excluded": result["family"]["n_cells_excluded"],
        "n_contrasts": result["family"]["n_contrasts"],
        "critical_value": result["bootstrap"]["critical_value"],
        "output_dir": result["output_dir"],
        "findings": result["findings_path"],
        "checksums": result["checksums_path"],
    }, indent=2), flush=True)
