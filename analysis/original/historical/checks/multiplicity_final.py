"""Fail-closed final max-|t| sensitivity interface for the locked 26-cell family.

The audited station-multiplier calculations are reused from
``multiplicity_sensitivity.py``.  This module owns only the final-family
registry, source and exact-key validation, and immutable output packaging.
Running without ``--run`` performs a preflight and never opens an OOF file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import assess_deployment_support as _qa_context_helper  # noqa: E402
import fast_cluster_bootstrap as _bootstrap_helper  # noqa: E402
import multiplicity_sensitivity as audited  # noqa: E402


REGISTRY_PATH = HERE.parent / "final_multiplicity_family_registry.json"
PROTOCOL_PATH = HERE.parent / "multiplicity_final_protocol_addendum_20261113.md"
AUDITED_MATH_SOURCE_PATH = HERE.parent / "multiplicity_sensitivity.py"
QA_CONTEXT_HELPER_PATH = Path(_qa_context_helper.__file__).resolve()
BOOTSTRAP_HELPER_PATH = Path(_bootstrap_helper.__file__).resolve()
EXPECTED_PROTOCOL_SHA256 = "b3777b73fab176dda6c75868d456c92921d048d57d8146a6331474ecda1efc86"
EXPECTED_AUDITED_MATH_SOURCE_SHA256 = "08e3f2f8568b2fe79192aa8fa00fa25a8838e5ee02b7c43f6e3a7b444f268be1"
EXPECTED_HGB_CONFIG = {
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "random_state": 0,
    "estimator_kind": "hgb",
    "ridge_alpha": 1.0,
}
FINAL_OVERLAY_STATUS = "geometry_verified_source_coverage_audited"
EXPECTED_CELL_COUNT = 26
EXPECTED_MAX_CONTRASTS = 78
PRIMARY_HISTORICAL_TASKS = (
    "T1_station",
    "T2_spatial",
    "T2_spatial_buffer50km",
    "T3_year_2022",
)
PROFILES = ("with_met", "no_met")
COUNTRIES = ("USA", "CAN")
ARMS = ("B", "B+AOD", "full")
IDENTITY_COLUMNS = (
    "country", "cohort_label", "profile", "task", "fold_id", "row_id",
    "site_id", "year", "month",
)
EXPECTED_KEY_COLUMNS = tuple(column for column in IDENTITY_COLUMNS if column != "profile")
ARTIFACT_NAMES = (
    "oof_predictions", "input_manifest", "model_manifest", "expected_keys",
    "expected_keys_manifest",
)
SOURCE_BINDING_FORMAT = "repair-v2-final-multiplicity-source-binding-v1"
EXPECTED_KEYS_FORMAT = "repair-v2-final-multiplicity-expected-test-keys-v1"
FINAL_EXPECTED_KEYS_FIELD = "final_multiplicity_expected_test_keys"
FINAL_ROW_SELECTION = (
    "external_features_with_assigned_target_and_finite_nonnegative_pm25_and_finite_"
    "coordinates_and_year_equal_external_year"
)


class IncompleteFamilyError(ValueError):
    """Raised when a fixed family member is still pending final inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}".encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_row_key_sha256(frame: pd.DataFrame) -> str:
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _cell_key(cell: dict[str, Any]) -> tuple[str, str, str]:
    return str(cell["country"]), str(cell["profile"]), str(cell["task"])


def _expected_registry_cells() -> dict[tuple[str, str, str], dict[str, Any]]:
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for country in COUNTRIES:
        for profile in PROFILES:
            for task in PRIMARY_HISTORICAL_TASKS:
                cell = {
                    "country": country, "profile": profile, "kind": "historical",
                    "task": task,
                }
                expected[_cell_key(cell)] = cell
            external_years = (2023, 2024, 2025) if country == "USA" else (2023, 2024)
            for year in external_years:
                task = f"T4_external_{year}_{profile}"
                cell = {
                    "country": country, "profile": profile,
                    "kind": "fixed_external", "task": task,
                    "train_through": 2022, "external_year": year,
                }
                expected[_cell_key(cell)] = cell
    return expected


def _resolve_path(raw_path: str | Path, workspace: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _resolve_protocol_path(registry: dict[str, Any], workspace: Path) -> Path:
    raw_path = registry.get("protocol_path")
    if not raw_path:
        raise ValueError("Locked registry lacks its protocol path")
    path = _resolve_path(raw_path, workspace)
    if not path.is_file():
        path = Path(workspace).resolve() / "work" / "repair_v2" / PROTOCOL_PATH.name
    return path.resolve()


def _integrity_code_paths() -> dict[str, Path]:
    actual_math_path = Path(audited.__file__).resolve()
    actual_qa_path = Path(_qa_context_helper.input_context.__code__.co_filename).resolve()
    actual_bootstrap_path = Path(_bootstrap_helper.__file__).resolve()
    if actual_math_path != AUDITED_MATH_SOURCE_PATH.resolve():
        raise RuntimeError("Imported audited multiplicity math resolved from an unexpected source path")
    if actual_qa_path != QA_CONTEXT_HELPER_PATH or actual_bootstrap_path != BOOTSTRAP_HELPER_PATH:
        raise RuntimeError("Imported QA/helper dependency path changed unexpectedly")
    return {
        "final_multiplicity_code": HERE,
        "audited_v2_multiplicity_math": actual_math_path,
        "qa_context_helper": actual_qa_path,
        "bootstrap_helper_dependency": actual_bootstrap_path,
    }


def _capture_integrity_snapshot(registry: dict[str, Any], *, registry_path: Path,
                                workspace: Path) -> dict[str, Any]:
    """Parse and bind start-of-run registry/protocol/code bytes."""
    registry_path = Path(registry_path).resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"Locked family registry is absent: {registry_path}")
    registry_bytes = registry_path.read_bytes()
    disk_registry = json.loads(registry_bytes.decode("utf-8"))
    if _json_bytes(registry) != _json_bytes(disk_registry):
        raise ValueError("In-memory final registry differs from its verified on-disk bytes")
    validate_locked_registry(disk_registry, workspace=workspace)

    protocol_path = _resolve_protocol_path(disk_registry, workspace)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Hash-bound final multiplicity protocol is absent: {protocol_path}")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if protocol_sha != EXPECTED_PROTOCOL_SHA256 or protocol_sha != disk_registry["protocol_sha256"]:
        raise ValueError("Hash-bound final multiplicity protocol differs from the locked start bytes")

    code_paths = _integrity_code_paths()
    code_hashes = {}
    for name, path in code_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Final multiplicity code dependency is absent: {path}")
        code_hashes[name] = {"path": str(path.resolve()), "sha256_start": sha256_file(path)}
    if code_hashes["audited_v2_multiplicity_math"]["sha256_start"] != EXPECTED_AUDITED_MATH_SOURCE_SHA256:
        raise ValueError("Audited v2 multiplicity math source is absent or its bound SHA-256 changed")
    return {
        "registry": disk_registry,
        "registry_path": str(registry_path),
        "registry_sha256_start": hashlib.sha256(registry_bytes).hexdigest(),
        "protocol_path": str(protocol_path),
        "protocol_sha256_start": protocol_sha,
        "code_dependencies": code_hashes,
    }


def _verify_integrity_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reject any registry, protocol or imported helper edit before publication."""
    registry_path = Path(snapshot["registry_path"])
    protocol_path = Path(snapshot["protocol_path"])
    registry_end = sha256_file(registry_path)
    protocol_end = sha256_file(protocol_path)
    if registry_end != snapshot["registry_sha256_start"]:
        raise RuntimeError("Locked final registry changed during execution")
    if protocol_end != snapshot["protocol_sha256_start"]:
        raise RuntimeError("Locked final protocol changed during execution")
    code_dependencies = {}
    for name, record in snapshot["code_dependencies"].items():
        path = Path(record["path"])
        digest = sha256_file(path)
        if digest != record["sha256_start"]:
            raise RuntimeError(f"Final multiplicity code dependency changed during execution: {path}")
        code_dependencies[name] = {
            "path": str(path), "sha256_start": record["sha256_start"],
            "sha256_end": digest,
        }
    return {
        "registry": {
            "path": str(registry_path), "sha256_start": snapshot["registry_sha256_start"],
            "sha256_end": registry_end,
        },
        "protocol": {
            "path": str(protocol_path), "sha256_start": snapshot["protocol_sha256_start"],
            "sha256_end": protocol_end,
        },
        "code_dependencies": code_dependencies,
    }


def _integrity_source_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        {"role": "locked_family_registry", "path": snapshot["registry_path"],
         "sha256": snapshot["registry_sha256_start"]},
        {"role": "locked_protocol", "path": snapshot["protocol_path"],
         "sha256": snapshot["protocol_sha256_start"]},
    ]
    records.extend(
        {"role": name, "path": record["path"], "sha256": record["sha256_start"]}
        for name, record in snapshot["code_dependencies"].items()
    )
    return records


def load_registry(path: Path = REGISTRY_PATH, workspace: Path = WORKSPACE) -> dict[str, Any]:
    path = Path(path).resolve()
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_locked_registry(registry, workspace=workspace)
    return registry


def validate_locked_registry(registry: dict[str, Any], *, workspace: Path = WORKSPACE) -> None:
    """Validate the hash-bound family definition and every expected cell key."""
    if registry.get("format") != "repair-v2-final-multiplicity-family-v1":
        raise ValueError("Unsupported locked multiplicity registry format")
    fixed = {
        "locked_local_date": "2026-11-13",
        "before_new_year_pm_model_scores": True,
        "qa_variant": "strict",
        "estimator_kind": "hgb",
        "target_strategy": "log_smear",
        "smear_strategy": "crossfit_station",
        "backtransform": "smearing",
        "station_multiplier_distribution": "Exp(1)",
        "replicates": 10_000,
        "seed": 20261113,
        "minimum_test_stations": 30,
        "expected_cells": EXPECTED_CELL_COUNT,
        "maximum_contrasts": EXPECTED_MAX_CONTRASTS,
    }
    for key, value in fixed.items():
        if registry.get(key) != value:
            raise ValueError(f"Locked registry metadata mismatch for {key}: {registry.get(key)!r}")
    expected_contrasts = [name for name, _, _ in audited.CONTRASTS]
    if registry.get("contrasts") != expected_contrasts:
        raise ValueError("Locked registry contrast order or names changed")
    if registry.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Locked protocol SHA-256 field changed")
    protocol_path = registry.get("protocol_path")
    if not protocol_path:
        raise ValueError("Locked registry lacks its protocol path")
    resolved_protocol = _resolve_path(protocol_path, workspace)
    if not resolved_protocol.is_file():
        # Keep the binding portable for a relocated workspace while still
        # checking the exact shipped protocol bytes.
        resolved_protocol = Path(workspace) / "work" / "repair_v2" / PROTOCOL_PATH.name
    if not resolved_protocol.is_file() or sha256_file(resolved_protocol) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Hash-bound final multiplicity protocol is absent or changed")
    if (not AUDITED_MATH_SOURCE_PATH.is_file()
            or sha256_file(AUDITED_MATH_SOURCE_PATH) != EXPECTED_AUDITED_MATH_SOURCE_SHA256):
        raise ValueError("Audited v2 multiplicity math source is absent or its bound SHA-256 changed")

    cells = registry.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Locked registry must retain exactly {EXPECTED_CELL_COUNT} cells")
    expected = _expected_registry_cells()
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Registry cells must be JSON objects")
        key = _cell_key(cell)
        if key in observed:
            raise ValueError(f"Duplicate expected registry cell: {key}")
        observed[key] = cell
        for field, expected_value in expected.get(key, {}).items():
            if cell.get(field) != expected_value:
                raise ValueError(f"Registry cell {key} changed locked field {field}")
        if key not in expected:
            raise ValueError(f"Unexpected cell in locked registry: {key}")
        status = cell.get("status")
        binding = cell.get("source_binding")
        if status == "pending_final_inputs":
            continue
        if status != "ready":
            raise ValueError(f"Unsupported final cell status {status!r} for {key}")
        if not isinstance(binding, dict):
            raise ValueError(f"Ready cell lacks source_binding: {key}")
        _validate_binding_identity(cell, binding)
    missing = set(expected) - set(observed)
    if missing:
        raise ValueError(f"Locked registry omitted expected cells: {sorted(missing)}")


def _validate_binding_identity(cell: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    if binding.get("format") != SOURCE_BINDING_FORMAT:
        raise ValueError(f"Unsupported source_binding format for {_cell_key(cell)}")
    identity = binding.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"source_binding lacks identity for {_cell_key(cell)}")
    expected_identity = {
        "country": cell["country"], "profile": cell["profile"],
        "task": cell["task"], "kind": cell["kind"],
        "qa_variant": "strict", "input_acceptance_stage": "audited",
        "estimator_kind": "hgb", "target_strategy": "log_smear",
        "smear_strategy": "crossfit_station", "backtransform": "smearing",
        "arms": list(ARMS), "train_through": 2022,
    }
    if cell["kind"] == "fixed_external":
        expected_identity["external_year"] = cell["external_year"]
    for field, value in expected_identity.items():
        if identity.get(field) != value:
            raise ValueError(
                f"source_binding identity mismatch for {_cell_key(cell)}: "
                f"{field}={identity.get(field)!r}, expected {value!r}"
            )
    cohort = identity.get("cohort_label")
    if not isinstance(cohort, str) or not cohort.strip():
        raise ValueError(f"source_binding needs an explicit cohort_label for {_cell_key(cell)}")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"source_binding lacks artifacts for {_cell_key(cell)}")
    missing = set(ARTIFACT_NAMES) - set(artifacts)
    if missing:
        raise ValueError(f"source_binding lacks artifacts {sorted(missing)} for {_cell_key(cell)}")
    for name in ARTIFACT_NAMES:
        record = artifacts[name]
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"Invalid {name} artifact reference for {_cell_key(cell)}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"Invalid {name} SHA-256 for {_cell_key(cell)}")
    return identity


def preflight_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Return a registry-only status; this function never opens an OOF file."""
    cells = registry["cells"]
    pending = [
        {"country": cell["country"], "task": cell["task"], "profile": cell["profile"],
         "binding_present": isinstance(cell.get("source_binding"), dict)}
        for cell in cells if cell["status"] == "pending_final_inputs"
    ]
    ready = [cell for cell in cells if cell["status"] == "ready"]
    return {
        "status": "preflight_incomplete" if pending else "preflight_ready",
        "mode": "preflight",
        "future_oof_read": False,
        "scoring_performed": False,
        "n_cells_expected": len(cells),
        "n_cells_ready": len(ready),
        "n_cells_pending": len(pending),
        "n_contrasts_maximum": len(cells) * len(audited.CONTRASTS),
        "audited_math_source_sha256": EXPECTED_AUDITED_MATH_SOURCE_SHA256,
        "pending_cells": pending,
    }


def _record_path(record: dict[str, Any], *, workspace: Path, label: str,
                 verified: set[tuple[str, str]]) -> Path:
    path = _resolve_path(record["path"], workspace)
    expected_hash = record["sha256"].lower()
    cache_key = (str(path), expected_hash)
    if cache_key not in verified:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {label}: {path}")
        verified.add(cache_key)
    return path


class _ArtifactReader:
    """Hash-verifying, per-run cache for the repeated cell-level source files."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.verified: set[tuple[str, str]] = set()
        self.json_cache: dict[Path, dict[str, Any]] = {}
        self.frame_cache: dict[tuple[Path, str], pd.DataFrame] = {}

    def path(self, record: dict[str, Any], label: str) -> Path:
        return _record_path(record, workspace=self.workspace, label=label,
                            verified=self.verified)

    def json(self, record: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
        path = self.path(record, label)
        if path not in self.json_cache:
            self.json_cache[path] = json.loads(path.read_text(encoding="utf-8"))
        return path, self.json_cache[path]

    def table(self, record: dict[str, Any], label: str, *, kind: str) -> tuple[Path, pd.DataFrame]:
        path = self.path(record, label)
        cache_key = (path, kind)
        if cache_key not in self.frame_cache:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                if kind == "historical_oof":
                    columns = [column for column in audited.OOF_COLUMNS if column != "profile"]
                    frame = pd.read_parquet(path, columns=columns)
                elif kind == "external_oof":
                    frame = pd.read_parquet(path, columns=audited.OOF_COLUMNS)
                else:
                    frame = pd.read_parquet(path)
            elif suffix == ".csv":
                frame = pd.read_csv(path, dtype={"site_id": "string", "country": "string"})
            else:
                raise ValueError(f"Unsupported {label} table format: {path.suffix}")
            self.frame_cache[cache_key] = frame
        return path, self.frame_cache[cache_key]


def _checked_source_table(reader: _ArtifactReader, record: dict[str, Any],
                          label: str) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError(f"{label} lacks a source path")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} lacks a SHA-256")
    path = reader.path(record, label)
    _, frame = reader.table(record, label, kind="source_table")
    return path, frame, {
        "role": label, "path": str(path), "bytes": path.stat().st_size,
        "sha256": digest.lower(),
    }


def _country_feature_frame(frame: pd.DataFrame, country: str, *, label: str,
                           allow_missing_country: bool) -> pd.DataFrame:
    required = {"site_id", "year", "month", "lat", "lon"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} feature source lacks columns {sorted(missing)}")
    result = frame.copy()
    if "country" in result:
        result = result.loc[result.country.astype("string").str.upper().eq(country)].copy()
    elif allow_missing_country:
        result["country"] = country
    else:
        raise ValueError(f"{label} feature source lacks country")
    if result.empty:
        raise ValueError(f"{label} feature source has no {country} rows")
    result["country"] = country
    if result.site_id.isna().any():
        raise ValueError(f"{label} feature source has null site IDs")
    result["site_id"] = result.site_id.astype(str)
    result["year"] = pd.to_numeric(result.year, errors="raise").astype(int)
    result["month"] = pd.to_numeric(result.month, errors="raise").astype(int)
    result["lat"] = pd.to_numeric(result.lat, errors="coerce")
    result["lon"] = pd.to_numeric(result.lon, errors="coerce")
    return result


def _target_override_values(feature_frame: pd.DataFrame, target_frame: pd.DataFrame,
                            *, country: str, target_column: str,
                            label: str) -> pd.DataFrame:
    """Apply the runner's exact station/year/month target override join."""
    target = target_frame.copy()
    if "country" in target:
        target = target.loc[target.country.astype("string").str.upper().eq(country)].copy()
    if target.empty:
        raise ValueError(f"{label} target override has no {country} rows")
    if target_column not in target:
        raise ValueError(f"{label} target override lacks {target_column!r}")
    key_columns = ["site_id", "year", "month"]
    if not set(key_columns).issubset(target.columns):
        raise ValueError(f"{label} target override lacks station/month keys")
    if target.site_id.isna().any():
        raise ValueError(f"{label} target override has null site IDs")
    target_keys = target[key_columns].copy()
    target_keys["site_id"] = target_keys.site_id.astype(str)
    target_keys["year"] = pd.to_numeric(target_keys.year, errors="raise").astype(int)
    target_keys["month"] = pd.to_numeric(target_keys.month, errors="raise").astype(int)
    if target_keys.duplicated().any():
        raise ValueError(f"{label} target override contains duplicate station/month keys")
    values = pd.to_numeric(target[target_column], errors="coerce").to_numpy(dtype="float64")
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError(f"{label} target override contains infinite or negative targets")

    source_keys = feature_frame[key_columns].copy()
    source_keys["site_id"] = source_keys.site_id.astype(str)
    source_keys["year"] = pd.to_numeric(source_keys.year, errors="raise").astype(int)
    source_keys["month"] = pd.to_numeric(source_keys.month, errors="raise").astype(int)
    if source_keys.duplicated().any():
        raise ValueError(f"{label} feature source has duplicate station/month keys")
    source_years = set(source_keys.year.astype(int))
    in_scope = target_keys.year.isin(source_years).to_numpy()
    target_scope = target_keys.loc[in_scope]
    lookup = {
        tuple(key): value
        for key, value in zip(target_scope.itertuples(index=False, name=None), values[in_scope])
    }
    keys = list(source_keys.itertuples(index=False, name=None))
    result = feature_frame.copy()
    result["pm25"] = [lookup.get(key, np.nan) for key in keys]
    return result


def _apply_target_source(feature_frame: pd.DataFrame, feature_record: dict[str, Any],
                         target_record: dict[str, Any] | None, *, country: str,
                         reader: _ArtifactReader, label: str,
                         historical: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if target_record is None:
        if historical:
            if "pm25" not in feature_frame:
                raise ValueError(f"{label} embedded feature target column pm25 is absent")
            result = feature_frame.copy()
            result["pm25"] = pd.to_numeric(result.pm25, errors="coerce")
            return result, []
        raise ValueError(f"{label} target source is absent")

    source_kind = target_record.get("source_kind")
    if not historical and source_kind not in {
        "embedded_feature_target_column", "standalone_target_override_table",
    }:
        raise ValueError(f"{label} target source must declare its source_kind")
    if not historical:
        expected_role = (
            f"{label.removeprefix('external_')}_target_embedded_in_feature_file"
            if source_kind == "embedded_feature_target_column"
            else f"{label.removeprefix('external_')}_target_override_table"
        )
        if target_record.get("role") != expected_role:
            raise ValueError(f"{label} target source has a mismatched provenance role")
    target_path = target_record.get("path")
    target_hash = target_record.get("sha256")
    feature_path = _resolve_path(feature_record["path"], reader.workspace)
    feature_hash = str(feature_record["sha256"]).lower()
    if source_kind == "embedded_feature_target_column" or (
        historical and not target_path and target_hash is None
    ):
        if (target_path and _resolve_path(target_path, reader.workspace) != feature_path) or (
            target_hash and str(target_hash).lower() != feature_hash
        ):
            raise ValueError(f"{label} embedded target is not bound to its feature source")
        target_column = str(target_record.get("target_column", "pm25"))
        if target_column != "pm25" or "pm25" not in feature_frame:
            raise ValueError(f"{label} embedded target must use the feature pm25 column")
        result = feature_frame.copy()
        result["pm25"] = pd.to_numeric(result.pm25, errors="coerce")
        return result, []

    if source_kind not in {None, "standalone_target_override_table"}:
        raise ValueError(f"{label} target source has an unsupported source_kind")
    if not target_path or not isinstance(target_hash, str):
        raise ValueError(f"{label} target override lacks a hash-bound path")
    target_record = {**target_record, "path": target_path, "sha256": target_hash}
    path, frame, source = _checked_source_table(reader, target_record, f"{label}_target_source")
    target_column = str(target_record.get("target_column", "pm25"))
    result = _target_override_values(
        feature_frame, frame, country=country, target_column=target_column,
        label=label,
    )
    source["path"] = str(path)
    return result, [source]


def _validate_final_qa_provenance(manifest: dict[str, Any], *, cell: dict[str, Any],
                                  external: bool) -> None:
    if external:
        records = [
            ("training", manifest.get("training_input_provenance")),
            ("external", manifest.get("external_input_provenance")),
        ]
    else:
        country_record = manifest.get(cell["country"], {})
        records = [("feature", country_record.get("feature_input_provenance"))]
    for role, record in records:
        provenance = record.get("manifest") if isinstance(record, dict) else None
        if not isinstance(provenance, dict):
            raise ValueError(f"Final {role} inputs lack hash-bound QA provenance")
        if (provenance.get("aod_variant") != "strict"
                or provenance.get("overlay_status") != FINAL_OVERLAY_STATUS
                or provenance.get("geometry_complete") is not True
                or provenance.get("source_coverage_audited") is not True
                or provenance.get("raw_source_gap_audit_pending") is not False):
            raise ValueError(
                f"Final {role} input provenance must declare strict QA, complete geometry/source audit, "
                "and no pending raw-source gap audit"
            )


def _reconstruct_historical_source(source_country: dict[str, Any], *, country: str,
                                   reader: _ArtifactReader) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = source_country.get("feature_tables")
    hashes = source_country.get("feature_sha256")
    if (not isinstance(paths, list) or not paths or not isinstance(hashes, list)
            or len(paths) != len(hashes)):
        raise ValueError("Historical source manifest must pair every feature path with its SHA-256")
    feature_frames = []
    source_records = []
    for index, (raw_path, digest) in enumerate(zip(paths, hashes)):
        record = {"path": raw_path, "sha256": digest}
        path, frame, source = _checked_source_table(
            reader, record, f"historical_feature_table_{index + 1}"
        )
        source_records.append(source)
        feature_frames.append(_country_feature_frame(
            frame, country, label=f"historical feature table {path.name}",
            allow_missing_country=True,
        ))
    features = pd.concat(feature_frames, ignore_index=True, sort=False)
    declared_source_rows = source_country.get("feature_rows_source")
    if not isinstance(declared_source_rows, int) or declared_source_rows != len(features):
        raise ValueError("Historical source row count disagrees with its feature tables")

    target_status = source_country.get("target_status", {})
    target_hash = source_country.get("target_source_sha256")
    target_path = target_status.get("path") if isinstance(target_status, dict) else None
    if target_path:
        if not isinstance(target_hash, str) or not target_status.get("sha256"):
            raise ValueError("Historical corrected target lacks a hash-bound source path")
        if str(target_status["sha256"]).lower() != str(target_hash).lower():
            raise ValueError("Historical target source SHA disagrees with source manifest")
        target_record = {
            "path": target_path, "sha256": target_hash,
            "target_column": target_status.get("target_column", "pm25"),
            "source_kind": "standalone_target_override_table",
        }
        features, target_records = _apply_target_source(
            features, {"path": paths[0], "sha256": hashes[0]}, target_record,
            country=country, reader=reader, label="historical", historical=True,
        )
        source_records.extend(target_records)
    else:
        if target_hash is not None:
            raise ValueError("Historical target source hash has no parseable path")
        features, _ = _apply_target_source(
            features, {"path": paths[0], "sha256": hashes[0]}, None,
            country=country, reader=reader, label="historical", historical=True,
        )

    if "pm25" not in features:
        raise ValueError("Historical feature tables lack their embedded pm25 target")
    target_values = pd.to_numeric(features.pm25, errors="coerce")
    coord_ok = np.isfinite(pd.to_numeric(features.lat, errors="coerce")) & np.isfinite(
        pd.to_numeric(features.lon, errors="coerce")
    )
    target_ok = np.isfinite(target_values) & (target_values >= 0)
    analysis = features.loc[coord_ok & target_ok].copy()
    if analysis.empty:
        raise ValueError(f"Historical source has no eligible target/coordinate rows for {country}")
    analysis["year"] = pd.to_numeric(analysis.year, errors="raise").astype(int)
    analysis["month"] = pd.to_numeric(analysis.month, errors="raise").astype(int)
    analysis["site_id"] = analysis.site_id.astype(str)
    analysis["pm25"] = target_values.loc[analysis.index].to_numpy(dtype="float64")
    analysis.reset_index(drop=True, inplace=True)
    analysis["_row_id"] = np.arange(len(analysis), dtype=np.int64)

    duplicate_mask = analysis.duplicated(["country", "site_id", "year", "month"], keep=False)
    collapsed = 0
    duplicate_keys = 0
    if duplicate_mask.any():
        compare_columns = [column for column in analysis.columns if column != "_row_id"]
        conflicts = []
        for key, group in analysis.loc[duplicate_mask].groupby(
            ["country", "site_id", "year", "month"], dropna=False, sort=False
        ):
            if len(group[compare_columns].drop_duplicates()) > 1:
                conflicts.append(key)
        if conflicts:
            raise ValueError("Historical source contains conflicting duplicate station-month rows")
        duplicate_keys = int(analysis.loc[duplicate_mask, ["country", "site_id", "year", "month"]].drop_duplicates().shape[0])
        before = len(analysis)
        analysis = analysis.drop_duplicates(["country", "site_id", "year", "month"], keep="first").copy()
        collapsed = before - len(analysis)
        analysis.reset_index(drop=True, inplace=True)
        analysis["_row_id"] = np.arange(len(analysis), dtype=np.int64)

    declared_analysis_rows = source_country.get("analysis_rows")
    if not isinstance(declared_analysis_rows, int) or declared_analysis_rows != len(analysis):
        raise ValueError("Historical eligible row count disagrees with its source manifest")
    declared_key_hash = source_country.get("analysis_row_key_sha256")
    if (not isinstance(declared_key_hash, str) or len(declared_key_hash) != 64
            or declared_key_hash != _canonical_row_key_sha256(analysis)):
        raise ValueError("Historical eligible row keys disagree with the source manifest commitment")
    duplicate_audit = source_country.get("duplicate_audit")
    if not isinstance(duplicate_audit, dict):
        raise ValueError("Historical source manifest lacks a duplicate-audit record")
    if (int(duplicate_audit.get("exact_rows_collapsed", -1)) != collapsed
            or int(duplicate_audit.get("duplicate_keys", -1)) != duplicate_keys
            or int(duplicate_audit.get("conflicting_keys", -1)) != 0):
        raise ValueError("Historical duplicate audit disagrees with source reconstruction")
    return analysis, source_records


def _reconstruct_external_sources(model_manifest: dict[str, Any], *, country: str,
                                  external_year: int, train_through: int,
                                  reader: _ArtifactReader) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    source_records = []
    role_frames: dict[str, pd.DataFrame] = {}
    for role, feature_name, target_name in (
        ("training", "training_features", "train_target_source"),
        ("external", "external_features", "external_target_source"),
    ):
        feature_record = model_manifest.get(feature_name)
        target_record = model_manifest.get(target_name)
        feature_path, feature_table, feature_source = _checked_source_table(
            reader, feature_record, f"external_{role}_features"
        )
        source_records.append(feature_source)
        features = _country_feature_frame(
            feature_table, country, label=f"external {role} features",
            allow_missing_country=False,
        )
        declared_rows = feature_record.get("rows_source") if isinstance(feature_record, dict) else None
        if declared_rows is not None and int(declared_rows) != len(features):
            raise ValueError(f"External {role} feature row count disagrees with its manifest")
        if not isinstance(target_record, dict):
            raise ValueError(f"External manifest lacks the {role} corrected target source")
        if not isinstance(target_record.get("status"), str) or not target_record.get("status").strip():
            raise ValueError(f"External {role} target source status is missing")
        assigned, target_sources = _apply_target_source(
            features, feature_record, target_record, country=country,
            reader=reader, label=f"external_{role}",
        )
        source_records.extend(target_sources)
        # Embedded targets are still input files and must appear in provenance
        # as an explicit role even when the feature record already has a record.
        if not target_sources:
            source_records.append({
                "role": f"external_{role}_target_embedded_in_features",
                "path": str(feature_path), "bytes": feature_path.stat().st_size,
                "sha256": str(feature_record["sha256"]).lower(),
            })
        role_frames[role] = assigned

    train = role_frames["training"].loc[
        role_frames["training"].year.le(int(train_through))
    ].copy()
    test = role_frames["external"].loc[
        role_frames["external"].year.eq(int(external_year))
    ].copy()
    if train.empty or test.empty:
        raise ValueError("External source reconstruction produced an empty train or test cohort")
    for role, frame in (("training", train), ("external", test)):
        targets = pd.to_numeric(frame.pm25, errors="coerce").to_numpy(dtype="float64")
        lat = pd.to_numeric(frame.lat, errors="coerce").to_numpy(dtype="float64")
        lon = pd.to_numeric(frame.lon, errors="coerce").to_numpy(dtype="float64")
        if (not np.isfinite(targets).all() or np.any(targets < 0)
                or not np.isfinite(lat).all() or not np.isfinite(lon).all()):
            raise ValueError(f"External {role} source has invalid target or coordinates")
    if test.duplicated(["site_id", "year", "month"]).any():
        raise ValueError("External expected test source contains duplicate station-month keys")
    test.reset_index(drop=True, inplace=True)
    test["row_id"] = np.arange(len(test), dtype=np.int64)
    test["obs"] = pd.to_numeric(test.pm25, errors="raise").to_numpy(dtype="float64")
    return train, test, source_records


def _external_expected_key_frame(test: pd.DataFrame, *, country: str,
                                 cohort_label: str, profiles: list[str],
                                 external_year: int) -> pd.DataFrame:
    parts = []
    for profile in profiles:
        part = pd.DataFrame({
            "country": country,
            "cohort_label": cohort_label,
            "profile": profile,
            "task": f"T4_external_{int(external_year)}_{profile}",
            "fold_id": str(int(external_year)),
            "row_id": test.row_id.to_numpy(dtype="int64"),
            "site_id": test.site_id.astype(str).to_numpy(dtype=object),
            "year": test.year.to_numpy(dtype="int64"),
            "month": test.month.to_numpy(dtype="int64"),
            "obs": test.obs.to_numpy(dtype="float64"),
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[[*IDENTITY_COLUMNS, "obs"]]


def _source_observations_for_historical_cell(
    cell: dict[str, Any], source_country: dict[str, Any], analysis: pd.DataFrame,
    expected_keys: pd.DataFrame,
) -> dict[tuple[str, ...], float]:
    if "role" in expected_keys:
        expected_keys = expected_keys.loc[
            expected_keys.role.astype(str).str.casefold().eq("test")
        ].copy()
    cell_rows = expected_keys.loc[
        expected_keys.country.astype(str).str.upper().eq(cell["country"])
        & expected_keys.cohort_label.astype(str).eq(source_country["cohort_label"])
        & expected_keys.task.astype(str).eq(cell["task"])
    ].copy()
    if cell_rows.empty:
        raise ValueError(f"Historical expected test artifact has no source rows for {_cell_key(cell)}")
    row_ids = pd.to_numeric(cell_rows.row_id, errors="raise").to_numpy(dtype="int64")
    if (row_ids < 0).any() or (row_ids >= len(analysis)).any():
        raise ValueError(f"Historical test row IDs are outside the reconstructed source for {_cell_key(cell)}")
    rebuilt = analysis.iloc[row_ids].reset_index(drop=True)
    expected_identity = cell_rows.reset_index(drop=True)
    for column in ("country", "site_id", "year", "month"):
        left = rebuilt[column].astype(str).to_numpy()
        right = expected_identity[column].astype(str).to_numpy()
        if column == "country":
            left = np.char.upper(left.astype(str))
            right = np.char.upper(right.astype(str))
        elif column in {"year", "month"}:
            left = pd.to_numeric(left, errors="raise").astype("int64")
            right = pd.to_numeric(right, errors="raise").astype("int64")
        if not np.array_equal(left, right):
            raise ValueError(
                f"Historical committed test keys do not reconstruct from source rows for {_cell_key(cell)}"
            )
    tokens = _key_tokens_in_order(
        cell_rows, profile=cell["profile"], label="historical expected source keys",
        allow_missing_profile=True,
    )
    values = rebuilt.pm25.to_numpy(dtype="float64")
    return dict(zip(tokens, values))


def _source_observations_for_external_cell(
    cell: dict[str, Any], commitment: dict[str, Any], expected_keys: pd.DataFrame,
    test: pd.DataFrame, model_manifest: dict[str, Any],
) -> dict[tuple[str, ...], float]:
    required_columns = [*IDENTITY_COLUMNS, "obs"]
    if list(expected_keys.columns) != required_columns:
        raise ValueError("External exact-test-key artifact must contain ordered identity columns plus obs")
    if (commitment.get("target_column") != "pm25"
            or commitment.get("observed_column") != "obs"):
        raise ValueError("External exact-test-key commitment lacks the pm25/obs target labels")
    artifact_obs = pd.to_numeric(expected_keys.obs, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(artifact_obs).all():
        raise ValueError("External exact-test-key artifact contains nonfinite observed targets")
    if commitment.get("target_values_sha256") != _array_sha256(artifact_obs):
        raise ValueError("External exact-test-key observed-target hash disagrees with its artifact")

    profiles = model_manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("External model manifest must declare unique expected-key profiles")
    artifact_profiles = set(expected_keys.profile.astype(str))
    if artifact_profiles != set(map(str, profiles)):
        raise ValueError("External expected-key artifact profile coverage disagrees with its manifest")
    reconstructed = _external_expected_key_frame(
        test, country=cell["country"], cohort_label=cell["source_binding"]["identity"]["cohort_label"],
        profiles=list(map(str, profiles)), external_year=int(cell["external_year"]),
    )
    artifact_target_map: dict[tuple[str, ...], float] = {}
    reconstructed_target_map: dict[tuple[str, ...], float] = {}
    for profile in map(str, profiles):
        artifact_rows = expected_keys.loc[expected_keys.profile.astype(str).eq(profile)].copy()
        reconstructed_rows = reconstructed.loc[reconstructed.profile.astype(str).eq(profile)].copy()
        if _key_tuples(
            reconstructed_rows, profile=profile, label="reconstructed external expected keys",
            allow_missing_profile=False,
        ) != _key_tuples(
            artifact_rows, profile=profile, label="external committed expected keys",
            allow_missing_profile=False,
        ):
            raise ValueError("External exact-test-key artifact differs from reconstructed clean source keys")
        artifact_tokens = _key_tokens_in_order(
            artifact_rows, profile=profile, label="external target hash keys",
            allow_missing_profile=False,
        )
        reconstructed_tokens = _key_tokens_in_order(
            reconstructed_rows, profile=profile, label="external reconstructed source targets",
            allow_missing_profile=False,
        )
        artifact_target_map.update(dict(zip(artifact_tokens, artifact_rows.obs.to_numpy(dtype="float64"))))
        reconstructed_target_map.update(dict(zip(
            reconstructed_tokens, reconstructed_rows.obs.to_numpy(dtype="float64")
        )))
    if artifact_target_map != reconstructed_target_map:
        raise ValueError("External committed observed targets disagree with the original clean source values")

    bound_profile = str(cell["profile"])
    bound_rows = expected_keys.loc[expected_keys.profile.astype(str).eq(bound_profile)].copy()
    return dict(zip(
        _key_tokens_in_order(bound_rows, profile=bound_profile, label="external cell source keys",
                             allow_missing_profile=False),
        pd.to_numeric(bound_rows.obs, errors="raise").to_numpy(dtype="float64"),
    ))


def _manifest_ref_matches(raw_path: Any, raw_hash: Any, artifact_path: Path,
                          artifact_hash: str, *, workspace: Path, label: str) -> None:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} manifest path commitment is missing")
    if _resolve_path(raw_path, workspace) != artifact_path.resolve():
        raise ValueError(f"{label} manifest path commitment disagrees with binding")
    if str(raw_hash).lower() != artifact_hash.lower():
        raise ValueError(f"{label} manifest SHA-256 commitment disagrees with binding")


def _verify_historical_split_keys(cell: dict[str, Any], source_country: dict[str, Any],
                                  split_manifest: dict[str, Any], assignments: pd.DataFrame) -> None:
    """Check every committed fold's row IDs, station-month keys, and source hashes."""
    required = set(EXPECTED_KEY_COLUMNS) | {"role"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Historical outer-fold assignment lacks {sorted(required - set(assignments.columns))}")
    cell_rows = assignments.loc[
        assignments.country.astype(str).str.upper().eq(cell["country"])
        & assignments.cohort_label.astype(str).eq(source_country["cohort_label"])
        & assignments.task.astype(str).eq(cell["task"])
        & assignments.role.astype(str).str.casefold().eq("test")
    ].copy()
    if cell_rows.empty:
        raise ValueError(f"Historical split manifest has no test-key assignments for {_cell_key(cell)}")
    records = [
        record for record in split_manifest.get("splits", [])
        if str(record.get("country", "")).upper() == cell["country"]
        and record.get("cohort_label") == source_country["cohort_label"]
        and record.get("task") == cell["task"]
    ]
    fold_map = {audited._fold_token(record.get("fold_id")): record for record in records}
    if len(fold_map) != len(records):
        raise ValueError(f"Duplicate fold records in historical split manifest for {_cell_key(cell)}")
    assignment_folds = set(cell_rows.fold_id.map(audited._fold_token))
    if set(fold_map) != assignment_folds:
        raise ValueError(
            f"Historical outer-fold manifest coverage mismatch for {_cell_key(cell)}: "
            f"assignments={sorted(assignment_folds)}, splits={sorted(fold_map)}"
        )
    features = source_country.get("feature_sha256")
    if not isinstance(features, list) or not features:
        raise ValueError("Historical source manifest lacks hash-bound feature source SHA values")
    target_hash = source_country.get("target_source_sha256")
    for fold_id, frame in cell_rows.groupby(cell_rows.fold_id.map(audited._fold_token), sort=True):
        record = fold_map[fold_id]
        row_ids = pd.to_numeric(frame.row_id, errors="raise").to_numpy(dtype="int64")
        row_key_hash = _canonical_row_key_sha256(frame)
        if (int(record.get("test_row_count", -1)) != len(frame)
                or int(record.get("test_station_count", -1)) != frame.site_id.astype(str).nunique()
                or record.get("test_row_ids_sha256") != _array_sha256(np.sort(row_ids))
                or record.get("test_row_key_sha256") != row_key_hash):
            raise ValueError(f"Historical committed test keys disagree for {_cell_key(cell)} fold {fold_id}")
        if record.get("feature_source_sha256") != features:
            raise ValueError(f"Historical fold feature hashes disagree with the source manifest for {_cell_key(cell)}")
        if record.get("target_source_sha256") != target_hash:
            raise ValueError(f"Historical fold target hash disagrees with the source manifest for {_cell_key(cell)}")
        if record.get("analysis_row_key_sha256") != source_country.get("analysis_row_key_sha256"):
            raise ValueError(f"Historical fold analysis-key hash disagrees with reconstructed source for {_cell_key(cell)}")


def _assert_hgb_config(config: Any, label: str) -> None:
    if not isinstance(config, dict) or _json_bytes(config) != _json_bytes(EXPECTED_HGB_CONFIG):
        raise ValueError(
            f"{label} does not match the locked primary HGB configuration "
            f"{EXPECTED_HGB_CONFIG}"
        )


def _assert_hgb_model(model_manifest: dict[str, Any], identity: dict[str, Any],
                      *, external: bool) -> None:
    kind = str(model_manifest.get("estimator_kind", "")).casefold()
    model_class = "".join(c for c in str(model_manifest.get("estimator_class", "")).casefold() if c.isalnum())
    config = model_manifest.get("model_config")
    if kind != "hgb" or not isinstance(config, dict) or config.get("estimator_kind") != "hgb":
        raise ValueError("Final multiplicity input must explicitly declare an HGB model/config")
    _assert_hgb_config(config, "Model manifest HGB configuration")
    if not model_class.endswith("histgradientboostingregressor"):
        raise ValueError("Final multiplicity input has a non-HGB estimator class")
    targets = model_manifest.get("target_strategies")
    smears = model_manifest.get("smear_strategies")
    if not isinstance(targets, list) or "log_smear" not in targets:
        raise ValueError("Model manifest does not declare log_smear")
    if not isinstance(smears, list) or "crossfit_station" not in smears:
        raise ValueError("Model manifest does not declare crossfit_station smearing")
    if external:
        model_records = model_manifest.get("models")
        if not isinstance(model_records, dict):
            raise ValueError("External manifest lacks explicit per-profile model records")
        record_key = f"{identity['profile']}:log_smear:crossfit_station"
        record = model_records.get(record_key)
        if not isinstance(record, dict):
            raise ValueError(f"External model manifest lacks {record_key}")
        if (record.get("estimator_kind") != "hgb"
                or record.get("target_strategy") != "log_smear"
                or record.get("smear_strategy") != "crossfit_station"):
            raise ValueError(f"External model record does not match the fixed HGB target pipeline: {record_key}")
        _assert_hgb_config(record.get("config"), f"External model record {record_key} configuration")
        record_class = "".join(c for c in str(record.get("estimator_class", "")).casefold() if c.isalnum())
        if not record_class.endswith("histgradientboostingregressor"):
            raise ValueError(f"External model record is not HGB: {record_key}")
    else:
        arms = model_manifest.get("crossfit_smear_arms")
        if not isinstance(arms, list) or not set(ARMS).issubset(set(arms)):
            raise ValueError("Historical run manifest does not bind crossfit predictions for all primary arms")


def _validate_historical_fold_model_configs(
    cell: dict[str, Any], source_country: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    """Require every committed fold's fitted full-arm record to match the lock."""
    records = [
        record for record in split_manifest.get("splits", [])
        if str(record.get("country", "")).upper() == cell["country"]
        and record.get("cohort_label") == source_country.get("cohort_label")
        and record.get("task") == cell["task"]
    ]
    if not records:
        raise ValueError(f"Historical split manifest has no model folds for {_cell_key(cell)}")
    _assert_hgb_config(split_manifest.get("model_config"), "Historical split-manifest HGB configuration")
    split_by_fold = {audited._fold_token(row.get("fold_id")): row for row in records}
    if len(split_by_fold) != len(records):
        raise ValueError(f"Historical split manifest has duplicate model folds for {_cell_key(cell)}")

    outer_models = source_country.get("outer_fold_models")
    if not isinstance(outer_models, list):
        raise ValueError("Historical source manifest lacks per-fold HGB model records")
    model_by_fold = {
        audited._fold_token(row.get("fold_id")): row
        for row in outer_models if row.get("task") == cell["task"]
    }
    if len(model_by_fold) != sum(row.get("task") == cell["task"] for row in outer_models):
        raise ValueError(f"Historical source manifest has duplicate per-fold models for {_cell_key(cell)}")
    if set(model_by_fold) != set(split_by_fold):
        raise ValueError(f"Historical per-fold HGB models disagree with committed folds for {_cell_key(cell)}")

    for fold, split in split_by_fold.items():
        model = model_by_fold[fold]
        if (model.get("estimator_kind") != "hgb"
                or model.get("target_strategy") != "log_smear"
                or model.get("smear_strategy") != "crossfit_station"):
            raise ValueError(f"Historical fold model does not match the fixed primary pipeline: {fold}")
        model_class = "".join(c for c in str(model.get("estimator_class", "")).casefold() if c.isalnum())
        if not model_class.endswith("histgradientboostingregressor"):
            raise ValueError(f"Historical fold model is not HGB: {fold}")
        _assert_hgb_config(model.get("config"), f"Historical fold {fold} HGB configuration")

        full_model = split.get("full_model")
        if not isinstance(full_model, dict) or not isinstance(full_model.get("binding"), dict):
            raise ValueError(f"Historical committed fold lacks a full-arm model binding: {fold}")
        binding = full_model["binding"]
        if full_model.get("binding_sha256") != _json_sha256(binding):
            raise ValueError(f"Historical full-arm fold binding hash mismatch: {fold}")
        if (binding.get("estimator_kind") != "hgb"
                or binding.get("target_strategy") != "log_smear"
                or binding.get("smear_strategy") != "crossfit_station"):
            raise ValueError(f"Historical full-arm fold binding has the wrong model pipeline: {fold}")
        _assert_hgb_config(binding.get("model_config"), f"Historical full-arm fold {fold} HGB configuration")
        if model.get("binding_sha256") != full_model.get("binding_sha256"):
            raise ValueError(f"Historical source/run model manifests disagree for fold {fold}")


def _validate_binding_manifests(cell: dict[str, Any], binding: dict[str, Any],
                                reader: _ArtifactReader) -> dict[str, Any]:
    identity = _validate_binding_identity(cell, binding)
    artifacts = binding["artifacts"]
    input_path, input_manifest = reader.json(artifacts["input_manifest"], "input_manifest")
    model_path, model_manifest = reader.json(artifacts["model_manifest"], "model_manifest")
    expected_manifest_path, expected_manifest = reader.json(
        artifacts["expected_keys_manifest"], "expected_keys_manifest"
    )
    expected_keys_path = reader.path(artifacts["expected_keys"], "expected_keys")
    oof_path = reader.path(artifacts["oof_predictions"], "oof_predictions")
    external = cell["kind"] == "fixed_external"

    contexts = audited.input_context(input_manifest)
    country_context = contexts.get(cell["country"])
    if not isinstance(country_context, dict):
        raise ValueError(f"Input manifest lacks QA context for {cell['country']}")
    if (country_context.get("qa_variant") != "strict"
            or country_context.get("input_acceptance_stage") != "audited"):
        raise ValueError("Final inputs require audited strict QA acceptance")
    if identity.get("qa_variant") != country_context["qa_variant"]:
        raise ValueError("Binding QA variant disagrees with hash-bound input manifest")
    if identity.get("input_acceptance_stage") != country_context["input_acceptance_stage"]:
        raise ValueError("Binding input-acceptance stage disagrees with hash-bound input manifest")
    _validate_final_qa_provenance(input_manifest, cell=cell, external=external)

    _assert_hgb_model(model_manifest, identity, external=external)
    raw_source_records: list[dict[str, Any]] = []
    source_observations: dict[tuple[str, ...], float]
    if external:
        if input_path != model_path or expected_manifest_path != model_path:
            raise ValueError("External input, model, and exact-key commitments must use one core external manifest")
        if (model_manifest.get("country") != cell["country"]
                or int(model_manifest.get("train_through", -1)) != 2022
                or int(model_manifest.get("external_year", -1)) != int(cell["external_year"])):
            raise ValueError("External manifest country/train cutoff/year mismatch")
        if cell["profile"] not in set(model_manifest.get("profiles", [])):
            raise ValueError("External manifest does not include the bound meteorology profile")
        if model_manifest.get("external_target_source", {}).get("status") != identity["cohort_label"]:
            raise ValueError("External target cohort disagrees with the hash-bound external manifest")
        train_cohort = identity.get("training_cohort_label")
        if not isinstance(train_cohort, str) or not train_cohort.strip():
            raise ValueError("External binding must declare its fixed training cohort label")
        if model_manifest.get("train_target_source", {}).get("status") != train_cohort:
            raise ValueError("External training cohort disagrees with the hash-bound external manifest")
        feature_hash = model_manifest.get("external_features", {}).get("sha256")
        target_hash = model_manifest.get("external_target_source", {}).get("sha256")
        if not isinstance(feature_hash, str) or len(feature_hash) != 64:
            raise ValueError("External manifest lacks a hash-bound feature source")
        if not isinstance(target_hash, str) or len(target_hash) != 64:
            raise ValueError("External manifest lacks a hash-bound corrected target source")
        for field, label in (
            ("training_features", "external_training_features"),
            ("external_features", "external_test_features"),
            ("train_target_source", "external_training_target_source"),
            ("external_target_source", "external_test_target_source"),
        ):
            source = model_manifest.get(field)
            if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
                raise ValueError(f"External manifest lacks hash-bound {field} path/SHA")
            reader.path(source, label)
        commitment = model_manifest.get(FINAL_EXPECTED_KEYS_FIELD)
        if not isinstance(commitment, dict) or commitment.get("format") != EXPECTED_KEYS_FORMAT:
            raise ValueError("External core manifest lacks the final exact-test-key commitment")
        _manifest_ref_matches(
            commitment.get("path"), commitment.get("sha256"), expected_keys_path,
            artifacts["expected_keys"]["sha256"], workspace=reader.workspace,
            label="External exact-test-key artifact",
        )
        if commitment.get("key_columns") != list(IDENTITY_COLUMNS):
            raise ValueError("External exact-test-key commitment has unexpected key columns")
        if commitment.get("row_selection") != FINAL_ROW_SELECTION:
            raise ValueError("External exact-test-key commitment has the wrong row-selection rule")
        if not isinstance(commitment.get("rows"), int) or commitment["rows"] < 0:
            raise ValueError("External exact-test-key commitment lacks a valid row count")
        if commitment.get("derived_from") != {
            "external_features_sha256": feature_hash,
            "external_target_source_sha256": target_hash,
        }:
            raise ValueError("External exact-test-key source hashes do not match the bound feature/target sources")
        _, expected_key_frame = reader.table(artifacts["expected_keys"], "expected_keys", kind="keys")
        if len(expected_key_frame) != commitment["rows"]:
            raise ValueError("External exact-test-key row count disagrees with the core manifest commitment")
        _, source_test, raw_source_records = _reconstruct_external_sources(
            model_manifest, country=cell["country"], external_year=int(cell["external_year"]),
            train_through=int(identity["train_through"]), reader=reader,
        )
        source_observations = _source_observations_for_external_cell(
            cell, commitment, expected_key_frame, source_test, model_manifest,
        )
    else:
        source_country = input_manifest.get(cell["country"])
        if not isinstance(source_country, dict):
            raise ValueError("Historical input manifest lacks its bound country record")
        if source_country.get("cohort_label") != identity["cohort_label"]:
            raise ValueError("Historical cohort disagrees with the hash-bound input manifest")
        if (source_country.get("country_met_profile_name") != cell["profile"]
                or source_country.get("country_met_profile_selection") not in {"source_declared", "no_met_override"}):
            raise ValueError("Historical profile disagrees with the hash-bound input manifest")
        if model_manifest.get("outer_fold_split_manifest") is None:
            raise ValueError("Historical run manifest lacks an outer-fold split-manifest commitment")
        split_ref = model_manifest["outer_fold_split_manifest"]
        split_ref_record = {
            "path": str(expected_manifest_path),
            "sha256": artifacts["expected_keys_manifest"]["sha256"],
        }
        _manifest_ref_matches(
            split_ref.get("path"), split_ref.get("sha256"), expected_manifest_path,
            split_ref_record["sha256"], workspace=reader.workspace,
            label="Historical split manifest",
        )
        _manifest_ref_matches(
            split_ref.get("assignment_path"), split_ref.get("assignment_sha256"),
            expected_keys_path, artifacts["expected_keys"]["sha256"],
            workspace=reader.workspace, label="Historical outer-fold assignment",
        )
        if (expected_manifest.get("format") != "repair-v2-outer-fold-split-manifest-v1"
                or expected_manifest.get("assignment_rows") != split_ref.get("assignment_rows")
                or expected_manifest.get("assignment_sha256") != artifacts["expected_keys"]["sha256"]):
            raise ValueError("Historical exact test keys disagree with the committed outer-fold split manifest")
        _, expected_key_frame = reader.table(artifacts["expected_keys"], "expected_keys", kind="keys")
        if len(expected_key_frame) != expected_manifest["assignment_rows"]:
            raise ValueError("Historical outer-fold assignment row count disagrees with its split manifest")
        _verify_historical_split_keys(cell, source_country, expected_manifest, expected_key_frame)
        _validate_historical_fold_model_configs(cell, source_country, expected_manifest)
        _manifest_ref_matches(
            expected_manifest.get("assignment_path"), expected_manifest.get("assignment_sha256"),
            expected_keys_path, artifacts["expected_keys"]["sha256"],
            workspace=reader.workspace, label="Historical split-manifest assignment",
        )
        source_analysis, raw_source_records = _reconstruct_historical_source(
            source_country, country=cell["country"], reader=reader,
        )
        source_observations = _source_observations_for_historical_cell(
            cell, source_country, source_analysis, expected_key_frame,
        )
    return {
        "identity": identity,
        "context": country_context,
        "input_manifest": input_manifest,
        "model_manifest": model_manifest,
        "expected_keys_manifest": expected_manifest,
        "paths": {
            "input_manifest": input_path, "model_manifest": model_path,
            "expected_keys_manifest": expected_manifest_path,
            "expected_keys": expected_keys_path, "oof_predictions": oof_path,
        },
        "raw_source_records": raw_source_records,
        "source_observations": source_observations,
    }


def _integer_token(value: Any, label: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {label} in exact test keys: {value!r}") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Non-integer {label} in exact test keys: {value!r}")
    return str(int(number))


def _key_tokens_in_order(frame: pd.DataFrame, *, profile: str, label: str,
                         allow_missing_profile: bool) -> list[tuple[str, ...]]:
    frame = frame.copy()
    required = set(EXPECTED_KEY_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks key columns: {sorted(missing)}")
    if "role" in frame:
        if not frame["role"].astype(str).str.casefold().eq("test").all():
            raise ValueError(f"{label} contains non-test outer-fold assignments")
    if "profile" not in frame:
        if not allow_missing_profile:
            raise ValueError(f"{label} must explicitly carry profile")
        frame["profile"] = profile
    if frame["profile"].astype(str).ne(profile).any():
        raise ValueError(f"{label} profile disagrees with its registry binding")
    if frame[list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError(f"{label} contains null exact test keys")
    tokens = []
    for row in frame[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None):
        country, cohort, row_profile, task, fold, row_id, site, year, month = row
        tokens.append((
            str(country).upper(), str(cohort), str(row_profile), str(task),
            audited._fold_token(fold), _integer_token(row_id, "row_id"),
            str(site), _integer_token(year, "year"), _integer_token(month, "month"),
        ))
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{label} contains duplicate exact test keys")
    return tokens


def _key_tuples(frame: pd.DataFrame, *, profile: str, label: str,
                allow_missing_profile: bool) -> list[tuple[str, ...]]:
    return sorted(_key_tokens_in_order(
        frame, profile=profile, label=label,
        allow_missing_profile=allow_missing_profile,
    ))


def _cell_expected_keys(cell: dict[str, Any], binding: dict[str, Any],
                        artifacts: dict[str, dict[str, Any]], reader: _ArtifactReader) -> list[tuple[str, ...]]:
    _, frame = reader.table(artifacts["expected_keys"], "expected_keys", kind="keys")
    identity = binding["identity"]
    if cell["kind"] == "historical" and "profile" not in frame:
        frame = frame.copy()
        frame["profile"] = cell["profile"]
        allow_missing_profile = True
    else:
        allow_missing_profile = False
    # Filter only after the complete artifact has been hash-checked.  The
    # expected-key manifest commits to the artifact row count before this cell
    # view is formed.
    cell_rows = frame.loc[
        frame["country"].astype(str).str.upper().eq(cell["country"])
        & frame["cohort_label"].astype(str).eq(identity["cohort_label"])
        & frame["profile"].astype(str).eq(cell["profile"])
        & frame["task"].astype(str).eq(cell["task"])
    ].copy()
    keys = _key_tuples(
        cell_rows, profile=cell["profile"], label="expected exact test keys",
        allow_missing_profile=allow_missing_profile,
    )
    if not keys:
        raise ValueError(f"Committed expected-key artifact has zero rows for fixed cell {_cell_key(cell)}")
    return keys


def _read_cell_predictions(cell: dict[str, Any], binding: dict[str, Any],
                           reader: _ArtifactReader) -> tuple[pd.DataFrame, dict[str, Any]]:
    details = _validate_binding_manifests(cell, binding, reader)
    identity = details["identity"]
    artifacts = binding["artifacts"]
    _, prediction = reader.table(
        artifacts["oof_predictions"], "oof_predictions",
        kind="historical_oof" if cell["kind"] == "historical" else "external_oof",
    )
    if cell["kind"] == "historical":
        if "profile" in prediction:
            raise ValueError("Historical OOF must use the profile declared by its source manifest")
        prediction = prediction.copy()
        prediction["profile"] = cell["profile"]
    else:
        if "profile" not in prediction:
            raise ValueError("External OOF must explicitly carry its meteorology profile")
    for required in [*IDENTITY_COLUMNS, "target_strategy", "smear_strategy", "backtransform", "arm", "obs", "pred"]:
        if required not in prediction:
            raise ValueError(f"OOF prediction file lacks required column {required!r}")
    cell_mask = (
        prediction["country"].astype(str).str.upper().eq(cell["country"])
        & prediction["cohort_label"].astype(str).eq(identity["cohort_label"])
        & prediction["profile"].astype(str).eq(cell["profile"])
        & prediction["task"].astype(str).eq(cell["task"])
    )
    if cell["kind"] == "historical":
        year = pd.to_numeric(prediction.loc[cell_mask, "year"], errors="coerce")
        if year.empty or not year.between(2018, 2022).all():
            raise ValueError("Historical OOF must contain only the bound 2018–2022 task years")
        if cell["task"] == "T3_year_2022" and set(year.astype(int).unique()) != {2022}:
            raise ValueError("Historical T3 cell must contain exactly the 2022 held-out year")
    else:
        year = pd.to_numeric(prediction.loc[cell_mask, "year"], errors="coerce")
        if year.empty or set(year.astype(int).unique()) != {int(cell["external_year"])}:
            raise ValueError("External OOF year is not the registered fixed year")
        if set(prediction.loc[cell_mask, "fold_id"].astype(str).unique()) != {str(cell["external_year"])}:
            raise ValueError("External OOF fold ID is not the registered external year")

    selected = prediction.loc[
        cell_mask
        & prediction["target_strategy"].astype(str).eq("log_smear")
        & prediction["smear_strategy"].astype(str).eq("crossfit_station")
        & prediction["backtransform"].astype(str).eq("smearing")
        & prediction["arm"].astype(str).isin(ARMS)
    ].copy()
    if selected.empty:
        raise ValueError(f"No bound primary three-arm OOF rows for fixed cell {_cell_key(cell)}")
    if not np.isfinite(selected[["obs", "pred"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite primary observations or predictions in {_cell_key(cell)}")
    if selected.duplicated(list(audited.ROW_IDENTITY) + ["arm"]).any():
        raise ValueError(f"Duplicate primary row/arm identity in {_cell_key(cell)}")

    expected_keys = _cell_expected_keys(cell, binding, artifacts, reader)
    actual_keys = _key_tuples(
        selected.drop_duplicates(list(audited.ROW_IDENTITY)), profile=cell["profile"],
        label="selected primary OOF keys", allow_missing_profile=False,
    )
    if actual_keys != expected_keys:
        missing = sorted(set(expected_keys) - set(actual_keys))[:5]
        unexpected = sorted(set(actual_keys) - set(expected_keys))[:5]
        raise ValueError(
            f"Exact expected-test-key mismatch for {_cell_key(cell)}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    source_observations = details.get("source_observations")
    if not isinstance(source_observations, dict) or not source_observations:
        raise ValueError(f"No independently reconstructed target observations for {_cell_key(cell)}")
    for arm in ARMS:
        arm_rows = selected.loc[selected.arm.astype(str).eq(arm)].copy()
        arm_tokens = _key_tokens_in_order(
            arm_rows, profile=cell["profile"], label=f"{arm} primary OOF target keys",
            allow_missing_profile=False,
        )
        observed = pd.to_numeric(arm_rows.obs, errors="raise").to_numpy(dtype="float64")
        bound = np.asarray([source_observations.get(token, np.nan) for token in arm_tokens], dtype="float64")
        if not np.isfinite(bound).all() or not np.array_equal(observed, bound):
            raise ValueError(
                f"{arm} OOF observations disagree with the reconstructed original target source "
                f"for {_cell_key(cell)}"
            )
    return selected, details


def _scope_for_cell(cell: dict[str, Any]) -> str:
    if cell["kind"] == "historical":
        return "historical_2018_2022"
    return f"fixed_external_{int(cell['external_year'])}"


def _prepare_final_cells(registry: dict[str, Any], workspace: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    incomplete = preflight_registry(registry)
    if incomplete["n_cells_pending"]:
        names = [f"{x['country']}/{x['task']}/{x['profile']}" for x in incomplete["pending_cells"]]
        raise IncompleteFamilyError(
            f"Refusing final scoring: {len(names)} locked cells remain pending; first={names[:5]}"
        )
    reader = _ArtifactReader(workspace)
    selected_frames = []
    details_by_cell: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    for cell in registry["cells"]:
        binding = cell["source_binding"]
        selected, details = _read_cell_predictions(cell, binding, reader)
        selected_frames.append(selected)
        key = _cell_key(cell)
        details_by_cell[key] = details
        for name in ARTIFACT_NAMES:
            record = binding["artifacts"][name]
            path = details["paths"].get(name)
            if path is None:
                path = _resolve_path(record["path"], workspace)
            source_records.append({
                "country": cell["country"], "task": cell["task"],
                "profile": cell["profile"], "role": name,
                "path": str(path), "sha256": record["sha256"].lower(),
                "bytes": path.stat().st_size,
            })
        source_records.extend(details.get("raw_source_records", []))
    selected = pd.concat(selected_frames, ignore_index=True)
    actual_groups = set(
        tuple(str(value) for value in row)
        for row in selected[["country", "profile", "task"]].drop_duplicates().itertuples(index=False, name=None)
    )
    expected_groups = set(_expected_registry_cells())
    if actual_groups != expected_groups:
        raise ValueError(
            f"Final OOF cell coverage differs from the locked 26-cell family: "
            f"missing={sorted(expected_groups - actual_groups)}, unexpected={sorted(actual_groups - expected_groups)}"
        )
    cells = audited._build_wide_cells(selected, {})
    if len(cells) != EXPECTED_CELL_COUNT:
        raise AssertionError("Audited cell builder did not retain all 26 expected cells")
    for cell in cells:
        key = (str(cell["country"]), str(cell["profile"]), str(cell["task"]))
        spec = next(item for item in registry["cells"] if _cell_key(item) == key)
        cell["scope"] = _scope_for_cell(spec)
        cell["kind"] = spec["kind"]
    return cells, details_by_cell, source_records


def _no_estimable_family_results(cells: list[dict[str, Any]], n_replicates: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Retain all expected denominators when every cell is ineligible."""
    cell_rows = []
    contrast_rows = []
    exclusions = []
    for cell in cells:
        reason = cell.get("eligibility_reason", "not estimable")
        rmse = np.asarray(cell.get("point_rmse", [np.nan, np.nan, np.nan]), dtype=float)
        base = {
            "country": cell["country"], "cohort_label": cell["cohort_label"],
            "scope": cell["scope"], "task": cell["task"], "profile": cell["profile"],
            "n_rows": cell["n_rows"], "n_test_stations": cell["n_test_stations"],
            "rmse_B": float(rmse[0]), "rmse_B_AOD": float(rmse[1]),
            "rmse_full": float(rmse[2]), "family_eligible": False,
            "eligibility_reason": reason,
        }
        cell_rows.append(base)
        exclusions.append({**base, "exclusion_reason": reason})
        for index, (name, numerator, denominator) in enumerate(audited.CONTRASTS):
            estimate = float(cell["point_contrasts"][index])
            contrast_rows.append({
                **base, "contrast": name, "numerator_arm": numerator,
                "denominator_arm": denominator, "delta_rmse": estimate,
                "station_multiplier_se": np.nan, "max_abs_t_critical_95": np.nan,
                "simultaneous_95_low": np.nan, "simultaneous_95_high": np.nan,
                "family_size_contrasts": 0, "replicates": 0,
                "interval_scope": "no estimable contrasts; fixed family retained without inference",
            })
    return pd.DataFrame(contrast_rows), pd.DataFrame(cell_rows), {
        "max_abs_t_critical_95": None,
        "n_cells_total": len(cells), "n_cells_eligible": 0,
        "n_cells_excluded": len(cells), "n_contrasts_in_family": 0,
        "n_draws": n_replicates, "max_abs_t_by_draw": None,
        "exclusions": pd.DataFrame(exclusions),
    }


def _retain_no_estimable_reasons(cells: list[dict[str, Any]], n_replicates: int) -> None:
    """Recover exclusion reasons from the unchanged audited eligibility rules."""
    for cell in cells:
        if not cell.get("sparse_eligible", True):
            reason = cell.get("eligibility_reason", "sparse_test_cell")
        else:
            boot = cell.get("bootstrap_contrasts")
            if boot is None or boot.shape != (n_replicates, len(audited.CONTRASTS)):
                reason = "missing_or_wrong_replicate_count"
            elif not np.isfinite(boot).all():
                invalid = int((~np.isfinite(boot)).any(axis=1).sum())
                reason = f"invalid_bootstrap_draws:{invalid}; no redraw/drop"
            else:
                se = boot.std(axis=0, ddof=1)
                unstable = [
                    contrast[0]
                    for contrast, value, estimate in zip(audited.CONTRASTS, se, cell["point_contrasts"])
                    if not audited.stable_se(float(value), float(estimate))
                ]
                reason = (
                    "zero_or_unstable_SE:" + ";".join(unstable)
                    if unstable else "not estimable"
                )
        cell["family_eligible"] = False
        cell["eligibility_reason"] = reason


def _enrich_result_frame(frame: pd.DataFrame, registry: dict[str, Any],
                         details_by_cell: dict[tuple[str, str, str], dict[str, Any]]) -> pd.DataFrame:
    enriched = frame.copy()
    stages, qa, estimator, target, smear, source_binding_hashes = [], [], [], [], [], []
    for row in enriched.itertuples(index=False):
        key = (str(row.country), str(row.profile), str(row.task))
        details = details_by_cell[key]
        stages.append(details["context"].get("input_acceptance_stage"))
        qa.append(details["context"].get("qa_variant"))
        estimator.append(details["model_manifest"].get("estimator_kind"))
        target.append("log_smear")
        smear.append("crossfit_station")
        cell = next(item for item in registry["cells"] if _cell_key(item) == key)
        source_binding_hashes.append(hashlib.sha256(_json_bytes(cell["source_binding"])).hexdigest())
    enriched["input_acceptance_stage"] = stages
    enriched["qa_variant"] = qa
    enriched["estimator_kind"] = estimator
    enriched["target_strategy"] = target
    enriched["smear_strategy"] = smear
    enriched["source_binding_sha256"] = source_binding_hashes
    return enriched


def _report_text(summary: dict[str, Any]) -> str:
    critical = summary["bootstrap"].get("critical_value")
    critical_text = "not estimable" if critical is None else f"{critical:.4f}"
    return "\n".join([
        "# Final max-|t| multiplicity sensitivity",
        "",
        "This is an execution-stage exploratory sensitivity registered before new-year PM model scores. It is conditional on fitted predictions and observed folds/years; it does not include model refitting or imply whole-matrix, spatial, event, or future-year coverage.",
        "",
        "The locked family contains 26 country × task × profile cells and at most 78 RMSE contrasts: `B+AOD − B`, `full − B+AOD`, and `full − B`. Every expected cell remains in the cell and contrast outputs. Cells with fewer than 30 test stations, invalid draws, or zero/unstable standard errors are excluded from the simultaneous family by the fixed rule and retain their reason and denominator.",
        "",
        f"Eligible cells: {summary['family']['n_cells_eligible']}; excluded cells: {summary['family']['n_cells_excluded']}; contrasts in the max-|t| family: {summary['family']['n_contrasts']}; 95% critical value: {critical_text}.",
        "",
        "The same Exp(1) station multiplier is reused across all included tasks, years, profiles, and contrasts within each country. Countries use independent streams, and each task cell uses its own weighted row-count denominator. The max-|t| calibration is centered on the unchanged point estimates and uses the 95th linear empirical quantile.",
        "",
        "See `cell_rmse_and_eligibility.csv`, `simultaneous_rmse_contrasts.csv`, `excluded_cells.csv`, and `max_abs_t_draws.csv` for the cell-level audit trail. The original pointwise confidence intervals are not modified.",
        "",
    ])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _new_output_target(*, out: Path | None, out_root: Path, mode: str) -> Path:
    if out is not None:
        target = Path(out).resolve()
    else:
        root = Path(out_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = root / f"multiplicity_final_{mode}_{stamp}_{secrets.token_hex(4)}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


class _AtomicOutput:
    """Build a complete immutable run directory and publish it by rename."""

    def __init__(self, target: Path):
        self.target = Path(target).resolve()
        if self.target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output path: {self.target}")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(
            prefix=f".{self.target.name}.partial.", dir=str(self.target.parent)
        ))
        self.published = False

    def publish(self) -> Path:
        if self.target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output path: {self.target}")
        os.rename(self.stage, self.target)
        self.published = True
        return self.target

    def discard(self) -> None:
        if not self.published and self.stage.exists():
            shutil.rmtree(self.stage)


def write_preflight(registry: dict[str, Any], *, registry_path: Path,
                    workspace: Path = WORKSPACE,
                    out: Path | None = None, out_root: Path | None = None) -> dict[str, Any]:
    started = _utc_now()
    registry_path = Path(registry_path).resolve()
    snapshot = _capture_integrity_snapshot(
        registry, registry_path=registry_path, workspace=workspace,
    )
    registry = snapshot["registry"]
    code_start = snapshot["code_dependencies"]["final_multiplicity_code"]["sha256_start"]
    payload = preflight_registry(registry)
    payload.update({
        "started_utc": started,
        "ended_utc": None,
        "code_sha256_start": code_start,
        "code_sha256_end": None,
        "registry_path": str(registry_path),
        "registry_sha256": snapshot["registry_sha256_start"],
        "protocol_path": snapshot["protocol_path"],
        "protocol_sha256": snapshot["protocol_sha256_start"],
        "audited_math_source_sha256": snapshot["code_dependencies"]["audited_v2_multiplicity_math"]["sha256_start"],
        "output_is_atomic": True,
    })
    target = _new_output_target(
        out=out,
        out_root=out_root or (WORKSPACE / "outputs/repair_v2/diagnostics/multiplicity_final"),
        mode="preflight",
    )
    transaction = _AtomicOutput(target)
    try:
        payload["output_dir"] = str(target)
        _write_json(transaction.stage / "preflight.json", payload)
        integrity = _verify_integrity_snapshot(snapshot)
        payload["integrity"] = integrity
        payload["code_sha256_end"] = integrity["code_dependencies"]["final_multiplicity_code"]["sha256_end"]
        payload["ended_utc"] = _utc_now()
        _write_json(transaction.stage / "preflight.json", payload)
        if _verify_integrity_snapshot(snapshot) != integrity:
            raise RuntimeError("Locked final inputs changed before preflight publication")
        result_path = transaction.publish()
    except Exception:
        transaction.discard()
        raise
    payload["output_dir"] = str(result_path)
    return payload


def _source_fingerprint(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def _write_final_outputs(stage: Path, registry: dict[str, Any], cells: list[dict[str, Any]],
                         details_by_cell: dict[tuple[str, str, str], dict[str, Any]],
                         source_records: list[dict[str, Any]], *, code_hash: str,
                         registry_path: Path, protocol_path: Path,
                         integrity_snapshot: dict[str, Any],
                         started_utc: str, replicates: int, seed: int,
                         chunk_size: int) -> dict[str, Any]:
    seed_children = np.random.SeedSequence(seed).spawn(2)
    country_diagnostics = []
    for country, child in zip(["CAN", "USA"], seed_children):
        _, diagnostics = audited._bootstrap_country_cells(
            cells, country=country, child_seed=child,
            n_replicates=replicates, chunk_size=chunk_size,
        )
        country_diagnostics.append(diagnostics)
    try:
        contrasts, cell_frame, assembled = audited._assemble_results(cells, replicates)
        max_abs = assembled.pop("max_abs_t_by_draw")
        exclusions = assembled.pop("exclusions")
        no_estimable_family = False
    except ValueError as exc:
        if "No estimable cells remain" not in str(exc):
            raise
        _retain_no_estimable_reasons(cells, replicates)
        contrasts, cell_frame, assembled = _no_estimable_family_results(cells, replicates)
        max_abs = None
        exclusions = assembled.pop("exclusions")
        no_estimable_family = True

    cell_frame = _enrich_result_frame(cell_frame, registry, details_by_cell)
    exclusions = _enrich_result_frame(exclusions, registry, details_by_cell) if not exclusions.empty else cell_frame.iloc[0:0].copy()
    if not exclusions.empty and "exclusion_reason" not in exclusions:
        exclusions["exclusion_reason"] = exclusions["eligibility_reason"]
    # Join source context to every contrast without changing the observed
    # point estimate or the legacy interval arithmetic.
    context_columns = [
        "country", "cohort_label", "task", "scope", "profile",
        "input_acceptance_stage", "qa_variant", "estimator_kind",
        "target_strategy", "smear_strategy", "source_binding_sha256",
    ]
    contrasts = contrasts.merge(
        cell_frame[context_columns],
        on=["country", "cohort_label", "task", "scope", "profile"],
        how="left", validate="many_to_one",
    )

    contrast_path = stage / "simultaneous_rmse_contrasts.csv"
    cell_path = stage / "cell_rmse_and_eligibility.csv"
    exclusion_path = stage / "excluded_cells.csv"
    max_path = stage / "max_abs_t_draws.csv"
    contrasts.to_csv(contrast_path, index=False)
    cell_frame.to_csv(cell_path, index=False)
    exclusions.to_csv(exclusion_path, index=False)
    if max_abs is None:
        max_draws = pd.DataFrame({
            "replicate": np.arange(1, replicates + 1),
            "max_abs_t": np.full(replicates, np.nan),
            "draw_status": "not_defined_no_estimable_contrasts",
        })
    else:
        max_draws = pd.DataFrame({
            "replicate": np.arange(1, replicates + 1),
            "max_abs_t": max_abs,
            "draw_status": "computed",
        })
    max_draws["qa_variant"] = "strict"
    max_draws["input_acceptance_stage"] = "audited"
    max_draws["diagnostic_code_sha256"] = code_hash
    max_draws["input_sources_sha256"] = _source_fingerprint(source_records)
    max_draws.to_csv(max_path, index=False)

    summary = {
        "status": "complete_no_estimable_family" if no_estimable_family else "complete_final_multiplicity_sensitivity",
        "mode": "run",
        "not_confirmatory": True,
        "family": {
            "n_cells_total": EXPECTED_CELL_COUNT,
            "n_cells_eligible": int(assembled["n_cells_eligible"]),
            "n_cells_excluded": int(assembled["n_cells_excluded"]),
            "n_contrasts_expected_maximum": EXPECTED_MAX_CONTRASTS,
            "n_contrasts": int(assembled["n_contrasts_in_family"]),
        },
        "bootstrap": {
            "distribution": "Exp(1)", "replicates": replicates, "seed": seed,
            "chunk_size": chunk_size,
            "same_station_weights_shared_across_all_cells_within_country": True,
            "country_streams_independent": True,
            "cell_specific_weighted_row_normalizer": True,
            "no_invalid_draw_redraw_or_silent_drop": True,
            "critical_value": assembled["max_abs_t_critical_95"],
            "country_diagnostics": country_diagnostics,
        },
        "input_sources_sha256": _source_fingerprint(source_records),
        "source_artifacts": source_records,
    }
    report_path = stage / "multiplicity_final_report.md"
    report_path.write_text(_report_text(summary), encoding="utf-8")
    output_records = []
    for path in (contrast_path, cell_path, exclusion_path, max_path, report_path):
        output_records.append({"path": path.name, "bytes": path.stat().st_size,
                               "sha256": sha256_file(path)})
    integrity_end = _verify_integrity_snapshot(integrity_snapshot)
    code_end = integrity_end["code_dependencies"]["final_multiplicity_code"]["sha256_end"]
    if code_end != code_hash:
        raise RuntimeError("Final multiplicity code changed during execution")
    audited_math_source_end = integrity_end["code_dependencies"]["audited_v2_multiplicity_math"]["sha256_end"]
    manifest = {
        **summary,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "integrity": integrity_end,
        "code_sha256_start": code_hash,
        "code_sha256_end": code_end,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": integrity_snapshot["registry_sha256_start"],
        "registry_sha256_start": integrity_snapshot["registry_sha256_start"],
        "registry_sha256_end": integrity_end["registry"]["sha256_end"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": integrity_snapshot["protocol_sha256_start"],
        "protocol_sha256_start": integrity_snapshot["protocol_sha256_start"],
        "protocol_sha256_end": integrity_end["protocol"]["sha256_end"],
        "audited_math_source_sha256": audited_math_source_end,
        "output_is_atomic": True,
        "outputs": output_records,
        "pointwise_intervals_modified": False,
        "scope_note": registry.get("scope"),
    }
    _write_json(stage / "provenance.json", manifest)
    return manifest


def run_final(registry: dict[str, Any], *, registry_path: Path = REGISTRY_PATH,
              workspace: Path = WORKSPACE, out: Path | None = None,
              out_root: Path | None = None, chunk_size: int = 256) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    registry_path = Path(registry_path).resolve()
    integrity_snapshot = _capture_integrity_snapshot(
        registry, registry_path=registry_path, workspace=workspace,
    )
    registry = integrity_snapshot["registry"]
    # The locked family fixes both values.  This entry point intentionally has
    # no user override for seed or replicate count.
    summary = preflight_registry(registry)
    if summary["n_cells_pending"]:
        pending = [f"{x['country']}/{x['task']}/{x['profile']}" for x in summary["pending_cells"]]
        raise IncompleteFamilyError(
            f"Refusing --run with {len(pending)} pending locked cells; first={pending[:5]}"
        )
    started = _utc_now()
    code_hash = integrity_snapshot["code_dependencies"]["final_multiplicity_code"]["sha256_start"]
    protocol_path = Path(integrity_snapshot["protocol_path"])
    cells, details, source_records = _prepare_final_cells(registry, workspace)
    source_records.extend(_integrity_source_records(integrity_snapshot))
    target = _new_output_target(
        out=out,
        out_root=out_root or (Path(workspace) / "outputs/repair_v2/diagnostics/multiplicity_final"),
        mode="run",
    )
    transaction = _AtomicOutput(target)
    try:
        manifest = _write_final_outputs(
            transaction.stage, registry, cells, details, source_records,
            code_hash=code_hash, registry_path=registry_path,
            protocol_path=protocol_path, integrity_snapshot=integrity_snapshot,
            started_utc=started, replicates=int(registry["replicates"]),
            seed=int(registry["seed"]), chunk_size=chunk_size,
        )
        if _verify_integrity_snapshot(integrity_snapshot) != manifest["integrity"]:
            raise RuntimeError("Locked final inputs changed before result publication")
        output_dir = transaction.publish()
    except Exception:
        transaction.discard()
        raise
    manifest["output_dir"] = str(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--out", type=Path, default=None,
                        help="Exact new output directory; must not already exist")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Parent for a uniquely named immutable output directory")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--run", action="store_true",
                        help="Run final inference; requires every locked cell binding to be ready")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    registry_path = args.registry.resolve()
    registry = load_registry(registry_path, workspace)
    if args.run:
        result = run_final(
            registry, registry_path=registry_path, workspace=workspace,
            out=args.out, out_root=args.out_root, chunk_size=args.chunk_size,
        )
    else:
        result = write_preflight(
            registry, registry_path=registry_path, out=args.out,
            workspace=workspace,
            out_root=args.out_root or (workspace / "outputs/repair_v2/diagnostics/multiplicity_final"),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
