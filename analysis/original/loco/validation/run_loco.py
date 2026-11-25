#!/usr/bin/env python
"""Historical-only leave-one-country-out sensitivity for repair-v2.

Input contract (repair-v2-loco-input-manifest-v1):

* ``country_universe`` is an explicit, duplicate-free roster. Its size is not
  fixed. Countries omitted from ``countries`` are reported as omitted.
* ``historical_years`` is exactly [2018, 2019, 2020, 2021, 2022],
  ``target_column`` names the cleaned target in each consolidated feature
  table, and ``qa_state`` is one of ``strict`` or ``sensitivity``.
* Each eligible country has ``feature_path``/``feature_sha256``,
  ``target_status``/``target_verified: true``, a hash-bound
  ``target_provenance_path``/``target_provenance_sha256``, a matching
  ``qa_state``, and a hash-bound ``qa_manifest_path``/
  ``qa_manifest_sha256``. Target provenance JSON records country,
  target_column, target_status, target_verified, feature_sha256, and a
  target_source_path/target_source_sha256 pair. The QA manifest's
  ``outputs[qa_state].sha256`` must equal the feature hash.
* ``met_profile_id`` identifies comparable units/source definitions. Actual
  completeness is derived from all six fixed meteorology columns over the
  historical rows; a manifest boolean cannot turn incomplete data into a
  with-met run.

The default CLI only writes a preflight ledger. Scoring requires ``--run`` and
always uses 2018-2022 rows, four fixed arms, fixed HGB settings, and the shared
validation_core fit/predict functions. No country is chosen from model scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, train_test_split

from validation_core import ESTIMATOR_CLASS_NAMES, ModelConfig, fit_model, predict_model


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[2]
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "repair_v2" / "validation" / "loco_preflight"
# Compatibility alias for callers that used the prior fixed directory constant.
DEFAULT_OUT = OUTPUT_ROOT
MANIFEST_FORMAT = "repair-v2-loco-input-manifest-v1"
TARGET_PROVENANCE_FORMAT = "repair-v2-loco-target-provenance-v1"
HISTORICAL_YEARS = tuple(range(2018, 2023))
GEO = ("lat_f", "lon_f", "lpop", "month_sin", "month_cos", "month_sin2", "month_cos2")
MET = ("blh", "wind10", "tp", "t2m", "rh", "vent")
AOD = ("aod_used",)
FIRE = ("lgfas", "lfrp", "lfn", "lgfas_r5", "lgfas_r15", "lfrp_r5")
ARMS = ("B", "B+AOD", "B+fire", "full")
TARGET_ALIASES = {
    "target", "target_pm25", "pm25", "pm25_clean", "pm25_final", "pm25_qc",
    "pm25_target", "pm25_obs", "pm25_obs_ug_m3",
}
COUNTRY_ALIASES = {
    "USA": {"USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"},
    "CAN": {"CAN", "CA", "CANADA"},
    "BRA": {"BRA", "BR", "BRAZIL"},
    "MEX": {"MEX", "MX", "MEXICO"},
    "AUS": {"AUS", "AU", "AUSTRALIA"},
    "ES": {"ES", "ESP", "SPAIN"},
    "PT": {"PT", "PRT", "PORTUGAL"},
    "GR": {"GR", "GRC", "GREECE"},
    "IT": {"IT", "ITA", "ITALY"},
    "CL": {"CL", "CHL", "CHILE"},
    "ZA": {"ZA", "ZAF", "SOUTH AFRICA"},
    "IN": {"IN", "IND", "INDIA"},
    "PK": {"PK", "PAK", "PAKISTAN"},
    "NP": {"NP", "NPL", "NEPAL"},
    "BD": {"BD", "BGD", "BANGLADESH"},
    "TH": {"TH", "THA", "THAILAND"},
    "VN": {"VN", "VNM", "VIETNAM", "VIET NAM"},
    "PH": {"PH", "PHL", "PHILIPPINES"},
    "IDN": {"ID", "IDN", "INDONESIA"},
}
ALIAS_TO_COUNTRY = {
    alias: country for country, aliases in COUNTRY_ALIASES.items() for alias in aliases
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_country(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError("country value is missing")
    raw = str(value).strip().upper()
    if not raw:
        raise ValueError("country value is empty")
    try:
        return ALIAS_TO_COUNTRY[raw]
    except KeyError as exc:
        raise ValueError(f"unsupported country alias: {value!r}") from exc


def _resolve_path(value: Any, relative_to: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value.strip())
    if not path.is_absolute():
        path = relative_to / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{field} does not exist: {path}") from exc


def _verify_hash(path_value: Any, expected: Any, relative_to: Path, field: str) -> tuple[Path, str]:
    path = _resolve_path(path_value, relative_to, field)
    expected_hash = str(expected or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"{field} requires a 64-character SHA-256")
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ValueError(f"{field} SHA-256 mismatch: expected {expected_hash}, observed {observed}")
    return path, observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be an object: {path}")
    return value


def _read_feature_table(path: Path) -> pd.DataFrame:
    """Read a supported feature file without numeric inference on station IDs."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix not in {".csv", ".tsv"}:
        raise ValueError(f"feature path must be parquet, csv, or tsv: {path}")
    sep = "\t" if suffix == ".tsv" else ","
    # Preserve station codes such as 001, 1, NA, and NULL as distinct strings.
    # Retain ordinary missing-value handling for every other field.
    header = pd.read_csv(path, sep=sep, nrows=0, keep_default_na=False)
    missing_tokens = ["", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none", "<NA>"]
    na_values = {str(column): missing_tokens for column in header.columns if str(column) != "site_id"}
    na_values["site_id"] = [""]
    return pd.read_csv(
        path, sep=sep, dtype={"site_id": "string"}, keep_default_na=False,
        na_values=na_values,
    )


def _verify_target_provenance(
    entry: Mapping[str, Any], country: str, target_column: str,
    feature_sha256: str, manifest_dir: Path,
) -> dict[str, Any]:
    path, digest = _verify_hash(
        entry.get("target_provenance_path"), entry.get("target_provenance_sha256"),
        manifest_dir, f"{country} target_provenance_path",
    )
    provenance = _read_json(path, f"{country} target provenance")
    if provenance.get("format") != TARGET_PROVENANCE_FORMAT:
        raise ValueError(f"{country} target provenance has unsupported format")
    if canonical_country(provenance.get("country")) != country:
        raise ValueError(f"{country} target provenance names another country")
    if provenance.get("target_column") != target_column:
        raise ValueError(f"{country} target provenance target_column disagrees with the input manifest")
    if provenance.get("target_status") != entry.get("target_status"):
        raise ValueError(f"{country} target provenance target_status disagrees with the input manifest")
    if provenance.get("target_verified") is not True or entry.get("target_verified") is not True:
        raise ValueError(f"{country} target is not explicitly verified")
    if str(provenance.get("feature_sha256", "")).lower() != feature_sha256:
        raise ValueError(f"{country} target provenance is not bound to the feature/target table hash")
    source_path, source_hash = _verify_hash(
        provenance.get("target_source_path"), provenance.get("target_source_sha256"),
        path.parent, f"{country} target_source_path",
    )
    return {
        "path": str(path), "sha256": digest,
        "target_source_path": str(source_path), "target_source_sha256": source_hash,
        "target_status": str(entry["target_status"]),
    }


def _verify_qa_provenance(
    entry: Mapping[str, Any], country: str, qa_state: str,
    feature_sha256: str, manifest_dir: Path,
) -> dict[str, Any]:
    if str(entry.get("qa_state", "")).strip() != qa_state:
        raise ValueError(f"{country} QA state differs from the manifest-wide QA state")
    path, digest = _verify_hash(
        entry.get("qa_manifest_path"), entry.get("qa_manifest_sha256"),
        manifest_dir, f"{country} qa_manifest_path",
    )
    provenance = _read_json(path, f"{country} QA provenance")
    if canonical_country(provenance.get("country")) != country:
        raise ValueError(f"{country} QA manifest names another country")
    output = provenance.get("outputs", {}).get(qa_state)
    if not isinstance(output, dict):
        raise ValueError(f"{country} QA manifest has no outputs[{qa_state!r}]")
    if str(output.get("sha256", "")).lower() != feature_sha256:
        raise ValueError(f"{country} QA output hash does not match the selected feature table")
    if str(output.get("qa_variant", "")) != qa_state:
        raise ValueError(f"{country} QA output variant does not match {qa_state!r}")
    return {"path": str(path), "sha256": digest, "qa_state": qa_state,
            "feature_sha256": feature_sha256}


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature column {column!r} must be numeric") from exc


def _check_target_aliases(frame: pd.DataFrame, target_column: str) -> list[str]:
    selected = _numeric_column(frame, target_column)
    verified: list[str] = []
    for alias in frame.columns:
        if alias == target_column or str(alias).strip().lower() not in TARGET_ALIASES:
            continue
        values = _numeric_column(frame, alias)
        same_missing = np.array_equal(np.isfinite(selected), np.isfinite(values))
        same_values = same_missing and np.allclose(
            selected[np.isfinite(selected)], values[np.isfinite(values)], rtol=0, atol=0,
        )
        if not same_values:
            raise ValueError(f"conflicting target alias {alias!r} vs selected {target_column!r}")
        verified.append(str(alias))
    return verified


def validate_country_frame(
    frame: pd.DataFrame, country: str, target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate schema/keys, then filter to the frozen historical window."""
    country = canonical_country(country)
    required = [
        "country", "site_id", "year", "month", "lat", "lon", target_column,
        *GEO, *AOD, *FIRE, "aod_raw",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{country} input is missing fixed schema columns: {missing}")
    if frame.empty:
        return frame.copy(), {"rows_source": 0, "rows_historical": 0, "outside_window_rows": 0,
                              "outside_window_year_rows": {}, "target_aliases_equal": [],
                              "met_complete_observed": False, "met_incomplete_reason": "empty_input"}

    raw_country = frame["country"].dropna().astype(str).str.strip().str.upper()
    if raw_country.empty:
        raise ValueError(f"{country} input has no country values")
    normalized_country = raw_country.map(canonical_country)
    if normalized_country.nunique() != 1 or normalized_country.iloc[0] != country:
        raise ValueError(f"{country} input has conflicting or mismatched country aliases")
    # Reject mixed aliases within one file, even if they normalize to the same code.
    if raw_country.nunique() != 1:
        raise ValueError(f"{country} input mixes country aliases: {sorted(raw_country.unique().tolist())}")
    if frame["country"].isna().any():
        raise ValueError(f"{country} input has missing country values")

    year = _numeric_column(frame, "year")
    if not np.isfinite(year).all() or not np.equal(year, np.floor(year)).all():
        raise ValueError(f"{country} input has invalid year values")
    year = year.astype("int64")
    outside = year[~np.isin(year, HISTORICAL_YEARS)]
    outside_counts = {str(int(k)): int(v) for k, v in pd.Series(outside).value_counts().sort_index().items()}
    historical = frame.loc[np.isin(year, HISTORICAL_YEARS)].copy()
    if historical.empty:
        return historical, {
            "rows_source": int(len(frame)), "rows_historical": 0,
            "outside_window_rows": int(len(frame)), "outside_window_year_rows": outside_counts,
            "target_aliases_equal": [], "met_complete_observed": False,
            "met_incomplete_reason": "no_rows_in_2018_2022",
        }

    historical["year"] = year[np.isin(year, HISTORICAL_YEARS)]
    month = _numeric_column(historical, "month")
    if not np.isfinite(month).all() or not np.equal(month, np.floor(month)).all():
        raise ValueError(f"{country} historical rows have invalid month values")
    if np.any((month < 1) | (month > 12)):
        raise ValueError(f"{country} historical rows have month outside 1..12")
    historical["month"] = month.astype("int64")

    historical["site_id"] = historical.site_id.astype("string")
    if historical.site_id.isna().any() or historical.site_id.str.strip().eq("").any():
        raise ValueError(f"{country} historical rows have missing/empty site_id")
    historical["site_id"] = historical.site_id.astype(str)
    historical["country"] = country
    for column in sorted(set(required) - {"country", "site_id", "year", "month"}):
        historical[column] = _numeric_column(historical, column)

    geo_columns = ["lat", "lon", *GEO]
    geo = historical[geo_columns].to_numpy(dtype="float64")
    if not np.isfinite(geo).all():
        raise ValueError(f"{country} has nonfinite geographic coordinates/features")
    if ((historical.lat < -90) | (historical.lat > 90)).any():
        raise ValueError(f"{country} latitude is outside [-90, 90]")
    if ((historical.lon < -180) | (historical.lon > 180)).any():
        raise ValueError(f"{country} longitude is outside [-180, 180]")

    alias_columns = _check_target_aliases(historical, target_column)
    target = _numeric_column(historical, target_column)
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError(f"{country} historical targets must be finite and nonnegative")
    historical[target_column] = target

    keys = historical[["country", "site_id", "year", "month"]]
    if keys.duplicated().any():
        examples = keys.loc[keys.duplicated(keep=False)].head(5).to_dict("records")
        raise ValueError(f"{country} has duplicate historical country/site/year/month keys: {examples}")

    missing_met = [column for column in MET if column not in historical]
    nonfinite_met = [column for column in MET if column in historical and
                     not np.isfinite(_numeric_column(historical, column)).all()]
    met_complete = not missing_met and not nonfinite_met
    met_reasons = []
    if missing_met:
        met_reasons.append(f"missing_columns:{','.join(missing_met)}")
    if nonfinite_met:
        met_reasons.append(f"nonfinite_values:{','.join(nonfinite_met)}")
    historical["_loco_site_group"] = historical.country.astype(str) + "::" + historical.site_id.astype(str)
    audit = {
        "rows_source": int(len(frame)), "rows_historical": int(len(historical)),
        "outside_window_rows": int(len(frame) - len(historical)),
        "outside_window_year_rows": outside_counts,
        "target_aliases_equal": alias_columns,
        "met_complete_observed": bool(met_complete),
        "met_incomplete_reason": ";".join(met_reasons) if met_reasons else None,
        "historical_sites": int(historical._loco_site_group.nunique()),
    }
    return historical.reset_index(drop=True), audit


def _normalize_met_profile_id(value: Any, country: str) -> str | None:
    """Normalize missing profile IDs without turning JSON null into "None"."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{country} met_profile_id must be a string or null")
    normalized = value.strip()
    if not normalized or normalized.casefold() in {"none", "null", "nan"}:
        return None
    return normalized

def load_manifest_inputs(manifest_path: Path | str) -> dict[str, Any]:
    """Verify hashes/provenance and load only eligible historical rows."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path, "LOCO input manifest")
    manifest_hash = sha256_file(manifest_path)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"unsupported manifest format; expected {MANIFEST_FORMAT}")
    if manifest.get("historical_years") != list(HISTORICAL_YEARS):
        raise ValueError(f"historical_years must be exactly {list(HISTORICAL_YEARS)}")
    target_column = manifest.get("target_column")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("manifest target_column must be a nonempty string")
    target_column = target_column.strip()
    if target_column != "pm25":
        raise ValueError("repair-v2 LOCO uses the fixed cleaned target column 'pm25'")
    qa_state = manifest.get("qa_state")
    if qa_state not in {"strict", "sensitivity"}:
        raise ValueError("manifest qa_state must be 'strict' or 'sensitivity'")
    universe_raw = manifest.get("country_universe")
    if not isinstance(universe_raw, list) or not universe_raw:
        raise ValueError("manifest country_universe must be a nonempty explicit list")
    universe: list[str] = []
    raw_universe = []
    for value in universe_raw:
        canonical = canonical_country(value)
        if canonical in universe:
            raise ValueError(f"country_universe has duplicate canonical country after alias normalization: {canonical}")
        universe.append(canonical)
        raw_universe.append(str(value))

    entries = manifest.get("countries")
    if not isinstance(entries, list):
        raise ValueError("manifest countries must be an explicit list")
    normalized_entries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each countries entry must be an object")
        country = canonical_country(entry.get("country"))
        if country in normalized_entries:
            raise ValueError(f"countries has duplicate canonical entry after alias normalization: {country}")
        if country not in universe:
            raise ValueError(f"country entry {country} is not listed in country_universe")
        status = entry.get("status")
        if status not in {"eligible", "ineligible"}:
            raise ValueError(f"{country} status must be 'eligible' or 'ineligible'")
        normalized_entries[country] = entry

    inputs: dict[str, pd.DataFrame] = {}
    input_ledger: dict[str, Any] = {}
    eligibility: list[dict[str, Any]] = []
    for country in universe:
        entry = normalized_entries.get(country)
        if entry is None:
            eligibility.append({"country": country, "status": "omitted_from_manifest",
                                "reason": "country_universe_member_has_no_input_entry"})
            input_ledger[country] = {"status": "omitted_from_manifest"}
            continue
        if entry["status"] == "ineligible":
            reason = str(entry.get("reason", "")).strip()
            if not reason:
                raise ValueError(f"{country} ineligible entry requires an explicit reason")
            eligibility.append({"country": country, "status": "ineligible", "reason": reason})
            input_ledger[country] = {"status": "ineligible", "reason": reason}
            continue

        target_status = entry.get("target_status")
        if not isinstance(target_status, str) or not target_status.strip():
            raise ValueError(f"{country} eligible entry requires an explicit target_status")
        if entry.get("target_verified") is not True:
            raise ValueError(f"{country} eligible entry requires target_verified: true")
        feature_path, feature_hash = _verify_hash(
            entry.get("feature_path"), entry.get("feature_sha256"),
            manifest_path.parent, f"{country} feature_path",
        )
        target_provenance = _verify_target_provenance(
            entry, country, target_column, feature_hash, manifest_path.parent,
        )
        qa_provenance = _verify_qa_provenance(
            entry, country, qa_state, feature_hash, manifest_path.parent,
        )
        frame = _read_feature_table(feature_path)
        frame, frame_audit = validate_country_frame(frame, country, target_column)
        if frame.empty:
            eligibility.append({"country": country, "status": "ineligible", "reason": "no_rows_in_2018_2022"})
            input_ledger[country] = {
                "status": "ineligible", "reason": "no_rows_in_2018_2022",
                "feature_path": str(feature_path), "feature_sha256": feature_hash,
                "target_status": target_status, "qa_state": qa_state,
                "frame_audit": frame_audit, "target_provenance": target_provenance,
                "qa_provenance": qa_provenance,
            }
            continue
        inputs[country] = frame
        input_ledger[country] = {
            "status": "eligible", "feature_path": str(feature_path),
            "feature_sha256": feature_hash, "feature_bytes": int(feature_path.stat().st_size),
            "target_status": target_status, "target_verified": True,
            "target_provenance": target_provenance,
            "qa_state": qa_state, "qa_provenance": qa_provenance,
            "met_profile_id": _normalize_met_profile_id(entry.get("met_profile_id"), country),
            # Recorded as submitted but never used to decide actual completeness.
            "met_complete_declared": entry.get("met_complete"),
            "frame_audit": frame_audit,
        }
        eligibility.append({
            "country": country, "status": "eligible",
            "reason": "historical_input_verified", "rows": int(len(frame)),
            "sites": int(frame._loco_site_group.nunique()),
            "met_complete_observed": frame_audit["met_complete_observed"],
        })

    # Re-check source hashes after loading, so an input changed during preflight
    # cannot silently be reported under the original digest.
    for country, record in input_ledger.items():
        if record.get("status") != "eligible" and "feature_path" not in record:
            continue
        feature = record.get("feature_path")
        if feature and sha256_file(Path(feature)) != record.get("feature_sha256"):
            raise RuntimeError(f"{country} feature input changed during load")
    return {
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
        "manifest": manifest, "qa_state": qa_state, "target_column": target_column,
        "country_universe": universe, "inputs": inputs, "input_ledger": input_ledger,
        "eligibility": eligibility,
    }


def metric_original_scale(y: Sequence[float], prediction: Sequence[float]) -> dict[str, float | int]:
    y_array = np.asarray(y, dtype="float64")
    p_array = np.asarray(prediction, dtype="float64")
    if len(y_array) != len(p_array) or not len(y_array):
        raise ValueError("metric arrays must have equal nonzero length")
    if not np.isfinite(y_array).all() or not np.isfinite(p_array).all():
        raise ValueError("original-scale metrics require finite outcomes and predictions")
    residual = p_array - y_array
    denominator = float(np.sum((y_array - y_array.mean()) ** 2))
    return {
        "n": int(len(y_array)),
        "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def _load_cluster_bootstrap():
    candidates = [HERE.parent / "fast_cluster_bootstrap.py", HERE / "fast_cluster_bootstrap.py"]
    helper_path = next((path for path in candidates if path.is_file()), None)
    if helper_path is None:
        raise FileNotFoundError("existing fast_cluster_bootstrap.py was not found in work/repair_v2")
    spec = importlib.util.spec_from_file_location("repair_v2_loco_fast_bootstrap", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load paired bootstrap helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "paired_bootstrap", None)
    if not callable(function):
        raise ImportError("fast_cluster_bootstrap.py has no callable paired_bootstrap")
    return function, helper_path.resolve(), sha256_file(helper_path)


def _seed_for(*parts: str, base_seed: int = 20261113) -> int:
    digest = hashlib.sha256("|".join((str(base_seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _feature_sets(with_met: bool) -> dict[str, list[str]]:
    base = list(GEO) + (list(MET) if with_met else [])
    return {
        "B": base,
        "B+AOD": base + list(AOD),
        "B+fire": base + list(FIRE),
        "full": base + list(AOD) + list(FIRE),
    }


def _met_eligibility(inputs: Mapping[str, pd.DataFrame], ledger: Mapping[str, Any]) -> tuple[bool, str | None]:
    countries = sorted(inputs)
    if len(countries) < 2:
        return False, "fewer_than_two_eligible_countries"
    incomplete = [
        country for country in countries
        if not ledger[country]["frame_audit"]["met_complete_observed"]
    ]
    if incomplete:
        detail = ",".join(
            f"{country}({ledger[country]['frame_audit']['met_incomplete_reason']})"
            for country in incomplete
        )
        return False, f"incomplete_observed_met_support:{detail}"
    profiles = {ledger[country].get("met_profile_id") for country in countries}
    if None in profiles or len(profiles) != 1:
        return False, "met_profile_id_missing_or_not_comparable_across_countries"
    return True, None


def _hgb_early_stopping_support(train: pd.DataFrame, config: ModelConfig) -> dict[str, Any]:
    """Check the final HGB fit and actual grouped-smear train folds for a valid holdout split."""
    if not config.early_stopping:
        return {"supported": True, "checked": False, "reason": None,
                "final_train_rows": int(len(train)), "inner_train_rows": []}
    group_values = train["_loco_site_group"].astype(str).to_numpy()
    n_groups = int(pd.Series(group_values).nunique())
    if n_groups < 2:
        return {"supported": False, "checked": True,
                "reason": "crossfit_smearing_requires_at_least_two_training_sites",
                "final_train_rows": int(len(train)), "inner_train_rows": []}
    n_splits = min(3, n_groups)
    inner_train_counts: list[int] = []
    failures: list[str] = []
    checks = [("final_fit", np.arange(len(train), dtype="int64"))]
    for fold_id, (inner_train_idx, _) in enumerate(
        GroupKFold(n_splits=n_splits).split(train, groups=group_values)
    ):
        checks.append((f"smear_inner_{fold_id}", inner_train_idx))
    for label, indices in checks:
        row_count = int(len(indices))
        if label.startswith("smear_inner_"):
            inner_train_counts.append(row_count)
        try:
            train_idx, validation_idx = train_test_split(
                np.arange(row_count, dtype="int64"),
                test_size=config.validation_fraction,
                random_state=config.random_state,
            )
            if not len(train_idx) or not len(validation_idx):
                failures.append(f"{label}:empty_early_stopping_partition(n={row_count})")
        except ValueError as exc:
            failures.append(f"{label}:early_stopping_partition_error(n={row_count};{exc})")
    return {
        "supported": not failures, "checked": True,
        "reason": ";".join(failures) if failures else None,
        "final_train_rows": int(len(train)), "inner_train_rows": inner_train_counts,
        "inner_splits": n_splits, "validation_fraction": config.validation_fraction,
        "early_stopping": bool(config.early_stopping),
    }


def evaluate_loco(
    inputs: Mapping[str, pd.DataFrame],
    ledger: Mapping[str, Any],
    *,
    target_column: str = "pm25",
    include_ridge: bool = False,
    bootstrap_replicates: int = 500,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Fit LOCO arms in memory; useful for deterministic synthetic checks."""
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    countries = sorted(inputs)
    if len(countries) < 2:
        rows = []
        for country in countries:
            rows.append({"country": country, "scenario": "noMET", "status": "ineligible",
                         "reason": "fewer_than_two_eligible_countries"})
            rows.append({"country": country, "scenario": "withMET", "status": "ineligible",
                         "reason": "fewer_than_two_eligible_countries"})
        if not countries:
            rows.append({"country": "ALL", "scenario": "all", "status": "not_estimable",
                         "reason": "no_eligible_countries_after_hash_QA_target_and_historical_checks"})
        return {"metrics": [], "pairwise": [], "predictions": [], "folds": [],
                "train_composition": [], "eligibility": rows, "status": "not_estimable"}

    met_ok, met_reason = _met_eligibility(inputs, ledger)
    scenarios = [("noMET", False)]
    if met_ok:
        scenarios.append(("withMET", True))
    models: list[tuple[str, ModelConfig]] = [("hgb", config or ModelConfig())]
    if models[0][1].estimator_kind != "hgb":
        raise ValueError("primary LOCO estimator config must be HGB")
    if include_ridge:
        models.append(("ridge", replace(models[0][1], estimator_kind="ridge", ridge_alpha=1.0)))

    metrics: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    bootstrap, bootstrap_path, bootstrap_hash = _load_cluster_bootstrap()

    for heldout in countries:
        training_countries = [country for country in countries if country != heldout]
        train = pd.concat([inputs[c] for c in training_countries], ignore_index=True, sort=False)
        test = inputs[heldout].copy().reset_index(drop=True)
        train_groups = set(train["_loco_site_group"].astype(str))
        test_groups = set(test["_loco_site_group"].astype(str))
        if train_groups & test_groups:
            raise AssertionError("country-qualified station groups overlap across LOCO train/test")
        row_n_train = int(len(train))
        site_n_train = int(train._loco_site_group.nunique())
        for country in training_countries:
            country_frame = inputs[country]
            composition.append({
                "heldout_country": heldout, "train_country": country,
                "n_train_rows": int(len(country_frame)),
                "n_train_sites": int(country_frame._loco_site_group.nunique()),
                "pooled_row_weight_fraction": float(len(country_frame) / row_n_train) if row_n_train else np.nan,
                "pooled_site_fraction": float(country_frame._loco_site_group.nunique() / site_n_train) if site_n_train else np.nan,
                "weighting_rule": "pooled_rows_unweighted; candidate country row fractions are implicit training weights",
            })
        if len(test_groups) == 0:
            for scenario, _ in scenarios:
                eligibility.append({"country": heldout, "scenario": scenario, "status": "ineligible",
                                    "reason": "no_test_sites", "n_train_rows": int(len(train)),
                                    "n_train_sites": int(len(train_groups)), "n_test_rows": int(len(test)),
                                    "n_test_sites": int(len(test_groups))})
            continue
        if len(train_groups) < 2:
            for scenario, _ in scenarios:
                eligibility.append({"country": heldout, "scenario": scenario, "status": "ineligible",
                                    "reason": "crossfit_smearing_requires_at_least_two_training_sites",
                                    "n_train_rows": int(len(train)), "n_train_sites": int(len(train_groups)),
                                    "n_test_rows": int(len(test)), "n_test_sites": int(len(test_groups))})
            continue
        hgb_support = _hgb_early_stopping_support(train, models[0][1])
        if not hgb_support["supported"]:
            reason = f"hgb_early_stopping_training_support_insufficient:{hgb_support['reason']}"
            for scenario, _ in scenarios:
                eligibility.append({
                    "country": heldout, "scenario": scenario, "status": "ineligible",
                    "reason": reason, "n_train_rows": int(len(train)),
                    "n_train_sites": int(len(train_groups)), "n_test_rows": int(len(test)),
                    "n_test_sites": int(len(test_groups)),
                    "hgb_early_stopping_support": hgb_support,
                })
            continue

        train_keys = train[["country", "site_id", "year", "month"]].astype(str).to_dict("records")
        test_keys = test[["country", "site_id", "year", "month"]].astype(str).to_dict("records")
        train_key_hash = _sha256_json(sorted(tuple(row[k] for k in ("country", "site_id", "year", "month"))
                                             for row in train_keys))
        test_key_hash = _sha256_json(sorted(tuple(row[k] for k in ("country", "site_id", "year", "month"))
                                            for row in test_keys))
        train_y_hash = _sha256_json(pd.to_numeric(train[target_column], errors="raise").to_numpy(float).tolist())
        for scenario, with_met in scenarios:
            if with_met and not met_ok:
                eligibility.append({"country": heldout, "scenario": scenario, "status": "ineligible",
                                    "reason": met_reason})
                continue
            feature_sets = _feature_sets(with_met)
            fold_prediction_hashes = set()
            fold_model_records = []
            for model_name, model_config in models:
                arm_predictions: dict[str, np.ndarray] = {}
                for arm in ARMS:
                    features = feature_sets[arm]
                    # The fit interface sees only other-country rows. In particular,
                    # the held-out target is never passed to preprocessing or smear.
                    bundle = fit_model(
                        train, features, target_col=target_column,
                        group_col="_loco_site_group", config=model_config,
                        smear_strategy="crossfit_station", smear_splits=3,
                        target_strategy="log_smear",
                    )
                    if bundle["smear_info"].get("residual_source") != "training-only group out-of-fold":
                        raise AssertionError("LOCO smearing did not use training-only grouped crossfit")
                    if bundle["fit_n"] != row_n_train or bundle["fit_stations"] != site_n_train:
                        raise AssertionError("shared fit bundle reports unexpected LOCO training support")
                    predicted = predict_model(bundle, test, backtransform="smearing")
                    if len(predicted) != len(test) or not np.isfinite(predicted).all():
                        raise ValueError(f"{heldout}/{scenario}/{model_name}/{arm} produced invalid predictions")
                    arm_predictions[arm] = predicted
                    prediction_key_hash = _sha256_json(test_keys)
                    fold_prediction_hashes.add(prediction_key_hash)
                    fold_model_records.append({
                        "estimator": model_name, "arm": arm,
                        "features": list(features), "model_config": asdict(model_config),
                        "fit_n": int(bundle["fit_n"]), "fit_sites": int(bundle["fit_stations"]),
                        "smear": float(bundle["smear"]), "smear_info": bundle["smear_info"],
                    })
                    for row_index, row in test.iterrows():
                        predictions.append({
                            "scenario": scenario, "estimator": model_name, "arm": arm,
                            "country": heldout, "site_id": str(row.site_id),
                            "year": int(row.year), "month": int(row.month),
                            "obs": float(row[target_column]), "pred": float(predicted[row_index]),
                            "site_group": str(row._loco_site_group),
                        })
                    metric = metric_original_scale(test[target_column].to_numpy(float), predicted)
                    metrics.append({
                        "scenario": scenario, "estimator": model_name, "arm": arm,
                        "country": heldout, "n_train": row_n_train,
                        "n_train_countries": int(len(training_countries)),
                        "n_train_sites": site_n_train, "train_key_sha256": train_key_hash,
                        "train_target_sha256": train_y_hash,
                        "test_key_sha256": test_key_hash,
                        "n_test_sites": int(len(test_groups)),
                        "target_status": ledger[heldout].get("target_status"),
                        **metric,
                    })
                if set(arm_predictions) != set(ARMS):
                    raise AssertionError("a LOCO estimator did not fit every fixed arm")
                for arm_a, arm_b in ((a, b) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]):
                    seed = _seed_for(scenario, model_name, heldout, arm_a, arm_b)
                    interval = bootstrap(
                        test["_loco_site_group"].astype(str).to_numpy(),
                        test[target_column].to_numpy(float), arm_predictions[arm_a],
                        arm_predictions[arm_b], bootstrap_replicates, seed,
                    )
                    ma = metric_original_scale(test[target_column].to_numpy(float), arm_predictions[arm_a])
                    mb = metric_original_scale(test[target_column].to_numpy(float), arm_predictions[arm_b])
                    pairwise.append({
                        "scenario": scenario, "estimator": model_name,
                        "country": heldout, "arm_a": arm_a, "arm_b": arm_b,
                        "contrast": f"{arm_a} - {arm_b}", "bootstrap_seed": seed,
                        "bootstrap_engine": "work/repair_v2/fast_cluster_bootstrap.py",
                        "delta_rmse": float(ma["rmse"] - mb["rmse"]),
                        "delta_mae": float(ma["mae"] - mb["mae"]),
                        "delta_bias": float(ma["bias"] - mb["bias"]),
                        "delta_r2": float(ma["r2"] - mb["r2"]),
                        **interval,
                    })
            if len(fold_prediction_hashes) != 1:
                raise AssertionError("fixed arms do not share identical LOCO test keys")
            folds.append({
                "scenario": scenario, "country": heldout,
                "train_countries": training_countries,
                "train_rows": row_n_train, "train_sites": site_n_train,
                "test_rows": int(len(test)), "test_sites": int(len(test_groups)),
                "train_key_sha256": train_key_hash, "train_target_sha256": train_y_hash,
                "test_key_sha256": test_key_hash,
                "arms_share_test_keys": True,
                "heldout_country_absent_from_fit": heldout not in training_countries,
                "preprocessing_fit_countries": training_countries,
                "crossfit_smearing_fit_countries": training_countries,
                "hgb_early_stopping_support": hgb_support,
                "estimator_fits": fold_model_records,
            })
            eligibility.append({"country": heldout, "scenario": scenario,
                                "status": "scored", "reason": "all_historical_test_rows_scored",
                                "n_train_rows": int(len(train)),
                                "n_train_sites": int(len(train_groups)),
                                "n_test_rows": int(len(test)), "n_test_sites": int(len(test_groups)),
                                "hgb_early_stopping_support": hgb_support})

    if not met_ok:
        for heldout in countries:
            eligibility.append({"country": heldout, "scenario": "withMET", "status": "ineligible",
                                "reason": met_reason})
    return {
        "metrics": metrics, "pairwise": pairwise, "predictions": predictions,
        "folds": folds, "train_composition": composition, "eligibility": eligibility,
        "status": "preliminary_historical_loco" if metrics else "not_estimable",
        "bootstrap_helper": {"path": str(bootstrap_path), "sha256": bootstrap_hash},
    }


def _runtime_code_snapshot() -> dict[str, dict[str, str]]:
    candidates = {
        "runner": HERE / "run_loco.py",
        "validation_core": HERE / "validation_core.py",
        "fast_cluster_bootstrap": HERE.parent / "fast_cluster_bootstrap.py",
    }
    snapshot = {}
    for name, path in candidates.items():
        resolved = path.resolve(strict=True)
        snapshot[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return snapshot


def _verify_code_snapshot(start: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    end = _runtime_code_snapshot()
    if dict(start) != end:
        raise RuntimeError(f"validation source changed during the run; start={dict(start)!r}; end={end!r}")
    return end


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }


def _write_json(path: Path, value: Any) -> None:
    """Publish JSON atomically so only complete manifest documents are visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"loco_{stamp}_{uuid.uuid4().hex[:8]}"


def _select_output_directory(requested: str | None) -> Path:
    root = OUTPUT_ROOT.resolve()
    if requested is None:
        out = root / _run_id()
    else:
        out = Path(requested).resolve()
        if out == root or not out.is_relative_to(root):
            raise ValueError(f"--out must be a unique directory under {root}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic ownership claim: an existing empty directory may belong to a
    # second launch that has not yet written its first status file.
    out.mkdir(exist_ok=False)
    return out


def _update_execution_status(out: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    state["partial_outputs"] = sorted(
        str(path.relative_to(out)) for path in out.rglob("*") if path.is_file() and path.name != "execution_status.json"
    )
    _write_json(out / "execution_status.json", state)


def write_preflight(
    loaded: Mapping[str, Any], out: Path, *,
    code_snapshot_start: Mapping[str, Mapping[str, str]], scoring_requested: bool,
) -> dict[str, dict[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "input_ledger.json", {
        "status": "inputs_verified",
        "model_scoring_requested": bool(scoring_requested),
        "manifest_path": loaded["manifest_path"],
        "manifest_sha256": loaded["manifest_sha256"],
        "qa_state": loaded["qa_state"], "historical_years": list(HISTORICAL_YEARS),
        "target_column": loaded["target_column"], "countries": loaded["input_ledger"],
        "runtime": _runtime_versions(), "code_snapshot_start": code_snapshot_start,
    })
    pd.DataFrame(loaded["eligibility"]).to_csv(out / "country_eligibility.csv", index=False)
    code_snapshot_end = _verify_code_snapshot(code_snapshot_start)
    _write_json(out / "preflight_manifest.json", {
        "format": "repair-v2-loco-preflight-v1", "stage": "preliminary",
        "status": "completed", "final_canonical_inputs_confirmed": False,
        "model_scoring_status": "requested_after_preflight" if scoring_requested else "not_requested",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "manifest_path": loaded["manifest_path"], "manifest_sha256": loaded["manifest_sha256"],
        "historical_years": list(HISTORICAL_YEARS), "qa_state": loaded["qa_state"],
        "country_universe": loaded["country_universe"],
        "eligible_countries": sorted(loaded["inputs"]),
        "runtime": _runtime_versions(),
        "code_snapshot_start": code_snapshot_start,
        "code_snapshot_end": code_snapshot_end,
        "code_unchanged": True,
        "limitations": [
            "Target provenance is hash-bound as supplied and is not an independent reconstruction of target cleaning.",
            "Canonical country input selection remains a separate input decision.",
        ],
    })
    return code_snapshot_end


def write_run_outputs(loaded: Mapping[str, Any], result: Mapping[str, Any], out: Path,
                      *, include_ridge: bool, bootstrap_replicates: int,
                      code_snapshot_start: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("metrics_original_scale.csv", "metrics"),
        ("paired_station_bootstrap.csv", "pairwise"),
        ("loco_predictions.csv", "predictions"),
        ("fold_provenance.csv", "folds"),
        ("training_country_weights.csv", "train_composition"),
        ("loco_fold_eligibility.csv", "eligibility"),
    ):
        pd.DataFrame(result[key]).to_csv(out / name, index=False)
    estimator_manifest = {"hgb": asdict(ModelConfig())}
    if include_ridge:
        estimator_manifest["ridge"] = {
            **asdict(ModelConfig(estimator_kind="ridge", ridge_alpha=1.0)),
            "ridge_alpha_fixed": 1.0,
        }
    code_snapshot_end = _verify_code_snapshot(code_snapshot_start)
    _write_json(out / "run_manifest.json", {
        "format": "repair-v2-loco-run-v1", "status": "completed",
        "analysis_status": result["status"],
        "stage": "preliminary_historical_only", "final_canonical_inputs_confirmed": False,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scored_years": list(HISTORICAL_YEARS), "new_2024_2025_outcomes_scored": False,
        "input_manifest_path": loaded["manifest_path"],
        "input_manifest_sha256": loaded["manifest_sha256"],
        "qa_state": loaded["qa_state"], "target_column": loaded["target_column"],
        "country_universe": loaded["country_universe"],
        "eligible_countries": sorted(loaded["inputs"]),
        "input_ledger": loaded["input_ledger"],
        "feature_arms": _feature_sets(False), "with_met_arms": _feature_sets(True),
        "estimators": estimator_manifest,
        "target_strategy": "log(pm25 + 0.1), train-only station-group crossfit smearing",
        "primary_inverse": "smear*exp(z)-0.1, clipped at zero via validation_core.predict_model",
        "smear_splits": 3,
        "smear_group_column": "country-qualified _loco_site_group",
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed_base": 20261113,
        "bootstrap_seed_derivation": "SHA-256(base|scenario|estimator|heldout_country|arm_a|arm_b), first 32 bits",
        "bootstrap_engine": result.get("bootstrap_helper"),
        "pooling": "pooled rows unweighted; per-country row and site fractions saved; no equal-country result",
        "runtime": _runtime_versions(),
        "code_snapshot_start": code_snapshot_start,
        "code_snapshot_end": code_snapshot_end,
        "code_unchanged": True,
        "limitations": [
            "Target provenance is hash-bound as supplied and is not an independent reconstruction of target cleaning.",
            "Canonical country input selection remains a separate input decision.",
        ],
    })
    return code_snapshot_end


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="hash-bound historical LOCO input JSON manifest")
    parser.add_argument(
        "--out", default=None,
        help="unique run directory under outputs/repair_v2/validation/loco_preflight; defaults to a timestamped run ID",
    )
    parser.add_argument("--run", action="store_true", help="score verified historical rows; default is preflight only")
    parser.add_argument("--include-ridge", action="store_true", help="add fixed Ridge alpha=1.0 sensitivity")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args(argv)

    try:
        code_snapshot_start = _runtime_code_snapshot()
        out = _select_output_directory(args.out)
    except Exception as exc:
        print(f"Cannot start LOCO output run: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2

    stage = "preflight"
    state: dict[str, Any] = {
        "format": "repair-v2-loco-execution-status-v1",
        "run_id": out.name,
        "output_dir": str(out),
        "input_manifest_path": str(Path(args.manifest).resolve()),
        "status": "preflight_in_progress",
        "stage": stage,
        "completed": False,
        "preflight_status": "in_progress",
        "scoring_status": "pending" if args.run else "not_requested",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "code_snapshot_start": code_snapshot_start,
    }
    preflight_complete = False
    run_manifest_complete = False
    try:
        _update_execution_status(out, state)
        loaded = load_manifest_inputs(args.manifest)
        code_snapshot_end = write_preflight(
            loaded, out, code_snapshot_start=code_snapshot_start,
            scoring_requested=args.run,
        )
        preflight_complete = True
        state.update({
            "status": "preflight_completed_scoring_pending" if args.run else "completed_preflight_only",
            "stage": "preflight_completed", "completed": not args.run,
            "preflight_status": "completed",
            "scoring_status": "pending" if args.run else "not_requested",
            "preflight_completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "code_snapshot_end": code_snapshot_end,
        })
        _update_execution_status(out, state)
        print(f"PREFLIGHT {out}; eligible={sorted(loaded['inputs'])}", flush=True)
        if not args.run:
            print("No model scoring was run. Add --run after canonical inputs are approved for historical rehearsal.", flush=True)
            return 0
        stage = "scoring"
        state.update({"status": "scoring_in_progress", "stage": stage,
                      "completed": False, "scoring_status": "in_progress"})
        _update_execution_status(out, state)
        result = evaluate_loco(
            loaded["inputs"], loaded["input_ledger"],
            target_column=loaded["target_column"], include_ridge=args.include_ridge,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        code_snapshot_end = write_run_outputs(
            loaded, result, out, include_ridge=args.include_ridge,
            bootstrap_replicates=args.bootstrap_replicates,
            code_snapshot_start=code_snapshot_start,
        )
        run_manifest_complete = True
        state.update({
            "status": "completed", "stage": "completed", "completed": True,
            "preflight_status": "completed", "scoring_status": "completed",
            "analysis_status": result["status"],
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "code_snapshot_end": code_snapshot_end,
            "completed_manifest": str(out / "run_manifest.json"),
        })
        try:
            _update_execution_status(out, state)
        except Exception as status_exc:
            # The completed atomic run manifest is authoritative if the mutable
            # convenience status file cannot be refreshed after publication.
            print(f"WARNING: completed run manifest published but execution status update failed: {status_exc}",
                  file=sys.stderr, flush=True)
        print(f"WROTE {out}; status={result['status']}; countries={sorted(loaded['inputs'])}", flush=True)
        return 0
    except Exception as exc:
        if run_manifest_complete:
            print(f"WARNING: completed manifest is present; later status update failed: {exc}",
                  file=sys.stderr, flush=True)
            return 0
        end_snapshot = None
        try:
            end_snapshot = _runtime_code_snapshot()
        except Exception:
            pass
        state.update({
            "status": "failed_partial", "stage": stage, "completed": False,
            "preflight_status": "completed" if preflight_complete else "failed",
            "scoring_status": (
                "failed" if stage == "scoring" else
                "not_reached" if args.run else "not_requested"
            ),
            "error_type": type(exc).__name__, "error": str(exc),
            "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "code_snapshot_end": end_snapshot,
        })
        try:
            _update_execution_status(out, state)
        except Exception as status_exc:
            print(f"Unable to write failure status: {status_exc}", file=sys.stderr, flush=True)
        print(f"LOCO preflight/run failed closed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
