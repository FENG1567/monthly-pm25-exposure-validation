#!/usr/bin/env python3
"""Re-summarize locked STOTEN results without refitting models or resampling."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUT = HERE / "csv"
V = Path("/external/project/validation")
SUP = V / "submission_editor_data_v2_20261115" / "Supplementary_Data"
FIG = HERE / "figure_source_inputs"
CATALOG_PATH = SUP / "data_catalog.json"

# SHA-256 values were read from the user's formal local package.  The
# Supplementary_Data manifest is also checked against its local-package hash.
EXPECTED_LOCAL_SHA256 = {
    "Supplementary_Data/data_catalog.json": "57f40d127def6cead517db4a6f85a1d3c49e4ea5d548a3ee9d6531f9e53a5c42",
    "Supplementary_Data/historical_metrics.csv": "d93385f68a85cfedbf7f5aa532af27c53e921c6033abfc6ace1e3a9bc121980a",
    "Supplementary_Data/joint_future.csv": "2f836b8c5205d13f10fa2fb16bfbe0cd78621039b40bca2f0680aa859323c6a8",
    "Supplementary_Data/annual_deltas.csv": "6a449dd8bb503b737b358e6ea4b6b355b7d7d25d4f19e0f52c72b054c46bb557",
    "Supplementary_Data/annual_metrics.csv": "679af31fcc4df1240cc346e8de629de4bf5ea1b2e3081ede784daf17016230b6",
    "Supplementary_Data/multiplicity.csv": "06f4a25b438275a286dff630b3e8af225f1a52a5224fa964dccc31f9ce519398",
    "Figure_Source_Data/Supplementary_Figure_S3_Panel_b_Source_Data.csv": "19f0ec0d60a0cf d984f4f185537d0a4703bee59520eeb2cae2af018f584de907".replace(" ", ""),
    "Figure_Source_Data/Supplementary_Figure_S2_Panel_a_Source_Data.csv": "d5a5095a91b0432ca6a764778b8a1c38db174bf822356953b0a22604673d2f32",
    "Figure_Source_Data/Supplementary_Figure_S2_Panel_b_Source_Data.csv": "0db301d33fac389d5c4dd0651bcca48c80260c8f42e52563809cef148d38db04",
    "Figure_Source_Data/Supplementary_Figure_S8_Panel_a_Source_Data.csv": "d78e5cf2de0c89e386ac5b8ae8b3278ec052aff275baf822a0c63922faae2044",
    "Figure_Source_Data/Supplementary_Figure_S8_Panel_b_Source_Data.csv": "39ffcdebf d9a54de9853b32983034d9199513290521ca07353d1ba4a71a63001".replace(" ", ""),
}
FIGURE_FILES = {
    "Supplementary_Figure_S3_Panel_b_Source_Data.csv": "inverse_transform_figure_source_crosscheck",
    "Supplementary_Figure_S2_Panel_a_Source_Data.csv": "conformal_figure_source",
    "Supplementary_Figure_S2_Panel_b_Source_Data.csv": "future_risk_figure_source_crosscheck",
    "Supplementary_Figure_S8_Panel_a_Source_Data.csv": "annual_comparator_figure_source",
    "Supplementary_Figure_S8_Panel_b_Source_Data.csv": "annual_comparator_figure_source",
}
KNOWN_ANALYSIS_SOURCE_MAP = {
    # These four labels were provided as exact S3a source literals by the root.
    # Other values, including blank profile/QA columns, remain unmapped.
    "can_hist_primary_strict_with_met_hgb": {
        "mapped_generation": "CAN historical primary strict with-MET HGB",
        "mapped_country": "CAN", "mapped_profile": "with_met", "mapped_qa_variant": "strict", "mapped_estimator": "hgb",
    },
    "can_hist_primary_strict_no_met_hgb": {
        "mapped_generation": "CAN historical primary strict no-MET HGB",
        "mapped_country": "CAN", "mapped_profile": "no_met", "mapped_qa_variant": "strict", "mapped_estimator": "hgb",
    },
    "usa_hist_strict_with_met_hgb": {
        "mapped_generation": "USA historical strict with-MET HGB",
        "mapped_country": "USA", "mapped_profile": "with_met", "mapped_qa_variant": "strict", "mapped_estimator": "hgb",
    },
    "usa_hist_strict_no_met_hgb": {
        "mapped_generation": "USA historical strict no-MET HGB",
        "mapped_country": "USA", "mapped_profile": "no_met", "mapped_qa_variant": "strict", "mapped_estimator": "hgb",
    },
}
SUPPLEMENTARY_FILES = {
    "historical_metrics.csv": "inverse_transform_metrics",
    "joint_future.csv": "future_risk_coverage",
    "annual_deltas.csv": "annual_comparator_deltas",
    "annual_metrics.csv": "annual_comparator_metrics_crosscheck",
    "multiplicity.csv": "multiplicity_family",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fields, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize(row.get(k)) for k in fieldnames})


def serialize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".15g")
    return value


def num(v: Any) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def int_num(v: Any) -> int | None:
    x = num(v)
    if x is None or not x.is_integer():
        return None
    return int(x)


def row_id(filename: str, idx: int) -> str:
    return f"{filename}#row={idx}"


def quartile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    x = sorted(sorted_values)
    pos = (len(x) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return x[lo]
    return x[lo] + (x[hi] - x[lo]) * (pos - lo)


def boolean_field(v: Any) -> bool | None:
    s = str(v or "").strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def literal_year(task: str) -> tuple[str, str]:
    # The year is extracted only from the explicit task token T4_external_YYYY.
    m = re.search(r"(?:^|_)external_(20\d{2})(?:_|$)", task or "")
    if m:
        return m.group(1), "task_label_exact_external_YYYY_token"
    return "", "not_explicit_in_task_label"


def source_binding(name: str, source_alias: str, source_path: Path, role: str, idx: int) -> dict[str, Any]:
    return {
        "source_file": name,
        "source_alias": source_alias,
        "source_role": role,
        "source_row_number": idx,
        "record_id": row_id(name, idx),
        "source_sha256": sha256(source_path),
    }


def copy_with_binding(row: dict[str, str], binding: dict[str, Any]) -> dict[str, Any]:
    out = dict(binding)
    out.update(row)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    qa: dict[str, Any] = {
        "status": "running",
        "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "execution": {
            "host": platform.node(),
            "python": sys.version,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "model_fits": 0,
            "bootstrap_runs": 0,
        },
        "input_root": str(SUP),
        "figure_source_input_root": str(FIG),
        "source_bindings": [],
        "exclusions": [],
        "checks": {},
    }
    expected = dict(EXPECTED_LOCAL_SHA256)
    input_paths: dict[str, Path] = {}
    source_aliases: dict[str, str] = {}
    source_roles: dict[str, str] = {}
    source_hashes: dict[str, str] = {}

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Required source catalog is absent: {CATALOG_PATH}")
    catalog_hash = sha256(CATALOG_PATH)
    qa["source_catalog_sha256_server"] = catalog_hash
    qa["checks"]["catalog_matches_local_formal_package"] = catalog_hash == expected["Supplementary_Data/data_catalog.json"]
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_entries = {str(x.get("file")): x for x in catalog.get("files", []) if x.get("file")}

    for name, role in SUPPLEMENTARY_FILES.items():
        path = SUP / name
        if not path.exists():
            qa["exclusions"].append({"file": name, "reason": "required_supplementary_source_missing"})
            continue
        input_paths[name] = path
        source_roles[name] = role
        entry = catalog_entries.get(name)
        source_aliases[name] = str(entry.get("source_alias", "")) if entry else ""
        source_hashes[name] = sha256(path)

    for name, role in FIGURE_FILES.items():
        path = FIG / name
        if not path.exists():
            qa["exclusions"].append({"file": name, "reason": "formal_figure_source_not_transferred"})
            continue
        input_paths[name] = path
        source_roles[name] = role
        source_aliases[name] = name
        source_hashes[name] = sha256(path)

    # Validate the formal Supplementary_Data file hashes from the local manifest,
    # and each Additional conformal file against the manifest's row-level hash.
    hash_rows: list[dict[str, Any]] = []
    for name, path in input_paths.items():
        expected_hash = expected.get(f"Supplementary_Data/{name}") or expected.get(f"Figure_Source_Data/{name}")
        actual_hash = source_hashes[name]
        entry = catalog_entries.get(name)
        manifest_hash = str(entry.get("sha256", "")).lower() if entry else ""
        compare_hash = expected_hash or manifest_hash
        hash_rows.append({
            "source_file": name,
            "source_role": source_roles[name],
            "source_alias": source_aliases[name],
            "sha256_server": actual_hash,
            "sha256_local_package_or_catalog": compare_hash,
            "hash_match": bool(compare_hash and actual_hash == compare_hash),
            "hash_binding": "local_formal_package_file" if expected_hash else ("data_catalog.json entry" if manifest_hash else "no_expected_hash"),
        })
    # Dynamically add every explicitly catalogued conformal summary source.
    conformal_entries = []
    for entry in catalog.get("files", []):
        name = str(entry.get("file", ""))
        if name.startswith("Additional_") and name.endswith(("_conformal_summary.csv", "_conformal_feasibility_summary.csv")):
            conformal_entries.append(entry)
    conformal_inputs: list[tuple[str, str, dict[str, Any], Path, list[str], list[dict[str, str]]]] = []
    for entry in conformal_entries:
        name = str(entry["file"])
        path = SUP / name
        alias = str(entry.get("source_alias", ""))
        if not alias:
            qa["exclusions"].append({"file": name, "reason": "catalog_source_alias_missing; no alias inferred"})
            continue
        if not path.exists():
            qa["exclusions"].append({"file": name, "reason": "catalogued_conformal_source_missing"})
            continue
        digest = sha256(path)
        expected_manifest = str(entry.get("sha256", "")).lower()
        hash_rows.append({
            "source_file": name,
            "source_role": "conformal_feasibility" if name.endswith("_conformal_feasibility_summary.csv") else "conformal_summary",
            "source_alias": alias,
            "sha256_server": digest,
            "sha256_local_package_or_catalog": expected_manifest,
            "hash_match": bool(expected_manifest and digest == expected_manifest),
            "hash_binding": "data_catalog.json entry",
        })
        fields, rows = load_csv(path)
        expected_rows = entry.get("rows")
        conformal_inputs.append((name, alias, entry, path, fields, rows))
        if expected_rows is not None and int(expected_rows) != len(rows):
            qa["exclusions"].append({"file": name, "reason": "catalog_row_count_mismatch", "expected": expected_rows, "observed": len(rows)})
    write_csv(OUT / "source_hashes.csv", hash_rows)
    qa["source_bindings"] = hash_rows
    qa["checks"]["all_source_hashes_match_local_or_catalog"] = bool(hash_rows) and all(r["hash_match"] for r in hash_rows)

    # Summary 1: historical inverse-transform sensitivity, exact source rows.
    h_name = "historical_metrics.csv"
    h_path = input_paths.get(h_name)
    h_fields, h_rows = load_csv(h_path) if h_path else ([], [])
    transform_rows: list[dict[str, Any]] = []
    transform_exclusions: Counter[str] = Counter()
    primary_refs: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, r in enumerate(h_rows, 1):
        if (r.get("qa_variant") == "strict" and r.get("estimator_kind") == "hgb"
                and r.get("arm") == "full" and r.get("profile") in {"no_met", "with_met"}):
            if r.get("target_strategy") == "log_smear" and r.get("backtransform") == "smearing":
                key = (r.get("country", ""), r.get("task", ""), r.get("profile", ""),
                       r.get("cohort_label", ""), r.get("dataset_key", ""), r.get("smear_strategy", ""))
                primary_refs[key].append((idx, r))
    transform_claims: list[dict[str, Any]] = []
    for idx, r in enumerate(h_rows, 1):
        if not (r.get("qa_variant") == "strict" and r.get("estimator_kind") == "hgb"
                and r.get("arm") == "full" and r.get("profile") in {"no_met", "with_met"}):
            transform_exclusions["not_strict_hgb_full_with_met_or_no_met"] += 1
            continue
        key = (r.get("country", ""), r.get("task", ""), r.get("profile", ""),
               r.get("cohort_label", ""), r.get("dataset_key", ""), r.get("smear_strategy", ""))
        refs = primary_refs.get(key, [])
        ref = refs[0][1] if len(refs) == 1 else None
        out = copy_with_binding(r, {
            "source_file": h_name,
            "source_alias": source_aliases.get(h_name, ""),
            "source_role": source_roles.get(h_name, ""),
            "source_row_number": idx,
            "record_id": row_id(h_name, idx),
            "reference_record_id": row_id(h_name, refs[0][0]) if len(refs) == 1 else "",
            "reference_candidates_n": len(refs),
            "reference_definition": "same country/task/profile/cohort/dataset_key/smear_strategy; target_strategy=log_smear; backtransform=smearing; strict HGB full arm",
            "difference_status": "unique_reference" if len(refs) == 1 else ("reference_missing" if not refs else "reference_ambiguous"),
            "delta_r2_vs_log_smear_smearing": (num(r.get("r2")) - num(ref.get("r2"))) if ref and num(r.get("r2")) is not None and num(ref.get("r2")) is not None else "",
            "delta_rmse_vs_log_smear_smearing": (num(r.get("rmse")) - num(ref.get("rmse"))) if ref and num(r.get("rmse")) is not None and num(ref.get("rmse")) is not None else "",
            "delta_bias_vs_log_smear_smearing": (num(r.get("bias")) - num(ref.get("bias"))) if ref and num(r.get("bias")) is not None and num(ref.get("bias")) is not None else "",
        })
        transform_rows.append(out)
        transform_claims.append({
            "id": f"transform_{idx}",
            "value": {k: r.get(k, "") for k in ["country", "task", "profile", "target_strategy", "smear_strategy", "backtransform", "r2", "rmse", "bias", "n", "n_stations"]},
            "row_binding": {"file": h_name, "source_alias": source_aliases.get(h_name, ""), "row_number": idx, "record_id": row_id(h_name, idx),
                            "filters": {"qa_variant": "strict", "estimator_kind": "hgb", "arm": "full", "profile": r.get("profile")}},
            "reference_binding": row_id(h_name, refs[0][0]) if len(refs) == 1 else "",
        })
    write_csv(OUT / "transform_sensitivity.csv", transform_rows)
    qa["transform_sensitivity"] = {"input_rows": len(h_rows), "selected_rows": len(transform_rows), "excluded_rows_by_reason": dict(transform_exclusions), "reference_branches_n": sum(map(len, primary_refs.values()))}

    # Summary 2: future monthly risk/coverage.  All rows retained; only explicitly
    # retained rows enter the risk table; false/sentinel rows get an exclusion reason.
    f_name = "joint_future.csv"
    f_path = input_paths.get(f_name)
    f_fields, f_rows = load_csv(f_path) if f_path else ([], [])
    future_rows: list[dict[str, Any]] = []
    future_risk_rows: list[dict[str, Any]] = []
    future_exclusions: list[dict[str, Any]] = []
    future_claims: list[dict[str, Any]] = []
    for idx, r in enumerate(f_rows, 1):
        p = boolean_field(r.get("retained"))
        policy = (r.get("dataset_policy") or "").strip().lower()
        exact_main = (r.get("qa_variant") == "strict" and r.get("estimator_kind") == "hgb"
                      and r.get("target_strategy") == "log_smear" and r.get("backtransform") == "smearing"
                      and r.get("profile") in {"no_met", "with_met"} and "primary" in policy)
        if exact_main:
            cls = "primary_strict_hgb_log_smear_smearing"
        elif r.get("qa_variant") == "sensitivity" or "sensitivity" in policy:
            cls = "sensitivity"
        else:
            cls = "other_or_unclassified"
        out = copy_with_binding(r, {
            "source_file": f_name, "source_alias": source_aliases.get(f_name, ""), "source_role": source_roles.get(f_name, ""),
            "source_row_number": idx, "record_id": row_id(f_name, idx), "summary_class": cls,
            "classification_rule": "explicit qa_variant/estimator_kind/target_strategy/backtransform/profile/dataset_policy fields; no analysis_source inference",
            "risk_table_included": p is True,
        })
        future_rows.append(out)
        if p is True:
            future_risk_rows.append(out)
            future_claims.append({
                "id": f"future_{idx}",
                "value": {k: r.get(k, "") for k in ["country", "task", "profile", "target_strategy", "smear_strategy", "backtransform", "rule", "fire_month_scope", "n", "n_stations", "eligible_rows", "eligible_stations", "row_coverage_fraction", "station_coverage_fraction", "full_rmse", "full_bias", "full_r2", "delta_rmse_full_minus_B", "dataset_policy", "validation_mode", "train_through"]},
                "row_binding": {"file": f_name, "source_alias": source_aliases.get(f_name, ""), "row_number": idx, "record_id": row_id(f_name, idx),
                                "filters": {"retained": "True", "summary_class": cls, "country": r.get("country"), "task": r.get("task"), "profile": r.get("profile"), "rule": r.get("rule"), "fire_month_scope": r.get("fire_month_scope"), "dataset_policy": r.get("dataset_policy")}},
            })
        elif p is False:
            future_exclusions.append({"source_file": f_name, "source_alias": source_aliases.get(f_name, ""), "source_row_number": idx,
                                      "record_id": row_id(f_name, idx), "country": r.get("country"), "task": r.get("task"), "profile": r.get("profile"),
                                      "rule": r.get("rule"), "fire_month_scope": r.get("fire_month_scope"), "dataset_policy": r.get("dataset_policy"),
                                      "reason": "retained=False sentinel; excluded from error-risk estimates; source row preserved in future_risk_coverage_rows.csv"})
        else:
            future_exclusions.append({"source_file": f_name, "source_alias": source_aliases.get(f_name, ""), "source_row_number": idx,
                                      "record_id": row_id(f_name, idx), "reason": "retained status missing/unrecognized; kept in all-rows table, excluded from risk summary"})
    write_csv(OUT / "future_risk_coverage_rows.csv", future_rows)
    write_csv(OUT / "future_risk_summary.csv", future_risk_rows)
    write_csv(OUT / "future_risk_exclusions.csv", future_exclusions)
    qa["future_risk_coverage"] = {
        "input_rows": len(f_rows), "all_rows_preserved": len(future_rows), "retained_risk_rows": len(future_risk_rows),
        "excluded_rows": len(future_exclusions), "summary_class_counts": dict(Counter(r["summary_class"] for r in future_rows)),
        "profile_values": sorted({r.get("profile", "") for r in f_rows}),
        "month_scope_values": sorted({r.get("fire_month_scope", "") for r in f_rows}),
        "support_rule_values": sorted({r.get("rule", "") for r in f_rows}),
        "validation_modes": sorted({r.get("validation_mode", "") for r in f_rows}),
        "country_task_fold_aggregation": "none; source records preserved row-by-row",
    }

    # Summary 3: conformal rows across all catalogued generations plus the
    # formal figure source.  Keep folds and source aliases separate.
    conformal_rows: list[dict[str, Any]] = []
    conformal_exclusions: list[dict[str, Any]] = []
    conformal_primary_mapped: list[dict[str, Any]] = []
    conformal_claims: list[dict[str, Any]] = []
    conformal_grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    figure_s3a = "Supplementary_Figure_S2_Panel_a_Source_Data.csv"
    s3a_source_rows: list[dict[str, str]] = []
    if figure_s3a in input_paths:
        fields, s3a_source_rows = load_csv(input_paths[figure_s3a])
        conformal_inputs.append((figure_s3a, figure_s3a, {"sha256": source_hashes[figure_s3a], "rows": len(s3a_source_rows)}, input_paths[figure_s3a], fields, s3a_source_rows))
    for name, alias, entry, path, fields, rows in conformal_inputs:
        figure_source = name == figure_s3a
        if name.endswith("_conformal_feasibility_summary.csv"):
            source_type = "feasibility_file"
        elif name.endswith("_conformal_summary.csv"):
            source_type = "summary_file"
        else:
            source_type = "figure_source_file"
        actual_source_sha = sha256(path)
        for idx, r in enumerate(rows, 1):
            split_label = r.get("split_label", "")
            if source_type == "feasibility_file":
                split_class = "feasible"
            elif split_label == "primary_80_20":
                split_class = "primary"
            else:
                split_class = "other_or_unclassified"
            qinf = boolean_field(r.get("q90_is_infinite"))
            q90_status = "infinite" if qinf is True else ("finite" if qinf is False else "not_reported")
            test_n = int_num(r.get("test_row_n"))
            point_cov = num(r.get("pointwise_coverage"))
            point_num = None
            point_num_status = "not_identifiable_from_source_fields"
            if test_n is not None and point_cov is not None:
                candidate = point_cov * test_n
                nearest = round(candidate)
                if abs(candidate - nearest) <= 1e-6:
                    point_num = int(nearest)
                    point_num_status = "derived_from_source_fraction_times_test_row_n_exact_integer_check"
                else:
                    point_num_status = "rounded_source_fraction_does_not_yield_integer_numerator"
            all_num = int_num(r.get("all_month_per_station_covered_n"))
            all_den = int_num(r.get("all_month_per_station_group_n"))
            all_fraction = (all_num / all_den) if all_num is not None and all_den not in (None, 0) else None
            target_policy = r.get("target_policy") or r.get("target_strategy") or ""
            if r.get("target_policy"):
                target_policy_binding = "source_column:target_policy"
            elif r.get("target_strategy"):
                target_policy_binding = "source_column:target_strategy"
            else:
                target_policy_binding = "not_reported; not inferred"
            analysis_source_raw = (r.get("analysis_source") or "").strip()
            explicit_analysis_map = KNOWN_ANALYSIS_SOURCE_MAP.get(analysis_source_raw)
            analysis_source_mapping_status = "mapped_exact_literal" if explicit_analysis_map else ("missing_unmapped" if not analysis_source_raw else "unrecognized_unmapped")
            binding = {
                "source_file": name, "source_alias": alias, "source_role": "conformal_" + source_type,
                "source_sha256": actual_source_sha, "source_row_number": idx, "record_id": row_id(name, idx),
                "split_class": split_class, "target_policy_normalized": target_policy,
                "target_policy_binding": target_policy_binding, "q90_status": q90_status,
                "analysis_source_mapping_status": analysis_source_mapping_status,
                "mapped_generation": explicit_analysis_map.get("mapped_generation", "") if explicit_analysis_map else "",
                "mapped_country": explicit_analysis_map.get("mapped_country", "") if explicit_analysis_map else "",
                "mapped_profile": explicit_analysis_map.get("mapped_profile", "") if explicit_analysis_map else "",
                "mapped_qa_variant": explicit_analysis_map.get("mapped_qa_variant", "") if explicit_analysis_map else "",
                "mapped_estimator": explicit_analysis_map.get("mapped_estimator", "") if explicit_analysis_map else "",
                "pointwise_covered_row_n_derived": point_num, "pointwise_numerator_status": point_num_status,
                "pointwise_coverage_recomputed_from_numerator_denominator": (point_num / test_n) if point_num is not None and test_n else None,
                "all_month_station_covered_n_source": all_num, "all_month_station_group_n_source": all_den,
                "all_month_station_coverage_recomputed_from_counts": all_fraction,
                "coverage_pooling_rule": "no pooling across folds or source generations; source numerators/denominators shown at record level",
            }
            out = copy_with_binding(r, binding)
            out["normalized_country_for_grouping"] = r.get("country") or (explicit_analysis_map.get("mapped_country", "") if explicit_analysis_map else "")
            out["normalized_profile_for_grouping"] = r.get("profile") or (explicit_analysis_map.get("mapped_profile", "") if explicit_analysis_map else "")
            out["normalized_qa_variant_for_grouping"] = r.get("qa_variant") or (explicit_analysis_map.get("mapped_qa_variant", "") if explicit_analysis_map else "")
            out["normalized_estimator_for_grouping"] = r.get("estimator_kind") or (explicit_analysis_map.get("mapped_estimator", "") if explicit_analysis_map else "")
            conformal_rows.append(out)
            key = (out["normalized_country_for_grouping"], r.get("task", ""), out["normalized_qa_variant_for_grouping"], out["normalized_profile_for_grouping"], target_policy,
                   split_class, split_label, r.get("arm", ""), name, alias)
            conformal_grouped[key].append(out)
            missing = []
            if not out["normalized_country_for_grouping"]: missing.append("country")
            if not r.get("task"): missing.append("task")
            if not out["normalized_profile_for_grouping"]: missing.append("MET/profile")
            if not out["normalized_qa_variant_for_grouping"]: missing.append("QA/policy")
            if not out["normalized_estimator_for_grouping"]: missing.append("estimator")
            if not split_label: missing.append("split_label")
            if test_n is None: missing.append("test_row_n denominator for pointwise coverage")
            if point_cov is None: missing.append("pointwise_coverage numerator/denominator")
            if all_num is None or all_den is None: missing.append("all_month_per_station_covered_n/group_n")
            if qinf is None: missing.append("q90_is_infinite")
            if missing:
                conformal_exclusions.append({"record_id": row_id(name, idx), "source_file": name, "source_alias": alias,
                                             "reason": "incomplete denominator/key fields for corresponding conformal summary: " + "; ".join(missing)})
            conformal_claims.append({
                "id": f"conformal_{len(conformal_claims)+1}",
                "value": {k: r.get(k, "") for k in ["country", "task", "qa_variant", "profile", "target_strategy", "target_policy", "split_label", "fold_id", "arm", "q90_is_infinite", "pointwise_coverage", "test_row_n", "all_month_per_station_covered_n", "all_month_per_station_group_n", "mean_interval_width", "median_interval_width", "calibration_station_n", "test_station_n", "calibration_row_n", "test_row_n", "test_months_per_station_min", "test_months_per_station_median", "test_months_per_station_max"]},
                "row_binding": {"file": name, "source_alias": alias, "row_number": idx, "record_id": row_id(name, idx),
                                "filters": {"source_type": source_type, "split_class": split_class, "country": r.get("country"), "task": r.get("task"), "profile": r.get("profile"), "split_label": split_label, "fold_id": r.get("fold_id"), "arm": r.get("arm")}},
            })
            if figure_source and analysis_source_raw in KNOWN_ANALYSIS_SOURCE_MAP and split_label == "primary_80_20" and r.get("arm") == "full":
                conformal_primary_mapped.append(out)
    write_csv(OUT / "conformal_fold_rows.csv", conformal_rows)
    policy_rows = []
    for key, rows in conformal_grouped.items():
        finite = sum(1 for r in rows if r.get("q90_status") == "finite")
        infinite = sum(1 for r in rows if r.get("q90_status") == "infinite")
        unknown = len(rows) - finite - infinite
        country, task, qa_variant, profile, policy, split_class, split_label, arm, source_file, source_alias = key
        folds = sorted({str(r.get("fold_id", "")) for r in rows})
        policy_rows.append({
            "country": country, "task": task, "qa_variant": qa_variant, "profile": profile,
            "target_policy_normalized": policy, "split_class": split_class, "split_label": split_label,
            "arm": arm, "source_file": source_file, "source_alias": source_alias,
            "record_count": len(rows), "fold_ids": folds, "finite_q90_record_n": finite,
            "infinite_q90_record_n": infinite, "unknown_q90_record_n": unknown,
            "finite_q90_record_fraction": finite / len(rows) if rows else None,
            "infinite_q90_record_fraction": infinite / len(rows) if rows else None,
            "coverage_values_pooled": False,
            "aggregation_note": "finite/infinite fractions use fold/source records as denominator; site-month and whole-station coverage remain in conformal_fold_rows.csv because folds/source generations may overlap",
        })
    write_csv(OUT / "conformal_policy_fold_counts.csv", policy_rows)
    write_csv(OUT / "conformal_exclusions.csv", conformal_exclusions)
    exclusion_counts: dict[tuple[str, str], int] = Counter((r.get("source_file", ""), r.get("reason", "")) for r in conformal_exclusions)
    write_csv(OUT / "conformal_exclusion_reason_counts.csv", [
        {"source_file": source, "reason": reason, "record_count": count}
        for (source, reason), count in sorted(exclusion_counts.items())
    ])
    write_csv(OUT / "conformal_primary_mapped_examples.csv", conformal_primary_mapped)
    primary_key_counts = Counter(
        "|".join(str(r.get(k, "")) for k in ["normalized_country_for_grouping", "task", "normalized_profile_for_grouping", "normalized_qa_variant_for_grouping", "normalized_estimator_for_grouping", "split_label", "fold_id", "arm"])
        for r in conformal_primary_mapped
    )
    conformal_primary_fold_summary = []
    for r in conformal_primary_mapped:
        out = {k: r.get(k, "") for k in ["source_file", "source_alias", "source_sha256", "record_id", "source_row_number", "analysis_source", "mapped_generation", "mapped_country", "mapped_profile", "mapped_qa_variant", "mapped_estimator", "normalized_country_for_grouping", "task", "normalized_profile_for_grouping", "normalized_qa_variant_for_grouping", "normalized_estimator_for_grouping", "target_policy_normalized", "split_label", "fold_id", "arm", "q90_status", "q90", "pointwise_covered_row_n_derived", "test_row_n", "pointwise_coverage", "all_month_station_covered_n_source", "all_month_station_group_n_source", "all_month_station_coverage_recomputed_from_counts", "all_month_per_station_coverage", "mean_interval_width", "median_interval_width", "calibration_station_n", "test_station_n", "calibration_row_n", "test_months_per_station_min", "test_months_per_station_median", "test_months_per_station_max"]}
        key = "|".join(str(r.get(k, "")) for k in ["normalized_country_for_grouping", "task", "normalized_profile_for_grouping", "normalized_qa_variant_for_grouping", "normalized_estimator_for_grouping", "split_label", "fold_id", "arm"])
        out["unique_key"] = key
        out["unique_key_record_count"] = primary_key_counts[key]
        out["mapping_basis"] = "exact analysis_source literal supplied by root; raw source fields retained unchanged"
        conformal_primary_fold_summary.append(out)
    write_csv(OUT / "conformal_primary_fold_summary.csv", conformal_primary_fold_summary)
    write_csv(OUT / "conformal_analysis_source_mapping.csv", [
        {"analysis_source_literal": literal, **mapping, "mapping_basis": "exact literal supplied by root for S3a; raw source fields retained unchanged"}
        for literal, mapping in sorted(KNOWN_ANALYSIS_SOURCE_MAP.items())
    ])
    s3a_exact_source_n = sum(1 for r in s3a_source_rows if r.get("analysis_source") in KNOWN_ANALYSIS_SOURCE_MAP)
    s3a_primary_split_n = sum(1 for r in s3a_source_rows if r.get("analysis_source") in KNOWN_ANALYSIS_SOURCE_MAP and r.get("split_label") == "primary_80_20")
    s3a_primary_full_n = len(conformal_primary_fold_summary)
    qa["conformal"] = {
        "input_file_count": len(conformal_inputs), "row_count": len(conformal_rows), "group_count": len(policy_rows),
        "excluded_incomplete_records": len(conformal_exclusions),
        "excluded_incomplete_records_by_source_and_reason": [{"source_file": source, "reason": reason, "record_count": count} for (source, reason), count in sorted(exclusion_counts.items())],
        "explicitly_mapped_primary_figure_rows": len(conformal_primary_mapped),
        "figure_s3a_total_rows": len(s3a_source_rows),
        "figure_s3a_exact_source_literal_match_rows": s3a_exact_source_n,
        "figure_s3a_rows_excluded_unmapped_source_literal": len(s3a_source_rows) - s3a_exact_source_n,
        "figure_s3a_primary_split_rows_under_exact_source_literals": s3a_primary_split_n,
        "figure_s3a_rows_excluded_nonprimary_split_after_source_map": s3a_exact_source_n - s3a_primary_split_n,
        "figure_s3a_primary_full_arm_rows_written": s3a_primary_full_n,
        "figure_s3a_primary_rows_excluded_nonfull_arm": s3a_primary_split_n - s3a_primary_full_n,
        "figure_s3a_primary_unique_keys": len(primary_key_counts),
        "figure_s3a_primary_duplicate_key_count": sum(1 for n in primary_key_counts.values() if n > 1),
        "source_generation_policy": "Supplementary_Data Additional sources and Figure_Source_Data S3a remain separate groups and are never deduplicated/pooled across file generations",
        "source_types": dict(Counter(r.get("source_role", "") for r in conformal_rows)),
        "fold_level_preserved": True, "coverage_averaged_across_folds": False,
        "finite_infinite_fractions_denominator": "number of source fold records within exact country/task/policy/profile/split/arm/source-file group",
    }

    # Summary 4: annual CAMS/SatPM outputs.  Keep all official supplementary rows
    # and the formal S4 figure source as separate generations; never merge them.
    annual_rows: list[dict[str, Any]] = []
    annual_claims: list[dict[str, Any]] = []
    annual_sources = [("annual_deltas.csv", "supplementary_summary")]
    for fn in ["Supplementary_Figure_S8_Panel_a_Source_Data.csv", "Supplementary_Figure_S8_Panel_b_Source_Data.csv"]:
        if fn in input_paths:
            annual_sources.append((fn, "figure_source"))
    for name, generation in annual_sources:
        path = input_paths[name]
        fields, rows = load_csv(path)
        alias = source_aliases.get(name, "")
        for idx, r in enumerate(rows, 1):
            year, year_rule = literal_year(r.get("task", ""))
            delta = num(r.get("delta_rmse"))
            ci_low = num(r.get("delta_rmse_ci_low") or r.get("delta_rmse_lo"))
            ci_high = num(r.get("delta_rmse_ci_high") or r.get("delta_rmse_hi"))
            if delta is None:
                sign = "not_applicable_or_missing"
            elif delta < 0:
                sign = "negative"
            elif delta > 0:
                sign = "positive"
            else:
                sign = "exact_zero"
            contrast = r.get("contrast", "")
            direction_note = ""
            if contrast.startswith("full minus "):
                direction_note = "negative favors full; positive favors named comparator"
            elif contrast.startswith("B+CAMS minus full"):
                direction_note = "positive favors full; negative favors B+CAMS"
            out = copy_with_binding(r, {
                "source_file": name, "source_alias": alias, "source_generation": generation,
                "source_role": source_roles.get(name, "annual_comparator_deltas"), "source_row_number": idx,
                "record_id": row_id(name, idx), "year": year, "year_binding": year_rule,
                "delta_rmse_sign": sign, "delta_rmse_ci_contains_zero": (ci_low <= 0 <= ci_high) if ci_low is not None and ci_high is not None else "",
                "contrast_direction_note": direction_note,
                "primary_filter_match": (r.get("qa_variant") == "strict" and r.get("estimator_kind") == "hgb"
                                         and r.get("arm") == "full" and r.get("target_strategy") == "log_smear"
                                         and r.get("backtransform") == "smearing" and r.get("profile") in {"no_met", "with_met"}),
                "support_group_key": {k: r.get(k, "") for k in ["country", "year", "task", "profile", "support_scope", "benchmark_version", "contrast"]},
            })
            annual_rows.append(out)
            annual_claims.append({
                "id": f"annual_{len(annual_claims)+1}",
                "value": {k: r.get(k, "") for k in ["country", "task", "profile", "arm", "target_strategy", "backtransform", "support_scope", "benchmark_version", "status", "contrast", "delta_rmse", "delta_rmse_ci_low", "delta_rmse_ci_high", "rows", "sites"]},
                "row_binding": {"file": name, "source_alias": alias, "row_number": idx, "record_id": row_id(name, idx), "source_generation": generation,
                                "filters": {"country": r.get("country"), "year": year, "task": r.get("task"), "profile": r.get("profile"), "support_scope": r.get("support_scope"), "benchmark_version": r.get("benchmark_version"), "contrast": contrast}},
            })
    annual_primary = [r for r in annual_rows if r.get("primary_filter_match") is True]
    write_csv(OUT / "annual_comparator_rows.csv", annual_rows)
    write_csv(OUT / "annual_comparator_primary_subtable.csv", annual_primary)

    # Summary 5: the fixed 78-contrast family.  These are existing intervals;
    # no bootstrap, interval recomputation, or refitting is performed.
    m_name = "multiplicity.csv"
    m_path = input_paths.get(m_name)
    m_fields, m_rows = load_csv(m_path) if m_path else ([], [])
    multiplicity_rows: list[dict[str, Any]] = []
    group_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
    multiplicity_claims: list[dict[str, Any]] = []
    for idx, r in enumerate(m_rows, 1):
        lo, hi, delta = num(r.get("simultaneous_95_low")), num(r.get("simultaneous_95_high")), num(r.get("delta_rmse"))
        if lo is None or hi is None:
            ci_class = "interval_missing"
        elif hi < 0:
            ci_class = "all_below_zero"
        elif lo > 0:
            ci_class = "all_above_zero"
        elif lo <= 0 <= hi:
            ci_class = "contains_zero"
        else:
            ci_class = "unclassified_interval"
        out = copy_with_binding(r, {
            "source_file": m_name, "source_alias": source_aliases.get(m_name, ""), "source_role": source_roles.get(m_name, ""),
            "source_row_number": idx, "record_id": row_id(m_name, idx), "simultaneous_interval_class": ci_class,
        })
        multiplicity_rows.append(out)
        group_m[r.get("contrast", "")].append(out)
        multiplicity_claims.append({
            "id": f"multiplicity_{idx}",
            "value": {k: r.get(k, "") for k in ["country", "task", "profile", "contrast", "delta_rmse", "simultaneous_95_low", "simultaneous_95_high", "family_size_contrasts", "replicates"]},
            "row_binding": {"file": m_name, "source_alias": source_aliases.get(m_name, ""), "row_number": idx, "record_id": row_id(m_name, idx),
                            "filters": {"contrast": r.get("contrast"), "qa_variant": r.get("qa_variant"), "estimator_kind": r.get("estimator_kind"), "target_strategy": r.get("target_strategy")}},
        })
    write_csv(OUT / "multiplicity_contrasts.csv", multiplicity_rows)
    multiplicity_summary = []
    expected_high_below = {"B+AOD - B": 24, "full - B+AOD": 12, "full - B": 26}
    for contrast, rows in sorted(group_m.items()):
        vals = [num(r.get("delta_rmse")) for r in rows]
        vals = [x for x in vals if x is not None]
        count = Counter(r.get("simultaneous_interval_class", "") for r in rows)
        multiplicity_summary.append({
            "contrast": contrast, "record_n": len(rows), "delta_rmse_nonmissing_n": len(vals),
            "all_below_zero_n": count["all_below_zero"], "all_above_zero_n": count["all_above_zero"],
            "contains_zero_n": count["contains_zero"], "interval_missing_n": count["interval_missing"],
            "all_below_zero_fraction": count["all_below_zero"] / len(rows) if rows else None,
            "delta_rmse_median_signed": quartile(vals, 0.5), "delta_rmse_q25": quartile(vals, 0.25),
            "delta_rmse_q75": quartile(vals, 0.75), "delta_rmse_median_absolute": quartile([abs(x) for x in vals], 0.5),
            "delta_rmse_min": min(vals) if vals else None, "delta_rmse_max": max(vals) if vals else None,
            "family_size_contrasts_values": sorted({r.get("family_size_contrasts", "") for r in rows}),
            "stored_replicate_values": sorted({r.get("replicates", "") for r in rows}),
        })
    write_csv(OUT / "multiplicity_contrast_summary.csv", multiplicity_summary)
    total_high_below = sum(r["all_below_zero_n"] for r in multiplicity_summary)
    qa["multiplicity"] = {
        "input_row_count": len(m_rows), "per_contrast": multiplicity_summary,
        "total_all_below_zero_n": total_high_below,
        "expected_counts_match_review": (len(m_rows) == 78 and all(len(group_m.get(k, [])) == 26 and sum(1 for r in group_m.get(k, []) if r.get("simultaneous_interval_class") == "all_below_zero") == v for k, v in expected_high_below.items()) and total_high_below == 62),
        "resampling_repeated": False,
    }

    # M04 saved-diagnostic search.  It is deliberately limited to identified
    # source/model artifact roots and text metadata; no raw arrays are loaded.
    search_roots = [
        V / "final_model_runtime_v4_20261114",
        V / "global_saved_diagnostics_v2",
        V / "global_saved_diagnostics_v3",
        V / "submission_publication_root_sources_v1_20261115",
        V / "submission_publication_root_sources_v2_20261115",
        V / "uscan_historical_execution",
    ]
    artifact_hits: list[dict[str, Any]] = []
    search_limit_files = 10000
    scanned = 0
    filename_tokens = ("prt", "portugal", "smear", "resid", "oof", "fold", "source_root", "manifest", "diagnostic")
    content_tokens = ("portugal", "prt", "source_root", "smear_factor", "smear", "oof_resid", "residual")
    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in {"__pycache__", ".git", "node_modules", "cache"}]
            for fn in filenames:
                scanned += 1
                if scanned > search_limit_files:
                    break
                p = Path(dirpath) / fn
                size = p.stat().st_size
                name_hit = any(tok in fn.lower() for tok in filename_tokens)
                content_hit = []
                if p.suffix.lower() in {".json", ".csv", ".txt", ".md", ".log"} and size <= 5 * 1024 * 1024:
                    try:
                        with p.open("r", encoding="utf-8", errors="replace") as f:
                            for line_no, line in enumerate(f, 1):
                                low = line.lower()
                                hits = [t for t in content_tokens if t in low]
                                if hits:
                                    content_hit.append({"line": line_no, "tokens": hits, "snippet": line.strip()[:500]})
                                    if len(content_hit) >= 3:
                                        break
                    except (OSError, UnicodeDecodeError):
                        pass
                if name_hit or content_hit:
                    artifact_hits.append({
                        "search_root": str(root), "artifact_path": str(p), "bytes": size,
                        "filename_match": name_hit, "content_hits": content_hit,
                        "looks_like_fold_smear_or_oof_diagnostic": any(t in (fn + " " + json.dumps(content_hit)).lower() for t in ("smear", "resid", "oof")),
                    })
            if scanned > search_limit_files:
                break
        if scanned > search_limit_files:
            break
    write_csv(OUT / "pt_m04_artifact_search.csv", artifact_hits)
    lookalikes = [r for r in artifact_hits if r["looks_like_fold_smear_or_oof_diagnostic"]]
    # The plan is carried forward with observed search evidence; no refit starts
    # automatically from this summary program.
    diag_plan_path = HERE / "diagnostic_plan.json"
    diag_plan: dict[str, Any] = {}
    if diag_plan_path.exists():
        try:
            diag_plan = json.loads(diag_plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            diag_plan = {}
    diag_plan["existing_artifact_search"] = {
        "roots_searched": [str(p) for p in search_roots if p.exists()],
        "files_scanned": scanned, "hit_records": len(artifact_hits),
        "smear_residual_oof_filename_or_content_hits": len(lookalikes),
        "candidate_artifacts": lookalikes[:100],
        "interpretation": "A filename/content match is only a candidate; exact source/split/model binding must be checked before reuse or any diagnostic refit.",
    }
    diag_plan["status"] = "existing-artifact search completed; equivalence gate not automatically asserted"
    diag_plan_path.write_text(json.dumps(diag_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Review example checks use exact file/key mappings and tolerant numeric equality.
    def find_one(rows: list[dict[str, str]], filters: dict[str, str]) -> dict[str, str] | None:
        matches = [r for r in rows if all(r.get(k, "") == v for k, v in filters.items())]
        return matches[0] if len(matches) == 1 else None
    def close(got: Any, expected_value: float, tol: float = 1e-9) -> bool:
        x = num(got)
        return x is not None and abs(x - expected_value) <= tol * max(1.0, abs(expected_value))

    pt_labels = {"pt", "prt", "portugal"}
    pt_h = [r for r in transform_rows if str(r.get("country", "")).strip().lower() in pt_labels]
    qa["pt_country_labels_observed"] = sorted({str(r.get("country", "")) for r in h_rows if str(r.get("country", "")).strip().lower() in pt_labels})
    m04_expected = {
        ("T1_station", "smearing"): (-13.40328274274915, 19.471243325149374, 12.655218131964762, 434),
        ("T1_station", "original"): (-0.2738920379897807, 5.79067582130596, -2.454352532811875, 434),
        ("T3_year_2022", "smearing"): (-26.19154945333162, 21.703864124787746, None, 95),
        ("T3_year_2022", "original"): (-0.4956068875246004, 5.090129274739622, None, 95),
    }
    m04_checks = []
    for (task, backtransform), exp in m04_expected.items():
        matches = [r for r in pt_h if r.get("profile") == "with_met" and r.get("task") == task and r.get("backtransform") == backtransform and r.get("target_strategy") == "log_smear"]
        exact = len(matches) == 1
        if exact:
            r = matches[0]
            exact = close(r.get("r2"), exp[0]) and close(r.get("rmse"), exp[1]) and int_num(r.get("n")) == exp[3]
            if exp[2] is not None:
                exact = exact and close(r.get("bias"), exp[2])
        m04_checks.append({"task": task, "backtransform": backtransform, "matches_n": len(matches), "matches_review_values": exact,
                           "expected_values": {"r2": exp[0], "rmse": exp[1], "bias": exp[2], "n": exp[3]},
                           "record_ids": [r.get("record_id") for r in matches]})
    qa["checks"]["m04_portugal_review_examples_match"] = all(x["matches_review_values"] for x in m04_checks)
    qa["m04_portugal_review_examples"] = m04_checks

    # M05 checks the exact source figure rows already accepted in the review.
    s3b_name = "Supplementary_Figure_S2_Panel_b_Source_Data.csv"
    s3b_rows = load_csv(input_paths[s3b_name])[1] if s3b_name in input_paths else []
    ca = [r for r in s3b_rows if r.get("country") == "CAN" and r.get("qa_variant") == "strict" and r.get("profile") == "with_met" and r.get("task") == "T3_fixed_2023" and r.get("target_strategy") == "log_smear" and r.get("backtransform") == "smearing" and boolean_field(r.get("retained")) is True]
    get_future = lambda scope, rule="all": next((r for r in ca if r.get("fire_month_scope") == scope and r.get("rule") == rule), None)
    m05all, m05p90, m05other = get_future("all_months"), get_future("P90_fire_proxy"), get_future("other_months")
    m05ok = bool(m05all and m05p90 and m05other and int_num(m05all.get("n")) == 2537 and close(m05all.get("full_rmse"), 6.457148052091692)
                 and int_num(m05p90.get("n")) == 725 and close(m05p90.get("full_rmse"), 11.735792221670255)
                 and int_num(m05other.get("n")) == 1812 and close(m05other.get("full_rmse"), 1.8084628665070936))
    qa["checks"]["m05_can_2023_figure_source_examples_match_review"] = m05ok

    # M06 checks exact primary figure-source folds.  Counts stay within a fold.
    s3a_name = "Supplementary_Figure_S2_Panel_a_Source_Data.csv"
    s3a_rows = s3a_source_rows
    m06_expected = [
        {"country": "CAN", "task": "T2_spatial_buffer50km", "fold_id": "2", "pointwise_coverage": 0.9681784930504757, "all_month_per_station_coverage": 0.0943396226415094, "mean_interval_width": 18.984710400231528, "all_month_per_station_covered_n": "5", "all_month_per_station_group_n": "53", "test_station_n": "53", "calibration_station_n": "38"},
        {"country": "USA", "task": "T2_spatial_buffer50km", "fold_id": "1", "pointwise_coverage": 0.885227010831989, "all_month_per_station_coverage": 0.2731707317073171, "mean_interval_width": 12.496801648674664, "all_month_per_station_covered_n": "56", "all_month_per_station_group_n": "205", "test_station_n": "205", "calibration_station_n": "133"},
    ]
    m06_checks = []
    explicit_source_for_country = {"CAN": "can_hist_primary_strict_with_met_hgb", "USA": "usa_hist_strict_with_met_hgb"}
    for e in m06_expected:
        matches = [r for r in s3a_rows if r.get("country") == e["country"] and r.get("task") == e["task"] and r.get("fold_id") == e["fold_id"] and r.get("split_label") == "primary_80_20" and r.get("analysis_source") == explicit_source_for_country[e["country"]]]
        ok = len(matches) == 1
        if ok:
            r = matches[0]
            ok = close(r.get("pointwise_coverage"), e["pointwise_coverage"]) and close(r.get("all_month_per_station_coverage"), e["all_month_per_station_coverage"]) and close(r.get("mean_interval_width"), e["mean_interval_width"]) and all(r.get(k) == e[k] for k in ["all_month_per_station_covered_n", "all_month_per_station_group_n", "test_station_n", "calibration_station_n"])
        m06_checks.append({"filters": {"country": e["country"], "task": e["task"], "fold_id": e["fold_id"], "split_label": "primary_80_20"}, "matches_n": len(matches), "matches_review_values": ok})
    qa["checks"]["m06_conformal_review_examples_match"] = all(x["matches_review_values"] for x in m06_checks)
    qa["m06_conformal_review_examples"] = m06_checks
    qa["explicit_analysis_source_mapping"] = {
        "known_literals": KNOWN_ANALYSIS_SOURCE_MAP,
        "unknown_or_blank_values": "preserved raw and left unmapped; no profile/generation guessed",
    }

    qa["checks"]["multiplicity_78_62_review_counts_match"] = qa["multiplicity"]["expected_counts_match_review"]
    qa["checks"]["no_coverage_averaging_across_folds"] = qa["conformal"]["coverage_averaged_across_folds"] is False
    qa["output_files"] = {}
    for p in sorted(OUT.glob("*.csv")):
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            qa["output_files"][p.name] = {"bytes": p.stat().st_size, "rows": max(0, sum(1 for _ in f) - 1), "sha256": sha256(p)}

    claims = {
        "title": "Structured re-summary of formal results for M04-M07 and m01",
        "source_mapping_rule": "Map files by exact path/file name and the formal data_catalog.json source_alias. Preserve any analysis_source column raw. Never infer a generation or country/profile from blank/unmapped analysis_source values.",
        "claims": transform_claims + future_claims + conformal_claims + annual_claims + multiplicity_claims,
        "summary_claims": [
            {"id": "multiplicity_overall", "values": {"contrast_rows": len(m_rows), "all_below_zero_count": total_high_below}, "row_bindings": [row_id(m_name, i) for i in range(1, len(m_rows)+1)],
             "filters": {"family_eligible": "true rows used in official fixed max-|t| family", "source_file": m_name, "classification": "simultaneous_95_high < 0"}},
            {"id": "conformal_policy_grouping", "values": {"source_row_count": len(conformal_rows), "policy_group_count": len(policy_rows), "coverage_pooled_across_folds": False}, "row_bindings": [r.get("record_id") for r in conformal_rows],
             "filters": {"source files": "all catalogued Additional conformal_summary/feasibility_summary plus exact S3 panel-a figure source", "grouping": "country/task/policy/profile/split/arm/source file; fold rows retained"}},
        ],
        "source_hashes_file": "csv/source_hashes.csv",
        "source_record_ids_are_1_based_data_row_numbers": True,
    }
    (HERE / "summary_claims.json").write_text(json.dumps(claims, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Dynamic English interpretation with explicit limits and exact table references.
    mult_lines = []
    for r in multiplicity_summary:
        mult_lines.append(f"- **{r['contrast']}**: {r['all_below_zero_n']}/{r['record_n']} intervals have upper limit below zero; median signed ΔRMSE {serialize(r['delta_rmse_median_signed'])}, IQR {serialize(r['delta_rmse_q25'])} to {serialize(r['delta_rmse_q75'])} µg m⁻³ (range {serialize(r['delta_rmse_min'])} to {serialize(r['delta_rmse_max'])}).")
    pt_examples = []
    for r in pt_h:
        if r.get("task") in {"T1_station", "T3_year_2022"} and r.get("profile") == "with_met" and r.get("target_strategy") == "log_smear":
            pt_examples.append(f"- PT {r.get('task')} with_met, {r.get('backtransform')}: R² {r.get('r2')}, RMSE {r.get('rmse')}, bias {r.get('bias')}, n={r.get('n')} sites={r.get('n_stations')} (historical_metrics.csv row {r.get('source_row_number')}).")
    canada_rows = [r for r in future_risk_rows if r.get("country") == "CAN" and r.get("task") == "T3_fixed_2023" and r.get("profile") == "with_met" and r.get("summary_class") == "primary_strict_hgb_log_smear_smearing" and r.get("fire_month_scope") in {"all_months", "P90_fire_proxy", "other_months"} and r.get("rule") == "all"]
    canada_lines = [f"- Canada 2023 (all-support rule), {r.get('fire_month_scope')}: n={r.get('n')}/{r.get('eligible_rows')} station-months, RMSE={r.get('full_rmse')}, bias={r.get('full_bias')}; row coverage={r.get('row_coverage_fraction')} (joint_future.csv row {r.get('source_row_number')})." for r in canada_rows]
    annual_lines = []
    for r in annual_primary:
        if r.get("country") == "USA" and r.get("profile") == "with_met" and r.get("contrast") == "full minus SatPM" and r.get("support_scope") == "primary_exact_12_months":
            annual_lines.append(f"- US {r.get('year') or 'historical task year not explicit'} {r.get('task')} {r.get('contrast')}: ΔRMSE={r.get('delta_rmse')} [{r.get('delta_rmse_ci_low') or r.get('delta_rmse_lo')}, {r.get('delta_rmse_ci_high') or r.get('delta_rmse_hi')}], n={r.get('rows') or r.get('rows')}, sites={r.get('sites')}; support={r.get('support_scope')}, source={r.get('source_file')} row {r.get('source_row_number')}.")
    s3_examples = []
    for r in conformal_rows:
        if r.get("source_file") == figure_s3a and r.get("split_label") == "primary_80_20" and r.get("country") in {"CAN", "USA"} and r.get("task") == "T2_spatial_buffer50km" and r.get("fold_id") in {"2", "1"} and r.get("arm") == "full":
            s3_examples.append(f"- {r.get('country')} T2 fold {r.get('fold_id')}: finite q90={r.get('q90_status')}; point coverage {r.get('pointwise_covered_row_n_derived')}/{r.get('test_row_n')} (source {r.get('pointwise_coverage')}); whole-station all-month {r.get('all_month_per_station_covered_n_source')}/{r.get('all_month_per_station_group_n_source')}; width={r.get('mean_interval_width')} (S3a row {r.get('source_row_number')}).")
    interpretation = [
        "# Structured result summary for manuscript revision",
        "",
        "Generated from the locked formal Supplementary_Data package and specified Figure_Source_Data tables. The script only re-summarizes saved values; it does not refit models, tune transforms, or rerun bootstrap intervals.",
        "",
        "## M04 — inverse-transform sensitivity",
        "",
        "`csv/transform_sensitivity.csv` retains strict HGB full-arm results for both MET profiles, all countries/tasks, and every recorded target/back-transform branch. Deltas are relative to a unique same-country/task/profile/cohort/dataset/smearing-strategy `log_smear` + `smearing` row. Blank or ambiguous references remain blank and are labeled.",
        "",
        *(pt_examples or ["No PT-labelled source rows matched the explicit country-label map {PT, PRT, Portugal}; no country label was guessed."]),
        "",
        "## M05 — future fire-proxy risk and support coverage",
        "",
        "`csv/future_risk_coverage_rows.csv` preserves every source row and classifies primary rows only from strict/HGB/log_smear/smearing/no_met-or-with_met and an explicit dataset_policy containing `primary`. `csv/future_risk_summary.csv` keeps only `retained=True` records for risk metrics; sentinels with `retained=False` are listed with reasons in `csv/future_risk_exclusions.csv`.",
        "",
        *(canada_lines or ["No exact Canada 2023 `rule=all` rows matched the required primary fields; inspect the row-level CSV and QA classifications."]),
        "",
        "P90_fire_proxy is the source-defined fire-feature stratum. It is not interpreted as measured smoke mass. Row/station coverage remains tied to the source denominators; no coverage percentage is averaged across folds.",
        "",
        "## M06 — conformal finite status and empirical coverage",
        "",
        "`csv/conformal_fold_rows.csv` is fold/source-row level. `csv/conformal_policy_fold_counts.csv` reports finite/infinite fractions with fold/source records as the denominator; it does not pool test stations across folds or source generations. Pointwise numerator is derived only when source pointwise fraction × test row count is an integer within 1e-6; station-wide coverage uses the source numerator and denominator. Incomplete denominators and keys are listed in `csv/conformal_exclusions.csv`.",
        "",
        *(s3_examples or ["The exact S3a primary fold examples were not found under the expected key; consult the row-level source table and QA checks."]),
        "",
        "`csv/conformal_primary_fold_summary.csv` is the main conformal table: it filters the 440-row S3a source to four exact, pre-mapped analysis_source literals, primary_80_20, and full arm; its unique key and input-to-output counts are in qa.json. Supplementary_Data Additional conformal rows remain separately keyed by source file/alias in the audit tables and are never merged with S3a. Blank or unknown source fields remain unmapped.",
        "These are empirical results for observed spatial/temporal splits. Finite intervals and their pointwise coverage do not establish transportable coverage under spatial or temporal shift.",
        "",
        "## M07 — annual CAMS/SatPM comparators",
        "",
        "`csv/annual_comparator_rows.csv` keeps Supplementary_Data and Figure_Source_Data rows separate, adds a year only when the task explicitly contains `external_YYYY`, and preserves blank/non-applicable metrics. The primary subtable is `csv/annual_comparator_primary_subtable.csv`; all original support, benchmark, contrast, and status fields remain available.",
        "",
        *(annual_lines or ["No US with-MET full-minus-SatPM rows matched the exact primary support key in the collected annual table."]),
        "",
        "For `full minus product`, negative ΔRMSE favors the full model and positive ΔRMSE favors the comparator; the contrast orientation is retained in each row. SatPM is not treated as an independent truth source where its calibration overlaps monitoring stations.",
        "",
        "## m01 — multiplicity decomposition",
        "",
        *mult_lines,
        "",
        f"The stored family has {len(m_rows)} contrast rows and {total_high_below} intervals with simultaneous upper bounds below zero. Signed median and quartiles are reported separately from the row-level deltas; no new bootstrap was run.",
        "",
        "## Provenance and limits",
        "",
        "`csv/source_hashes.csv` records server-side file hashes against either the local formal-package hashes or the package catalog. Source rows are bound by exact file name and catalog alias; any `analysis_source` column is kept raw and is not used to infer missing generation/profile values. Inspect `qa.json` for failed hashes, row-count mismatches, unfulfilled denominators, and review-example checks.",
        "",
        "M04 fold smearing factors and training OOF residual tails are not inferred from held-out metrics. Candidate saved artifacts, if any, are listed in `csv/pt_m04_artifact_search.csv`; a diagnostic refit remains gated on proof of identical source, outer folds, and locked parameters.",
        "",
    ]
    (HERE / "summary_interpretation_en.md").write_text("\n".join(interpretation), encoding="utf-8")

    qa["status"] = "complete_with_warnings" if qa["exclusions"] or conformal_exclusions or not all(qa["checks"].values()) else "complete"
    qa["finished_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    (HERE / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qa["status"], "outputs": len(qa["output_files"]), "transform_rows": len(transform_rows),
                      "future_rows": len(future_rows), "conformal_rows": len(conformal_rows), "annual_rows": len(annual_rows),
                      "multiplicity_rows": len(multiplicity_rows), "checks": qa["checks"], "exclusions": len(qa["exclusions"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
