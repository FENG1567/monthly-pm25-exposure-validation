#!/usr/bin/env python3
"""Build key-only Future T4 target subsets from independently source-QC'd masters.

The manifest selects the cohort keys and one of the frozen source policies. This
program never reads predictor columns from features, never filters on target
values, and does not create candidate acceptance or scoring artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROVENANCE_FORMAT = "repair-v2-future-target-provenance-v1"
MANIFEST_FORMAT = "repair-v2-target-cohort-materializer-manifest-v1"
PENDING_STATUS = "sourceQCverified_cohort_candidate_pending_independent_candidate_acceptance"
KEY_COLUMNS = ("country", "site_id", "year", "month")
TARGET_COLUMNS = (*KEY_COLUMNS, "pm25")
POLICIES = {
    "USA": {
        "EPA_QC_exclude_advised_methods": "epa.master_monthly",
    },
    "CAN": {
        "NAPS_all_published_primary": "naps.master_all_published",
        "NAPS_all_published_on_predefined_common_keys": "naps.master_all_published",
        "NAPS_exclude_236_238_on_predefined_common_keys": "naps.master_exclude_236_238",
    },
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class MaterializerError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _pinned_ref(ref: Any, label: str, checks: dict[Path, str]) -> dict[str, Any]:
    if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
        raise MaterializerError(f"{label} must pin path and SHA-256")
    path = Path(str(ref["path"])).expanduser().resolve()
    expected = str(ref["sha256"]).lower()
    if not SHA_RE.fullmatch(expected) or not path.is_file():
        raise MaterializerError(f"{label} has invalid hash or missing file: {path}")
    actual = _sha(path)
    if actual != expected:
        raise MaterializerError(f"{label} hash mismatch: {path}")
    previous = checks.setdefault(path, expected)
    if previous != expected:
        raise MaterializerError(f"conflicting pinned hashes for {path}")
    return {"path": str(path), "sha256": actual}


def _read_json_ref(ref: Any, label: str, checks: dict[Path, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    verified = _pinned_ref(ref, label, checks)
    try:
        raw = Path(verified["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != verified["sha256"]:
            raise MaterializerError(f"{label} changed between pinning and parsing")
        record = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise MaterializerError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(record, dict):
        raise MaterializerError(f"{label} root must be a JSON object")
    return record, verified


def _source_input_refs(qc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every source file pinned by the reviewed source-QC manifest."""
    refs: dict[str, dict[str, Any]] = {}
    epa = qc.get("epa")
    naps = qc.get("naps")
    if not isinstance(epa, dict) or not isinstance(naps, dict):
        raise MaterializerError("source-QC manifest lacks EPA/NAPS sections")
    for name in ("daily", "old_daily", "master_monthly", "old_monthly"):
        if name in epa:
            refs[f"epa.{name}"] = epa[name]
    years = naps.get("years")
    if not isinstance(years, list):
        raise MaterializerError("source-QC manifest lacks NAPS annual input pins")
    for item in years:
        if not isinstance(item, dict) or not isinstance(item.get("year"), int):
            raise MaterializerError("invalid NAPS annual source reference")
        refs[f"naps.{item['year']}"] = item
    for name in ("master_all_published", "master_exclude_236_238", "provenance_all_published", "provenance_exclude_236_238"):
        if name in naps:
            refs[f"naps.{name}"] = naps[name]
    if len(refs) != 15:
        raise MaterializerError(f"expected 15 source-QC input pins, found {len(refs)}")
    return refs


def _verify_source_qc(source_qc: Any, checks: dict[Path, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source_qc, dict):
        raise MaterializerError("manifest source_qc must be an object")
    qc, qc_ref = _read_json_ref(source_qc.get("manifest"), "source-QC manifest", checks)
    report, report_ref = _read_json_ref(source_qc.get("audit_report"), "source-QC replay report", checks)
    review, review_ref = _read_json_ref(source_qc.get("qa_record"), "source-QC QA record", checks)
    if review.get("decision") != "approved_source_QC_only":
        raise MaterializerError("QA record did not approve source QC")
    if review.get("manifest_sha256") != qc_ref["sha256"] or review.get("audit_report_sha256") != report_ref["sha256"]:
        raise MaterializerError("QA record does not bind the pinned manifest/report bytes")
    if report.get("audit_type") != "source_qc_replay_only" or report.get("source_QC_reads_target_values") is not True:
        raise MaterializerError("pinned report is not the source-QC replay report")
    if report.get("all_daily_monthly_and_naps_replays_match") is not True or report.get("input_hashes_stable") is not True:
        raise MaterializerError("source-QC replay checks are not all passing")
    if report.get("predictive_scores_computed") is not False or report.get("candidate_pinned_target_acceptance_yet") is not False:
        raise MaterializerError("source-QC report contains scoring or candidate acceptance")
    if report.get("manifest_sha256") != qc_ref["sha256"] or report.get("manifest_sha256_post") != qc_ref["sha256"]:
        raise MaterializerError("source-QC report pre/post manifest pins do not match")
    if report.get("input_hashes_pre") != report.get("input_hashes_post"):
        raise MaterializerError("source-QC report input pre/post hashes differ")
    if review.get("all_15_current_inputs_match_manifest_and_audit_pre_post") is not True:
        raise MaterializerError("QA record did not verify all current source-QC inputs")
    inputs = _source_input_refs(qc)
    expected_hashes: dict[str, str] = {}
    for name, ref in inputs.items():
        verified = _pinned_ref(ref, f"source-QC input {name}", checks)
        expected_hashes[name] = verified["sha256"]
    if report.get("input_hashes_pre") != expected_hashes or review.get("current_input_hashes") != expected_hashes:
        raise MaterializerError("source-QC report/review input hashes do not match manifest pins")
    # The source-QC QA record approved these exact two code files.
    code_hashes = review.get("reviewed_script_hashes")
    if not isinstance(code_hashes, dict):
        raise MaterializerError("QA record lacks recorded code hashes")
    code_refs: dict[str, dict[str, Any]] = {}
    root = Path(qc_ref["path"]).parent
    for name in ("audit_targets.py", "prepare_actual_manifest.py"):
        expected = str(code_hashes.get(name, "")).lower()
        if not SHA_RE.fullmatch(expected):
            raise MaterializerError(f"review lacks a valid hash for {name}")
        path = (root / name).resolve()
        actual = _sha(path) if path.is_file() else ""
        if actual != expected:
            raise MaterializerError(f"source-QC reviewed code bytes differ: {path}")
        checks[path] = expected
        code_refs[name] = {"path": str(path), "sha256": actual}
    if code_hashes.get("source_QC_manifest_v1.json") != qc_ref["sha256"]:
        raise MaterializerError("reviewed source-QC manifest hash differs from pinned manifest")
    evidence = {
        "manifest": qc_ref,
        "audit_report": report_ref,
        "qa_record": review_ref,
        "reviewed_code": code_refs,
    }
    masters: dict[str, dict[str, Any]] = {}
    for label, key in (("epa.master_monthly", "epa.master_monthly"), ("epa.old_monthly", "epa.old_monthly"),
                       ("naps.master_all_published", "naps.master_all_published"),
                       ("naps.master_exclude_236_238", "naps.master_exclude_236_238")):
        if key in inputs:
            masters[label] = _pinned_ref(inputs[key], f"master {label}", checks)
    return evidence, masters


def _check_stable(checks: dict[Path, str]) -> None:
    for path, expected in checks.items():
        if not path.is_file() or _sha(path) != expected:
            raise MaterializerError(f"pinned input changed during materialization: {path}")


def _path_overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _validate_output(out: Path, checks: dict[Path, str]) -> Path:
    resolved = out.expanduser().resolve()
    if resolved.exists():
        raise MaterializerError(f"output directory must be fresh: {resolved}")
    for source in checks:
        if _path_overlaps(resolved, source.parent.resolve()):
            raise MaterializerError(f"output overlaps protected input parent: {source.parent}")
    return resolved


def _parquet_bytes(path: Path, label: str, checks: dict[Path, str]) -> bytes:
    raw = path.read_bytes()
    expected = checks.get(path.resolve())
    if expected is None or hashlib.sha256(raw).hexdigest() != expected:
        raise MaterializerError(f"{label} changed between hash verification and Parquet read")
    return raw


def _keys(path: Path, label: str, *, feature: bool, checks: dict[Path, str]) -> list[tuple[str, str, int, int]]:
    raw = _parquet_bytes(path, label, checks)
    schema = pq.read_schema(pa.BufferReader(raw))
    required = set(KEY_COLUMNS if feature else TARGET_COLUMNS)
    if not required.issubset(schema.names):
        raise MaterializerError(f"{label} lacks columns: {sorted(required - set(schema.names))}")
    # Explicitly read only identity/calendar fields from features.
    columns = list(KEY_COLUMNS) if feature else list(TARGET_COLUMNS)
    table = pq.read_table(pa.BufferReader(raw), columns=columns)
    frame = table.to_pydict()
    rows: list[tuple[str, str, int, int]] = []
    for i in range(table.num_rows):
        raw_country, raw_site, raw_year, raw_month = (frame[name][i] for name in KEY_COLUMNS)
        if raw_country is None or raw_site is None or raw_site == "":
            raise MaterializerError(f"{label} has null/empty country or site identity")
        country, site = str(raw_country).upper(), str(raw_site)
        try:
            year_num, month_num = float(raw_year), float(raw_month)
        except (TypeError, ValueError):
            raise MaterializerError(f"{label} has a missing/non-numeric year or month")
        if not math.isfinite(year_num) or not math.isfinite(month_num) or not year_num.is_integer() or not month_num.is_integer():
            raise MaterializerError(f"{label} has missing or non-integer calendar values")
        year, month = int(year_num), int(month_num)
        if month < 1 or month > 12:
            raise MaterializerError(f"{label} has out-of-range month")
        rows.append((country, site, year, month))
    if len(set(rows)) != len(rows):
        raise MaterializerError(f"{label} has duplicate source-qualified station-month keys")
    return rows


def _master_rows(path: Path, label: str, checks: dict[Path, str]) -> dict[tuple[str, str, int, int], float]:
    raw = _parquet_bytes(path, f"master {label}", checks)
    schema = pq.read_schema(pa.BufferReader(raw))
    if not set(TARGET_COLUMNS).issubset(schema.names):
        raise MaterializerError(f"master {label} lacks key/pm25 columns")
    table = pq.read_table(pa.BufferReader(raw), columns=list(TARGET_COLUMNS))
    data = table.to_pydict()
    result: dict[tuple[str, str, int, int], float] = {}
    for i in range(table.num_rows):
        raw_country, raw_site, raw_year, raw_month = (data[name][i] for name in KEY_COLUMNS)
        if raw_country is None or raw_site is None or raw_site == "":
            raise MaterializerError(f"master {label} has null/empty identity")
        try:
            y, m = float(raw_year), float(raw_month)
        except (TypeError, ValueError):
            raise MaterializerError(f"master {label} has non-numeric calendar values")
        if not math.isfinite(y) or not math.isfinite(m) or not y.is_integer() or not m.is_integer() or not 1 <= int(m) <= 12:
            raise MaterializerError(f"master {label} has invalid calendar values")
        key = (str(raw_country).upper(), str(raw_site), int(y), int(m))
        if key in result:
            raise MaterializerError(f"master {label} has duplicate station-month keys")
        try:
            value = float(data["pm25"][i])
        except (TypeError, ValueError):
            raise MaterializerError(f"master {label} has missing/non-numeric pm25")
        if not math.isfinite(value):
            raise MaterializerError(f"master {label} has nonfinite pm25")
        result[key] = value
    return result


def _write_parquet(path: Path, keys: list[tuple[str, str, int, int]], values: list[float]) -> None:
    table = pa.table({
        "country": pa.array([r[0] for r in keys], type=pa.string()),
        "site_id": pa.array([r[1] for r in keys], type=pa.string()),
        "year": pa.array([r[2] for r in keys], type=pa.int64()),
        "month": pa.array([r[3] for r in keys], type=pa.int64()),
        "pm25": pa.array(values, type=pa.float64()),
    })
    pq.write_table(table, path)


def prepare(manifest_path: Path, manifest_sha256: str, out_dir: Path) -> dict[str, Any]:
    checks: dict[Path, str] = {}
    manifest, manifest_ref = _read_json_ref({"path": str(manifest_path), "sha256": manifest_sha256}, "cohort manifest", checks)
    _check_stable(checks)
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise MaterializerError(f"manifest format must be {MANIFEST_FORMAT}")
    evidence, masters = _verify_source_qc(manifest.get("source_qc"), checks)
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise MaterializerError("manifest cohorts must be a non-empty list")
    ids = [c.get("id") if isinstance(c, dict) else None for c in cohorts]
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", x) for x in ids) or len(set(ids)) != len(ids):
        raise MaterializerError("cohort IDs must be unique safe path components")
    resolved_cohorts: list[dict[str, Any]] = []
    for raw in cohorts:
        country = str(raw.get("country", "")).upper()
        policy = raw.get("target_policy")
        if country not in POLICIES or policy not in POLICIES[country]:
            raise MaterializerError(f"unsupported frozen country/target policy: {country}/{policy}")
        label = raw.get("master_label")
        allowed_label = POLICIES[country][policy]
        if label == "epa.old_monthly":
            if country != "USA" or policy != "EPA_QC_exclude_advised_methods":
                raise MaterializerError("old EPA master is allowed only for USA EPA policy")
        elif label != allowed_label:
            raise MaterializerError(f"master label {label!r} does not match frozen policy {policy!r}")
        years = raw.get("period_years")
        if (not isinstance(years, list) or not years or any(type(y) is not int for y in years)
                or years != sorted(set(years))):
            raise MaterializerError(f"{raw['id']} period_years must be sorted unique integers")
        if label == "epa.old_monthly" and any(y < 2018 or y > 2023 for y in years):
            raise MaterializerError("old EPA master may be used only for 2018-2023")
        if label not in masters:
            raise MaterializerError(f"source-QC manifest does not bind requested master: {label}")
        feature_ref = _pinned_ref(raw.get("feature_table"), f"{raw['id']} feature table", checks)
        resolved_cohorts.append({**raw, "country": country, "feature_ref": feature_ref, "master_ref": masters[label], "master_label": label})

    out = _validate_output(out_dir, checks)
    # All feature keys and masters are fully pinned before any read or output write.
    for cohort in resolved_cohorts:
        feature = Path(cohort["feature_ref"]["path"])
        keys = _keys(feature, f"feature {cohort['id']}", feature=True, checks=checks)
        years = cohort["period_years"]
        if any(key[0] != cohort["country"] for key in keys):
            raise MaterializerError(f"feature {cohort['id']} has mixed/wrong country")
        if any(key[2] not in years for key in keys):
            raise MaterializerError(f"feature {cohort['id']} has rows outside declared period")
        if sorted({key[2] for key in keys}) != years:
            raise MaterializerError(f"feature {cohort['id']} observed years differ from declared period")
        master = _master_rows(Path(cohort["master_ref"]["path"]), cohort["master_label"], checks)
        missing = [key for key in keys if key not in master]
        if missing:
            raise MaterializerError(f"master {cohort['master_label']} misses {len(missing)} selected feature keys")
        cohort["keys"] = keys
        cohort["values"] = [master[key] for key in keys]

    # Create fresh output roots only after all input validation, then stage targets.
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for cohort in resolved_cohorts:
        subdir = out / cohort["id"]
        subdir.mkdir(exist_ok=False)
        target_path = subdir / "key-matchedtarget.parquet"
        _write_parquet(target_path, cohort["keys"], cohort["values"])
        target_ref = {"path": str(target_path.resolve()), "sha256": _sha(target_path)}
        master_ref = cohort["master_ref"]
        ordered = [list(row) for row in cohort["keys"]]
        provenance = {
            "format": PROVENANCE_FORMAT,
            "country": cohort["country"],
            "target_policy": cohort["target_policy"],
            "target_column": "pm25",
            "period_years": cohort["period_years"],
            "target_verified": True,
            "target_status": PENDING_STATUS,
            "target_table": target_ref,
            "key_commitment": {"algorithm": "sha256-canonical-json-array-v1", "row_count": len(ordered), "sha256": _digest(ordered)},
            "key_sha256": _digest([list(row) for row in sorted(cohort["keys"])]),
            "cleaning_chain": [{
                "step": "select_source_qc_verified_master_values_for_feature_keys",
                "operation": "exact station-month key subset in feature key order; no imputation or target-value filtering",
                "input": master_ref,
                "output": target_ref,
            }],
            "source_QC_evidence": evidence,
            "master_label": cohort["master_label"],
            "feature_table": cohort["feature_ref"],
            "candidate_acceptance_generated": False,
            "model_scoring_allowed": False,
            "predictive_scores_computed": False,
        }
        provenance_path = subdir / "target_provenance.json"
        with provenance_path.open("xb") as stream:
            stream.write(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        provenance_ref = {"path": str(provenance_path.resolve()), "sha256": _sha(provenance_path)}
        results.append({"id": cohort["id"], "target_table": target_ref, "target_provenance": provenance_ref,
                        "row_count": len(ordered), "key_commitment_sha256": provenance["key_commitment"]["sha256"]})

    # Stage the durable inventory; it becomes visible as the success record only
    # after both inputs and all generated artifacts pass their post-write checks.
    output_refs = [ref for item in results for ref in (item["target_table"], item["target_provenance"])]
    _check_stable(checks)
    for ref in output_refs:
        if _sha(Path(ref["path"])) != ref["sha256"]:
            raise MaterializerError(f"generated output changed before inventory: {ref['path']}")
    inventory = {"format": "repair-v2-future-target-cohort-materialization-v1",
                 "status": PENDING_STATUS, "manifest": manifest_ref, "cohorts": results,
                 "model_scoring_allowed": False, "predictive_scores_computed": False}
    staged_inventory = out / ".materialization_manifest.json.tmp"
    with staged_inventory.open("xb") as stream:
        stream.write(json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
    _check_stable(checks)
    for ref in output_refs:
        if _sha(Path(ref["path"])) != ref["sha256"]:
            raise MaterializerError(f"generated output changed during final checks: {ref['path']}")
    final_inventory = out / "materialization_manifest.json"
    if final_inventory.exists():
        raise MaterializerError("refusing to overwrite materialization manifest")
    staged_inventory.rename(final_inventory)
    try:
        _check_stable(checks)
        for ref in output_refs:
            if _sha(Path(ref["path"])) != ref["sha256"]:
                raise MaterializerError(f"generated output changed after inventory publication: {ref['path']}")
        inventory_ref = {"path": str(final_inventory.resolve()), "sha256": _sha(final_inventory)}
    except Exception:
        final_inventory.unlink(missing_ok=True)
        raise
    return {"status": PENDING_STATUS, "manifest": manifest_ref, "out_dir": str(out), "cohorts": results,
            "materialization_manifest": inventory_ref, "model_scoring_allowed": False, "predictive_scores_computed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.manifest, args.manifest_sha256.lower(), args.out_dir)
    except Exception as exc:
        parser.exit(2, f"prepare_targets: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
