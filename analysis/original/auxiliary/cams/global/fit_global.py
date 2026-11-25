#!/usr/bin/env python3
"""Acceptance-gated, no-score global GEO+MET+CAMS OOF fit adapter."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

BASE = Path("/external/project")
EXPECTED_RUNTIME = BASE / "validation/global_union_v2_remaining_models_retry1_20261114/source_snapshot/validation"
OUTPUT_ROOT = BASE / "validation/global_current_fit_20261114"
CORE_SHA = "027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157"
RUNNER_SHA = "05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c"
LEGACY_RUNNER_SHA = "3ba6a7fd3830cd10c50c030871d06405eed85cd2bc380e06f31891fbd2d43539"
GEO = ["lat_f", "lon_f", "lpop", "month_sin", "month_cos", "month_sin2", "month_cos2"]
MET = ["blh", "wind10", "tp", "t2m", "rh", "vent"]
CAM = "cams_pm25_ug_m3"
COUNTRIES = {"AUS", "CL", "ES", "GR", "IT", "MEX", "PK", "PT", "TH", "ZA"}
GUARD_COUNTRIES = {"BD", "NP", "VN"}
STRATEGIES = {
    "smear": ["crossfit_station"], "splits": 3,
    "target": ["log_smear", "original_scale"],
    "primary_backtransform": "smear*exp(z)-0.1",
    "sensitivity_backtransform": "exp(z)-0.1",
}
FIT_SPECS = [
    ("log_smear", "crossfit_station", "smearing"),
    ("original_scale", "not_applicable", "model_native"),
]
COMMON_CONFIG = {
    "max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 31,
    "min_samples_leaf": 25, "l2_regularization": 1.0, "early_stopping": True,
    "validation_fraction": 0.15, "random_state": 0, "ridge_alpha": 1.0,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, default=str, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_acceptance(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, str]]:
    if sha256_file(path).lower() != expected_sha.lower():
        raise ValueError("acceptance JSON SHA256 mismatch")
    a = read_json(path)
    if a.get("format") != "global-current-cams-fit-acceptance-v1" or a.get("status") != "accepted_global_current_cams_fit":
        raise ValueError("root acceptance format/status is not accepted")
    bound = a.get("bound_files")
    if not isinstance(bound, dict) or not bound:
        raise ValueError("acceptance.bound_files must be a nonempty absolute-path SHA256 map")
    checked: dict[str, str] = {}
    for raw, expected in bound.items():
        p = Path(str(raw))
        if not p.is_absolute() or not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected)):
            raise ValueError(f"invalid bound_files entry: {raw!r}")
        resolved = p.resolve(strict=True)
        got = sha256_file(resolved)
        if got.lower() != str(expected).lower():
            raise ValueError(f"bound SHA256 mismatch: {resolved}")
        checked[str(resolved)] = got
    return a, checked


def require_bound(path_value: str, bound: dict[str, str], expected: str | None = None) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        raise ValueError(f"input path must be absolute: {path_value}")
    p = p.resolve(strict=True)
    got = bound.get(str(p))
    if got is None or (expected is not None and got.lower() != expected.lower()):
        raise ValueError(f"input is absent from bound_files or has unexpected SHA256: {p}")
    return p


def allowed_config(kind: str) -> dict[str, Any]:
    if kind not in {"hgb", "ridge"}:
        raise ValueError(f"unsupported estimator: {kind}")
    return {**COMMON_CONFIG, "estimator_kind": kind}


def check_report(report: dict[str, Any], a: dict[str, Any]) -> None:
    if (report.get("format") != "global-current-cams-preflight-v2" or report.get("status") != "PASS"
            or report.get("fit_performed") is not False or report.get("pass_jobs") != 80
            or report.get("blocked_jobs") != 0 or report.get("oof_jobs_expected") != 80):
        raise ValueError("preflight is not complete PASS/80 with fit_performed=false; partial blockers are fatal")
    jobs = report.get("jobs")
    guards = report.get("guard_rows")
    if not isinstance(jobs, list) or len(jobs) != 80 or not isinstance(guards, list) or len(guards) != 24:
        raise ValueError("preflight job/guard census differs from 80 OOF and 24 guard rows")
    if report.get("guard_rows_count") != 24 or report.get("runtime_core_sha256") != CORE_SHA or report.get("runtime_runner_sha256") != RUNNER_SHA:
        raise ValueError("preflight guard/runtime binding differs from the frozen contract")
    if any(str(x.get("country", "")) not in GUARD_COUNTRIES for x in guards):
        raise ValueError("guard registry contains a non-guard country")
    ids = [str(x.get("job_id", "")) for x in jobs]
    if len(set(ids)) != 80 or any(x.get("status") != "PASS" for x in jobs):
        raise ValueError("preflight has duplicate IDs or a non-PASS OOF job")
    if {str(x.get("country")) for x in jobs} != COUNTRIES:
        raise ValueError("preflight country census differs from the frozen 10-country set")
    expected={(c,q,p,e) for c in COUNTRIES for q in ('strict','sensitivity') for p in ('with_met','no_met') for e in ('hgb','ridge')}
    if {(x['country'],x['qa'],x['profile'],x['estimator']) for x in jobs}!=expected:
        raise ValueError('complete country/QA/profile/estimator census differs')
    if Path(str(a.get("runtime_dir", ""))).resolve(strict=True) != EXPECTED_RUNTIME.resolve(strict=True):
        raise ValueError("acceptance runtime_dir differs from frozen runtime")
    if not isinstance(a.get("job_inputs"), dict) or set(a["job_inputs"]) != set(ids):
        raise ValueError("acceptance.job_inputs must cover exactly all 80 report job IDs")


def verify_runtime_versions(expected: dict[str, str]) -> dict[str, str]:
    actual = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "numpy_version": importlib.metadata.version("numpy"),
        "pandas_version": importlib.metadata.version("pandas"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
    }
    if actual != expected:
        raise RuntimeError(f"runtime version mismatch: expected {expected}, got {actual}")
    return actual


def resolve_evidence(job: dict[str, Any], inputs: dict[str, Any], bound: dict[str, str]):
    paths = {}
    expected = {
        "run_manifest": job.get("run_manifest_sha256"),
        "schema_manifest": job.get("schema_manifest_sha256"),
        "split_manifest": job.get("split_manifest_sha256"),
        "assignment": job.get("assignment_sha256"),
        "source_manifest": job.get("source_manifest_sha256"),
    }
    for key in ("run_manifest", "schema_manifest", "source_manifest", "split_manifest", "assignment"):
        value = inputs.get(key)
        if not isinstance(value, str):
            raise ValueError(f"acceptance input path missing for {job['job_id']}/{key}")
        paths[key] = require_bound(value, bound, expected.get(key))
    paths["source"] = require_bound(str(job["source_path"]), bound, str(job["source_sha256"]))
    paths["frame"] = require_bound(str(job["frame_path"]), bound, str(job["frame_sha256"]))
    paths["membership"] = require_bound(str(job["membership_path"]), bound, str(job['membership_sha256']))
    return paths


def verify_metadata(job: dict[str, Any], paths: dict[str, Path], runtime_versions: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rm, sm = read_json(paths["run_manifest"]), read_json(paths["source_manifest"])
    schema = read_json(paths["schema_manifest"])
    split = read_json(paths["split_manifest"])
    country, profile, estimator = str(job["country"]), str(job["profile"]), str(job["estimator"])
    expected_features = [*GEO, *(MET if profile == "with_met" else []), CAM]
    cfg = allowed_config(estimator)
    if job.get("base_features") != expected_features or job.get("model_config") != cfg:
        raise ValueError(f"feature/config contract differs for {job['job_id']}")
    if job.get("smear_strategies") != STRATEGIES or rm.get("model_config") != cfg or rm.get("estimator_kind") != estimator:
        raise ValueError(f"fit configuration differs for {job['job_id']}")
    if rm.get("runtime") != job.get("runtime"):
        raise ValueError(f"run-manifest runtime differs from report for {job['job_id']}")
    runtime = job.get("runtime", {})
    if {k: runtime.get(k) for k in runtime_versions} != runtime_versions:
        raise ValueError(f"job runtime versions differ for {job['job_id']}")
    code_sha = runtime.get("code_sha256", {})
    recorded_runner=code_sha.get('runner')
    compatible=(recorded_runner==RUNNER_SHA or (recorded_runner==LEGACY_RUNNER_SHA and rm.get('country_meteorology_profile_selection') in {'source_declared_fixed_profile','no_met_override'}))
    if code_sha.get("validation_core") != CORE_SHA or not compatible or recorded_runner!=job.get('accepted_runner_sha256'):
        raise ValueError(f"frozen code SHA mismatch for {job['job_id']}")
    if rm.get("smear_strategies") != STRATEGIES["smear"] or rm.get("smear_splits") != 3 or rm.get("target_strategies") != STRATEGIES["target"]:
        raise ValueError(f"smear/target strategy differs for {job['job_id']}")
    sm_country = sm.get(country)
    if not isinstance(sm_country, dict) or sm_country.get("target_source_sha256") is not None:
        raise ValueError(f"target override is present or source manifest is missing for {job['job_id']}")
    if sm_country.get("feature_tables") != [str(paths["source"])] or sm_country.get("feature_sha256") != [job["source_sha256"]]:
        raise ValueError(f"source manifest feature binding differs for {job['job_id']}")
    if sm_country.get("country_met_profile_name") != profile or sm_country.get("country_met_profile") != (MET if profile == "with_met" else []):
        raise ValueError(f"source MET profile differs for {job['job_id']}")
    b_schema = schema.get(f"{country}:B", {})
    if b_schema.get("base_features") != expected_features[:-1]:
        raise ValueError(f"schema manifest base features differ for {job['job_id']}")
    if split.get("format") != "repair-v2-outer-fold-split-manifest-v1":
        raise ValueError(f"unsupported saved split manifest for {job['job_id']}")
    rm_split = rm.get("outer_fold_split_manifest", {})
    if rm_split.get("sha256") != sha256_file(paths["split_manifest"]) or rm_split.get("assignment_sha256") != sha256_file(paths["assignment"]):
        raise ValueError(f"run-manifest split/assignment SHA differs for {job['job_id']}")
    return rm, sm_country, split, schema


def target_hash(frame: Any) -> str:
    b = frame[["row_id", "pm25"]].to_csv(index=False, na_rep="<NA>", float_format="%.17g", lineterminator="\n").encode()
    return hashlib.sha256(b).hexdigest()


def verify_job_data(job: dict[str, Any], paths: dict[str, Path], split: dict[str, Any], core: Any, runner: Any, np: Any, pd: Any):
    frame = pd.read_parquet(paths["frame"])
    base = job["base_features"]
    required = {"row_id", "country", "site_id", "year", "month", "pm25", *base}
    if required - set(frame.columns):
        raise ValueError(f"frame columns missing for {job['job_id']}: {sorted(required-set(frame.columns))}")
    frame["row_id"] = pd.to_numeric(frame.row_id, errors="raise").astype("int64")
    if frame.row_id.duplicated().any() or not frame.row_id.is_monotonic_increasing:
        raise ValueError(f"frame row_id order/uniqueness invalid for {job['job_id']}")
    if not np.isfinite(pd.to_numeric(frame.pm25, errors="coerce").to_numpy(dtype="float64")).all():
        raise ValueError(f"frame targets are nonfinite for {job['job_id']}")
    schema = core.schema_for(base)
    if schema != job.get("schema"):
        raise ValueError(f"runtime-derived feature schema differs for {job['job_id']}")
    matrix = core.feature_matrix(frame, schema)
    mh = hashlib.sha256()
    mh.update(np.ascontiguousarray(matrix).tobytes())
    mh.update(frame[["row_id", "country", "site_id", "year", "month", "pm25"]].to_csv(index=False, float_format="%.17g", lineterminator="\n").encode())
    matrix_sha = mh.hexdigest()
    if matrix_sha != job.get("matrix_fingerprint_sha256"):
        raise ValueError(f"matrix fingerprint mismatch for {job['job_id']}")
    membership = read_json(paths["membership"])
    records = membership.get("folds")
    report_splits = job.get("splits")
    saved_splits = split.get("splits")
    if membership.get("job_id") != job["job_id"] or not isinstance(records, list) or not isinstance(saved_splits, list):
        raise ValueError(f"saved membership/split records absent for {job['job_id']}")
    report_by_key = {(str(x["task"]), str(x["fold_id"])): x for x in report_splits}
    manifest_by_key = {(str(x["task"]), str(x["fold_id"])): x for x in saved_splits}
    mem_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        key = (str(rec.get("task")), str(rec.get("fold_id")))
        if key not in report_by_key or key not in manifest_by_key or rec.get("role") not in {"train", "test"}:
            raise ValueError(f"unexpected membership identity for {job['job_id']}: {key}")
        slot = mem_by_key.setdefault(key, {})
        if rec["role"] in slot or rec.get("job_id") != job["job_id"] or str(rec.get("country", "")).upper() != str(job["country"]).upper():
            raise ValueError(f"duplicate/mismatched membership role for {job['job_id']}: {key}")
        slot[rec["role"]] = rec
    if set(mem_by_key) != set(report_by_key) or set(manifest_by_key) != set(report_by_key) or any(set(v) != {"train", "test"} for v in mem_by_key.values()):
        raise ValueError(f"saved eligible fold census differs for {job['job_id']}")
    fp_folds = []
    for key in [(str(x["task"]), str(x["fold_id"])) for x in report_splits]:
        rep, sp = report_by_key[key], manifest_by_key[key]
        train_ids_by_role = {}
        for role in ("train", "test"):
            rec = mem_by_key[key][role]
            ids = rec.get("row_ids")
            if not isinstance(ids, list) or len(ids) != len(set(ids)) or ids != sorted(ids):
                raise ValueError(f"invalid ordered row IDs for {job['job_id']}/{key}/{role}")
            part = frame.loc[frame.row_id.isin(ids)].copy().sort_values("row_id", kind="mergesort").reset_index(drop=True)
            if len(part) != len(ids) or runner.canonical_row_key_sha256(part) != rec.get("row_key_sha256") or target_hash(part) != rec.get("ordered_target_sha256"):
                raise ValueError(f"saved membership key/target mismatch for {job['job_id']}/{key}/{role}")
            if len(part) != int(rep[f"{role}_rows"]) or rec.get("row_key_sha256") != rep[f"{role}_key_sha256"] or rec.get("ordered_target_sha256") != rep[f"{role}_target_sha256"]:
                raise ValueError(f"report row census/hash mismatch for {job['job_id']}/{key}/{role}")
            saved_ids = sp.get(f"{role}_ids") or []
            unit = sp.get(f"{role}_ids_unit")
            actual_ids = set(part.site_id.astype(str)) if unit == "site_id" else set(part.row_id.astype(int)) if unit == "row_id" else None
            expected_ids = {str(x) for x in saved_ids} if unit == "site_id" else {int(x) for x in saved_ids} if unit == "row_id" else None
            if actual_ids is None or actual_ids != expected_ids or len(part) != int(sp.get(f"{role}_row_count", -1)):
                raise ValueError(f"saved source split IDs/count mismatch for {job['job_id']}/{key}/{role}")
            if sp.get(f"{role}_row_key_sha256") != rec["row_key_sha256"]:
                raise ValueError(f"original split row-key mismatch for {job['job_id']}/{key}/{role}")
            train_ids_by_role[role] = set(ids)
        if train_ids_by_role["train"] & train_ids_by_role["test"]:
            raise ValueError(f"train/test overlap for {job['job_id']}/{key}")
        tr = frame.loc[frame.row_id.isin(mem_by_key[key]["train"]["row_ids"])].copy().sort_values("row_id", kind="mergesort")
        binding = sp.get("full_model", {}).get("binding", {})
        y = pd.to_numeric(tr.pm25, errors="raise").to_numpy(dtype="float64")
        if binding.get("raw_target_y_sha256") != runner._array_sha256(y) or binding.get("ordered_group_ids_sha256") != runner._json_sha256(tr.site_id.astype(str).tolist()):
            raise ValueError(f"saved target/group binding mismatch for {job['job_id']}/{key}")
        fs = sp.get("feature_source_sha256")
        if not isinstance(fs, list) or fs != [job["source_sha256"]]:
            raise ValueError(f"saved feature-source binding mismatch for {job['job_id']}/{key}")
        for role in ("train", "test"):
            rec = mem_by_key[key][role]
            fp_folds.append({k: v for k, v in rec.items() if k != "job_id"})
    recipe = {"country": job["country"], "profile": job["profile"], "estimator": job["estimator"],
              "base_features": base, "model_config": job["model_config"], "runtime": job["runtime"],
              "strategies": job["smear_strategies"], "core_sha": CORE_SHA, "runner_sha": RUNNER_SHA,
              "folds": fp_folds, "matrix_sha": matrix_sha}
    fingerprint = json_sha(recipe)
    if fingerprint != job.get("exact_dedup_fingerprint_sha256"):
        raise ValueError(f"exact dedup fingerprint mismatch for {job['job_id']}")
    return frame, membership, fingerprint


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fit_all(groups: dict[str, list[dict[str, Any]]], bound: dict[str, str], acceptance_sha: str, core: Any, np: Any, pd: Any) -> None:
    import joblib
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    claim = OUTPUT_ROOT / ".global_fit.claim"
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, f"pid={os.getpid()} run_id={run_id} acceptance_sha256={acceptance_sha}\n".encode())
    os.fsync(fd)
    os.close(fd)
    run_dir = OUTPUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run = {"format": "global-current-cams-fit-v1", "status": "running", "acceptance_sha256": acceptance_sha,
           "bound_files_sha256": json_sha(bound), "fit_performed": True, "artifacts": [], "completed_fits": 0}
    run['expected_fits']=sum(len(aliases[0]['membership']['folds'])//2*len(FIT_SPECS) for aliases in groups.values())
    atomic_json(run_dir / "run_result.json", run)
    try:
        for fingerprint, aliases in sorted(groups.items()):
            rep = aliases[0]
            frame = rep["frame"]
            split_records = rep['membership']['folds']
            split_map: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
            for r in split_records:
                split_map.setdefault((str(r["task"]), str(r["fold_id"])), {})[str(r["role"])] = r
            for (task, fold), roles in sorted(split_map.items()):
                train = frame.loc[frame.row_id.isin(roles["train"]["row_ids"])].copy().sort_values("row_id", kind="mergesort").reset_index(drop=True)
                test = frame.loc[frame.row_id.isin(roles["test"]["row_ids"])].copy().sort_values("row_id", kind="mergesort").reset_index(drop=True)
                for target, smear_meta, backtransform in FIT_SPECS:
                    key = f"{fingerprint[:16]}__{safe(task)}__{safe(fold)}__{target}"
                    fit_dir = run_dir / "fits" / key
                    fit_dir.mkdir(parents=True, exist_ok=False)
                    manifest_path = fit_dir / "fit_manifest.json"
                    aliases_out = [{"job_id": a["job_id"], "qa": a["qa"]} for a in aliases]
                    fm = {"status": "started", "fingerprint": fingerprint, "aliases": aliases_out,
                          "task": task, "fold_id": fold, "target_strategy": target,
                          "smear_strategy": smear_meta, "backtransform": backtransform,
                          "train_rows": len(train), "test_rows": len(test)}
                    fm.update({'acceptance_sha256':acceptance_sha,'frame_sha256':rep['frame_sha256'],
                               'membership_sha256':rep['membership_sha256'],'schema':rep['schema'],
                               'model_config':rep['model_config'],'runtime':rep['runtime'],
                               'train_ordered_target_sha256':target_hash(train),'test_ordered_target_sha256':target_hash(test)})
                    atomic_json(manifest_path, fm)
                    bundle = core.fit_model(train, rep["base_features"], target_col="pm25", group_col="site_id",
                        config=core.ModelConfig(**rep["model_config"]), smear_strategy="crossfit_station",
                        smear_splits=3, target_strategy=target)
                    prediction = np.asarray(core.predict_model(bundle, test, backtransform=backtransform), dtype="float64")
                    if len(prediction) != len(test) or not np.isfinite(prediction).all():
                        raise ValueError(f"invalid prediction vector: {key}")
                    model_path = fit_dir / "model.joblib"
                    tmp_model = fit_dir / "model.joblib.tmp"
                    joblib.dump(bundle, tmp_model, compress=3)
                    os.replace(tmp_model, model_path)
                    pred = pd.DataFrame({"country": rep["country"], "profile": rep["profile"], "estimator": rep["estimator"],
                        "task": task, "fold_id": fold, "role": "test", "row_id": test.row_id.astype("int64"),
                        "site_id": test.site_id.astype(str), "year": test.year, "month": test.month,
                        "observed_pm25": test.pm25.astype("float64"), "prediction_pm25": prediction,
                        "target_strategy": target, "smear_strategy": bundle["smear_info"]["method"],
                        "backtransform": backtransform, "exact_dedup_fingerprint_sha256": fingerprint,
                        "job_aliases_json": json.dumps(aliases_out, sort_keys=True)})
                    pred_path = fit_dir / "oof_predictions.parquet"
                    pred.to_parquet(pred_path, index=False)
                    fm.update({"status": "complete", "model_path": str(model_path), "model_sha256": sha256_file(model_path),
                               "predictions_path": str(pred_path), "predictions_sha256": sha256_file(pred_path),
                               "prediction_rows": len(pred), "smear_info": bundle["smear_info"]})
                    atomic_json(manifest_path, fm)
                    run["artifacts"].append(str(manifest_path))
                    run["completed_fits"] += 1
                    atomic_json(run_dir / "run_result.json", run)
        if run['completed_fits']!=run['expected_fits']:
            raise ValueError('completed fit census differs from validated membership plan')
        run.update({"status": "complete", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        atomic_json(run_dir / "run_result.json", run)
    except BaseException as exc:
        run.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        atomic_json(run_dir / "run_result.json", run)
        raise


def main() -> int:
    if sys.flags.optimize:
        raise RuntimeError("optimized Python prohibited")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acceptance", required=True, type=Path)
    p.add_argument("--acceptance-sha", required=True)
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--plan-only", action="store_true")
    m.add_argument("--run", action="store_true")
    args = p.parse_args()
    a, bound = verify_acceptance(args.acceptance.resolve(strict=True), args.acceptance_sha)
    require_bound(str(Path(__file__).resolve()),bound)
    report_path = require_bound(str(a.get("preflight_report_path", "")), bound)
    runtime_dir = Path(str(a.get("runtime_dir", ""))).resolve(strict=True)
    core_path = require_bound(str(runtime_dir / "validation_core.py"), bound, CORE_SHA)
    runner_path = require_bound(str(runtime_dir / "run_validation.py"), bound, RUNNER_SHA)
    report = read_json(report_path)
    check_report(report, a)
    expected_versions = a.get("expected_runtime_versions")
    if not isinstance(expected_versions, dict) or expected_versions != report.get("actual_runtime"):
        raise ValueError("acceptance runtime versions must equal the reviewed report")
    prepared = []
    for job in report["jobs"]:
        paths = resolve_evidence(job, a["job_inputs"][job["job_id"]], bound)
        rm, smc, split, schema = verify_metadata(job, paths, expected_versions)
        prepared.append((job, paths, split, schema))
    versions = verify_runtime_versions(expected_versions)
    sys.path.insert(0, str(runtime_dir))
    import numpy as np
    import pandas as pd
    import validation_core as core
    import run_validation as runner
    if Path(core.__file__).resolve() != core_path or Path(runner.__file__).resolve() != runner_path:
        raise RuntimeError("frozen runtime imported from unexpected path")
    groups: dict[str, list[dict[str, Any]]] = {}
    frames: dict[str, Any] = {}
    for job, paths, split, schema in prepared:
        frame, membership, fingerprint = verify_job_data(job, paths, split, core, runner, np, pd)
        item = dict(job)
        item.update({"paths": paths, "frame": frame, "membership": membership})
        groups.setdefault(fingerprint, []).append(item)
        frames[str(paths["frame"])] = frame
    # Reconstruct the report's exact dedup index; aliases may differ only by QA.
    computed_groups: dict[str, dict[str, list[str]]] = {}
    report_frames = report.get("fingerprints", {}).get("frames", {})
    for fingerprint, aliases in groups.items():
        labels = {(x["country"], x["profile"], x["estimator"]) for x in aliases}
        if len(labels) != 1 or len({x["qa"] for x in aliases}) != len(aliases):
            raise ValueError("dedup would cross a country/profile/estimator or repeat a QA label")
        label = "|".join(next(iter(labels)))
        computed_groups.setdefault(label, {})[fingerprint] = sorted(x["job_id"] for x in aliases)
        frame_rec = report_frames.get(fingerprint)
        if not isinstance(frame_rec, dict) or set(frame_rec.get("job_labels", [])) != set(computed_groups[label][fingerprint]):
            raise ValueError(f"report frame alias index mismatch: {fingerprint}")
    expected_groups = {str(k): {str(f): sorted(map(str, ids)) for f, ids in v.items()}
                       for k, v in report.get("fingerprints", {}).get("groups", {}).items()}
    if computed_groups != expected_groups or len(groups) != report.get("fingerprints", {}).get("unique_frames"):
        raise ValueError("independently rebuilt exact dedup index differs from preflight report")
    fits = sum(len(v[0]['membership']['folds']) // 2 * len(FIT_SPECS) for v in groups.values())
    if args.plan_only:
        print(json.dumps({"status": "plan_validated", "fit_performed": False, "jobs": 80, "guards": 24,
                          "unique_fit_groups": len(groups), "unique_group_fold_strategy_fits": fits,
                          "target_strategies": STRATEGIES["target"], "runtime_versions": versions,
                          "acceptance_sha256": args.acceptance_sha.lower(), "preflight_report_sha256": bound[str(report_path)]}, sort_keys=True))
        return 0
    if a.get("fit_authorized") is not True:
        raise ValueError("--run requires fit_authorized=true in the root acceptance JSON")
    fit_all(groups, bound, args.acceptance_sha.lower(), core, np, pd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        print(f"fit_global failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
