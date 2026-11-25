"""Prepare a descriptive, held-out support/P90-fire-proxy audit for US/CAN.

This module is inert on import. Any later table-reading entry point must first
validate the accepted historical disposition and reconciliation pins. The
helpers below are intentionally testable using synthetic frames only.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DISPOSITION_SHA256 = "35a143e4bdbbc3721709b4df5c920b4156e8825e8493eb174e2a0529d1dbb6fa"
RECONCILIATION_SHA256 = "c582bc8ec51bdea32e9c44ef3fc763888d0b86c6ee94327e7604bfee7177d232"
SMOKE_ACCEPTANCE_SHA256 = "38741850360c288d2ca3eecce48b6b3143c6b7ab1e3bf8dfa9fe3c89aacbfa08"
SMOKE_REPORT_SHA256 = "0c4dfbe6eda816f0cf3ead71cf4a310a8a11d0407c111db2f8473af486757a51"
SUPPORT_RULES_SHA256 = "b7c88d905b5ece5a57f26fae9cbd4236b2da41248b37d70ddd85dce5367353c8"

COUNTRIES = ("USA", "CAN")
QAS = ("strict", "sensitivity")
PROFILES = ("with_met", "no_met")
TASKS = ("T1_station", "T2_spatial", "T2_spatial_buffer50km", "T3_year_2022")
TARGET_BRANCHES = (("log_smear", "smearing"), ("original_scale", "model_native"))
FIRE_SCOPES = ("all_months", "P90_fire_proxy", "other_months")
PAIR_KEYS = (
    "country", "task", "fold_id", "site_id", "year", "month", "profile",
    "qa_variant", "target_strategy", "backtransform",
)


class AuditError(ValueError):
    """A bound input or synthetic audit contract differs from expectation."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise AuditError(message)


def validate_current_artifact_pins(
    pins: Mapping[str, str], required_paths: Iterable[str], hash_file: Callable[[str], str]
) -> None:
    """Check reconciliation pins before the caller opens any OOF/flag table."""
    require(isinstance(pins, Mapping) and bool(pins), "accepted current_artifact_pins are missing")
    for path in required_paths:
        require(path in pins, f"accepted reconciliation has no pin for {path}")
        require(hash_file(path) == pins[path], f"current artifact pin differs: {path}")


def select_primary_hgb_refs(refs: Sequence[Mapping[str, Any]], *, country: str,
                            expected_count: int = 16) -> Mapping[str, Any]:
    """Return the one primary-HGB core binding repeated across a profile's folds.

    CAN common-cohort and every Ridge binding are excluded. USA's primary cohort
    uses the historical USA naming convention without a literal ``primary``.
    """
    require(country in COUNTRIES, f"unsupported country: {country}")
    require(len(refs) == expected_count, f"{country}: unexpected split binding census")
    unique = {}
    for ref in refs:
        require(isinstance(ref, Mapping), "malformed core manifest reference")
        path = str(ref.get("run_manifest_path", "")).lower().replace("\\", "/")
        require(path and "hgb" in path and "ridge" not in path,
                f"{country}: core reference is not HGB")
        require("common" not in path, f"{country}: CAN-common/non-primary reference excluded")
        if country == "CAN":
            require("primary" in path, "CAN core reference must be the accepted primary cohort")
        unique[str(ref.get("run_manifest_path"))] = ref
    require(len(unique) == 1, f"{country}: ambiguous primary-HGB core job for profile")
    return next(iter(unique.values()))


def select_supported_target_branches(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the two target/backtransform pairs shared with smoke output."""
    require({"target_strategy", "backtransform"} <= set(frame.columns),
            "target branch fields are missing")
    keep = np.zeros(len(frame), dtype=bool)
    for strategy, backtransform in TARGET_BRANCHES:
        keep |= (frame.target_strategy.astype(str).to_numpy() == strategy) & \
                (frame.backtransform.astype(str).to_numpy() == backtransform)
    return frame.loc[keep].copy()


def exact_p90_pair(core_pairs: pd.DataFrame, smoke: pd.DataFrame) -> pd.DataFrame:
    """Exact one-to-one match across protocol identity, preserving signed values."""
    require(set(PAIR_KEYS) <= set(core_pairs.columns), "core/support identity keys missing")
    required = set(PAIR_KEYS) | {"obs", "pred"}
    require(required <= set(core_pairs.columns), "joined support rows missing required fields")
    smoke_required = set(PAIR_KEYS) | {
        "gate_quantile", "obs", "predicted_total", "affected", "training_threshold", "lgfas_r15",
    }
    require(smoke_required <= set(smoke.columns), "smoke P90 rows missing required fields")
    p90 = smoke.loc[np.isclose(pd.to_numeric(smoke.gate_quantile, errors="raise"), .90)].copy()
    p90 = select_supported_target_branches(p90)
    core = select_supported_target_branches(core_pairs)
    require(not p90.duplicated(list(PAIR_KEYS)).any(), "duplicate smoke P90 identity")
    require(not core.duplicated(list(PAIR_KEYS)).any(), "duplicate core/support identity")
    matched = core.merge(p90[list(PAIR_KEYS) + ["obs", "predicted_total", "affected", "training_threshold", "lgfas_r15"]],
                         on=list(PAIR_KEYS), how="outer", validate="one_to_one", indicator=True,
                         suffixes=("_core", "_smoke"))
    require(matched["_merge"].eq("both").all(), "core/support and smoke identities differ")
    require(np.array_equal(matched.obs_core.to_numpy(), matched.obs_smoke.to_numpy()),
            "core and smoke targets differ")
    require(np.array_equal(matched.pred.to_numpy(), matched.predicted_total.to_numpy()),
            "core full prediction differs from smoke full prediction")
    computed = matched.lgfas_r15.to_numpy(float) > matched.training_threshold.to_numpy(float)
    require(np.array_equal(computed, matched.affected.to_numpy(bool)), "P90 event flag differs from fixed threshold")
    matched['gate_affected'] = computed
    matched['obs'] = matched.obs_core
    return matched.drop(columns="_merge")


def summarize_coverage(pairs: pd.DataFrame, rules: Sequence[str],
                       metrics: Callable[..., Mapping[str, float]]) -> pd.DataFrame:
    """Summarize retained/excluded rows and station denominators descriptively."""
    group_cols = ["country", "qa_variant", "profile", "task", "target_strategy",
                  "smear_strategy", "backtransform"]
    require(set(group_cols + ["site_id", "gate_affected", "obs", "pred", "baseline_pred"]) <= set(pairs.columns),
            "support pair schema incomplete")
    rows: list[dict[str, Any]] = []
    for group_key, part in pairs.groupby(group_cols, sort=True, dropna=False):
        identity = dict(zip(group_cols, group_key))
        scopes = (("all_months", np.ones(len(part), dtype=bool)),
                  ("P90_fire_proxy", part.gate_affected.to_numpy(dtype=bool)),
                  ("other_months", ~part.gate_affected.to_numpy(dtype=bool)))
        for rule in rules:
            gate_name = "gate_" + rule
            require(gate_name in part, f"frozen support rule flag missing: {gate_name}")
            gate = part[gate_name].to_numpy(dtype=bool)
            for scope, scope_mask in scopes:
                population = part.loc[scope_mask]
                denominator_stations = population.site_id.nunique()
                for retained, keep_mask in ((True, gate), (False, ~gate)):
                    selected = part.loc[scope_mask & keep_mask]
                    full = metrics(selected)
                    baseline = metrics(selected, prediction="baseline_pred")
                    n, stations = len(selected), selected.site_id.nunique()
                    denom = len(population)
                    rows.append({**identity, "rule": rule, "fire_month_scope": scope,
                        "retained": retained, "eligible_rows": denom,
                        "eligible_stations": denominator_stations, "n": n,
                        "n_stations": stations,
                        "row_coverage_fraction": n / denom if denom else np.nan,
                        "station_coverage_fraction": stations / denominator_stations if denominator_stations else np.nan,
                        "full_rmse": full["rmse"], "full_mae": full["mae"],
                        "full_bias": full["bias"], "full_r2": full["r2"],
                        "baseline_rmse": baseline["rmse"], "baseline_mae": baseline["mae"],
                        "baseline_bias": baseline["bias"], "baseline_r2": baseline["r2"],
                        "delta_rmse_full_minus_B": full["rmse"] - baseline["rmse"],
                        "interpretation": "descriptive held-out error under a fixed support rule and P90 fire-feature proxy; not identified smoke mass"})
    table = pd.DataFrame(rows)
    for _, part in table.groupby(group_cols + ["rule", "fire_month_scope"], sort=False, dropna=False):
        require(len(part) == 2 and set(part.retained) == {True, False},
                "retained/excluded pair missing")
        require(int(part.n.sum()) == int(part.eligible_rows.iloc[0]),
                "retained/excluded denominator mismatch")
    return table


def filter_smoke_p90(smoke: pd.DataFrame, *, country: str, qa: str, profile: str) -> pd.DataFrame:
    """Select the exact P90 event slice for one accepted smoke job/profile."""
    require(country in COUNTRIES and qa in QAS and profile in PROFILES, "unknown smoke identity")
    selected = smoke.loc[
        smoke.country.astype(str).eq(country) & smoke.qa_variant.astype(str).eq(qa) &
        smoke.profile.astype(str).eq(profile) &
        np.isclose(pd.to_numeric(smoke.gate_quantile, errors="raise"), .90) &
        smoke.task.astype(str).isin(TASKS)
    ].copy()
    return select_supported_target_branches(selected)


def pin_required_before_table_read(path: Path, pins: Mapping[str, str], hash_file: Callable[[str], str]) -> None:
    """Single-file guard to call immediately before read_parquet on a core table."""
    validate_current_artifact_pins(pins, [str(path)], hash_file)


V = Path('/external/project/validation')
ROOT = V/'uscan_smoke_isolated_v2_20261115'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'refusing existing joint-fire audit output')
    sources = {}
    def pinned(path, digest):
        path = Path(path)
        require(sha(path) == digest, 'source hash mismatch: '+str(path))
        sources[str(path)] = digest
        return path
    disposition = load_json(pinned(V/'historical32_final_disposition_v1_20261115/independent_disposition.json', DISPOSITION_SHA256))
    require(disposition['independent_postfit_pass'] is True and disposition['audited_jobs'] == 32, 'historical32 not accepted')
    ref = disposition['supplement']
    require(ref['sha256'] == RECONCILIATION_SHA256, 'historical reconciliation ref mismatch')
    reconciliation = load_json(pinned(ref['path'], RECONCILIATION_SHA256))
    core_pins = reconciliation['current_artifact_pins']
    accepted = load_json(pinned(V/'uscan_smoke_core_acceptance_v1_20261115/independent_acceptance_v1.json', SMOKE_ACCEPTANCE_SHA256))
    report = load_json(pinned(ROOT/'postfit_audit_v2/audit_result.json', SMOKE_REPORT_SHA256))
    require(accepted['decision'] == 'accepted_core_postfit_only' and accepted['report_sha256'] == SMOKE_REPORT_SHA256 and
            report['status'] == 'historical4_core_postfit_replay_passed', 'smoke core not accepted')
    helper_path = pinned(V/'final_model_runtime_v4_20261114/source_snapshot/checks/assess_deployment_support.py', SUPPORT_RULES_SHA256)
    sys.path.insert(0, str(helper_path.parent))
    spec = importlib.util.spec_from_file_location('frozen_joint_support', helper_path)
    helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
    all_pairs, all_summaries = [], []
    jobs = {(j['country'], j['qa']): j for j in report['jobs']}
    require(set(jobs) == {(c, q) for c in COUNTRIES for q in QAS}, 'smoke job roster mismatch')
    for (country, qa), job in jobs.items():
        out = ROOT/'results'/f'{country}_{qa}'
        manifest_path = pinned(out/'smoke_manifest.json', job['source_manifest_sha256'])
        manifest = load_json(manifest_path)
        smoke_path = out/'smoke_oof_predictions.parquet'
        pinned(smoke_path, job['artifact_sha256'][str(smoke_path)])
        smoke = pd.read_parquet(smoke_path)
        for profile in PROFILES:
            refs = [r['core_manifest_refs'] for r in manifest['split_bindings'] if r['profile'] == profile]
            ref = select_primary_hgb_refs(refs, country=country)
            core_root = Path(ref['run_manifest_path']).parent
            expected = f'usa_hist_{qa}_{profile}_hgb' if country == 'USA' else f'can_hist_primary_{qa}_{profile}_hgb'
            require(core_root.name == expected, 'core profile/QA/policy job mismatch')
            op, fp = core_root/'oof_predictions.parquet', core_root/'support_row_flags.parquet'
            for path in (Path(ref['run_manifest_path']), Path(ref['source_manifest_path']), op, fp):
                require(str(path) in core_pins, 'historical artifact not accepted')
                pinned(path, core_pins[str(path)])
            predictions, flags = pd.read_parquet(op), pd.read_parquet(fp)
            predictions = select_supported_target_branches(predictions.loc[predictions.task.isin(TASKS)].copy())
            predictions['profile'] = profile; flags['profile'] = profile
            pairs, _ = helper.join_support(predictions, flags)
            pairs['qa_variant'] = qa
            pairs.fold_id = pairs.fold_id.astype(str)
            event = filter_smoke_p90(smoke, country=country, qa=qa, profile=profile)
            event.fold_id = event.fold_id.astype(str)
            merged = exact_p90_pair(pairs, event)
            summary = summarize_coverage(merged, helper.RULES, helper.metrics)
            require(len(summary) == 192, 'per-profile coverage summary census mismatch')
            all_pairs.append(merged); all_summaries.append(summary)
            print(json.dumps({'joined': [country, qa, profile], 'rows': len(merged)}), flush=True)
    table = pd.concat(all_summaries, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    require(len(table) == 1536, 'whole coverage summary census mismatch')
    for path, digest in sources.items():
        require(sha(path) == digest, 'source changed during joint audit')
    args.out.mkdir(parents=True)
    table.to_csv(args.out/'coverage_summary.csv', index=False)
    pairs.to_parquet(args.out/'exact_core_P90_support_pairs.parquet', index=False)
    result = {'status': 'historical4_joint_fire_support_passed', 'summary_rows': len(table), 'paired_rows': len(pairs),
        'sources_sha256': sources, 'rules': helper.RULES, 'scopes': FIRE_SCOPES,
        'interpretation': 'fixed train-only support and P90 fire-feature proxy; descriptive held-out errors, not identified smoke mass',
        'fits_performed': False, 'future_inputs_or_scores_read': False, 'whole_project_complete': False,
        'auditor_sha256': sha(__file__), 'output_sha256': {name: sha(args.out/name) for name in
         ('coverage_summary.csv', 'exact_core_P90_support_pairs.parquet')}}
    (args.out/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'paired_rows': len(pairs), 'summary_rows': len(table)}), flush=True)


if __name__ == '__main__':
    main()

