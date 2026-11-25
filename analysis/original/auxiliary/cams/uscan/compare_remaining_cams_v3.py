#!/usr/bin/env python3
"""Acceptance-gated exact pairing plan and separately accepted CAMS metrics.

The protocol-only path is static and never opens source, fit, or prediction files.
Plan/run paths refuse all source/output reads until the immutable remaining-28
postfit acceptance and the audit it pins have both been checked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

BASE = Path('/external/project')
V = BASE / 'validation'
PACK = V / 'uscan_cams_remaining_v1_20261115/pack/no_fit_pack_manifest.json'
PACK_SHA = '7d2d2c7f6ef4918ab5e99357e3c713cc8e4f1833c7cc85f6283d3ba8583feba2'
RUN_DIR = V / 'uscan_cams_fit_remaining_v1_20261115/runs/run_20261115T042810Z_237880'
COMPARISON_HELPER = V / 'global_current_comparison_20261114/compare_cams.py'
COMPARISON_HELPER_SHA = 'adfe83af59cc9ba7e6889a8229487e3961268b0b71f69bfe7b4bcc739462444c'
HISTORICAL32_DISPOSITION_SHA = '35a143e4bdbbc3721709b4df5c920b4156e8825e8493eb174e2a0529d1dbb6fa'
RUNTIME = V / 'final_model_runtime_v4_20261114/source_snapshot/validation'
METRIC_RUNNER = RUNTIME / 'run_validation.py'
METRIC_RUNNER_SHA = '05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c'
CORE_SHA = '027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157'
BOOTSTRAP_SHA = 'f8cbbbb53ce94a79b3da6b506788583880d81b817f18a5eba2b0f92fd1eb9332'
FEATURE_QA_ACCEPTANCE_SHA = 'f4e32b4b9b40c1938da90d3329694dfc099668fa655a1ac744c7d09ff8105351'
EXPECTED_RUNTIME = {'python_version': '3.11.16', 'numpy_version': '2.4.6',
                    'pandas_version': '2.3.3', 'scikit_learn_version': '1.9.1'}
EXPECTED_ALIASES = frozenset(
    f'{dataset}_{qa}_{profile}_{estimator}'
    for dataset, estimator in (
        ('can_hist_primary', 'hgb'), ('can_hist_common_all_published', 'hgb'),
        ('can_hist_common_exclude_236_238', 'hgb'), ('usa_hist', 'ridge'),
        ('can_hist_primary', 'ridge'), ('can_hist_common_all_published', 'ridge'),
        ('can_hist_common_exclude_236_238', 'ridge'))
    for qa in ('strict', 'sensitivity') for profile in ('with_met', 'no_met'))
TASK_FOLDS = {
    'T1_station': tuple(str(i) for i in range(5)),
    'T2_spatial': tuple(str(i) for i in range(5)),
    'T2_spatial_buffer50km': tuple(str(i) for i in range(5)),
    'T3_year_2022': ('2022',),
}
TARGETS = {'log_smear': ('smearing', 'crossfit_station'),
           'original_scale': ('model_native', 'not_applicable')}
KEYS = ['country', 'task', 'fold_id', 'row_id', 'site_id', 'year', 'month']
METRICS = ('rmse', 'mae', 'bias', 'r2')
BOOTSTRAP_REPLICATES = 300


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_pandas():
    global pd
    if pd is None:
        import pandas as pd_module
        pd = pd_module
    return pd


def protocol() -> dict[str, Any]:
    return {
        'format': 'uscan-cams-remaining-comparison-protocol-v1',
        'status': 'static_contract_only_no_source_reads',
        'pack_manifest_sha256': PACK_SHA,
        'expected_aliases': sorted(EXPECTED_ALIASES), 'alias_count': 28,
        'policy_scoped_fit_groups': 14, 'accepted_fit_count': 448,
        'target_strategies': {k: {'backtransform': v[0], 'smear_strategy': v[1]}
                              for k, v in TARGETS.items()},
        'tasks_and_folds': {k: list(v) for k, v in TASK_FOLDS.items()},
        'pairing_keys': KEYS, 'required_equalities': ['all keys', 'observed target'],
        'arms': ['B+CAMS', 'B', 'full'],
        'contrasts': ['B+CAMS minus B', 'B+CAMS minus full'],
        'metric_direction': {
            'delta': 'metric(B+CAMS) - metric(control), preserving signed bias',
            'rmse_mae': 'negative favors B+CAMS', 'r2': 'positive favors B+CAMS',
            'bias': 'signed difference only; not an absolute-bias benefit'},
        'bootstrap': {'engine': 'cluster_stats', 'replicates': BOOTSTRAP_REPLICATES,
                      'unit': 'site_id', 'seed_rule': 'frozen compare_cams.seed_for',
                      'labels': ['country', 'qa', 'profile', 'estimator', 'target_strategy',
                                 'backtransform', 'task', 'control_arm']},
        'metric_engine': {'runner_sha256': METRIC_RUNNER_SHA, 'core_sha256': CORE_SHA,
                          'bootstrap_helper_sha256': BOOTSTRAP_SHA},
        'expected_runtime': EXPECTED_RUNTIME,
        'expected_plan_cells': 224, 'expected_metric_rows': 672,
        'expected_paired_contrasts': 448,
        'fit_or_score_execution': False, 'metric_computation': False,
        'future_year_data_or_scores': False,
        'scope_note': 'historical 2018-2022 only; fold-conditional exploratory station-cluster intervals; no pooled-global significance',
    }


def postfit_gate(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate accepted refs before touching the pack, fit run, or predictions."""
    p = path.resolve(strict=True)
    require(re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha or '') is not None,
            'postfit acceptance SHA must be 64 hex characters')
    require(sha(p).lower() == expected_sha.lower(), 'postfit acceptance SHA mismatch')
    accepted = read_json(p)
    require(accepted.get('format') == 'uscan-cams-remaining-postfit-acceptance-v1'
            and accepted.get('decision') == 'accepted'
            and accepted.get('independent_postfit_pass') is True,
            'remaining-CAMS postfit acceptance is not accepted')
    require(accepted.get('pack_manifest_sha256') == PACK_SHA
            and accepted.get('audited_job_count') == 28
            and accepted.get('audited_fit_count') == 448
            and set(accepted.get('audited_aliases', [])) == EXPECTED_ALIASES
            and accepted.get('counts', {}).get('fit_manifests_rehashed') == 448
            and accepted.get('counts', {}).get('model_prediction_files_rehashed') == 896
            and accepted.get('counts', {}).get('bound_files_rehashed') == 243
            and accepted.get('counts', {}).get('policy_scoped_groups') == 14
            and accepted.get('actual_replay_runtime') == EXPECTED_RUNTIME,
            'postfit acceptance does not bind the exact remaining-28/448/pack scope')
    ref = accepted.get('postfit_audit', {})
    audit_path = Path(str(ref.get('path', '')))
    require(audit_path.is_absolute() and re.fullmatch(r'[0-9a-fA-F]{64}', str(ref.get('sha256', ''))) is not None,
            'postfit acceptance must supply an absolute audit path and SHA-256')
    audit_path = audit_path.resolve(strict=True)
    require(sha(audit_path).lower() == str(ref['sha256']).lower(), 'accepted postfit audit SHA mismatch')
    audit = read_json(audit_path)
    require(audit.get('format') == 'uscan-cams-remaining-postfit-audit-v1'
            and audit.get('status') == 'PASS'
            and audit.get('fits_verified') == 448
            and audit.get('aliases_verified') == 28
            and audit.get('policy_scoped_groups') == 14
            and audit.get('pack') == {'path': str(PACK), 'sha256': PACK_SHA}
            and audit.get('run_result', {}).get('path') == str(RUN_DIR / 'run_result.json')
            and re.fullmatch(r'[0-9a-fA-F]{64}', str(audit.get('run_result', {}).get('sha256', ''))) is not None,
            'accepted audit does not certify the pinned remaining-CAMS fit census')
    require(audit.get('pack', {}).get('path') == str(PACK)
            and audit.get('pack', {}).get('sha256') == PACK_SHA
            and audit.get('run_result') == accepted.get('run_result'),
            'accepted audit pack/run-result references differ from QA acceptance')
    manifests = audit.get('fit_manifest_sha256', {})
    outputs = audit.get('output_sha256', {})
    bound_files = audit.get('bound_files_sha256', {})
    require(isinstance(manifests, dict) and len(manifests) == 448
            and isinstance(outputs, dict) and len(outputs) == 896
            and isinstance(bound_files, dict) and len(bound_files) == 243
            and all(re.fullmatch(r'[0-9a-fA-F]{64}', str(x)) is not None for x in manifests.values())
            and all(re.fullmatch(r'[0-9a-fA-F]{64}', str(x)) is not None for x in outputs.values())
            and all(re.fullmatch(r'[0-9a-fA-F]{64}', str(x)) is not None for x in bound_files.values()),
            'accepted audit must pin exactly 448 fit manifests')
    runtime = audit.get('actual_replay_runtime', {})
    require(all(runtime.get(k) == v for k, v in EXPECTED_RUNTIME.items()),
            'accepted audit replay runtime differs from the pinned fit runtime')
    return accepted, audit


def historical32_gate(path: Path, expected_sha: str) -> dict[str, Any]:
    """Require accepted prior-32 disposition before any original OOF is read."""
    require(expected_sha.lower() == HISTORICAL32_DISPOSITION_SHA
            and sha(path) == HISTORICAL32_DISPOSITION_SHA,
            'historical32 independent disposition SHA mismatch')
    disposition = read_json(path)
    require(disposition.get('decision') == 'accepted'
            and disposition.get('independent_postfit_pass') is True
            and disposition.get('audited_jobs') == 32,
            'historical32 postfit disposition is not accepted for exact OOF reuse')
    supplement = disposition.get('supplement', {})
    sp = Path(str(supplement.get('path', '')))
    require(sp.is_absolute() and supplement.get('sha256') ==
            'c582bc8ec51bdea32e9c44ef3fc763888d0b86c6ee94327e7604bfee7177d232' and
            sha(sp) == supplement['sha256'], 'accepted historical reconciliation hash mismatch')
    reconciliation = read_json(sp)
    require(reconciliation.get('corrected_metadata_checks_passed') == 32 and
            reconciliation.get('matrix_sha256') == disposition['audited_matrix_sha256'] and
            reconciliation.get('candidate_manifest_sha256') == disposition['candidate_manifest_sha256'],
            'historical reconciliation identity mismatch')
    pins = reconciliation.get('current_artifact_pins')
    require(isinstance(pins, dict) and len(pins) > 0, 'historical artifact digest census missing')
    return {'disposition': disposition, 'pins': pins, 'supplement': supplement}


def verify_historical_artifacts(paths, pins):
    """Validate accepted bytes before a table reader can consume an old output."""
    sources = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        expected = pins.get(str(path))
        require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected) is not None and
                sha(path) == expected, 'accepted historical artifact SHA mismatch: '+str(path))
        sources.append({'path': str(path), 'sha256': expected})
    return sources


def import_helpers() -> tuple[Any, Any]:
    require(sha(COMPARISON_HELPER) == COMPARISON_HELPER_SHA, 'frozen comparison helper SHA mismatch')
    spec = importlib.util.spec_from_file_location('frozen_remaining_compare_cams', COMPARISON_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load frozen comparison helper')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require(all(callable(getattr(helper, x, None)) for x in ('canonical', 'exact_join', 'seed_for')),
            'frozen helper API differs')
    require(sha(METRIC_RUNNER) == METRIC_RUNNER_SHA and sha(RUNTIME / 'validation_core.py') == CORE_SHA,
            'frozen metric runtime SHA mismatch')
    actual_runtime = {'python_version': sys.version.split()[0],
                      'numpy_version': importlib.metadata.version('numpy'),
                      'pandas_version': importlib.metadata.version('pandas'),
                      'scikit_learn_version': importlib.metadata.version('scikit-learn')}
    require(actual_runtime == EXPECTED_RUNTIME, f'comparison runtime differs from accepted pack: {actual_runtime}')
    sys.path.insert(0, str(RUNTIME))
    import run_validation as runner
    require(Path(runner.__file__).resolve() == METRIC_RUNNER.resolve(), 'wrong frozen metric runner imported')
    boot = RUNTIME / 'fast_cluster_bootstrap.py'
    require(sha(boot) == BOOTSTRAP_SHA, 'frozen bootstrap helper SHA mismatch')
    function, imported, imported_sha = runner._load_fast_bootstrap()
    require(imported.resolve() == boot.resolve() and imported_sha == BOOTSTRAP_SHA
            and Path(function.__globals__['__file__']).resolve() == boot.resolve(),
            'dynamically loaded bootstrap helper differs')
    return helper, runner


def assert_exact_key_targets(*frames: list[dict[str, Any]]) -> None:
    """Require identical unique key/observed maps across CAMS, B, and full."""
    maps = []
    for rows in frames:
        current = {}
        for row in rows:
            key = tuple(row[k] for k in KEYS)
            require(key not in current, 'duplicate key in a comparison arm')
            current[key] = row['obs']
        maps.append(current)
    require(all(set(maps[0]) == set(other) for other in maps[1:]),
            'matched comparison arms do not have identical keys')
    require(all(all(maps[0][key] == other[key] for key in maps[0]) for other in maps[1:]),
            'matched comparison arms do not have identical observed targets')


def identity_fields(job: dict[str, Any], target: str, task: str) -> dict[str, str]:
    """Metadata required on every saved matched-prediction row."""
    return {'alias': str(job['job_id']), 'country': str(job['country']), 'qa': str(job['qa']),
            'profile': str(job['profile']), 'estimator': str(job['estimator']),
            'dataset_key': str(job['dataset_key']), 'target_policy': str(job['target_policy']),
            'target_strategy': target, 'backtransform': TARGETS[target][0], 'task': task}


def validate_pack(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(pack.get('format') == 'repair-v2-uscan-historical-b-cams-remaining-no-fit-pack-v1'
            and pack.get('status') == 'PASS' and pack.get('fit_performed') is False
            and pack.get('scores_or_predictions_performed') is False,
            'pinned file is not the remaining-CAMS no-fit pack')
    jobs = pack.get('jobs', [])
    require(len(jobs) == 28 and {str(x.get('job_id')) for x in jobs} == EXPECTED_ALIASES,
            'pack aliases differ from the expected remaining-28 roster')
    require(pack.get('dedup', {}).get('unique_fingerprints') == 14
            and pack.get('dedup', {}).get('policy_scoped_groups_prevent_cross_dataset_or_target_reuse') is True,
            'pack dedup census/scope differs')
    require(len(pack.get('dedup', {}).get('fingerprint_groups', {})) == 14,
            'pack must declare exactly 14 policy-scoped fingerprint groups')
    for job in jobs:
        observed = {}
        for split in job.get('folds', []):
            observed.setdefault(str(split.get('task')), []).append(str(split.get('fold_id')))
        require({k: tuple(sorted(v)) for k, v in observed.items()} == TASK_FOLDS,
                f'{job.get("job_id")}: task/fold census differs')
    return {str(j['job_id']): j for j in jobs}


def validate_bound_sources(pack: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, str]]:
    pack_bound = pack.get('bound_files', {})
    audit_bound = audit.get('bound_files_sha256', {})
    require(isinstance(pack_bound, dict) and isinstance(audit_bound, dict)
            and len(pack_bound) == len(audit_bound) == 243,
            'pack/postfit bound-source hash census must contain 243 files')
    normalize = lambda d: {str(Path(k).resolve(strict=True)): str(v).lower() for k, v in d.items()}
    pack_norm, audit_norm = normalize(pack_bound), normalize(audit_bound)
    require(pack_norm == audit_norm, 'postfit bound-source hashes differ from the immutable pack')
    sources = []
    for path_text, digest in sorted(pack_norm.items()):
        path = Path(path_text)
        require(re.fullmatch(r'[0-9a-f]{64}', digest) is not None and sha(path) == digest,
                f'pack-bound source SHA mismatch: {path}')
        sources.append({'path': path_text, 'sha256': digest})
    return sources


def prepare_pairs(acceptance_path: Path, acceptance_sha: str,
                  hist32_path: Path, hist32_sha: str):
    # This must remain the first function called before source, fit, or prediction reads.
    accepted, audit = postfit_gate(acceptance_path, acceptance_sha)
    historical_evidence = historical32_gate(hist32_path, hist32_sha)
    load_pandas()
    require(sha(PACK) == PACK_SHA, 'remaining-CAMS no-fit pack SHA mismatch')
    pack = read_json(PACK)
    jobs = validate_pack(pack)
    bound_sources = validate_bound_sources(pack, audit)
    run_result_path = RUN_DIR / 'run_result.json'
    run_result_sha = str(audit['run_result']['sha256']).lower()
    require(sha(run_result_path) == run_result_sha, 'accepted run_result SHA mismatch')
    helper, _ = import_helpers()
    pairs, roster = [], []
    historical: dict[str, Any] = {}
    sources = [{'path': str(PACK.resolve()), 'sha256': PACK_SHA},
               {'path': str(acceptance_path.resolve()), 'sha256': acceptance_sha.lower()},
               {'path': str(Path(accepted['postfit_audit']['path']).resolve()),
                'sha256': accepted['postfit_audit']['sha256'].lower()},
               {'path': str(run_result_path.resolve()), 'sha256': run_result_sha},
               {'path': str(hist32_path.resolve()), 'sha256': HISTORICAL32_DISPOSITION_SHA}]
    sources.extend(bound_sources)
    sources.append(historical_evidence['supplement'])
    for jid, job in jobs.items():
        old_run = Path(job['run_manifest_path']).resolve(strict=True)
        old_split = Path(job['split_manifest_path']).resolve(strict=True)
        old_path, elig_path = old_run.parent / 'oof_predictions.parquet', old_run.parent / 'eligibility.csv'
        sources.extend(verify_historical_artifacts(
            [old_run, old_split, old_path, elig_path], historical_evidence['pins']))
        split = read_json(old_split)
        eligible = {(str(s['task']), str(s['fold_id'])) for s in split.get('splits', [])
                    if int(s.get('train_row_count', 0)) > 0 and int(s.get('test_row_count', 0)) > 0}
        expected_eligible = {(t, f) for t, fs in TASK_FOLDS.items() for f in fs}
        require(eligible == expected_eligible, f'{jid}: historical split membership differs')
        eligibility = pd.read_csv(elig_path, dtype={'fold_id': 'string'})
        actual_eligible = {(str(r['task']), str(r['fold_id'])) for r in eligibility.to_dict('records')
                           if str(r['status']) == 'eligible'}
        require(actual_eligible == eligible, f'{jid}: eligibility roster differs from split manifest')
        historical[jid] = pd.read_parquet(old_path)
        sources.extend([{'path': str(old_path.resolve()), 'sha256': sha(old_path)},
                        {'path': str(old_split), 'sha256': sha(old_split)},
                        {'path': str(elig_path.resolve()), 'sha256': sha(elig_path)}])
    seen: set[tuple[str, str, str, str]] = set()
    run_dir = RUN_DIR.resolve(strict=True)
    audited_outputs = {str(Path(k).resolve()): str(v).lower()
                       for k, v in audit['output_sha256'].items()}
    for manifest_text, expected in sorted(audit['fit_manifest_sha256'].items()):
        manifest_path = Path(manifest_text).resolve(strict=True)
        require(manifest_path.is_relative_to(run_dir), 'fit manifest escaped the accepted run directory')
        require(sha(manifest_path) == str(expected).lower(), 'fit manifest differs from accepted audit')
        fm = read_json(manifest_path)
        require(fm.get('status') == 'complete' and fm.get('acceptance_sha256') == FEATURE_QA_ACCEPTANCE_SHA,
                'fit manifest status/feature-QA acceptance reference differs')
        target = str(fm.get('target_strategy'))
        require(target in TARGETS and fm.get('smear_strategy') == TARGETS[target][1]
                and fm.get('backtransform') == TARGETS[target][0], 'noncanonical target/smearing branch')
        prediction_path = Path(str(fm.get('predictions_path', ''))).resolve(strict=True)
        require(prediction_path.is_relative_to(run_dir), 'prediction file escaped accepted run directory')
        require(sha(prediction_path) == str(fm.get('predictions_sha256', '')).lower(),
                'saved prediction differs from its accepted fit manifest')
        require(audited_outputs.get(str(prediction_path)) == str(fm.get('predictions_sha256', '')).lower(),
                'saved prediction is absent/different in the postfit output hash census')
        raw = pd.read_parquet(prediction_path)
        required = set(KEYS + ['profile', 'estimator', 'observed_pm25', 'prediction_pm25',
                               'target_strategy', 'smear_strategy', 'backtransform'])
        require(required.issubset(raw.columns), 'CAMS prediction schema is incomplete')
        require(set(raw.target_strategy.astype(str)) == {target}
                and set(raw.smear_strategy.astype(str)) == {TARGETS[target][1]}
                and set(raw.backtransform.astype(str)) == {TARGETS[target][0]},
                'saved prediction target branches differ from fit manifest')
        for alias in fm.get('aliases', []):
            jid = str(alias.get('job_id'))
            require(jid in jobs and str(alias.get('qa')) == str(jobs[jid].get('qa')),
                    f'fit manifest alias differs from accepted pack: {jid}')
            job = jobs[jid]
            require(set(raw.profile.astype(str)) == {str(job['profile'])}
                    and set(raw.estimator.astype(str)) == {str(job['estimator'])},
                    f'prediction alias model identity differs: {jid}')
            for task, folds in TASK_FOLDS.items():
                task_frame = raw[raw.task.astype(str) == task]
                for fold in folds:
                    part = task_frame[task_frame.fold_id.astype(str) == fold]
                    if part.empty:
                        continue
                    key = (jid, target, task, fold)
                    require(key not in seen, f'duplicate CAMS alias/task/fold/target artifact: {key}')
                    seen.add(key)
                    item = part.copy()
                    for column in ('qa', 'profile', 'estimator', 'dataset_key', 'target_policy'):
                        item[column] = str(job.get(column, job.get('qa') if column == 'qa' else ''))
                    item['target_strategy'] = target
                    item['backtransform'] = TARGETS[target][0]
                    item['country'] = str(job['country'])
                    # Historical originals are sourced from the pinned core run/split paths in the pack.
                    old = historical[jid]
                    for arm, dest in (('B', 'pred_b'), ('full', 'pred_full')):
                        baseline = old[(old.arm.astype(str) == arm) & (old.task.astype(str) == task)
                                       & (old.fold_id.astype(str) == fold)
                                       & (old.target_strategy.astype(str) == target)
                                       & (old.backtransform.astype(str) == TARGETS[target][0])]
                        require(not baseline.empty and set(baseline.smear_strategy.astype(str)) == {TARGETS[target][1]},
                                f'{jid}: missing/noncanonical original {arm} branch')
                        canon = helper.canonical(baseline, 'obs', 'pred')
                        if arm == 'B':
                            branch_b = canon
                        else:
                            branch_full = canon
                    cams = helper.canonical(item, 'observed_pm25', 'prediction_pm25')
                    assert_exact_key_targets(cams.to_dict('records'), branch_b.to_dict('records'),
                                             branch_full.to_dict('records'))
                    joined = helper.exact_join(cams, branch_b, 'pred_b')
                    joined = helper.exact_join(joined, branch_full, 'pred_full')
                    for name, value in identity_fields(job, target, task).items():
                        joined[name] = value
                    pairs.append((job, target, task, fold, joined))
                    sources.extend([{'path': str(manifest_path), 'sha256': str(expected).lower()},
                                    {'path': str(prediction_path), 'sha256': str(fm['predictions_sha256']).lower()}])
    expected = {(jid, t, task, fold) for jid in EXPECTED_ALIASES for t in TARGETS
                for task, folds in TASK_FOLDS.items() for fold in folds}
    require(seen == expected, f'CAMS output alias/target/task/fold census differs: missing={len(expected-seen)} extra={len(seen-expected)}')
    # Consolidate by alias × target × task and verify exact key/observed row census in all folds.
    cells = {}
    for job, target, task, fold, part in pairs:
        key = (str(job['job_id']), target, task)
        cells.setdefault(key, []).append(part)
    cells = consolidate_cells(cells)
    for (jid, target, task), full in cells.items():
        require(not full.duplicated(KEYS).any(), f'duplicate paired keys: {jid}/{target}/{task}')
        roster.append({'alias': jid, 'country': jobs[jid]['country'], 'qa': jobs[jid]['qa'],
                       'profile': jobs[jid]['profile'], 'estimator': jobs[jid]['estimator'],
                       'dataset_key': jobs[jid]['dataset_key'], 'target_policy': jobs[jid]['target_policy'],
                       'target_strategy': target, 'backtransform': TARGETS[target][0], 'task': task,
                       'folds': len(TASK_FOLDS[task]), 'rows': len(full),
                       'key_sha256': frame_sha(full, KEYS),
                       'observed_sha256': frame_sha(full, KEYS + ['obs']),
                       'paired_predictions_sha256': frame_sha(full, KEYS + ['obs', 'pred', 'pred_b', 'pred_full'])})
    require(len(roster) == 224, 'remaining-CAMS pairing census is not 224 alias/target/task cells')
    source_payload = json.dumps(sorted({x['path']: x for x in sources}.values(), key=lambda x: x['path']),
                                indent=2, sort_keys=True).encode()
    proof = {'format': 'uscan-cams-remaining-matched-pairing-proof-v1',
             'status': 'exact_pairing_validated_no_metrics', 'postfit_acceptance_sha256': acceptance_sha.lower(),
             'postfit_audit_sha256': accepted['postfit_audit']['sha256'].lower(),
             'pack_manifest_sha256': PACK_SHA, 'comparison_helper_sha256': COMPARISON_HELPER_SHA,
             'aliases': sorted(EXPECTED_ALIASES), 'alias_count': 28, 'fit_groups': 14,
             'target_strategies': list(TARGETS), 'tasks_and_folds': {k: list(v) for k, v in TASK_FOLDS.items()},
             'pairing_cells': len(roster), 'same_keys_and_observed_targets': True, 'row_loss': 0,
             'metric_computation_performed': False, 'bootstrap_performed': False}
    return cells, roster, source_payload, proof, jobs


def frame_sha(frame, columns: list[str]) -> str:
    payload = frame[columns].to_csv(index=False, lineterminator='\n', float_format='%.17g').encode()
    return hashlib.sha256(payload).hexdigest()


def consolidate_cells(cells):
    load_pandas()
    return {key: pd.concat(pieces, ignore_index=True).sort_values(KEYS, kind='mergesort').reset_index(drop=True)
            for key, pieces in cells.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--protocol-only', action='store_true', help='print static contract; read no data files')
    mode.add_argument('--plan-only', action='store_true', help='exactly pair accepted historical outputs; no metrics')
    mode.add_argument('--run', action='store_true', help='compute metrics only with separate recorded acceptance')
    parser.add_argument('--postfit-acceptance', type=Path)
    parser.add_argument('--postfit-acceptance-sha')
    parser.add_argument('--historical32-disposition', type=Path)
    parser.add_argument('--historical32-disposition-sha')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run-acceptance', type=Path)
    parser.add_argument('--run-acceptance-sha')
    parser.add_argument('--pairing-proof', type=Path)
    parser.add_argument('--source-roster', type=Path)
    a = parser.parse_args()
    if a.protocol_only:
        print(json.dumps(protocol(), sort_keys=True))
        return 0
    require(a.postfit_acceptance is not None and a.postfit_acceptance_sha is not None,
            'plan/run require immutable accepted postfit reference and expected SHA')
    require(a.historical32_disposition is not None and a.historical32_disposition_sha is not None,
            'plan/run require the accepted historical32 disposition and its SHA before original OOF reads')
    require(a.out is not None, 'plan/run require --out')
    # Fail closed before checking paths, importing numerical packages, or opening source files.
    accepted, audit = postfit_gate(a.postfit_acceptance, a.postfit_acceptance_sha)
    if a.out.exists():
        raise FileExistsError(f'refusing to reuse output directory: {a.out}')
    if a.run:
        require(a.run_acceptance is not None and a.run_acceptance_sha is not None
                and a.pairing_proof is not None and a.source_roster is not None,
                '--run requires separate root metric gate, pairing proof, and source roster')
        gate_path = a.run_acceptance.resolve(strict=True)
        require(sha(gate_path).lower() == a.run_acceptance_sha.lower(), 'metric run acceptance SHA mismatch')
        gate = read_json(gate_path)
        require(gate.get('format') == 'uscan-cams-remaining-comparison-run-acceptance-v1'
                and gate.get('status') == 'accepted_uscan_cams_remaining_comparison'
                and gate.get('comparison_authorized') is True
                and gate.get('driver_sha256') == sha(Path(__file__))
                and gate.get('postfit_acceptance_sha256') == a.postfit_acceptance_sha.lower()
                and gate.get('pack_manifest_sha256') == PACK_SHA
                and gate.get('comparison_helper_sha256') == COMPARISON_HELPER_SHA
                and gate.get('bootstrap_helper_sha256') == BOOTSTRAP_SHA
                and gate.get('pairing_proof_sha256') == sha(a.pairing_proof.resolve(strict=True))
                and gate.get('source_roster_sha256') == sha(a.source_roster.resolve(strict=True)),
                'separate root-reviewed remaining-CAMS metric authorization missing/mismatched')
    cells, roster, source_payload, proof, jobs = prepare_pairs(
        a.postfit_acceptance, a.postfit_acceptance_sha,
        a.historical32_disposition, a.historical32_disposition_sha)
    source_sha = hashlib.sha256(source_payload).hexdigest()
    if a.plan_only:
        load_pandas()
        a.out.mkdir(parents=True, exist_ok=False)
        proof['source_roster_sha256'] = source_sha
        (a.out / 'pairing_roster.csv').write_text(pd.DataFrame(roster).to_csv(index=False), encoding='utf-8')
        (a.out / 'source_roster.json').write_bytes(source_payload)
        (a.out / 'pairing_proof.json').write_text(json.dumps(proof, indent=2, sort_keys=True), encoding='utf-8')
        print(json.dumps({'status': proof['status'], 'pairing_cells': len(roster),
                          'metrics_computed': False, 'source_roster_sha256': source_sha}, sort_keys=True))
        return 0
    proof_path = a.pairing_proof.resolve(strict=True)
    source_path = a.source_roster.resolve(strict=True)
    approved = read_json(proof_path)
    require(approved.get('format') == proof['format'] and approved.get('status') == proof['status']
            and approved.get('source_roster_sha256') == source_sha
            and gate.get('source_roster_sha256') == source_sha
            and gate.get('pairing_roster_sha256') == sha(proof_path.parent / 'pairing_roster.csv')
            and sha(source_path) == source_sha, 'current plan proof/roster differs from reviewed plan')
    helper, runner = import_helpers()
    metric_rows, delta_rows = [], []
    load_pandas()
    for (jid, target, task), part in sorted(cells.items()):
        job = jobs[jid]
        y = part.obs.to_numpy(float)
        labels = (str(job['country']), str(job['qa']), str(job['profile']), str(job['estimator']),
                  target, TARGETS[target][0], task)
        for arm, pred_col in (('B+CAMS', 'pred'), ('B', 'pred_b'), ('full', 'pred_full')):
            row = {'alias': jid, 'country': labels[0], 'qa': labels[1], 'profile': labels[2],
                   'estimator': labels[3], 'target_strategy': target, 'backtransform': labels[5], 'task': task}
            row['arm'] = arm
            row.update(runner._metric(y, part[pred_col].to_numpy(float)))
            row.update({'sites': int(part.site_id.nunique()), 'years': ','.join(map(str, sorted(part.year.unique())))})
            metric_rows.append(row)
        for arm, pred_col in (('B', 'pred_b'), ('full', 'pred_full')):
            seed = helper.seed_for(labels + (arm,))
            boot = runner._paired_bootstrap(part.site_id.to_numpy(str), y, part.pred.to_numpy(float),
                                            part[pred_col].to_numpy(float), BOOTSTRAP_REPLICATES,
                                            seed, engine='cluster_stats')
            a_metric, b_metric = runner._metric(y, part.pred.to_numpy(float)), runner._metric(y, part[pred_col].to_numpy(float))
            row = {'alias': jid, 'country': labels[0], 'qa': labels[1], 'profile': labels[2],
                   'estimator': labels[3], 'target_strategy': target, 'backtransform': labels[5], 'task': task}
            row.update({'contrast': f'B+CAMS minus {arm}', 'n': len(part), 'sites': int(part.site_id.nunique()),
                        'bootstrap_seed': seed, 'bootstrap_replicates': BOOTSTRAP_REPLICATES})
            for metric in METRICS:
                row[f'delta_{metric}'] = a_metric[metric] - b_metric[metric]
                row[f'delta_{metric}_ci_low'] = boot[f'delta_{metric}_lo']
                row[f'delta_{metric}_ci_high'] = boot[f'delta_{metric}_hi']
            row.update(runner._uncertainty_scope(task, int(part.fold_id.nunique())))
            row['inference_scope'] = protocol()['scope_note']
            delta_rows.append(row)
    require(len(metric_rows) == 672 and len(delta_rows) == 448, 'metric/contrast census differs from frozen protocol')
    a.out.mkdir(parents=True, exist_ok=False)
    all_pairs = pd.concat(list(cells.values()), ignore_index=True)
    all_pairs.to_parquet(a.out / 'matched_predictions.parquet', index=False)
    pd.DataFrame(metric_rows).to_csv(a.out / 'matched_metrics.csv', index=False)
    pd.DataFrame(delta_rows).to_csv(a.out / 'paired_deltas.csv', index=False)
    pd.DataFrame(roster).to_csv(a.out / 'pairing_roster.csv', index=False)
    manifest = {'status': 'complete', 'metrics_computed': True, 'analysis_class': 'secondary_exploratory',
                'driver_sha256': sha(Path(__file__)), 'postfit_acceptance_sha256': a.postfit_acceptance_sha.lower(),
                'pack_manifest_sha256': PACK_SHA, 'comparison_helper_sha256': COMPARISON_HELPER_SHA,
                'bootstrap': {'engine': 'cluster_stats', 'replicates': BOOTSTRAP_REPLICATES,
                              'seed_rule': 'frozen seed_for(country, QA, profile, estimator, target, task, control arm)'},
                'metric_direction': protocol()['metric_direction'], 'source_roster_sha256': source_sha,
                'pairing_cells': len(roster), 'metric_rows': len(metric_rows), 'paired_contrast_rows': len(delta_rows),
                'metrics_computed_before_gate': False, 'fits_performed': False,
                'future_year_data_or_scores': False}
    manifest['output_sha256'] = {name: sha(a.out / name) for name in
                                 ('matched_predictions.parquet', 'matched_metrics.csv', 'paired_deltas.csv', 'pairing_roster.csv')}
    (a.out / 'completion_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return 0


pd = None


if __name__ == '__main__':
    raise SystemExit(main())
