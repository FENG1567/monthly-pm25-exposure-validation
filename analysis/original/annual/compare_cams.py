#!/usr/bin/env python3
"""Acceptance-gated matched comparison of historical B+CAMS, B, and full OOF."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

BASE = Path('/data/admin/fengshuo/cand1_repair_v2_20261112')
FIT_ROOT = BASE / 'validation/global_current_fit_20261114'
RUN_ID = 'run_20261114T024137Z_4192412'
ACCEPTANCE_SHA = '0fd47c7640c9add462674aee6d627b7578a63fede37afe88eb5c6be0ac7a11dc'
AUDIT_SHA = '452dacae19743082ae623b6374533af620bb9aced7e6677dc3cd6301350d7ca1'
RUNNER_SHA = '05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c'
CORE_SHA = '027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157'
KEYS = ['country', 'task', 'fold_id', 'row_id', 'site_id', 'year', 'month']
TARGETS = {'log_smear': 'smearing', 'original_scale': 'model_native'}


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


def verify_hash_map(path: Path, expected_sha: str) -> dict[str, str]:
    if sha(path) != expected_sha.lower():
        raise ValueError('source-hash map SHA256 mismatch')
    value = read_json(path)
    if value.get('format') != 'global-current-cams-comparison-source-hashes-v1':
        raise ValueError('unsupported source-hash map format')
    files = value.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('source-hash map requires files={absolute_path: sha256}')
    checked = {}
    for name, expected in files.items():
        p = Path(name).resolve(strict=True)
        if not p.is_absolute() or not isinstance(expected, str) or sha(p) != expected.lower():
            raise ValueError(f'source hash mismatch: {p}')
        checked[str(p)] = expected.lower()
    return checked


def require_hash(path: Path, hashes: dict[str, str]) -> Path:
    p = path.resolve(strict=True)
    if str(p) not in hashes or sha(p) != hashes[str(p)]:
        raise ValueError(f'input is missing from verified source-hash map: {p}')
    return p


def _family(task: str) -> str | None:
    if task.startswith('T1_station'):
        return 'T1'
    if task == 'T2_spatial':
        return 'T2'
    if task.startswith('T2_spatial_buffer'):
        return 'T2buffer'
    if task.startswith('T3_year_'):
        return 'T3'
    return None


def seed_for(labels: tuple[str, ...]) -> int:
    raw = ('cams-matched-station-bootstrap-v1\0' + '\0'.join(labels)).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def canonical(frame: pd.DataFrame, obs: str, pred: str) -> pd.DataFrame:
    d = frame.copy()
    if d[KEYS].isna().any().any():
        raise ValueError('missing identity key before string normalization')
    d['country'] = d.country.astype(str).str.upper()
    d['task'] = d.task.astype(str)
    d['fold_id'] = d.fold_id.astype(str)
    d['row_id'] = pd.to_numeric(d.row_id, errors='raise').astype('int64')
    d['site_id'] = d.site_id.astype(str)
    d['year'] = pd.to_numeric(d.year, errors='raise').astype('int64')
    d['month'] = pd.to_numeric(d.month, errors='raise').astype('int64')
    d['obs'] = pd.to_numeric(d[obs], errors='raise').astype('float64')
    d['pred'] = pd.to_numeric(d[pred], errors='raise').astype('float64')
    if d[KEYS].isna().any().any() or not np.isfinite(d[['obs', 'pred']].to_numpy()).all():
        raise ValueError('matched predictions contain missing keys or non-finite values')
    if d.duplicated(KEYS).any():
        raise ValueError('duplicate held-out identity keys')
    return d[KEYS + ['obs', 'pred']].sort_values(KEYS, kind='mergesort').reset_index(drop=True)


def exact_join(left: pd.DataFrame, right: pd.DataFrame, pred_name: str) -> pd.DataFrame:
    if len(left) != len(right) or not left[KEYS].equals(right[KEYS]):
        raise ValueError('strict 1:1 held-out keys/order mismatch; row drops are prohibited')
    if not np.array_equal(left.obs.to_numpy(), right.obs.to_numpy()):
        raise ValueError('observed raw targets differ across matched sources')
    return left.assign(**{pred_name: right.pred.to_numpy()})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-acceptance', type=Path, required=True)
    p.add_argument('--fit-acceptance-sha', required=True)
    p.add_argument('--audit-result', type=Path, required=True)
    p.add_argument('--source-hash-map', type=Path, required=True)
    p.add_argument('--source-hash-map-sha', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    p.add_argument('--run-acceptance', type=Path)
    p.add_argument('--run-acceptance-sha')
    a = p.parse_args()

    if a.fit_acceptance_sha.lower() != ACCEPTANCE_SHA or sha(a.fit_acceptance.resolve(strict=True)) != ACCEPTANCE_SHA:
        raise ValueError('fit acceptance differs from frozen accepted fit')
    if sha(a.audit_result.resolve(strict=True)) != AUDIT_SHA:
        raise ValueError('post-fit audit differs from frozen PASS audit')
    if a.run_dir.resolve(strict=True) != (FIT_ROOT / 'runs' / RUN_ID).resolve(strict=True):
        raise ValueError('run directory differs from accepted completed run')
    hashes = verify_hash_map(a.source_hash_map.resolve(strict=True), a.source_hash_map_sha)
    for path in (a.fit_acceptance, a.audit_result, Path(__file__)):
        require_hash(path, hashes)
    if a.run:
        if not a.run_acceptance or not a.run_acceptance_sha:
            raise ValueError('--run requires separate root-reviewed --run-acceptance and --run-acceptance-sha')
        if sha(a.run_acceptance.resolve(strict=True)) != a.run_acceptance_sha.lower():
            raise ValueError('run acceptance SHA mismatch')
        gate = read_json(a.run_acceptance)
        if (gate.get('format') != 'global-current-cams-comparison-acceptance-v1'
                or gate.get('status') != 'accepted_global_current_cams_comparison'
                or gate.get('comparison_authorized') is not True
                or gate.get('fit_acceptance_sha256') != ACCEPTANCE_SHA
                or gate.get('audit_result_sha256') != AUDIT_SHA
                or gate.get('source_hash_map_sha256') != a.source_hash_map_sha.lower()):
            raise ValueError('separate root-reviewed comparison acceptance is absent or mismatched')
        # The separate gate binds this already frozen map by digest; including
        # the gate inside that map would create a circular digest dependency.
        a.out.mkdir(parents=True, exist_ok=False)
        with (a.out / '.comparison_claim').open('x') as claim:
            json.dump({'pid':os.getpid(),'driver_sha256':sha(Path(__file__)),'acceptance_sha256':a.run_acceptance_sha.lower()},claim)

    acceptance = read_json(a.fit_acceptance)
    audit = read_json(a.audit_result)
    run_result = read_json(a.run_dir / 'run_result.json')
    require_hash(a.run_dir / 'run_result.json', hashes)
    if (audit.get('status') != 'PASS' or audit.get('model_count') != 1240
            or audit.get('job_labels') != 80 or audit.get('qa_dedup_groups') != 40
            or audit.get('guard_rows_no_fit') != 24 or audit.get('prediction_rows_replayed') != 312464
            or audit.get('max_absolute_replay_difference') != 0):
        raise ValueError('post-fit replay audit census is not the accepted PASS census')
    if run_result.get('status') != 'complete' or run_result.get('completed_fits') != 1240:
        raise ValueError('accepted CAMS fit run is incomplete')

    runtime = Path(acceptance['runtime_dir']).resolve(strict=True)
    runner_path, core_path = runtime / 'run_validation.py', runtime / 'validation_core.py'
    boot_path = runtime / 'fast_cluster_bootstrap.py'
    for src, expected in ((runner_path, RUNNER_SHA), (core_path, CORE_SHA)):
        if sha(require_hash(src, hashes)) != expected:
            raise ValueError(f'frozen helper changed: {src}')
    require_hash(boot_path, hashes)
    versions = {'python_version': '.'.join(map(str, sys.version_info[:3])),
                'numpy_version': importlib.metadata.version('numpy'),
                'pandas_version': importlib.metadata.version('pandas'),
                'scikit_learn_version': importlib.metadata.version('scikit-learn')}
    if versions != acceptance.get('expected_runtime_versions'):
        raise ValueError('comparison runtime versions differ from accepted frozen runtime')

    report_path = Path(acceptance['preflight_report_path']).resolve(strict=True)
    require_hash(report_path, hashes)
    report = read_json(report_path)
    jobs = {str(j['job_id']): j for j in report.get('jobs', [])}
    if len(jobs) != 80 or len(report.get('guard_rows', [])) != 24:
        raise ValueError('accepted job/guard census mismatch')
    sys.path.insert(0, str(runtime))
    import run_validation as runner
    if Path(runner.__file__).resolve() != runner_path:
        raise ValueError('wrong frozen metric/bootstrap helper imported')

    run_artifacts = {str(Path(x).resolve()) for x in run_result.get('artifacts', [])}
    audit_artifacts = audit.get('artifacts', [])
    if len(audit_artifacts) != 1240 or {str(Path(x['fit_manifest']).resolve()) for x in audit_artifacts} != run_artifacts:
        raise ValueError('audited fit manifest inventory differs from complete run')
    new_rows: dict[tuple[str, str], pd.DataFrame] = {}
    artifact_census = []
    for rec in audit_artifacts:
        mp = require_hash(Path(rec['fit_manifest']), hashes)
        fm = read_json(mp)
        pp = require_hash(Path(fm['predictions_path']), hashes)
        if sha(pp) != rec['predictions_sha256']:
            raise ValueError(f'prediction bytes differ from post-fit audit: {pp}')
        raw = pd.read_parquet(pp)
        if not {'country','task','fold_id','row_id','site_id','year','month','observed_pm25','prediction_pm25',
                'target_strategy','backtransform','smear_strategy'}.issubset(raw.columns):
            raise ValueError(f'new CAMS prediction schema incomplete: {pp}')
        target = str(fm['target_strategy'])
        expected_smear = 'crossfit_station' if target == 'log_smear' else 'not_applicable'
        if (fm.get('smear_strategy') != expected_smear or raw.target_strategy.astype(str).nunique() != 1
                or raw.smear_strategy.astype(str).nunique() != 1 or not raw.smear_strategy.astype(str).eq(expected_smear).all()):
            raise ValueError('noncanonical new-fit smearing branch')
        base = canonical(raw, 'observed_pm25', 'prediction_pm25')
        for alias in fm.get('aliases', []):
            jid, qa = str(alias['job_id']), str(alias['qa'])
            if jid not in jobs:
                raise ValueError(f'unknown QA alias in fit manifest: {jid}')
            key = (jid, target)
            lab = jobs[jid]
            part = base.copy()
            part['qa'], part['profile'], part['estimator'] = qa, str(lab['profile']), str(lab['estimator'])
            part['target_strategy'], part['backtransform'] = target, str(fm['backtransform'])
            slot = new_rows.setdefault(key, [])
            slot.append(part)
        artifact_census.append({'fit_manifest': str(mp), 'fit_manifest_sha256': sha(mp),
                                'predictions': str(pp), 'predictions_sha256': sha(pp)})

    eligible_rows, matched, metric_rows, delta_rows = [], [], [], []
    original_eligibility_rows=0
    for jid, job in sorted(jobs.items()):
        inp = acceptance['job_inputs'][jid]
        original_path = Path(inp['run_manifest']).resolve(strict=True).parent / 'oof_predictions.parquet'
        eligibility_path = original_path.with_name('eligibility.csv')
        split_path = Path(inp['split_manifest']).resolve(strict=True)
        original_path, eligibility_path, split_path = [require_hash(x, hashes) for x in (original_path, eligibility_path, split_path)]
        old = pd.read_parquet(original_path)
        elig = pd.read_csv(eligibility_path, dtype={'fold_id':'string'})
        split = read_json(split_path)
        if not {'task','fold_id','status'}.issubset(elig.columns):
            raise ValueError(f'original eligibility schema incomplete: {eligibility_path}')
        original_eligibility_rows+=len(elig)
        for ordinal,row in enumerate(elig.to_dict('records')):
            eligible_rows.append({'job_id': jid, 'country': job['country'], 'qa': job['qa'],
                                  'profile': job['profile'], 'estimator': job['estimator'], **row,
                                  'source_eligibility_file':str(eligibility_path),'source_row_ordinal':ordinal})
        eligible_folds = {(str(s['task']), str(s['fold_id'])) for s in split.get('splits', [])
                          if int(s.get('train_row_count', 0)) > 0 and int(s.get('test_row_count', 0)) > 0}
        if not eligible_folds:
            raise ValueError(f'no accepted eligible folds: {jid}')
        for task, fold in eligible_folds:
            er = elig[(elig.task.astype(str) == task) & (elig.fold_id.astype(str) == fold)]
            if len(er) != 1 or str(er.iloc[0].status) != 'eligible':
                raise ValueError(f'saved split is not explicitly eligible in source table: {jid}/{task}/{fold}')
        for target, back in TARGETS.items():
            aliases = new_rows.get((jid, target), [])
            if not aliases:
                raise ValueError(f'missing CAMS predictions for logical job/target: {jid}/{target}')
            cams_all = pd.concat(aliases, ignore_index=True)
            if cams_all.task.astype(str).nunique() == 0:
                raise ValueError('empty CAMS prediction branch')
            for task, fold in sorted(eligible_folds):
                cams_raw = cams_all[(cams_all.task.astype(str) == task) & (cams_all.fold_id.astype(str) == fold)]
                if cams_raw.empty:
                    raise ValueError(f'missing CAMS task/fold: {jid}/{target}/{task}/{fold}')
                old_task = old[(old.task.astype(str) == task) & (old.fold_id.astype(str) == fold)
                               & (old.target_strategy.astype(str) == target)
                               & (old.backtransform.astype(str) == back)]
                if not {'arm','smear_strategy','obs','pred'}.issubset(old_task.columns):
                    raise ValueError(f'original OOF schema incomplete: {original_path}')
                for arm, pred_col in (('B', 'pred_b'), ('full', 'pred_full')):
                    part = old_task[old_task.arm.astype(str) == arm]
                    original_smear='crossfit_station' if target=='log_smear' else 'not_applicable'
                    if part.empty or set(part.smear_strategy.astype(str)) != {original_smear}:
                        raise ValueError(f'noncanonical/missing original {arm} branch: {jid}/{target}/{task}/{fold}')
                    baseline = canonical(part, 'obs', 'pred')
                    cams = canonical(cams_raw, 'obs', 'pred')
                    if arm == 'B':
                        joined = exact_join(cams, baseline, pred_col)
                    else:
                        joined = exact_join(joined, baseline, pred_col)
                labels = (str(job['country']), str(job['qa']), str(job['profile']), str(job['estimator']),
                          target, back, task)
                joined['country'], joined['qa'], joined['profile'], joined['estimator'] = labels[:4]
                joined['target_strategy'], joined['backtransform'], joined['arm'] = target, back, 'matched'
                joined['fold_id'] = joined.fold_id.astype(str)
                matched.append(joined)

    for guard in report['guard_rows']:
        eligible_rows.append({**guard,'qa':guard.get('qa_label'),'profile':guard.get('profile_label'),
                              'estimator':guard.get('estimator_label'),'task':'all','fold_id':'not_applicable',
                              'status':'guard_only_no_fit','reason':'Accepted original job has no eligible OOF; retained without CAMS fitting'})
    pairs = pd.concat(matched, ignore_index=True)
    groupcols = ['country','qa','profile','estimator','target_strategy','backtransform','task']
    if len(pairs)!=624928 or len(pairs[groupcols].drop_duplicates())!=624:
        raise ValueError('matched row/task census differs from frozen complete OOF support')
    if sum(x.get('status')=='guard_only_no_fit' for x in eligible_rows)!=24:
        raise ValueError('24 accepted guard rows were not retained')
    if len(eligible_rows)!=original_eligibility_rows+24:
        raise ValueError('source eligibility rows were dropped')
    if a.plan_only:
        print(json.dumps({'status':'plan_validated_no_metrics','logical_jobs':len(jobs),
                          'matched_rows':len(pairs),'matched_task_cells':624,'guard_rows_retained':24,
                          'original_eligibility_rows':original_eligibility_rows,'eligibility_rows_retained':len(eligible_rows),
                          'metric_computation_performed':False}, sort_keys=True))
        return 0
    for labels, part in pairs.groupby(groupcols, dropna=False, sort=True):
        y = part.obs.to_numpy(float)
        for arm, pred in (('B+CAMS', 'pred'), ('B', 'pred_b'), ('full', 'pred_full')):
            row = dict(zip(groupcols, labels)); row['arm'] = arm
            row.update(runner._metric(y, part[pred].to_numpy(float)))
            row.update({'sites': int(part.site_id.nunique()), 'years': ','.join(map(str, sorted(part.year.unique())))})
            metric_rows.append(row)
        for arm, pred in (('B', 'pred_b'), ('full', 'pred_full')):
            seed = seed_for(tuple(map(str, labels)) + (arm,))
            boot = runner._paired_bootstrap(part.site_id.to_numpy(str), y, part.pred.to_numpy(float),
                                            part[pred].to_numpy(float), 300, seed, engine='cluster_stats')
            a_metric, b_metric = runner._metric(y, part.pred.to_numpy(float)), runner._metric(y, part[pred].to_numpy(float))
            row = dict(zip(groupcols, labels)); row.update({'contrast': f'B+CAMS minus {arm}', 'n': len(part),
                'sites': int(part.site_id.nunique()), 'bootstrap_seed': seed, 'bootstrap_replicates': 300})
            for m in ('rmse','mae','bias','r2'):
                row[f'delta_{m}'] = a_metric[m] - b_metric[m]
                row[f'delta_{m}_ci_low'] = boot[f'delta_{m}_lo']
                row[f'delta_{m}_ci_high'] = boot[f'delta_{m}_hi']
            row.update(runner._uncertainty_scope(str(labels[-1]), int(part.fold_id.nunique())))
            row['inference_scope'] = 'exploratory station-cluster interval conditional on observed folds and a single held-out year where applicable; no pooled-global significance'
            delta_rows.append(row)

    if not a.plan_only:
        if len(metric_rows)!=1872 or len(delta_rows)!=1248:
            raise ValueError('metric/contrast census differs from the complete 624 task cells')
        pairs.to_parquet(a.out / 'matched_predictions.parquet', index=False)
        pd.DataFrame(metric_rows).to_csv(a.out / 'matched_metrics.csv', index=False)
        pd.DataFrame(delta_rows).to_csv(a.out / 'paired_deltas.csv', index=False)
        pd.DataFrame(eligible_rows).to_csv(a.out / 'eligibility.csv', index=False)
        manifest = {'status': 'complete', 'comparison': 'secondary exploratory historical CAMS comparison',
            'locked_26_cell_family_extended': False, 'metrics_computed': True,
            'fit_acceptance_sha256': ACCEPTANCE_SHA, 'audit_result_sha256': AUDIT_SHA,
            'source_hash_map_sha256': a.source_hash_map_sha.lower(), 'run_id': RUN_ID,
            'new_fit_count_expected_actual': [1240, len(artifact_census)],
            'logical_jobs_expected_actual': [80, len(jobs)], 'qa_dedup_groups_expected': 40,
            'guard_rows_retained_no_fit': 24, 'pair_rows': len(pairs),
            'original_eligibility_rows':original_eligibility_rows,'eligibility_rows_retained':len(eligible_rows),
            'metric_rows': len(metric_rows), 'paired_delta_rows': len(delta_rows),
            'bootstrap': {'engine': 'cluster_stats', 'replicates': 300,
                          'seed_rule': 'first 8 hex digits of SHA256(domain + NUL-separated comparison labels)'},
            'source_hash_map_path':str(a.source_hash_map.resolve()),'driver_sha256':sha(Path(__file__)),
            'output_sha256':{name:sha(a.out/name) for name in ['matched_predictions.parquet','matched_metrics.csv','paired_deltas.csv','eligibility.csv']},
            'limitations': ['intervals condition on observed spatial folds and a single held-out year where applicable',
                            'no pooled-global significance', 'CAMS is a secondary exploratory covariate comparator']}
        (a.out / 'completion_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    else:
        print(json.dumps({'status':'plan_validated_no_metrics','logical_jobs':len(jobs),
                          'matched_rows':len(pairs),'metric_computation_performed':False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    raise SystemExit(main())
