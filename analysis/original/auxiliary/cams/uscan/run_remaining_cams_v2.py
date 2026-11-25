#!/usr/bin/env python3
"""Check the pinned remaining USA/Canada historical B+CAMS pack; fits need a release approval."""
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

BASE = Path('/external/project')
V = BASE / 'validation'
RUNTIME = V / 'final_model_runtime_v4_20261114'
SNAPSHOT = RUNTIME / 'source_snapshot/validation'
PACK_ROOT = V / 'uscan_cams_remaining_v1_20261115/pack'
PACK = PACK_ROOT / 'no_fit_pack_manifest.json'
PACK_SHA = '7d2d2c7f6ef4918ab5e99357e3c713cc8e4f1833c7cc85f6283d3ba8583feba2'
FIT_HELPER = V / 'global_current_fit_20261114/fit_global.py'
FIT_HELPER_SHA = 'd4e3110900d291e9d13966dd88d5539cbbaea2c51c69c2132c51ba080e720a12'
MATRIX_SHA = '5a341aee2eb0c4f018e6510580d44876861d2f55bd9cde09adef0f89225f330f'
CORE_SHA = '027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157'
RUNNER_SHA = '05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c'
ACCEPTANCE_SHA = 'f4e32b4b9b40c1938da90d3329694dfc099668fa655a1ac744c7d09ff8105351'
CANDIDATE_SHA = '1c8e49668f7225ef9526b7ebcbd06af3ba8bd34efd9cab8f8cc0eb5b3f657048'
TARGET_CONTRACT_SHA = '9a4dab79615cc217aca53022094713b3b4c1044854dcc8af972a42c2d3c7cfe0'
OUTPUT_ROOT = V / 'uscan_cams_fit_remaining_v1_20261115'
COMPLETED = {
    'usa_hist_strict_with_met_hgb', 'usa_hist_strict_no_met_hgb',
    'usa_hist_sensitivity_with_met_hgb', 'usa_hist_sensitivity_no_met_hgb',
}
GEO = ['lat_f', 'lon_f', 'lpop', 'month_sin', 'month_cos', 'month_sin2', 'month_cos2']
MET = ['blh', 'wind10', 'tp', 't2m', 'rh', 'vent']
CAM = 'cams_pm25_ug_m3'
COMMON_CONFIG = {
    'max_iter': 400, 'learning_rate': 0.06, 'max_leaf_nodes': 31,
    'min_samples_leaf': 25, 'l2_regularization': 1.0, 'early_stopping': True,
    'validation_fraction': 0.15, 'random_state': 0, 'ridge_alpha': 1.0,
}
STRATEGIES = {
    'smear': ['crossfit_station'], 'splits': 3,
    'target': ['log_smear', 'original_scale'],
    'primary_backtransform': 'smear*exp(z)-0.1',
    'sensitivity_backtransform': 'exp(z)-0.1',
}
FIT_SPECS = [
    ['log_smear', 'crossfit_station', 'smearing'],
    ['original_scale', 'not_applicable', 'model_native'],
]
EXPECTED_RUNTIME = {
    'python_version': '3.11.16', 'numpy_version': '2.4.6',
    'pandas_version': '2.3.3', 'scikit_learn_version': '1.9.1',
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, default=str, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(value, dict), f'expected JSON object: {path}')
    return value


def load_frozen(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f'cannot load frozen module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config_for(kind: str) -> dict[str, Any]:
    require(kind in {'hgb', 'ridge'}, f'unsupported estimator {kind}')
    return {**COMMON_CONFIG, 'estimator_kind': kind}


def pin_job_paths(job: dict[str, Any], bound: dict[str, str]) -> None:
    for path_key, sha_key in (
        ('frame_path', 'frame_sha256'), ('membership_path', 'membership_sha256'),
        ('split_manifest_path', 'split_manifest_sha256'), ('run_manifest_path', 'run_manifest_sha256'),
        ('schema_manifest_path', 'schema_manifest_sha256'), ('source_manifest_path', 'source_manifest_sha256'),
        ('assignment_path', 'assignment_sha256'),
    ):
        path, digest = str(job.get(path_key, '')), str(job.get(sha_key, '')).lower()
        require(Path(path).is_absolute() and re.fullmatch(r'[0-9a-f]{64}', digest or '')
                and bound.get(path) == digest, f'{job.get("job_id")}: {path_key} is not bound')


def validate(args: argparse.Namespace) -> dict[str, Any]:
    require(not sys.flags.optimize, 'optimized Python is prohibited')
    require(sha(PACK) == PACK_SHA, 'no-fit pack manifest SHA mismatch')
    report = read_json(PACK)
    require(report.get('format') == 'repair-v2-uscan-historical-b-cams-remaining-no-fit-pack-v1'
            and report.get('status') == 'PASS' and report.get('fit_performed') is False
            and report.get('scores_or_predictions_performed') is False,
            'manifest is not the completed no-fit remaining pack')
    require(report.get('candidate') == {'path': str(V / 'uscan_final_feature_candidates_v2_20261115/final_candidate.json'),
            'sha256': CANDIDATE_SHA}, 'candidate pin differs')
    require(report.get('acceptance', {}).get('path') == str(V / 'uscan_actual_feature_target_qa_v1_20261115/feature_target_qa_acceptance.json')
            and report.get('acceptance', {}).get('sha256') == ACCEPTANCE_SHA
            and report.get('acceptance', {}).get('accepted_artifacts_checked_by_exact_path_and_sha') is True,
            'feature/target acceptance pin differs')
    require(report.get('target_identity_contract', {}).get('sha256') == TARGET_CONTRACT_SHA,
            'target identity contract pin differs')
    require(report.get('historical_job_matrix', {}).get('sha256') == MATRIX_SHA
            and report.get('historical_job_matrix', {}).get('remaining_alias_count') == 28
            and set(report.get('historical_job_matrix', {}).get('excluded_completed_aliases', [])) == COMPLETED,
            'historical job matrix/exclusion scope differs')
    require(report.get('core_source_sha256') == CORE_SHA and report.get('runner_source_sha256') == RUNNER_SHA
            and report.get('fit_helper_path') == str(FIT_HELPER)
            and report.get('fit_helper_sha256') == FIT_HELPER_SHA,
            'frozen source/helper pins differ')
    require(report.get('runtime_versions') == EXPECTED_RUNTIME, 'pack runtime differs')
    contract = report.get('fit_contract', {})
    expected_configs = {k: config_for(k) for k in ('hgb', 'ridge')}
    require(contract.get('model_config_by_estimator') == expected_configs
            and contract.get('base_features') == {'with_met': [*GEO, *MET, CAM], 'no_met': [*GEO, CAM]}
            and contract.get('smear_and_target_strategies') == STRATEGIES
            and contract.get('fit_specs') == FIT_SPECS,
            'model, feature, target, or fit strategy contract differs')
    require(report.get('cams', {}).get('period') == [2018, 2022], 'CAMS period is not historical 2018-2022')

    # Rehash every declared byte before importing numerical packages or the fit helper.
    bound = report.get('bound_files')
    require(isinstance(bound, dict) and len(bound) >= 200, 'complete bound_files map is absent')
    for raw, expected in bound.items():
        p, digest = Path(raw), str(expected).lower()
        require(p.is_absolute() and re.fullmatch(r'[0-9a-f]{64}', digest or ''), f'invalid bound file entry: {raw}')
        require(sha(p) == digest, f'bound file SHA mismatch: {raw}')
    require(sha(FIT_HELPER) == FIT_HELPER_SHA, 'reviewed fit helper bytes changed')
    core_path, runner_path = SNAPSHOT / 'validation_core.py', SNAPSHOT / 'run_validation.py'
    require(sha(core_path) == CORE_SHA and sha(runner_path) == RUNNER_SHA, 'frozen core/runner bytes changed')

    candidate_path = Path(report['candidate']['path'])
    acceptance_path = Path(report['acceptance']['path'])
    candidate, acceptance = read_json(candidate_path), read_json(acceptance_path)
    require(acceptance.get('decision') == 'accepted' and acceptance.get('independent_qa_pass') is True
            and acceptance.get('candidate_manifest_sha256') == CANDIDATE_SHA,
            'candidate acceptance decision does not bind the pinned candidate')
    accepted = {str(Path(x['path']).resolve(strict=True)): str(x['sha256']).lower()
                for x in acceptance.get('accepted_artifacts', []) if isinstance(x, dict) and x.get('path') and x.get('sha256')}
    require(accepted.get(str(candidate_path.resolve())) == CANDIDATE_SHA, 'candidate is not in accepted artifacts')
    target_path = Path(report['target_identity_contract']['path']).resolve(strict=True)
    require(accepted.get(str(target_path)) == TARGET_CONTRACT_SHA, 'target contract is not in accepted artifacts')
    matrix_path = Path(report['historical_job_matrix']['path'])
    require(sha(matrix_path) == MATRIX_SHA, 'historical matrix bytes changed')
    matrix = read_json(matrix_path)
    matrix_jobs = [x for kind in ('hgb', 'ridge') for x in matrix.get('estimators', {}).get(kind, {}).get('jobs', [])]
    require(len(matrix_jobs) == 32 and len({x.get('job_id') for x in matrix_jobs}) == 32,
            'historical matrix is not the pinned 32-job roster')
    removed = [x for x in matrix_jobs if x.get('job_id') in COMPLETED]
    require(len(removed) == 4 and all(x.get('dataset_key') == 'usa_hist' and x.get('country') == 'USA'
            and x.get('estimator_kind') == 'hgb' for x in removed), 'completed USA/HGB exclusion rows differ')
    expected_ids = [str(x['job_id']) for x in matrix_jobs if x.get('job_id') not in COMPLETED]
    jobs = report.get('jobs')
    require(isinstance(jobs, list) and [str(x.get('job_id')) for x in jobs] == expected_ids
            and report.get('dedup', {}).get('all_aliases') == expected_ids,
            'remaining 28 job aliases/order differ from the matrix')
    statuses = report.get('controller_job_statuses', {})
    require(set(statuses) == set(expected_ids), 'controller receipts do not cover all remaining jobs')
    for job_id, status in statuses.items():
        require(status.get('job_id') == job_id and status.get('status') == 'succeeded'
                and status.get('returncode') == 0, f'core job has not succeeded: {job_id}')

    for job in jobs:
        job_id, country, estimator = str(job['job_id']), str(job['country']), str(job['estimator'])
        profile, qa = str(job['profile']), str(job['qa'])
        require(country in {'USA', 'CAN'} and estimator in {'hgb', 'ridge'} and profile in {'with_met', 'no_met'}
                and qa in {'strict', 'sensitivity'}, f'{job_id}: invalid job identity')
        ds = candidate.get('datasets', {}).get(job.get('dataset_key'), {})
        require(ds.get('country') == country and ds.get('target_policy') == job.get('target_policy'),
                f'{job_id}: dataset/target policy differs from accepted candidate')
        recorded_runtime = job.get('runtime', {})
        require({key: recorded_runtime.get(key) for key in EXPECTED_RUNTIME} == EXPECTED_RUNTIME,
                f'{job_id}: runtime versions differ')
        expected_code = {'runner': RUNNER_SHA, 'run_validation_helpers': RUNNER_SHA, 'validation_core': CORE_SHA,
            'support_diagnostics': '27b1e546971bbcbcfad4539ca1010c3aa83db51cb7230a35651024d25bcc91bc',
            'conformal_sensitivity': '0b11184d6d89566851666ca25a4164c98fdddbed0156068c45dabc687ddac5c5',
            'fast_cluster_bootstrap': 'f8cbbbb53ce94a79b3da6b506788583880d81b817f18a5eba2b0f92fd1eb9332'}
        require(recorded_runtime.get('code_sha256') == expected_code and
                recorded_runtime.get('bootstrap', {}).get('engine') == 'cluster_stats' and
                recorded_runtime.get('bootstrap', {}).get('helper_sha256') == expected_code['fast_cluster_bootstrap'],
                f'{job_id}: core runtime source/bootstrap pins differ')
        expected_features = [*GEO, *(MET if profile == 'with_met' else []), CAM]
        require(job.get('base_features') == expected_features and job.get('model_config') == config_for(estimator)
                and job.get('smear_strategies') == STRATEGIES
                and job.get('fold_count') == 16 and len(job.get('folds', [])) == 16,
                f'{job_id}: feature/config/runtime/fold contract differs')
        pin_job_paths(job, bound)
        require(accepted.get(str(Path(job['source_path']).resolve())) == str(job['source_sha256']).lower(),
                f'{job_id}: selected source table is not covered by accepted artifacts')

    # Numerical imports follow byte pins, as in the reviewed USA wrapper.
    import numpy as np
    import pandas as pd
    runtime = {'python_version': '.'.join(map(str, sys.version_info[:3])),
               'numpy_version': importlib.metadata.version('numpy'),
               'pandas_version': importlib.metadata.version('pandas'),
               'scikit_learn_version': importlib.metadata.version('scikit-learn')}
    require(runtime == EXPECTED_RUNTIME, f'fit runtime differs: {runtime}')
    sys.path.insert(0, str(SNAPSHOT))
    core = load_frozen('uscan_cams_remaining_frozen_core', core_path)
    runner = load_frozen('uscan_cams_remaining_frozen_runner', runner_path)
    fit = load_frozen('uscan_cams_remaining_existing_fit_helper', FIT_HELPER)
    require(fit.CORE_SHA == CORE_SHA and fit.RUNNER_SHA == RUNNER_SHA, 'fit helper runtime pins differ')

    groups: dict[str, list[dict[str, Any]]] = {}
    reuse_groups: dict[str, list[str]] = {}
    exact_groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        split = read_json(Path(job['split_manifest_path']))
        item = {**job, 'splits': job['folds']}
        frame, membership, exact = fit.verify_job_data(
            item, {'frame': Path(job['frame_path']), 'membership': Path(job['membership_path'])},
            split, core, runner, np, pd)
        require(exact == job.get('exact_dedup_fingerprint_sha256'), f'{job["job_id"]}: exact fingerprint differs')
        policy_key = json_hash({'dataset_key': job['dataset_key'], 'target_policy': job['target_policy'],
                                'country': job['country'], 'fit_fingerprint': exact})
        require(policy_key == job.get('reuse_group_fingerprint_sha256'), f'{job["job_id"]}: reuse group differs')
        years = pd.to_numeric(frame.year, errors='raise')
        countries = frame.country.astype(str).str.upper().replace({'US': 'USA', 'CA': 'CAN'})
        require(years.between(2018, 2022).all() and countries.eq(job['country']).all(),
                f'{job["job_id"]}: nonhistorical or wrong-country frame row')
        item.update({'frame': frame, 'membership': membership, 'frame_sha256': job['frame_sha256'],
                     'membership_sha256': job['membership_sha256']})
        groups.setdefault(exact, []).append(item)
        exact_groups.setdefault(exact, []).append(job)
        reuse_groups.setdefault(policy_key, []).append(job['job_id'])

    expected_reuse = {str(k): sorted(map(str, v)) for k, v in report['dedup']['fingerprint_groups'].items()}
    rebuilt_reuse = {k: sorted(v) for k, v in sorted(reuse_groups.items())}
    require(rebuilt_reuse == expected_reuse and len(rebuilt_reuse) == 14
            and report['dedup'].get('unique_fingerprints') == 14
            and report['dedup'].get('policy_scoped_groups_prevent_cross_dataset_or_target_reuse') is True,
            'policy-scoped dedup census differs from manifest')
    require(len(groups) == 14 and len(exact_groups) == 14, 'exact fingerprint census is not 14')
    for exact, aliases in exact_groups.items():
        require(len(aliases) == 2 and len({x['reuse_group_fingerprint_sha256'] for x in aliases}) == 1
                and len({x['qa'] for x in aliases}) == 2
                and len({(x['dataset_key'], x['target_policy'], x['country'], x['profile'], x['estimator'])
                         for x in aliases}) == 1,
                f'exact group crosses scope or is not a strict/sensitivity pair: {exact}')
        require({x['qa'] for x in aliases} == {'strict', 'sensitivity'}, f'exact group QA labels differ: {exact}')
    fits = sum(len(aliases[0]['membership']['folds']) // 2 * len(FIT_SPECS) for aliases in groups.values())
    require(fits == 448, f'expected 448 fits, reconstructed {fits}')
    require(not OUTPUT_ROOT.exists(), f'refusing to reuse output root: {OUTPUT_ROOT}')
    wrapper_sha = sha(Path(__file__).resolve())
    if args.execute:
        require(args.go is not None and args.go_sha is not None, '--execute requires --go and --go-sha')
        check_go(args.go, args.go_sha, wrapper_sha, expected_ids, fits, rebuilt_reuse)
    elif args.go is not None or args.go_sha is not None:
        require(args.go is not None and args.go_sha is not None, '--go and --go-sha must be supplied together')
    summary = {'status': 'PASS', 'mode': 'execute_authorized' if args.execute else 'check_only',
               'fit_performed': False, 'scores_or_predictions_performed': False, 'jobs': len(jobs),
               'aliases': expected_ids, 'dedup_groups': len(groups), 'policy_scoped_groups': len(rebuilt_reuse),
               'fits_authorized': fits, 'future_years_included': False, 'pack_manifest': str(PACK),
               'pack_manifest_sha256': PACK_SHA, 'wrapper_sha256': wrapper_sha,
               'fit_helper_sha256': FIT_HELPER_SHA, 'runtime_versions': runtime, 'output_root': str(OUTPUT_ROOT)}
    if args.execute:
        claim = create_claim(args.go_sha, wrapper_sha)
        fit.OUTPUT_ROOT = OUTPUT_ROOT
        fit.fit_all(groups, bound, ACCEPTANCE_SHA, core, np, pd)
        summary['fit_performed'] = True
        summary['scores_or_predictions_performed'] = True
        summary['exclusive_claim'] = str(claim)
    return summary


def check_go(path: Path, expected_sha: str, wrapper_sha: str, aliases: list[str],
             fit_count: int, groups: dict[str, list[str]]) -> None:
    resolved = path.resolve(strict=True)
    require(sha(resolved) == expected_sha.lower(), 'GO JSON SHA-256 mismatch')
    go = read_json(resolved)
    expected_scope = {'aliases': aliases, 'dedup_groups': 14, 'fits': fit_count,
                      'countries': ['CAN', 'USA'], 'future_years': False}
    expected_pins = {'pack_manifest_path': str(PACK), 'pack_manifest_sha256': PACK_SHA,
        'acceptance_sha256': ACCEPTANCE_SHA, 'wrapper_sha256': wrapper_sha,
        'fit_helper_path': str(FIT_HELPER), 'fit_helper_sha256': FIT_HELPER_SHA,
        'core_sha256': CORE_SHA, 'runner_sha256': RUNNER_SHA,
        'runtime_versions': EXPECTED_RUNTIME, 'output_root': str(OUTPUT_ROOT),
        'policy_scoped_groups': groups}
    require(go.get('format') == 'uscan-cams-fit-remaining-go-v1' and go.get('authorized') is True,
            'GO must be uscan-cams-fit-remaining-go-v1 with authorized=true')
    require(go.get('scope') == expected_scope and go.get('pins') == expected_pins,
            'GO scope or pins differ from validated remaining pack')
    require(re.fullmatch(r'[0-9a-fA-F]{64}', str(go.get('qa_record_ref', ''))) is not None,
            'GO must include a pinned qa_record_ref SHA-256')
    review_path = Path(go.get('qa_record_path', ''))
    require(review_path.is_absolute() and review_path.is_file() and
            sha(review_path) == go['qa_record_ref'].lower(), 'QA record path/bytes mismatch')
    review = read_json(review_path)
    require(review.get('decision') == 'approve_fit_contract' and review.get('findings') == [] and
            review.get('scope') == expected_scope and review.get('pack_manifest_sha256') == PACK_SHA and
            review.get('wrapper_sha256') == wrapper_sha and review.get('fit_helper_sha256') == FIT_HELPER_SHA,
            'QA record does not approve exact remaining fit scope/pack/wrapper/helper')
    require(not OUTPUT_ROOT.exists(), 'output root was created before execute authorization')


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_claim(go_sha: str, wrapper_sha: str) -> Path:
    OUTPUT_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
    fsync_dir(OUTPUT_ROOT.parent)
    claim = OUTPUT_ROOT / '.remaining_fit_authorization.claim'
    payload = (json.dumps({'pid': os.getpid(), 'go_sha256': go_sha.lower(),
                           'pack_sha256': PACK_SHA, 'wrapper_sha256': wrapper_sha},
                          sort_keys=True) + '\n').encode()
    fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(OUTPUT_ROOT)
    return claim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', dest='execute', action='store_false', help='validate only (default)')
    mode.add_argument('--execute', dest='execute', action='store_true', help='execute only with pinned release approval')
    parser.set_defaults(execute=False)
    parser.add_argument('--go', type=Path, help='separately pinned release approval JSON')
    parser.add_argument('--go-sha', help='expected SHA-256 of GO JSON')
    args = parser.parse_args()
    try:
        result = validate(args)
    except BaseException as exc:
        print(json.dumps({'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True), file=sys.stderr, flush=True)
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
