#!/usr/bin/env python3
"""Validate the accepted USA HGB + CAMS pack; run only with a pinned GO file."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

BASE = Path('/external/project')
V = BASE / 'validation'
RUNTIME = V / 'final_model_runtime_v4_20261114'
SNAPSHOT = RUNTIME / 'source_snapshot/validation'
PACK_ROOT = V / 'uscan_cams_current_v2_20261115/pack'
PACK = PACK_ROOT / 'no_fit_pack_manifest.json'
PACK_SHA = '57cb1ebed6acd3a31109b26f14ae40555b3826722762d49898f89410cceb7b20'
FIT_HELPER = V / 'global_current_fit_20261114/fit_global.py'
FIT_HELPER_SHA = 'd4e3110900d291e9d13966dd88d5539cbbaea2c51c69c2132c51ba080e720a12'
CORE_SHA = '027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157'
RUNNER_SHA = '05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c'
ACCEPTANCE_SHA = 'f4e32b4b9b40c1938da90d3329694dfc099668fa655a1ac744c7d09ff8105351'
OUTPUT_ROOT = V / 'uscan_cams_fit_USA_HGB_v1_20261115'
JOBS = [
    'usa_hist_strict_with_met_hgb', 'usa_hist_strict_no_met_hgb',
    'usa_hist_sensitivity_with_met_hgb', 'usa_hist_sensitivity_no_met_hgb',
]
GEO = ['lat_f', 'lon_f', 'lpop', 'month_sin', 'month_cos', 'month_sin2', 'month_cos2']
MET = ['blh', 'wind10', 'tp', 't2m', 'rh', 'vent']
CAM = 'cams_pm25_ug_m3'
CONFIG = {
    'max_iter': 400, 'learning_rate': 0.06, 'max_leaf_nodes': 31,
    'min_samples_leaf': 25, 'l2_regularization': 1.0, 'early_stopping': True,
    'validation_fraction': 0.15, 'random_state': 0, 'estimator_kind': 'hgb',
    'ridge_alpha': 1.0,
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
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(value, dict), f'expected JSON object: {path}')
    return value


def load_frozen(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f'cannot load frozen module {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_go(path: Path, expected_sha: str, wrapper_sha: str, fit_count: int) -> None:
    resolved = path.resolve(strict=True)
    require(sha(resolved) == expected_sha.lower(), 'GO JSON SHA-256 mismatch')
    go = read_json(resolved)
    expected_scope = {
        'country': 'USA', 'aliases': JOBS, 'dedup_groups': 2,
        'fits': 64, 'future_years': False,
    }
    expected_pins = {
        'pack_manifest_path': str(PACK), 'pack_manifest_sha256': PACK_SHA,
        'acceptance_sha256': ACCEPTANCE_SHA,
        'wrapper_sha256': wrapper_sha, 'fit_helper_path': str(FIT_HELPER),
        'fit_helper_sha256': FIT_HELPER_SHA, 'core_sha256': CORE_SHA,
        'runner_sha256': RUNNER_SHA, 'runtime_versions': EXPECTED_RUNTIME,
        'output_root': str(OUTPUT_ROOT),
    }
    require(go.get('format') == 'uscan-cams-fit-go-v1' and go.get('authorized') is True,
            'GO must be format uscan-cams-fit-go-v1 with authorized=true')
    require(go.get('scope') == expected_scope, 'GO scope differs from USA/4 aliases/2 groups/64 fits/no futures')
    require(go.get('pins') == expected_pins, 'GO pins differ from the validated pack, wrapper, helper, or runtime')
    require(fit_count == expected_scope['fits'], 'validated fit census is not 64')


def validate(args: argparse.Namespace) -> dict[str, Any]:
    require(not sys.flags.optimize, 'optimized Python is prohibited')
    require(sha(PACK) == PACK_SHA, 'no-fit pack manifest SHA mismatch')
    report = read_json(PACK)
    require(report.get('format') == 'repair-v2-uscan-historical-b-cams-no-fit-pack-v1'
            and report.get('status') == 'PASS' and report.get('fit_performed') is False
            and report.get('scores_or_predictions_performed') is False,
            'pack is not the expected completed no-fit pack')
    require(report.get('candidate') == {
        'path': str(V / 'uscan_final_feature_candidates_v2_20261115/final_candidate.json'),
        'sha256': '1c8e49668f7225ef9526b7ebcbd06af3ba8bd34efd9cab8f8cc0eb5b3f657048'},
        'candidate pin differs')
    require(report.get('acceptance', {}).get('sha256') == ACCEPTANCE_SHA
            and report.get('acceptance', {}).get('accepted_artifacts_checked_by_exact_path_and_sha') is True,
            'candidate/feature acceptance pin differs')
    require(report.get('fit_helper_path') == str(FIT_HELPER)
            and report.get('fit_helper_sha256') == FIT_HELPER_SHA,
            'reviewed fit helper pin differs')
    require(report.get('core_source_sha256') == CORE_SHA
            and report.get('runner_source_sha256') == RUNNER_SHA,
            'frozen source snapshot pins differ')
    require(report.get('runtime_versions') == EXPECTED_RUNTIME, 'pack runtime versions differ')

    contract = report.get('fit_contract', {})
    require(contract.get('model_config') == CONFIG, 'HGB400 config differs')
    require(contract.get('base_features') == {
        'with_met': [*GEO, *MET, CAM], 'no_met': [*GEO, CAM]},
        'GEO/MET/CAMS feature contract differs')
    require(contract.get('smear_and_target_strategies') == STRATEGIES
            and contract.get('fit_specs') == FIT_SPECS,
            'log_smear/crossfit3/original_scale strategy contract differs')

    jobs = report.get('jobs')
    require(isinstance(jobs, list) and [j.get('job_id') for j in jobs] == JOBS,
            'accepted USA job roster differs')
    dedup = report.get('dedup', {})
    expected_groups = {
        '494dc29a4870e1b2991326e851f467d90fad83213e7773e3fb1eb75169c2374c': [
            'usa_hist_sensitivity_with_met_hgb', 'usa_hist_strict_with_met_hgb'],
        '5c006fe18c270fa95c9a34283476e13edb998da9f271fe8360c5237f4c6ea3ff': [
            'usa_hist_sensitivity_no_met_hgb', 'usa_hist_strict_no_met_hgb'],
    }
    require(dedup.get('all_aliases') == JOBS and dedup.get('unique_fingerprints') == 2
            and dedup.get('fingerprint_groups') == expected_groups,
            'exact dedup groups differ from the accepted two-group census')

    # Revalidate every declared input byte before importing NumPy, pandas, sklearn, or the fitter.
    bound = report.get('bound_files')
    require(isinstance(bound, dict) and bound, 'pack bound_files is absent')
    for raw_path, expected in bound.items():
        path = Path(raw_path)
        require(path.is_absolute() and len(str(expected)) == 64, f'invalid bound file declaration: {raw_path}')
        require(sha(path) == str(expected).lower(), f'bound file SHA mismatch: {raw_path}')
    require(sha(FIT_HELPER) == FIT_HELPER_SHA, 'reviewed fitter bytes changed')

    # The GO file is separate from the reviewed fit implementation and is mandatory only for --run.
    current_wrapper_sha = sha(Path(__file__).resolve())
    if args.run:
        require(args.go is not None and args.go_sha is not None,
                '--run requires both --go and --go-sha')
    if args.go is not None or args.go_sha is not None:
        require(args.go is not None and args.go_sha is not None, '--go and --go-sha must be supplied together')

    # Imports follow the complete bound-file revalidation above.
    import numpy as np
    import pandas as pd

    runtime = {
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'numpy_version': importlib.metadata.version('numpy'),
        'pandas_version': importlib.metadata.version('pandas'),
        'scikit_learn_version': importlib.metadata.version('scikit-learn'),
    }
    require(runtime == EXPECTED_RUNTIME, f'fit runtime differs: {runtime}')
    core_path, runner_path = SNAPSHOT / 'validation_core.py', SNAPSHOT / 'run_validation.py'
    require(sha(core_path) == CORE_SHA and sha(runner_path) == RUNNER_SHA,
            'frozen core/runner bytes changed')
    sys.path.insert(0, str(SNAPSHOT))
    core = load_frozen('uscan_cams_fit_frozen_core', core_path)
    runner = load_frozen('uscan_cams_fit_frozen_runner', runner_path)
    fit = load_frozen('uscan_cams_existing_fit_helper', FIT_HELPER)
    require(fit.CORE_SHA == CORE_SHA and fit.RUNNER_SHA == RUNNER_SHA,
            'unchanged fit helper runtime pins differ')
    fit.OUTPUT_ROOT = OUTPUT_ROOT

    checked_jobs: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        features = job.get('base_features')
        require(features in ([*GEO, *MET, CAM], [*GEO, CAM]),
                f"unexpected feature set for {job.get('job_id')}")
        lowered = [str(x).lower() for x in features]
        require(not any(('fire' in x or 'aod' in x) for x in lowered),
                f"fire/AOD feature present for {job.get('job_id')}")
        require(job.get('country') == 'USA' and job.get('estimator') == 'hgb'
                and job.get('model_config') == CONFIG and job.get('smear_strategies') == STRATEGIES,
                f"job config/country differs for {job.get('job_id')}")
        require(job.get('fold_count') == 16, f"fold count differs for {job.get('job_id')}")
        frame_path, membership_path = Path(job['frame_path']), Path(job['membership_path'])
        split_path = Path(job['split_manifest_path'])
        require(bound.get(str(frame_path)) == job.get('frame_sha256')
                and bound.get(str(membership_path)) == job.get('membership_sha256'),
                f"pack frame/membership are not bound for {job['job_id']}")
        checked_job = {**job, 'splits': job['folds']}
        frame, membership, fingerprint = fit.verify_job_data(
            checked_job, {'frame': frame_path, 'membership': membership_path},
            read_json(split_path), core, runner, np, pd)
        require(fingerprint == job.get('exact_dedup_fingerprint_sha256'),
                f"dedup fingerprint changed for {job['job_id']}")
        years = pd.to_numeric(frame['year'], errors='raise')
        countries = frame['country'].astype(str).str.upper().replace({'US': 'USA'})
        require(years.between(2018, 2022).all(), f"nonhistorical year found for {job['job_id']}")
        require(countries.eq('USA').all(), f"non-USA frame row found for {job['job_id']}")
        checked_job.update({
            'frame': frame, 'membership': membership,
            'frame_sha256': job['frame_sha256'], 'membership_sha256': job['membership_sha256'],
        })
        groups.setdefault(fingerprint, []).append(checked_job)
        checked_jobs.append(job)

    rebuilt = {fp: sorted(item['job_id'] for item in aliases) for fp, aliases in sorted(groups.items())}
    require(rebuilt == expected_groups, 'reconstructed exact fingerprint groups differ from manifest')
    fit_count = sum(len(aliases[0]['membership']['folds']) // 2 * len(FIT_SPECS)
                    for aliases in groups.values())
    require(fit_count == 64, f'expected 64 fits, reconstructed {fit_count}')
    require(not OUTPUT_ROOT.exists(), f'refusing to reuse output root: {OUTPUT_ROOT}')
    if args.run:
        check_go(args.go, args.go_sha, current_wrapper_sha, fit_count)

    summary = {
        'status': 'PASS', 'mode': 'run_authorized' if args.run else 'check_only',
        'fit_performed': False, 'scores_or_predictions_performed': False,
        'jobs': len(checked_jobs), 'aliases': JOBS, 'dedup_groups': len(groups),
        'fits_authorized': fit_count, 'future_years_included': False,
        'pack_manifest': str(PACK), 'pack_manifest_sha256': PACK_SHA,
        'wrapper_sha256': current_wrapper_sha, 'fit_helper_sha256': FIT_HELPER_SHA,
        'runtime_versions': runtime, 'output_root': str(OUTPUT_ROOT),
    }
    if args.run:
        summary['fit_performed'] = True
        fit.fit_all(groups, bound, PACK_SHA, core, np, pd)
        summary['scores_or_predictions_performed'] = True
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', dest='run', action='store_false', help='validate only (default)')
    mode.add_argument('--run', dest='run', action='store_true', help='run only with a pinned GO JSON')
    parser.set_defaults(run=False)
    parser.add_argument('--go', type=Path, help='separate authorized GO JSON; required with --run')
    parser.add_argument('--go-sha', help='expected SHA-256 of --go')
    args = parser.parse_args()
    try:
        result = validate(args)
    except BaseException as exc:
        print(json.dumps({'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True),
              file=sys.stderr, flush=True)
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
