"""Gated, no-fit comparator replay for the four accepted USA/CAN smoke jobs.

This script is preparation for a postfit comparator audit. It cannot fit models.
It verifies the accepted core-audit receipts and artifact hashes before opening
any OOF or saved metric/pair table. The replay delegates metric/bootstrap logic
to the pinned frozen smoke runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


V = Path('/external/project/validation')
ROOT = V / 'uscan_smoke_isolated_v2_20261115'
PREPARATION_SHA256 = 'ce5f5926b20f7a646eeb7eae8960cff18b2814d5b26bc7b195321d70cee85c5f'
POSTFIT_REPORT_SHA256 = '0c4dfbe6eda816f0cf3ead71cf4a310a8a11d0407c111db2f8473af486757a51'
CORE_ACCEPTANCE_SHA256 = '38741850360c288d2ca3eecce48b6b3143c6b7ab1e3bf8dfa9fe3c89aacbfa08'
RUNNER_SHA256 = '3624d294dde1702e8fa5c8918584a13ebba45c4347683284c1171c69148c1679'
ACCEPTANCE = V / 'uscan_smoke_core_acceptance_v1_20261115/independent_acceptance_v1.json'
PREPARATION = ROOT / 'preparation_result.json'
POSTFIT_REPORT = ROOT / 'postfit_audit_v2/audit_result.json'
RUNNER = ROOT / 'runtime/work/repair_v2/smoke/run_smoke_validation.py'
COUNTRIES = ('USA', 'CAN')
QAS = ('strict', 'sensitivity')
JOB_ROSTER = {(country, qa) for country in COUNTRIES for qa in QAS}
REPS = 500
HMS_UNKNOWN = {'status': 'historical_coverage_unknown', 'matched_pairs': None}


class ComparatorAuditError(ValueError):
    """A comparator replay gate or saved artifact does not match its contract."""


def require(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ComparatorAuditError(message)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'expected JSON object: {path}')
    return value


def cli_value(argv: Sequence[str], name: str) -> str:
    matches: list[str] = []
    prefix = name + '='
    for index, arg in enumerate(argv):
        if arg == name:
            require(index + 1 < len(argv), f'missing CLI value for {name}')
            matches.append(argv[index + 1])
        elif arg.startswith(prefix):
            matches.append(arg[len(prefix):])
    require(len(matches) == 1, f'expected exactly one {name} in prepared CLI')
    return matches[0]


def validate_preparation(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(stage.get('status') == 'prepared_no_fit' and
            stage.get('model_fitting_allowed') is False and
            stage.get('fits_performed') is False and stage.get('scores_read') is False,
            'preparation is not the pinned no-fit stage')
    jobs = stage.get('jobs')
    require(isinstance(jobs, list) and
            {(str(j.get('country')), str(j.get('qa'))) for j in jobs} == JOB_ROSTER and len(jobs) == 4,
            'prepared job roster is not exactly USA/CAN strict/sensitivity')
    for job in jobs:
        argv = list(job.get('cli_argv', []))
        require(cli_value(argv, '--bootstrap-replicates') == str(REPS),
                'prepared comparator bootstrap is not the frozen 500-replicate protocol')
        require('--skip-oc' in argv, 'OC was not explicitly excluded by the prepared job')
        require(not any('satpm' in str(arg).lower() for arg in argv),
                'SatPM input is outside this monthly station-comparator replay')
        require(Path(cli_value(argv, '--out')) == ROOT / 'results' /
                f"{job['country']}_{job['qa']}", 'prepared result directory differs from pinned root')
        for option in ('--qfed-path', '--gfed-path'):
            require(Path(cli_value(argv, option)).name == option[2:-5] + '_station_month.parquet',
                    f'{option} is not an explicit station-month comparator path')
    comparator_hashes = stage.get('comparators')
    require(isinstance(comparator_hashes, dict) and comparator_hashes,
            'prepared comparator input hash map is absent')
    return jobs


def validate_core_acceptance(stage: Mapping[str, Any], report: Mapping[str, Any],
                             acceptance: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(acceptance.get('decision') == 'accepted_core_postfit_only' and
            acceptance.get('independent_metadata_hash_pass') is True and
            acceptance.get('report_sha256') == POSTFIT_REPORT_SHA256,
            'independent core-postfit acceptance gate failed')
    require(acceptance.get('models_refitted') is False and
            acceptance.get('model_replay_repeated') is False and
            acceptance.get('scientific_scores_interpreted') is False and
            acceptance.get('whole_project_complete') is False,
            'core acceptance receipt exceeds its no-fit/no-interpretation scope')
    require(report.get('format') == 'uscan-historical-smoke4-core-postfit-audit-v1' and
            report.get('status') == 'historical4_core_postfit_replay_passed' and
            report.get('whole_project_complete') is False and
            report.get('models_refitted') is False and report.get('future_scoring_seen') is False,
            'core postfit audit report is not an accepted historical no-fit replay')
    jobs = report.get('jobs')
    require(isinstance(jobs, list) and len(jobs) == 4 and
            {(str(j.get('country')), str(j.get('qa'))) for j in jobs} == JOB_ROSTER,
            'core postfit report does not cover the exact four jobs')
    require(stage.get('status') == 'prepared_no_fit', 'preparation gate changed before comparator replay')
    return jobs


def validate_hms_unknown(record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep historical HMS unavailable distinct from observed zero coverage."""
    if record is None:
        return dict(HMS_UNKNOWN)
    require(record.get('status') == HMS_UNKNOWN['status'] and record.get('matched_pairs') is None,
            'historical HMS coverage must remain unknown/null, never zero')
    return dict(HMS_UNKNOWN)


def validate_childs_pair_limits(frame: Any, *, country_scope: str = 'USA') -> dict[str, Any]:
    """Validate the county-only USA 2018–2020 comparison without fixed counts."""
    required = {'year', 'GEOID_primary', 'comparator_value'}
    require(required <= set(frame.columns), 'Childs pair table lacks county-month keys')
    if len(frame):
        years = frame.year.astype(int)
        require(country_scope == 'USA', 'Childs county-month pairs must be USA only')
        if 'country' in frame:
            require(frame.country.astype(str).str.upper().eq('USA').all(), 'Childs country field is not USA')
        require(years.between(2018, 2020).all(), 'Childs county-month pairs exceed 2018–2020 support')
        require(frame.GEOID_primary.notna().all(), 'Childs comparison lacks official county identifiers')
    return {
        'status': 'matched' if len(frame) else 'no_overlap',
        'matched_county_months': int(len(frame)),
        'matched_years': sorted(map(int, frame.year.unique())) if len(frame) else [],
        'country_scope': 'USA_only', 'maximum_year': 2020,
    }


def after_core_acceptance_gate(preparation_path: Path, report_path: Path,
                               acceptance_path: Path,
                               replay: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any],
                               *, preparation_sha256: str = PREPARATION_SHA256,
                               report_sha256: str = POSTFIT_REPORT_SHA256,
                               acceptance_sha256: str = CORE_ACCEPTANCE_SHA256) -> Any:
    """Validate all signed gates before handing control to any OOF reader."""
    require(sha256(preparation_path) == preparation_sha256, 'pinned no-fit preparation hash differs')
    require(sha256(report_path) == report_sha256, 'independently reviewed core postfit report hash differs')
    require(sha256(acceptance_path) == acceptance_sha256, 'QA acceptance receipt hash differs')
    stage = read_json(preparation_path)
    report = read_json(report_path)
    acceptance = read_json(acceptance_path)
    validate_preparation(stage)
    validate_core_acceptance(stage, report, acceptance)
    return replay(stage, report, acceptance)


def _report_job_map(report_jobs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in report_jobs:
        key = (str(item['country']), str(item['qa']))
        require(key in JOB_ROSTER and key not in result, f'unexpected/duplicate postfit job {key}')
        result[key] = item
    require(set(result) == JOB_ROSTER, 'postfit job roster is incomplete')
    return result


def _hash_selected_outputs(stage_jobs: Sequence[Mapping[str, Any]], report_jobs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Path]]:
    """Hash OOF/metrics/pair outputs first; their values remain unopened here."""
    reports = _report_job_map(report_jobs)
    selected: dict[tuple[str, str], dict[str, Path]] = {}
    for job in stage_jobs:
        key = (str(job['country']), str(job['qa']))
        out = Path(cli_value(list(job['cli_argv']), '--out'))
        report_job = reports[key]
        require(report_job.get('source_manifest_sha256'), f'{key}: source manifest hash missing')
        manifest = out / 'smoke_manifest.json'
        require(sha256(manifest) == report_job['source_manifest_sha256'],
                f'{key}: source manifest changed after core acceptance')
        refs = report_job.get('artifact_sha256')
        require(isinstance(refs, dict), f'{key}: core report has no artifact hash map')
        candidates = {
            name: [Path(path) for path in refs if Path(path).name == name]
            for name in ('smoke_oof_predictions.parquet', 'smoke_comparator_metrics.csv')
        }
        for name, paths in candidates.items():
            require(len(paths) == 1 and paths[0].parent == out,
                    f'{key}: core report does not bind exactly one {name}')
        selected_paths = {name: paths[0] for name, paths in candidates.items()}
        # Pair artifacts are conditional on actual overlap; when present, hash
        # every saved pair table before any prediction/metric table is opened.
        for raw_path, digest in refs.items():
            path = Path(raw_path)
            if path.name in {
                'qfed_matched_station_pairs.parquet', 'gfed_matched_station_pairs.parquet',
                'childs_county_month_matched_pairs.parquet',
            }:
                require(path.parent == out, f'{key}: comparator pair artifact escaped result directory')
                selected_paths[path.name] = path
        for name, path in selected_paths.items():
            require(path.is_file() and sha256(path) == refs[str(path)],
                    f'{key}: postfit-bound artifact hash differs: {name}')
        selected[key] = selected_paths
    return selected


def _verify_comparator_sources(stage: Mapping[str, Any]) -> dict[str, Path]:
    hashes = stage['comparators']
    paths: dict[str, Path] = {}
    for raw_path, digest in hashes.items():
        path = Path(raw_path)
        require(path.is_file() and sha256(path) == digest,
                f'prepared comparator source hash differs: {path.name}')
        name = path.name.lower()
        if name in {'childs_county_month.parquet', 'usa_station_county_assignment.csv',
                    'global_comparator_provenance.json'}:
            paths.setdefault(name, path)
    for required in ('childs_county_month.parquet', 'usa_station_county_assignment.csv'):
        require(required in paths, f'prepared comparator input missing: {required}')
    return paths


def compare_table(actual: Any, saved: Any, keycols: Sequence[str]) -> int:
    import numpy as np
    import pandas as pd

    require(len(actual) == len(saved), 'comparator metric row census differs')
    require(set(actual.columns) == set(saved.columns), 'comparator metric schema differs')
    require(set(keycols) <= set(actual.columns), 'comparator metric sort keys are missing')
    actual = actual.sort_values(list(keycols)).reset_index(drop=True)
    saved = saved.sort_values(list(keycols)).reset_index(drop=True)
    comparisons = 0
    for column in actual:
        if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(saved[column]):
            require(np.allclose(actual[column].to_numpy(dtype=float), saved[column].to_numpy(dtype=float),
                                atol=1e-10, rtol=0, equal_nan=True),
                    f'comparator metric replay differs: {column}')
            comparisons += len(actual)
        else:
            require(actual[column].fillna('').astype(str).equals(saved[column].fillna('').astype(str)),
                    f'comparator metric replay differs: {column}')
    return comparisons


def compare_pair_table(actual: Any, saved: Any, keycols: Sequence[str]) -> int:
    import numpy as np
    import pandas as pd

    require(len(actual) == len(saved), 'matched pair row census differs')
    require(set(actual.columns) == set(saved.columns), 'matched pair schema differs')
    require(set(keycols) <= set(actual.columns), 'matched pair identity keys are missing')
    sort_keys = list(keycols)
    actual = actual.sort_values(sort_keys).reset_index(drop=True)
    saved = saved.sort_values(sort_keys).reset_index(drop=True)
    for column in actual:
        if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(saved[column]):
            require(np.array_equal(actual[column].to_numpy(dtype=float), saved[column].to_numpy(dtype=float),
                                   equal_nan=True), f'matched pair metadata differs: {column}')
        else:
            require(actual[column].fillna('').astype(str).equals(saved[column].fillna('').astype(str)),
                    f'matched pair metadata differs: {column}')
    return len(actual) * len(actual.columns)


def _validate_global_pair_metadata(frame: Any, runner: Any, product: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    required = {'country', 'site_id', 'year', 'month', 'lat', 'lon', 'comparator_lat',
                'comparator_lon', 'valid_daily_values', 'calendar_days',
                'complete_calendar_support', 'units', 'units_provenance'}
    require(required <= set(frame.columns), f'{product}: matched pairs lack coordinate/calendar/version provenance')
    require(frame.complete_calendar_support.astype(bool).all(), f'{product}: incomplete calendar support was matched')
    require(frame.units.astype(str).eq('ug/m3').all(), f'{product}: matched units are not publisher-verified ug/m3')
    valid = pd.to_numeric(frame.valid_daily_values, errors='raise').to_numpy(float)
    days = pd.to_numeric(frame.calendar_days, errors='raise').to_numpy(float)
    require(np.array_equal(valid, days), f'{product}: matched monthly calendar support differs')
    geometry = np.maximum(
        np.abs(frame.lat.to_numpy(float) - frame.comparator_lat.to_numpy(float)),
        np.abs(frame.lon.to_numpy(float) - frame.comparator_lon.to_numpy(float)),
    )
    require(np.isfinite(geometry).all() and
            (geometry <= runner.TARGET_COORDINATE_TOLERANCE_DEG).all(),
            f'{product}: matched station coordinate geometry differs')
    require(frame.units_provenance.notna().all() and frame.units_provenance.astype(str).str.strip().ne('').all(),
            f'{product}: matched unit evidence is absent')
    return {
        'matched_pairs': int(len(frame)), 'matched_years': sorted(map(int, frame.year.unique())),
        'max_coordinate_delta_degrees': float(geometry.max()) if len(geometry) else None,
        'calendar_complete_rows': int(len(frame)), 'units': 'ug/m3',
        'unit_provenance_values': sorted(frame.units_provenance.astype(str).unique().tolist()),
    }


def _load_runner(path: Path) -> Any:
    require(path.is_file() and sha256(path) == RUNNER_SHA256, 'frozen comparator runner changed or missing')
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location('historical4_frozen_comparator_runner', path)
    require(spec is not None and spec.loader is not None, 'cannot import frozen comparator runner')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay_after_gates(root: Path, stage: dict[str, Any], report: dict[str, Any],
                       acceptance: dict[str, Any], audit_out: Path) -> dict[str, Any]:
    import pandas as pd

    jobs = validate_preparation(stage)
    report_jobs = validate_core_acceptance(stage, report, acceptance)
    source_paths = _verify_comparator_sources(stage)
    # This hashes OOF, saved metrics and any saved pair tables without opening
    # their values. Every output hash comes from the independently accepted report.
    bound_outputs = _hash_selected_outputs(jobs, report_jobs)
    runner_path = root / 'runtime/work/repair_v2/smoke/run_smoke_validation.py'
    runner = _load_runner(runner_path)
    for name, rel in [('validation_core', 'validation/validation_core.py'),
                      ('run_validation', 'validation/run_validation.py'), ('core_split_binding', 'smoke/core_split_binding.py')]:
        module = sys.modules.get(name)
        path = root/'runtime/work/repair_v2'/rel
        require(module is not None and Path(module.__file__).resolve() == path.resolve() and
                sha256(path) == stage['runtime_cloned_exactly'][str(path)], 'helper origin/hash mismatch: '+name)
    def no_fit(*args, **kwargs):
        raise ComparatorAuditError('fitting forbidden in comparator replay')
    runner.fit_model = no_fit
    require(not audit_out.exists(), f'audit output exists; refusing overwrite: {audit_out}')
    # Childs/county assets are staged into the isolated runtime rather than
    # the frozen runner's earlier AUS-root defaults. QFED/GFED remain explicit
    # CLI inputs, each independently bound by the preparation source hashes.
    runner.COMPARATOR_PATHS['CHILDS'] = source_paths['childs_county_month.parquet']
    runner.COMPARATOR_PATHS['USA_COUNTY'] = source_paths['usa_station_county_assignment.csv']
    report_by_job = _report_job_map(report_jobs)
    audit_out.mkdir(parents=True, exist_ok=False)
    job_results = []
    for job in jobs:
        key = (str(job['country']), str(job['qa']))
        out = Path(cli_value(list(job['cli_argv']), '--out'))
        oof_path = bound_outputs[key]['smoke_oof_predictions.parquet']
        saved_metrics_path = bound_outputs[key]['smoke_comparator_metrics.csv']
        # No prediction data is read until all preparation, independent-review,
        # frozen-code, input and result artifact hashes have passed above.
        oof = pd.read_parquet(oof_path)
        qfed = Path(cli_value(list(job['cli_argv']), '--qfed-path'))
        gfed = Path(cli_value(list(job['cli_argv']), '--gfed-path'))
        replay_dir = audit_out / f'{key[0]}_{key[1]}'
        replay_dir.mkdir()
        metrics, records = runner.run_station_comparators(
            oof, replay_dir, REPS,
            comparator_paths={'QFED': qfed, 'GFED': gfed},
        )
        saved_metrics = pd.read_csv(saved_metrics_path, float_precision='round_trip')
        metrics_frame = pd.DataFrame(metrics)
        numeric_values = compare_table(
            metrics_frame, saved_metrics,
            ['scope', 'comparator', 'qa_variant', 'profile', 'gate_quantile'],
        )
        pair_summaries: dict[str, Any] = {}
        for product in ('QFED', 'GFED'):
            rec = records.get(product, {'status': 'unavailable'})
            if rec.get('status') == 'matched':
                replay_pairs_path = replay_dir / f'{product.lower()}_matched_station_pairs.parquet'
                saved_pairs_path = out / replay_pairs_path.name
                require(saved_pairs_path.is_file() and saved_pairs_path == bound_outputs[key].get(saved_pairs_path.name),
                        f'{key}/{product}: matched pair artifact was not bound by the accepted report')
                saved_pairs = pd.read_parquet(saved_pairs_path)
                replay_pairs = pd.read_parquet(replay_pairs_path)
                pair_values = compare_pair_table(
                    replay_pairs, saved_pairs, ['cohort', 'country', 'task', 'qa_variant', 'profile',
                      'gate_quantile', 'gate_role', 'target_strategy', 'site_id', 'year', 'month'],
                )
                provenance = _validate_global_pair_metadata(replay_pairs, runner, product)
                pair_summaries[product] = {
                    'status': 'matched', 'smoke_pairs_available': int(rec['smoke_pairs_available']),
                    'matched_pairs': int(rec['matched_smoke_pairs']),
                    'unmatched_smoke_pairs': int(rec['unmatched_smoke_pairs']),
                    'pair_table_values_compared': pair_values,
                    'source_path': rec.get('input_path'), 'source_sha256': rec.get('input_sha256'),
                    'verified_source': rec.get('verified_source'), **provenance,
                }
            else:
                require(not (replay_dir / f'{product.lower()}_matched_station_pairs.parquet').exists(),
                        f'{key}/{product}: unexpected pair table for nonmatched comparator')
                pair_summaries[product] = {
                    'status': rec.get('status', 'unavailable'),
                    'smoke_pairs_available': rec.get('smoke_pairs_available'),
                    'matched_pairs': rec.get('matched_smoke_pairs'),
                    'unmatched_smoke_pairs': rec.get('unmatched_smoke_pairs'),
                    'verified_source': rec.get('verified_source'),
                }
        childs_path = replay_dir / 'childs_county_month_matched_pairs.parquet'
        if records.get('CHILDS', {}).get('status') == 'matched':
            saved_childs_path = out / childs_path.name
            require(saved_childs_path.is_file() and saved_childs_path == bound_outputs[key].get(childs_path.name),
                    f'{key}/Childs: pair artifact was not bound by the accepted report')
            saved_childs = pd.read_parquet(saved_childs_path)
            replay_childs = pd.read_parquet(childs_path)
            pair_values = compare_pair_table(
                replay_childs, saved_childs,
                ['cohort', 'task', 'qa_variant', 'profile', 'gate_quantile',
                 'gate_role', 'target_strategy', 'GEOID_primary', 'year', 'month'],
            )
            childs_summary = validate_childs_pair_limits(replay_childs, country_scope=key[0])
            childs_summary['pair_table_values_compared'] = pair_values
            childs_summary['smoke_county_months_available'] = int(
                records['CHILDS']['smoke_county_months_available'])
            childs_summary['unmatched_smoke_county_months'] = int(
                records['CHILDS']['unmatched_smoke_county_months'])
        else:
            require(not childs_path.exists(), f'{key}: unexpected Childs table without matched status')
            childs_summary = {'status': records.get('CHILDS', {}).get('status', 'unavailable'),
                              'matched_county_months': 0, 'matched_years': [],
                              'country_scope': 'USA_only', 'maximum_year': 2020}
        metrics_frame.to_csv(replay_dir / 'replayed_comparator_metrics.csv', index=False)
        report_job = report_by_job[key]
        job_results.append({
            'country': key[0], 'qa': key[1],
            'source_manifest_sha256': report_job['source_manifest_sha256'],
            'source_oof_sha256': sha256(oof_path),
            'saved_comparator_metrics_sha256': sha256(saved_metrics_path),
            'metric_rows_replayed': int(len(metrics_frame)),
            'metric_numeric_values_recomputed': numeric_values,
            'unfiltered_metric_rows_preserved': int(len(metrics_frame)),
            'comparator_pairs': pair_summaries,
            'childs': childs_summary,
            'hms': validate_hms_unknown(),
            'bootstrap_replicates': REPS,
            'bootstrap_unit': 'station; descriptive conditional modeled-consistency comparison',
            'models_refitted': False, 'scores_interpreted': False,
            'satpm_monthly_comparison': False, 'oc_fit_or_comparison': False,
        })
        print(json.dumps({'audited': key, 'metric_rows': len(metrics_frame)}), flush=True)
    _verify_comparator_sources(stage)
    _hash_selected_outputs(jobs, report_jobs)
    result = {
        'format': 'uscan-historical-smoke4-comparator-replay-v1',
        'status': 'historical4_comparator_replay_passed',
        'scope': 'US_CAN_historical_smoke_4_jobs_comparator_replay_only',
        'core_postfit_report_sha256': POSTFIT_REPORT_SHA256,
        'core_acceptance_sha256': CORE_ACCEPTANCE_SHA256,
        'preparation_sha256': PREPARATION_SHA256,
        'runner_sha256': RUNNER_SHA256,
        'jobs': job_results,
        'models_refitted': False, 'scientific_scores_interpreted': False,
        'whole_project_complete': False,
    }
    with (audit_out / 'comparator_audit_result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def run(root: Path, audit_out: Path) -> dict[str, Any]:
    preparation = root / 'preparation_result.json'
    report = root / 'postfit_audit_v2/audit_result.json'
    acceptance = V / 'uscan_smoke_core_acceptance_v1_20261115/independent_acceptance_v1.json'
    return after_core_acceptance_gate(
        preparation, report, acceptance,
        lambda stage, audit, receipt: replay_after_gates(root, stage, audit, receipt, audit_out),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit-out', type=Path)
    args = parser.parse_args(argv)
    root = args.root
    audit_out = args.audit_out or (root / 'comparator_audit_v1')
    require(audit_out.resolve().parent == root.resolve(), 'audit output must be a fresh child of the smoke root')
    require(sys.executable == '/external/fire_downscaling_20261107/env/bin/python3.11',
            'canonical Python 3.11 runtime required')
    result = run(root, audit_out)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
