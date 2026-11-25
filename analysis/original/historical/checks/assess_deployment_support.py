"""Evaluate fixed joint support rules and their held-out coverage cost.

Execution-stage exploratory addition: all rules are retained, never optimized
against held-out outcomes. Support flags must originate from outer training.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from fast_cluster_bootstrap import paired_bootstrap

ROW_KEYS = ['country', 'cohort_label', 'task', 'fold_id', 'row_id', 'site_id']
GROUP_KEYS = ['country', 'cohort_label', 'task', 'target_strategy',
              'smear_strategy', 'backtransform']
RULES = ['all', 'distance_50km', 'distance_50km_non_geo_minmax',
         'distance_50km_non_geo_p01_p99']
PRELIMINARY_STAGES = {'base_support_pending_augmentation',
                      'verified_old_support_pending_expansion',
                      'available_source_geometry_verified_pending_raw_gap_audit'}
FINAL_STAGE = 'geometry_verified_source_coverage_audited'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _bool_column(frame, name):
    if name not in frame or frame[name].isna().any():
        raise ValueError(f'Missing support decision: {name}')
    if not frame[name].isin([True, False]).all():
        raise ValueError(f'Non-boolean support decision: {name}')
    return frame[name].to_numpy(dtype=bool)


def input_context(manifest):
    """Carry explicit QA and acceptance stages into standalone outputs."""
    def stage(record):
        value = record.get('manifest', {})
        status, variant = value.get('overlay_status'), value.get('aod_variant')
        if variant not in {'strict', 'sensitivity'}:
            raise ValueError('Missing or unknown QA variant in input provenance')
        if status not in PRELIMINARY_STAGES | {FINAL_STAGE}:
            raise ValueError('Missing or unrecognized QA overlay stage')
        if status == FINAL_STAGE and (value.get('geometry_complete') is not True
                                      or value.get('source_coverage_audited') is not True):
            raise ValueError('Final overlay stage lacks explicit geometry/source audit')
        return status, variant
    contexts = {}
    if 'training_input_provenance' in manifest:
        records = [(manifest['country'], manifest['training_input_provenance'],
                    manifest['external_input_provenance'])]
    else:
        records = [(country, record.get('feature_input_provenance', {}),
                    record.get('feature_input_provenance', {}))
                   for country, record in manifest.items()]
    for country, training, evaluation in records:
        train_stage, train_qa = stage(training)
        eval_stage, eval_qa = stage(evaluation)
        if train_qa != eval_qa:
            raise ValueError('Training and evaluation QA variants differ')
        contexts[country] = {'training_overlay_status':train_stage,
                             'evaluation_overlay_status':eval_stage,
                             'qa_variant':train_qa,
                             'input_acceptance_stage':
                                 'audited' if train_stage==eval_stage==FINAL_STAGE else 'preliminary',
                             'distance_rule':'finite nearest-training distance strictly less than 50 km'}
    return contexts


def join_support(oof, flags):
    """Bind each full-arm prediction and baseline to one support record."""
    keys = ROW_KEYS + (['profile'] if 'profile' in oof else [])
    groups = GROUP_KEYS + (['profile'] if 'profile' in oof else [])
    required = set(keys+groups+['arm', 'obs', 'pred'])
    if not required.issubset(oof.columns):
        raise ValueError(f'Prediction columns missing: {required-set(oof.columns)}')
    if oof[list(dict.fromkeys(keys+groups))].isna().any().any():
        raise ValueError('Null prediction identity or strategy')
    selected = oof.loc[((oof.target_strategy.eq('log_smear')
                        & oof.smear_strategy.eq('crossfit_station')
                        & oof.backtransform.eq('smearing'))
                       | (oof.target_strategy.eq('original_scale')
                          & oof.backtransform.eq('model_native')))
                      & oof.arm.isin(['full', 'B'])].copy()
    if selected.empty:
        raise ValueError('No primary or original-scale prediction pairs')
    identity = list(dict.fromkeys(keys+groups))
    if selected.duplicated(identity+['arm']).any():
        raise ValueError('Duplicate prediction identities')
    full = selected.loc[selected.arm.eq('full')].drop(columns='arm')
    base = selected.loc[selected.arm.eq('B'), identity+['obs', 'pred']]
    paired = full.merge(base.rename(columns={'obs':'baseline_obs', 'pred':'baseline_pred'}),
                        on=identity, how='outer', validate='one_to_one', indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Full/baseline support differs')
    paired = paired.drop(columns='_merge')
    if not np.array_equal(paired.obs.to_numpy(), paired.baseline_obs.to_numpy()):
        raise ValueError('Full/baseline outcomes differ')
    support = flags.loc[flags.arm.eq('full')
                        & flags.feature_group.eq('non_geography')].copy()
    if not set(keys).issubset(support.columns) or support[keys].isna().any().any():
        raise ValueError('Missing or null support identity')
    if support.duplicated(keys).any():
        raise ValueError('Duplicate support identities')
    keep = ['n_selected_features', 'n_missing_features', 'n_no_training_reference_features',
            'all_selected_finite_within_train_minmax',
            'all_selected_finite_within_train_p01_p99', 'within_50km_train']
    if not set(keys+keep).issubset(support.columns):
        raise ValueError('Incomplete support schema')
    paired = paired.merge(support[keys+keep], on=keys, how='left',
                          validate='many_to_one', indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Prediction without support flags')
    paired = paired.drop(columns='_merge')
    # A missing flag, no selected variables, or absent training reference can
    # never pass the joint rules.
    distance = _bool_column(paired, 'within_50km_train')
    supported = (paired.n_selected_features.gt(0)
                 & paired.n_missing_features.eq(0)
                 & paired.n_no_training_reference_features.eq(0)).to_numpy()
    paired['gate_all'] = True
    paired['gate_distance_50km'] = distance
    for column, suffix in [('all_selected_finite_within_train_minmax', 'minmax'),
                           ('all_selected_finite_within_train_p01_p99', 'p01_p99')]:
        paired['gate_distance_50km_non_geo_'+suffix] = (
            distance & supported & _bool_column(paired, column))
    if not np.isfinite(paired[['obs', 'pred', 'baseline_pred']]).all().all():
        raise ValueError('Nonfinite evaluation observations/predictions')
    return paired, groups


def metrics(frame, prediction='pred'):
    if frame.empty:
        return dict(rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan)
    y = frame.obs.to_numpy(dtype=float)
    error = frame[prediction].to_numpy(dtype=float)-y
    variance = np.sum((y-y.mean())**2)
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                r2=float(1-np.sum(error**2)/variance) if variance>0 else np.nan)


def assess(pairs, groups, replicates=500):
    records = []
    for key, frame in pairs.groupby(groups, sort=False, dropna=False):
        info = dict(zip(groups, key))
        n, stations = len(frame), frame.site_id.nunique()
        # Site coverage is within this task/year support, never a geographic
        # area coverage estimate.
        for rule in RULES:
            gate = frame['gate_'+rule]
            site_all = frame.assign(_gate=gate).groupby('site_id')._gate.all()
            for label, mask in [('retained', gate), ('not_retained', ~gate)]:
                part = frame.loc[mask]
                model, baseline = metrics(part), metrics(part, 'baseline_pred')
                record = {**info, 'rule':rule, 'selection':label,
                          'n':len(part), 'n_all':n,
                          'row_fraction':float(len(part)/n),
                          'n_stations':int(part.site_id.nunique()),
                          'n_all_stations':int(stations),
                          'station_any_fraction':float(part.site_id.nunique()/stations),
                          'station_all_months_retained_fraction':float(site_all.mean()),
                          **model, **{'baseline_'+k:v for k,v in baseline.items()},
                          **{'delta_'+k:model[k]-baseline[k] for k in model},
                          'obs_mean':float(part.obs.mean()) if len(part) else np.nan,
                          'obs_sd_population':float(part.obs.std(ddof=0)) if len(part) else np.nan,
                          'obs_p95':float(part.obs.quantile(.95)) if len(part) else np.nan,
                          'obs_max':float(part.obs.max()) if len(part) else np.nan,
                          'rmse_reduction_vs_same_subset_baseline_fraction':
                              float(1-model['rmse']/baseline['rmse'])
                              if baseline['rmse']>0 else np.nan,
                          'interpretation':'fixed exploratory support rule; not a guarantee of reliability',
                          'interval_scope':'station resampling conditional on observed task/years; no spatial or new-year guarantee'}
                if len(part) and part.site_id.nunique()>=2:
                    record.update(paired_bootstrap(part.site_id, part.obs, part.pred,
                                                   part.baseline_pred, replicates, 963))
                records.append(record)
    return pd.DataFrame(records)


def assess_run(run, out, replicates=500):
    run, out = Path(run), Path(out)
    candidates = [run/name for name in ['oof_predictions.parquet', 'external_oof_predictions.parquet']]
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates)!=1:
        raise ValueError('Expected exactly one historical or external prediction file')
    prediction = candidates[0]
    support = run/'support_row_flags.parquet'
    manifests = [run/name for name in ['source_manifest.json', 'external_run_manifest.json']
                 if (run/name).is_file()]
    if len(manifests)!=1:
        raise ValueError('Expected one run provenance manifest')
    context = input_context(json.loads(manifests[0].read_text(encoding='utf-8')))
    pairs, groups = join_support(pd.read_parquet(prediction), pd.read_parquet(support))
    results = assess(pairs, groups, replicates)
    if not set(results.country).issubset(context):
        raise ValueError('Prediction countries lack input-stage provenance')
    for field in next(iter(context.values())):
        results[field] = results.country.map({country:values[field] for country, values in context.items()})
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out/'paired_support_flags.parquet', index=False)
    results.to_csv(out/'support_risk_coverage.csv', index=False)
    record = {'status':'exploratory execution-stage analysis; preserve all rules',
              'run_directory':str(run.resolve()),
              'sources':[{'path':str(path.resolve()), 'sha256':sha256(path)}
                         for path in [prediction, support, *manifests]],
              'code_sha256':sha256(__file__), 'rules':RULES,
              'bootstrap_replicates':replicates,
              'input_context':context,
              'prediction_pairs':len(pairs), 'metric_rows':len(results),
              'coverage_unit':'held-out station-month and physical station; not land area',
              'no_test_outcome_rule_selection':True,
              'conformal_intervals_assessed':False}
    (out/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=500)
    args = parser.parse_args()
    record = assess_run(args.run, args.out, args.bootstrap_replicates)
    print(json.dumps(record,indent=2),flush=True)
