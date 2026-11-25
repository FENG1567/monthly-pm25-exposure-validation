"""No-fit replay of saved 2023 OC tracer models, pairs and correlations."""
from pathlib import Path
import hashlib
import json
import os
import sys

for name in ('REPAIR_V2_SMOKE_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'

CORE_AUDIT_SHA='94cf87dbcf8f970eeba9abba5da764b1172dc27b096d57bc402dc400168a5c0a'
shared_path=Path(__file__).with_name('audit_future4_v1.py')
if hashlib.sha256(shared_path.read_bytes()).hexdigest()!=CORE_AUDIT_SHA:
    raise RuntimeError('reviewed shared audit source changed')
import audit_future4_v1 as shared
import joblib
import numpy as np
import pandas as pd


def training_after_global_exclusion(frame,oc,runner):
    history=frame.loc[frame.year<=2022].copy()
    pm_sites=history[['site_id','lat','lon']].drop_duplicates('site_id')
    oc_sites=oc[['site_id','lat','lon']].drop_duplicates('site_id')
    excluded=set()
    for row in pm_sites.itertuples(index=False):
        distance=runner._haversine(np.full(len(oc_sites),float(row.lat)),
            np.full(len(oc_sites),float(row.lon)),oc_sites.lat.to_numpy(float),oc_sites.lon.to_numpy(float))
        if len(distance) and float(np.nanmin(distance))<=10.:
            excluded.add(str(row.site_id))
    train=history.loc[~history.site_id.isin(excluded)].copy()
    return train,excluded


def main():
    root=shared.ROOT
    out=root/'oc_postfit_audit_v2'
    shared.require(not out.exists(),'fresh OC audit output required')
    shared.remember(Path(__file__))
    shared.remember(shared_path,CORE_AUDIT_SHA)
    controller=shared.read_json(root/'future_controller_run/controller.status.json')
    shared.validate_controller_status(controller)
    shared.remember(shared.PREPARATION,shared.PREPARATION_SHA256)
    stage=shared.read_json(shared.PREPARATION)
    shared.validate_execution_lineage(root,stage,controller)
    for path,digest in stage['source_pins'].items():
        shared.remember(path,digest)
    runner=shared._load_runner(shared.RUNNER)
    def no_fit(*args,**kwargs):
        raise RuntimeError('fitting/OC producer rerun forbidden in saved OC audit')
    runner.fit_model=no_fit;runner.run_oc_validation=no_fit
    core=sys.modules['validation_core'];core.fit_model=no_fit
    diagnostics,_=shared.load_pinned_diagnostics(Path(__file__))
    config=runner.ModelConfig(max_iter=400)
    source=Path(runner.COMPARATOR_PATHS['OC'])
    shared.remember(source,'05a0666dac829bca3755af6ab2d4116f0fc36a7bb1317876d7ed51e73af11117')
    oc_all=pd.read_parquet(source)
    oc=oc_all.loc[oc_all.year==2023].copy()
    physical=['site_id','year','month']
    unique=oc[physical+['lat','lon']].drop_duplicates()
    shared.require(not unique.duplicated(physical).any(),'OC site-month coordinates conflict')
    reports=[]
    for job in stage['jobs']:
        if job['country']!='USA':
            continue
        qa=job['qa'];result_dir=root/'results'/f'USA_{qa}'
        manifest=shared.read_json(result_dir/'smoke_manifest.json')
        shared._verify_manifest_refs(manifest,'OC/'+qa)
        record=manifest['OC_comparison']
        shared.require(record['status']=='matched' and record['qa_variant']==qa
            and record['training_cutoff_year']==2022 and record['training_spatial_exclusion_km']==10.
            and record['input_sha256']==stage['source_pins'][str(source)],'OC source/protocol record differs')
        _,frame,_,_=runner.load_dataset('USA',{'USA':[Path(job['feature']['path'])]},Path(job['target']['path']))
        train,excluded=training_after_global_exclusion(frame,unique,runner)
        shared.require(train.year.le(2022).all() and train.site_id.nunique()>=10,
                       'OC historical training eligibility differs')
        shared.require(record['training_stations_excluded']==len(excluded),'global OC station exclusion count differs')
        feature_ref=job['OC_grid_features']
        shared.remember(feature_ref['path'],feature_ref['sha256'])
        sampled=pd.read_parquet(feature_ref['path'])
        shared.require(len(sampled)==3039 and sampled.maiac_qa_definition.eq(qa).all()
            and sampled.maiac_cell_geometry_verified.eq(True).all(), 'accepted OC grid QA/geometry differs')
        paired_path=Path(record['paired_rows_path'])
        shared.remember(paired_path,record['paired_rows_sha256'])
        saved_pairs=pd.read_parquet(paired_path)
        records=record['threshold_and_model_records']
        indexed={(str(r['profile']),str(r['target_strategy']),round(float(r['gate_quantile']),2)):r for r in records}
        expected={(p,t,q) for p in shared.PROFILES for t in shared.TARGET_STRATEGIES for q in shared.QUANTILES}
        shared.require(len(records)==12 and set(indexed)==expected,'exact OC 12 model/gate records required')
        expected_pairs=[];models=set();thresholds=set()
        for profile,strategy,q in sorted(expected):
            fit_record=indexed[(profile,strategy,q)]
            threshold=float(np.quantile(train[runner.GATE_FEATURE].to_numpy(float),q,method='linear'))
            low=train.loc[train[runner.GATE_FEATURE]<=threshold].copy()
            path=result_dir/'thresholds'/f'USA_OC_2023_10km_exclusion_{qa}_{profile}_{strategy}_{runner._quantile_tag(q)}.json'
            metadata=shared.read_json(path);thresholds.add(str(path.resolve()))
            required={'threshold_value_train_only':threshold,'training_rows_after_10km_exclusion':len(train),
                'training_stations_after_10km_exclusion':int(train.site_id.nunique()),
                'training_stations_excluded_within_10km':len(excluded),'background_rows':len(low),
                'background_stations':int(low.site_id.nunique()),'exclusion_distance_km':10.,
                'train_years':sorted(map(int,train.year.unique())),'threshold_feature':runner.GATE_FEATURE,
                'profile':profile,'target_strategy':strategy,'gate_quantile':q,'gate_role':runner._gate_role(q)}
            for name,value in required.items():
                shared.require(metadata.get(name)==value,'OC threshold/training metadata differs: '+name)
                if name in fit_record:
                    shared.require(fit_record[name]==value,'OC manifest fit-record differs: '+name)
            full_features,background_features=runner.feature_profiles(sampled,profile)
            prediction=[]
            inverse='smearing' if strategy=='log_smear' else 'model_native'
            for kind,features,rows in (('full',full_features,train),('background',background_features,low)):
                model_path=Path(metadata[kind+'_model_path']).resolve()
                shared.require(result_dir.resolve() in model_path.parents,'OC model escaped its result directory')
                shared.remember(model_path,metadata[kind+'_model_sha256']);models.add(str(model_path))
                bundle=joblib.load(model_path)
                diagnostics.validate_bundle(bundle,core,config.__dict__,features,rows,target_strategy=strategy)
                residual='training-only group out-of-fold' if strategy=='log_smear' else 'direct original-scale squared-error fit'
                shared.require(bundle['smear_info']['residual_source']==residual,'OC model smearing source differs')
                shared.require(float(bundle['smear'])==float(metadata[kind+'_smear']),'OC smearing factor differs')
                prediction.append(runner.predict_model(bundle,sampled,backtransform=inverse))
            full,background=prediction
            gate=sampled[runner.GATE_FEATURE].to_numpy(float)
            raw=np.asarray(full)-np.asarray(background);clipped=np.maximum(raw,0.);affected=gate>threshold
            sampled_predictions=sampled[physical+['grid_lat','grid_lon',runner.GATE_FEATURE]].assign(
                predicted_total=full,predicted_background=background,component_unclipped=raw,
                component_clipped=clipped,affected=affected,component_prediction=np.where(affected,clipped,0.))
            expected_pair=oc.merge(sampled_predictions,on=physical,how='inner',validate='many_to_one')
            expected_pair['cohort']='USA_OC_2023_10km_exclusion';expected_pair['country']='USA'
            expected_pair['task']='OC_grid_2023_10km_exclusion';expected_pair['qa_variant']=qa
            expected_pair['profile']=profile;expected_pair['gate_quantile']=float(q)
            expected_pair['gate_role']=runner._gate_role(q);expected_pair['target_strategy']=strategy
            expected_pair['backtransform']=inverse;expected_pair['obs']=expected_pair['value']
            expected_pair['comparator_label']='organic carbon tracer; not smoke-specific or PM2.5 mass-equivalent'
            actual=saved_pairs.loc[saved_pairs.profile.eq(profile)&saved_pairs.target_strategy.eq(strategy)
                &np.isclose(saved_pairs.gate_quantile.to_numpy(float),q)].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,expected_pair.reset_index(drop=True),check_exact=True)
            shared.require(len(expected_pair)==metadata['OC_rows'] and
                expected_pair.site_id.nunique()==metadata['OC_sites'] and
                len(expected_pair[['parameter_code','method_code']].drop_duplicates())==metadata['OC_parameter_method_series'],
                'OC supported row/site/series census differs')
            expected_pairs.append(expected_pair)
            print(json.dumps({'phase':'OC_model_gate_replayed','qa':qa,'profile':profile,'target':strategy,'q':q}),flush=True)
        combined=pd.concat(expected_pairs,ignore_index=True)
        shared.require(len(combined)==len(saved_pairs)==record['oc_rows_supported'], 'OC total paired row count differs')
        actual_models={str(p.resolve()) for p in (result_dir/'models/USA_OC_2023_10km_exclusion').rglob('*.joblib')}
        actual_thresholds={str(p.resolve()) for p in (result_dir/'thresholds').glob('USA_OC_2023_10km_exclusion_*.json')}
        shared.require(actual_models==models and len(models)==16 and actual_thresholds==thresholds,
                       'OC model/threshold filesystem census differs')
        metrics=[]
        for keys,group in combined.groupby(['cohort','task','qa_variant','profile','gate_quantile','gate_role',
                'target_strategy','parameter_code','method_code'],dropna=False):
            score=runner._pair_scores(group.component_prediction.to_numpy(float),group.obs.to_numpy(float),False)
            row={'scope':':'.join(map(str,keys)),'comparator':'EPA_OC_tracer','qa_variant':qa,
                'profile':group.iloc[0].profile,'gate_quantile':float(group.iloc[0].gate_quantile),
                'gate_role':group.iloc[0].gate_role,'n_pairs':len(group),'n_groups':int(group.site_id.nunique()),
                'interval_group':'site_id',**score,
                'note':'correlation only; OC is a nonspecific tracer and is not identified smoke mass; scale metrics omitted'}
            row.update(runner._bootstrap_pair_ci(group,'component_prediction','obs','site_id',False,500,500+len(metrics)))
            metrics.append(row)
        expected_metrics=pd.DataFrame(metrics)
        metric_path=result_dir/'smoke_comparator_metrics.csv';shared.remember(metric_path)
        saved_metrics=pd.read_csv(metric_path,float_precision='round_trip')
        saved_metrics=saved_metrics.loc[saved_metrics.comparator.eq('EPA_OC_tracer')].copy()
        shared.require(set(expected_metrics.columns).issubset(saved_metrics.columns),'OC correlation metric columns missing')
        for extra in set(saved_metrics.columns)-set(expected_metrics.columns):
            shared.require(saved_metrics[extra].isna().all(),'OC has an unexpected scale/metadata score: '+extra)
        count=shared.compare_table(expected_metrics,saved_metrics[expected_metrics.columns],
            ['scope','comparator','qa_variant','profile','gate_quantile','gate_role'])
        shared._verify_manifest_refs(manifest,'OC/'+qa)
        reports.append({'qa':qa,'models':len(models),'model_gate_records':12,'pairs':len(combined),
            'excluded_PM_stations':sorted(excluded),'training_rows':len(train),
            'training_stations':int(train.site_id.nunique()),'metric_rows':len(metrics),'numeric_metric_cells_recomputed':count})
    shared.require(len(reports)==2,'both USA QA OC runs required')
    for path,digest in shared.ARTIFACTS.items():
        shared.require(shared.sha256(path)==digest,'OC audit source/output changed: '+path)
    out.mkdir()
    report={'format':'future-smoke-OC2023-saved-model-correlation-audit-v1','status':'PASS','findings':[],
        'jobs':reports,'models_replayed':32,'model_gate_records':24,'fits_performed':False,
        'all_inspected_artifact_pins':shared.ARTIFACTS,'auditor_sha256':shared.sha256(Path(__file__)),
        'scope':'2023 OC tracer associations; historical PM training only; global 10km exclusion; not identified smoke mass'}
    with (out/'audit.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASS','models':32,'output':str(out/'audit.json')}),flush=True)


if __name__=='__main__':
    main()
