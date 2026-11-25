"""Date-aware HMS consistency diagnostics on independently accepted saved predictions."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import pandas as pd
import future_core_gate_v1 as api

HELPER_SHA='e0861e57d28ac2953e93c93e15d1cd7842308d872c5b485807025b13e7f19b5b'
ORIGINAL_SHA='be8563fba7c6969a2a8f41100f2cf0a88a858fbf744fc4871ff9d279d701b1b7'
OC_SHA='1955744395a78859ebce2ced349abd4173805800256a153661728b2fcadd48cc'
RAW_OC_SHA='1b8555b999d2b96f7d74022492747056794f796f375a8e661ede648bfa442c56'
PROTOCOLS={'hms_2023_protocol.md':'8ac95fbd92365aca386dcc432f82459d5fd5f041385bf1900f29ab80e8bf6f8b',
           'hms_future_protocol_addendum_20261113.md':'5b8b136ae72623a29f1e3ccaea0deb8e7f06146b7dcfdc03fb219d7bd1fb00a3'}
ALIGNMENTS=('same_named_date','plusminus1day')
DEFINITIONS=('any_plume','medium_heavy')
MODEL_GROUP=['model_scope','country','cohort','qa_variant','task','profile','target_strategy',
             'backtransform','gate_quantile','gate_role','year','parameter_code','method_code']
POINT_KEYS=['country','site_id','lat','lon']
INTERVAL_SCOPE='500 independent-physical-station bootstrap replicates, seed610; conditional on fixed predictions and year; regional shared episodes/spatial dependence unquantified'


def module(name,path,digest,gate):
    path=api.pinned(path,digest,gate['used'])
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def load_sources():
    gate=api.load_gate()
    here=Path(__file__).resolve().parent
    api.pinned(api.__file__,'93f7f5903ecb73092bca57df9a386e8c6899ce7b2fb0bc57c5b33f1058716d00',gate['used'])
    api.pinned(__file__,api.sha(Path(__file__)),gate['used'])
    for name,digest in PROTOCOLS.items():
        api.pinned(here/name,digest,gate['used'])
    original=api.pinned(here/'original_validate_hms_2023.py',ORIGINAL_SHA,gate['used'])
    helper_path=api.pinned(here/'historical_hms_pure_helpers_v1.py',HELPER_SHA,gate['used'])
    old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
    copied={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(helper_path.read_text()).body if isinstance(n,ast.FunctionDef)}
    api.require(set(copied)=={'oc_contrasts','correlation','station_correlation_interval'}
                and all(old[k]==v for k,v in copied.items()),'historical pure HMS function AST changed')
    helpers=module('frozen_hms_pure_helpers',helper_path,HELPER_SHA,gate)
    receipt=api.V/'uscan_future_smoke_OC_independent_acceptance_v1_20261115/independent_acceptance.json'
    oc_accept=json.loads(api.pinned(receipt,OC_SHA,gate['used']).read_text())
    api.require(oc_accept['format']=='future-smoke-independent-OC-tracer-acceptance-v1'
        and oc_accept['decision']=='accepted_OC_tracer_only' and oc_accept['findings']==[]
        and oc_accept['core_acceptance_sha256']==api.ACCEPTANCE_SHA,'accepted OC tracer gate differs')
    ref=oc_accept['report'];api.pinned(ref['path'],ref['sha256'],gate['used'])
    raw=api.V/'uscan_future_HMS_inputs_v1_20261115/oc_station_day_2018_2025.parquet'
    api.pinned(raw,RAW_OC_SHA,gate['used'])
    return gate,helpers,raw


def exact_geometry_join(left,right,keys):
    """Retain every left row, accepting only an unambiguous coordinate match."""
    api.require(not ((set(left)&set(right))-set(keys)-{'lat','lon'}),'HMS columns overlap saved model identity')
    api.require(not left.empty and np.isfinite(left[['lat','lon']].to_numpy(float)).all(),'invalid left station geometry')
    api.require(np.isfinite(right[['lat','lon']].to_numpy(float)).all(),'invalid HMS station geometry')
    api.require(not right.duplicated(keys+['lat','lon']).any(),'duplicate HMS station coordinate key')
    api.require('_hms_left_order' not in left and '_hms_left_order' not in right,'reserved join key present')
    l=left.copy();l['_hms_left_order']=np.arange(len(l),dtype=np.int64)
    r=right.rename(columns={'lat':'hms_lat','lon':'hms_lon'})
    candidates=l[['_hms_left_order']+keys+['lat','lon']].merge(r,on=keys,how='inner',validate='many_to_many')
    good=(np.isclose(candidates.lat,candidates.hms_lat,atol=1e-6,rtol=0)
          &np.isclose(candidates.lon,candidates.hms_lon,atol=1e-6,rtol=0))
    candidates=candidates.loc[good].drop(columns=keys+['lat','lon'])
    api.require(not candidates._hms_left_order.duplicated().any(),'ambiguous HMS coordinate match')
    out=l.merge(candidates,on='_hms_left_order',how='left',sort=False,validate='one_to_one',indicator='_hms_match')
    out['hms_geometry_matched']=out._hms_match.eq('both')
    out=out.sort_values('_hms_left_order',kind='stable').drop(columns=['_hms_left_order','_hms_match']).reset_index(drop=True)
    api.require(len(out)==len(left),'HMS join lost modeled station-month rows')
    return out


def support_hash(frame,columns):
    ordered=frame[columns].sort_values(columns,kind='stable').reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered,index=False).to_numpy().tobytes()).hexdigest()


def day_bins(values):
    return pd.cut(values,[-.1,.1,5,15,31],labels=['0','1-5','6-15','16+'])


def monthly_statistics(pairs,helpers):
    records=[];coverage=[];bins=[];cache={}
    grouping=MODEL_GROUP+['alignment','plume_definition']
    for labels,group in pairs.groupby(grouping,dropna=False,sort=False):
        identity=dict(zip(grouping,labels))
        matched=group.hms_geometry_matched
        eligible=matched&group.support_80pct.fillna(False).astype(bool)
        supported=group.loc[eligible].copy()
        coverage.append({**identity,'source_rows':len(group),'geometry_matched_rows':int(matched.sum()),
            'geometry_unmatched_rows':int((~matched).sum()),'availability_below_80pct_rows':int((matched&~eligible).sum()),
            'retained_rows':len(supported),'retained_sites':supported.site_id.nunique(),
            'known_days_matched':int(group.loc[matched,'n_known_classification_days'].sum()),
            'unknown_days_matched':int(group.loc[matched,'n_unknown_classification_days'].sum()),
            'available_calendar_days_matched':int(group.loc[matched,'n_available_days'].sum()),
            'calendar_days_matched':int(group.loc[matched,'calendar_days'].sum())})
        for indicator in ('n_plume_days_with_available_calendar','plume_fraction_available_calendar'):
            d=supported.loc[np.isfinite(supported[indicator])&np.isfinite(supported.component_prediction)].copy()
            d=d.sort_values(['country','site_id','year','month','lat','lon'],kind='stable').reset_index(drop=True)
            api.require(not d.duplicated(['country','site_id','year','month']).any(),'monthly model/HMS support duplicated within model/method stratum')
            nsites=d.site_id.nunique()
            sh=support_hash(d,['country','site_id','year','month','lat','lon'])
            evidence=support_hash(d,['country','site_id','year','month','lat','lon','component_prediction',indicator])
            if evidence not in cache:
                r=helpers.correlation(d,'component_prediction',indicator)
                if nsites>=2 and len(d)>=3:
                    interval,nfinite=helpers.station_correlation_interval(d,'component_prediction',indicator)
                else:
                    interval,nfinite=[np.nan,np.nan],0
                cache[evidence]=(r,interval,nfinite)
            r,interval,nfinite=cache[evidence]
            records.append({**identity,'HMS_indicator':indicator,'n_pairs':len(d),'sites':nsites,
                'pearson_r':r,'ci_low':interval[0],'ci_high':interval[1],'valid_bootstrap_reps':nfinite,
                'status':'eligible' if np.isfinite(r) else 'ineligible_insufficient_or_constant_support',
                'uncertainty_status':'eligible' if nsites>=2 and nfinite>=475 else 'ineligible_fewer_than_two_sites_or_insufficient_finite_replicates',
                'support_sha256':sh,'model_HMS_evidence_sha256':evidence,'interval_scope':INTERVAL_SCOPE})
        supported['hms_day_bin']=day_bins(supported.n_plume_days_with_available_calendar)
        for label in ('0','1-5','6-15','16+'):
            d=supported.loc[supported.hms_day_bin.eq(label)]
            bins.append({**identity,'hms_day_bin':label,'n':len(d),'sites':d.site_id.nunique(),
                'mean_component':d.component_prediction.mean(),'mean_total':d.predicted_total.mean()})
    return pd.DataFrame(records),pd.DataFrame(coverage),pd.DataFrame(bins)


def prepare(args):
    gate,_,_=load_sources()
    normalizer=module('hms_monthly_source_v1',Path(__file__).with_name('hms_monthly_source_v1.py'),args.normalizer_sha,gate)
    monthly,audit=normalizer.load_monthly(gate)
    required={'country','site_id','year','month','lat','lon','alignment','plume_definition','support_80pct'}
    api.require(required.issubset(monthly),'normalizer schema differs')
    api.require(set(monthly.year)=={2023,2024,2025} and set(monthly.alignment)==set(ALIGNMENTS)
        and set(monthly.plume_definition)==set(DEFINITIONS),'complete HMS years/definitions required')
    api.require(not monthly.duplicated(['country','site_id','lat','lon','year','month','alignment','plume_definition']).any(),
                'normalized HMS coordinate-month definitions duplicated')
    api.assert_unchanged(gate)
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/'normalized_HMS_monthly.parquet';monthly.to_parquet(path,index=False)
    plan={'format':'future-HMS-frozen-date-aware-plan-v1','status':'prepared_no_metrics_no_fit',
        'source_audit':audit,'source_pins':gate['used'],'normalizer_sha256':args.normalizer_sha,
        'normalized_monthly':{'path':str(path),'sha256':api.sha(path),'rows':len(monthly)},
        'definitions':list(DEFINITIONS),'alignments':list(ALIGNMENTS),'monthly_minimum_availability':.80,
        'monthly_indicators':['n_plume_days_with_available_calendar','plume_fraction_available_calendar'],
        'uncertainty':{'station_bootstrap_reps':500,'seed':610,'minimum_finite_fraction':.95},
        'model_inputs':'all accepted future core smoke4 and USA OC2023 saved rows, bothQA/MET/target/gates/modes',
        'OC_daily_protocol':'2023 only; >=2distinct dates both classes, equal station-month deltas within station then station bootstrap',
        'no_model_selection_or_tuning':True,'fits_performed':False,'driver_sha256':api.sha(Path(__file__))}
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':plan['status'],'normalized_rows':len(monthly),'plan':str(args.out/'plan.json')}),flush=True)


def observed_oc(gate,helpers,raw_path,out):
    daily=pd.read_parquet(raw_path)
    daily=daily.loc[daily.year.eq(2023)].copy().reset_index(drop=True)
    daily.date=pd.to_datetime(daily.date)
    daily['parameter_code']=daily.parameter_code.astype(str);daily['method_code']=daily.method_code.astype(str)
    api.require(not daily.duplicated(['site_id','parameter_code','method_code','date']).any(),'OC station-method-date duplicated')
    api.require(daily.date.dt.year.eq(daily.year).all() and daily.date.dt.month.eq(daily.month).all()
                and np.isfinite(daily.value).all(),'OC dates/finite observations differ')
    base=api.V/'uscan_final_comparator_inputs_v1_20261115/hms_2023'
    pins=gate['configuration']['HMS_postfit_input_pins']
    points=pd.read_parquet(api.pinned(base/'station_points.parquet',pins[str(base/'station_points.parquet')],gate['used']))
    d=pd.read_parquet(api.pinned(base/'hms_station_day_2023.parquet',pins[str(base/'hms_station_day_2023.parquet')],gate['used']))
    joined=exact_geometry_join(daily,points,['country','site_id'])
    joined=joined.merge(d,on=['point_id','date'],how='left',validate='many_to_one',sort=False,indicator='_day_match')
    api.require(len(joined)==len(daily),'OC daily matching lost rows')
    joined['hms_daily_matched']=joined._day_match.eq('both')&joined.hms_geometry_matched
    joined=joined.drop(columns='_day_match')
    for col in ('archive_available','archive_available_plusminus1day'):
        # Unknown unmatched dates cannot enter either observed OC contrast.
        joined[col]=joined[col].fillna(False).astype(bool)
    paired,metrics=helpers.oc_contrasts(joined)
    if metrics.empty:
        metrics=pd.DataFrame(columns=['parameter_code','method_code','alignment','plume_definition',
            'station_month_contrasts','sites','mean_station_oc_difference','ci_low','ci_high','interval_scope'])
    metrics['status']='eligible'
    expected=daily[['parameter_code','method_code']].drop_duplicates()
    existing=set(tuple(row) for row in metrics[['parameter_code','method_code','alignment','plume_definition']].itertuples(index=False,name=None))
    extras=[]
    for parameter,method in expected.itertuples(index=False,name=None):
        for alignment in ('same_named_date_EPA_local_HMS_UTC','plusminus1day_window_expansion'):
            for definition in ('any_plume','medium_or_heavy'):
                key=(parameter,method,alignment,definition)
                if key in existing:continue
                selected=paired.loc[paired.parameter_code.eq(parameter)&paired.method_code.eq(method)
                                    &paired.alignment.eq(alignment)&paired.plume_definition.eq(definition)]
                extras.append({'parameter_code':parameter,'method_code':method,'alignment':alignment,
                    'plume_definition':definition,'station_month_contrasts':len(selected),'sites':selected.site_id.nunique(),
                    'mean_station_oc_difference':selected.groupby('site_id').delta_oc.mean().mean(),
                    'ci_low':np.nan,'ci_high':np.nan,'status':'ineligible_fewer_than_two_sites',
                    'interval_scope':INTERVAL_SCOPE})
    if extras:metrics=pd.concat([metrics,pd.DataFrame(extras)],ignore_index=True)
    joined.to_parquet(out/'OC2023_all_daily_HMS_pairs.parquet',index=False)
    paired.to_parquet(out/'OC2023_within_station_month_contrasts.parquet',index=False)
    metrics.to_csv(out/'OC2023_observed_contrast_metrics.csv',index=False)
    return {'raw_2023_observation_rows':len(daily),'daily_geometry_matched_rows':int(joined.hms_geometry_matched.sum()),
        'daily_classification_matched_rows':int(joined.hms_daily_matched.sum()),
        'same_named_available_rows':int(joined.archive_available.sum()),
        'plusminus1day_complete_rows':int(joined.archive_available_plusminus1day.sum()),
        'station_month_contrasts':len(paired),'metric_rows':len(metrics),
        'minimum_distinct_dates_each_class':2,'station_first_equal_month_weighting':True,
        'medium_heavy_control':'light or no delineated plume',
        'same_named_date_interpretation':'nominal EPA Date Local to HMS UTC same named date; exact sampling window unavailable'}


def execute(args):
    gate,helpers,raw=load_sources()
    plan=json.loads(api.pinned(args.plan,args.plan_sha,gate['used']).read_text())
    api.require(plan['format']=='future-HMS-frozen-date-aware-plan-v1' and plan['status']=='prepared_no_metrics_no_fit'
        and plan['driver_sha256']==api.sha(Path(__file__)) and plan['no_model_selection_or_tuning'] is True,'HMS execution plan differs')
    for path,digest in plan['source_pins'].items():api.pinned(path,digest,gate['used'])
    approval=json.loads(api.pinned(args.review,args.review_sha,gate['used']).read_text())
    go=json.loads(api.pinned(args.root_go,args.root_go_sha,gate['used']).read_text())
    for item,decision in ((approval,'approve_HMS_date_aware_execution'),(go,'GO_HMS_date_aware_execution')):
        api.require(item['decision']==decision and item['findings']==[] and item['plan_sha256']==args.plan_sha
            and item['driver_sha256']==api.sha(Path(__file__)),'HMS execution approval differs')
    ref=plan['normalized_monthly'];monthly=pd.read_parquet(api.pinned(ref['path'],ref['sha256'],gate['used']))
    api.require(len(monthly)==ref['rows'],'normalized HMS census changed')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'pairs').mkdir()
    observed=observed_oc(gate,helpers,raw,args.out)
    all_metrics=[];all_coverage=[];all_bins=[];source_census=[];output_pins={}
    for job in gate['configuration']['jobs']:
        country,qa=job['country'],job['qa']
        result=api.ROOT/'results'/f'{country}_{qa}'
        inputs=[('PM_future',result/'smoke_oof_predictions.parquet')]
        if country=='USA':
            manifest=api.accepted_json(gate,result/'smoke_manifest.json')
            inputs.append(('OC2023_10km_PM_training_exclusion',Path(manifest['OC_comparison']['paired_rows_path'])))
        for scope,path in inputs:
            saved=pd.read_parquet(api.accepted_file(gate,path))
            api.require(saved.country.eq(country).all() and saved.qa_variant.eq(qa).all(),'saved model country/QA identity differs')
            api.require(np.isfinite(saved[['component_prediction','predicted_total','predicted_background']].to_numpy(float)).all(),
                        'saved model predictions nonfinite')
            saved['model_scope']=scope
            if scope=='PM_future':
                saved['parameter_code']='not_applicable';saved['method_code']='not_applicable'
                saved['validation_mode']=np.where(saved.task.str.contains('_rolling_'),'rolling','fixed')
                api.require(saved.task.str.match(r'T3_(fixed|rolling)_202[345]$').all(),'future model task identity differs')
                identity=MODEL_GROUP+['site_id','month','row_id']
            else:
                api.require(saved.year.eq(2023).all(),'OC prediction year differs')
                saved['parameter_code']=saved.parameter_code.astype(str);saved['method_code']=saved.method_code.astype(str)
                saved['validation_mode']='fixed_history_through_2022_OC_10km_exclusion'
                identity=MODEL_GROUP+['site_id','month']
            api.require(not saved.duplicated(identity).any(),'saved model subgroup station-month identity duplicated')
            source_census.append({'scope':scope,'country':country,'qa':qa,'path':str(path),'rows':len(saved),
                'groups':len(saved.groupby(MODEL_GROUP,dropna=False))})
            for alignment in ALIGNMENTS:
                for definition in DEFINITIONS:
                    hms=monthly.loc[monthly.alignment.eq(alignment)&monthly.plume_definition.eq(definition)].copy()
                    joined=exact_geometry_join(saved,hms,['country','site_id','year','month'])
                    # Definition identity is retained also on unmatched rows.
                    joined['alignment']=alignment;joined['plume_definition']=definition
                    joined['hms_monthly_retained']=joined.hms_geometry_matched&joined.support_80pct.fillna(False).astype(bool)
                    joined['hms_disposition']=np.select([~joined.hms_geometry_matched,~joined.hms_monthly_retained],
                        ['unknown_unmatched_actual_coordinate','excluded_less_than_80pct_available_calendar'],default='retained')
                    metrics,coverage,bins=monthly_statistics(joined,helpers)
                    all_metrics.append(metrics);all_coverage.append(coverage);all_bins.append(bins)
                    dest=args.out/'pairs'/f'{country}_{qa}_{scope}_{alignment}_{definition}.parquet'
                    joined.to_parquet(dest,index=False);output_pins[str(dest)]=api.sha(dest)
                    print(json.dumps({'scope':scope,'country':country,'qa':qa,'alignment':alignment,'definition':definition,
                                      'paired_rows':len(joined),'retained_rows':int(joined.hms_monthly_retained.sum())}),flush=True)
    metrics=pd.concat(all_metrics,ignore_index=True);coverage=pd.concat(all_coverage,ignore_index=True);bins=pd.concat(all_bins,ignore_index=True)
    api.require(len(source_census)==6 and sum(r['rows'] for r in source_census if r['scope']=='PM_future')==1383216
                and sum(r['rows'] for r in source_census if r['scope']!='PM_future')==109584,'complete saved model census differs')
    metrics.to_csv(args.out/'monthly_HMS_correlations.csv',index=False)
    coverage.to_csv(args.out/'monthly_HMS_coverage_all_strata.csv',index=False)
    bins.to_csv(args.out/'monthly_HMS_day_bins.csv',index=False)
    for path in args.out.iterdir():
        if path.is_file():output_pins[str(path)]=api.sha(path)
    api.assert_unchanged(gate)
    audit={'format':'future-HMS-date-aware-actual-diagnostics-v1','status':'PASS','findings':[],
        'source_pins':gate['used'],'plan_sha256':args.plan_sha,'source_normalization':plan['source_audit'],
        'model_source_census':source_census,'observed_OC2023':observed,'monthly_correlation_rows':len(metrics),
        'coverage_rows':len(coverage),'bin_rows':len(bins),'outputs':output_pins,'fits_performed':False,
        'scope':'all saved future PM smoke and OC2023 component-to-HMS consistency; all alignments/definitions/QA/MET/targets/gates/modes',
        'interpretation':'satellite-visible plumes can include elevated smoke; OC has other sources; no identified ground smoke PM mass or causal timing',
        'reporting':'all signs, constant/ineligible and unmatched groups retained; repeated model_HMS_evidence_sha256 values denote repeated model evidence',
        'driver_sha256':api.sha(Path(__file__))}
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','correlation_rows':len(metrics),'coverage_rows':len(coverage),'bin_rows':len(bins)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--prepare',action='store_true');modes.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--normalizer-sha')
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha');p.add_argument('--review',type=Path);p.add_argument('--review-sha')
    p.add_argument('--root-go',type=Path);p.add_argument('--root-go-sha')
    a=p.parse_args()
    if a.prepare:
        api.require(a.normalizer_sha is not None,'explicit normalizer source hash required');prepare(a)
    else:
        api.require(all((a.plan,a.plan_sha,a.review,a.review_sha,a.root_go,a.root_go_sha)),'explicit plan and approvals required');execute(a)
