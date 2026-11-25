"""Fixed exploratory metrics on independently accepted exact annual support."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import pandas as pd
import shared_support_v1 as support
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'future28_actual_acceptance'))
import future28_gate_v1 as future_gate

V=Path('/data/admin/fengshuo/cand1_repair_v2_20261112/validation')
PAIR=V/'fair_totalpm_actual_pairing_v1_20261115'
PAIR_SHA='4fd1e822f0df75f636423cffb241939140f20cecffa0235770e61e699c8c8b85'
ACCEPTANCE=V/'fair_totalpm_pairing_independent_acceptance_v1_20261115/independent_acceptance_gate_v1.json'
ACCEPTANCE_SHA='c1b18350a74e490dc68715adfde7da886b285c270355167a40042907beed1ad0'
RUNTIME=V/'final_model_runtime_v4_20261114/source_snapshot/validation'
RUNNER_SHA='05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c'
HELPER=V/'global_current_comparison_20261114/compare_cams.py'
HELPER_SHA='adfe83af59cc9ba7e6889a8229487e3961268b0b71f69bfe7b4bcc739462444c'
VERSIONS={'python_version':'3.11.16','numpy_version':'2.4.6','pandas_version':'2.3.3','scikit_learn_version':'1.9.1'}
METRICS=('rmse','mae','bias','r2')
SUPPORTS={'primary_exact_12_months':'primary_month_support','sensitivity_at_least_10_months':'sensitivity_month_support'}
REPS=500
GROUPS=[c for c in support.MODEL_IDENTITY if c!='fold_id']+['country']


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def pinned(path,digest,bound):
    path=Path(path).resolve()
    require(sha(path)==digest,'file hash mismatch: '+str(path))
    bound[str(path)]=digest
    return path


def load_module(name,path,digest,bound):
    path=pinned(path,digest,bound)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def acceptance_gate(digest,bound):
    require(digest==ACCEPTANCE_SHA,'independent acceptance SHA differs from actual accepted gate')
    acceptance=json.loads(pinned(ACCEPTANCE,digest,bound).read_text())
    require(acceptance.get('format')=='fair-totalpm-independent-pairing-acceptance-gate-v1'
        and acceptance.get('decision')=='accepted' and acceptance.get('independent_pairing_pass') is True
        and acceptance.get('audited_cells')==176 and acceptance.get('pairing_report')==
        {'path':str(PAIR/'pairing_report.json'),'sha256':PAIR_SHA},'independent pairing acceptance differs')
    for path,digest in acceptance['accepted_artifact_pins'].items():
        pinned(path,digest,bound)
    report=json.loads(pinned(PAIR/'pairing_report.json',PAIR_SHA,bound).read_text())
    require(report['status']=='paired_no_metrics' and report['historical_cells']==64 and report['future_cells']==112
        and report['monthly_rows']==1867592 and report['annual_rows']==166352
        and report['model_fitting_performed'] is False and report['metrics_computed'] is False,
        'accepted pairing report census differs')
    for path,value in report['outputs'].items():
        pinned(path,value,bound)
    for path,value in report['bound_sources'].items():
        pinned(path,value,bound)
    return acceptance,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--go',type=Path)
    parser.add_argument('--go-sha')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    require(not args.out.exists(),'fresh fair metric output required')
    bound={}
    acceptance,report=acceptance_gate(args.acceptance_sha,bound)
    protocol={'cells':176,'support_rules':SUPPORTS,'benchmark_version':'V6GL03','arms':['full','raw_CAMS','SatPM'],
        'contrasts':['full minus raw_CAMS','full minus SatPM'],'metric_rows':1056,'contrast_rows':704,
        'bootstrap_replicates':REPS,'bootstrap_engine':'cluster_stats','clusters':'physical site_id within each cell',
        'annual_weighting':'calendar days on exactly shared finite monthly core/CAMS/SatPM support',
        'metric_weighting':'equal accepted station-year observations within a cell/support rule',
        'scope':'secondary exploratory conditional on fitted predictions and observed years/stations; outside locked maxT family',
        'uncertainty_ineligible_if':'fewer than two distinct eligible physical sites; point metrics still retained',
        'NA01':'retained as ineligible: accepted monthly values/month-set unavailable; no substitution',
        '2025':'SatPM unavailable; all source loss retained in paired reports and ineligible metric rows',
        'unfavorable_results_retained':True,'model_fitting_performed':False}
    protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if not args.run:
        print(json.dumps({'status':'check_only_passed','protocol':protocol,'protocol_sha256':protocol_sha,
            'pairing_acceptance_sha256':args.acceptance_sha,'metrics_computed':False}),flush=True)
        return
    require(args.go is not None and args.go_sha is not None,'separate reviewed root metric GO required')
    go=json.loads(pinned(args.go,args.go_sha,bound).read_text())
    require(go.get('format')=='fair-totalpm-metrics-root-go-v1' and go.get('decision')=='go'
        and go.get('driver_sha256')==sha(Path(__file__)) and go.get('protocol_sha256')==protocol_sha
        and go.get('pairing_acceptance_sha256')==args.acceptance_sha
        and go.get('output_root')==str(args.out),'root fair metrics GO differs')
    pinned(Path(__file__),go['driver_sha256'],bound)
    rr=go['independent_review_ref']
    review=json.loads(pinned(rr['path'],rr['sha256'],bound).read_text())
    require(review.get('decision')=='approve_fair_totalpm_metrics' and review.get('findings')==[]
        and review.get('driver_sha256')==go['driver_sha256'],'independent metric-code review differs')
    # No annual prediction or target values are opened until every metric gate passed.
    annual=pd.read_parquet(PAIR/'annual_shared_support.parquet')
    require(len(annual)==report['annual_rows'] and not annual.duplicated(GROUPS+['fold_id','site_id','year']).any(),
             'annual paired row identity differs')
    sys.path.insert(0,str(RUNTIME))
    future=future_gate.load_gate()
    matrix=future_gate.accepted_json(future,V/'future_bound_matrix_v2_20261115/bound_no_fit_matrix.json',future_gate.MATRIX_SHA)
    for source in matrix['source_bindings']['runtime']['sources'].values():
        future_gate.accepted_file(future,source['path'],source['sha256'])
    bound.update(future['used'])
    runner=load_module('run_validation',RUNTIME/'run_validation.py',RUNNER_SHA,bound)
    helper=load_module('fair_totalpm_frozen_seed',HELPER,HELPER_SHA,bound)
    runtime=runner.runtime_provenance(RUNTIME/'run_validation.py','cluster_stats')
    require({k:runtime.get(k) for k in VERSIONS}==VERSIONS,'canonical metric runtime versions differ')
    for name,digest in runtime['code_sha256'].items():
        path=Path(runtime['bootstrap']['helper_path']) if name=='fast_cluster_bootstrap' else {
            'runner':RUNTIME/'run_validation.py','run_validation_helpers':RUNTIME/'run_validation.py',
            'validation_core':RUNTIME/'validation_core.py','support_diagnostics':RUNTIME/'support_diagnostics.py',
            'conformal_sensitivity':RUNTIME/'conformal_sensitivity.py'}[name]
        # Expected helper hashes are inherited from the accepted original-core runtime receipts.
        require(str(path.resolve()) in bound and bound[str(path.resolve())]==digest,
                 'metric runtime code lacks accepted source pin: '+str(path))
    metrics=[];deltas=[];roster=[]
    groups=list(annual.groupby(GROUPS,sort=True,dropna=False))
    require(len(groups)==176,'all 176 model cells must remain present')
    for labels,part in groups:
        identity=dict(zip(GROUPS,labels))
        require(part.benchmark_version.eq('V6GL03').all(),'annual benchmark version differs')
        for scope,flag in SUPPORTS.items():
            selected=part.loc[part[flag].eq(True)].copy()
            require(not selected.duplicated(['site_id','year']).any(),'annual station-year crosses held-out folds')
            rows=len(selected);sites=int(selected.site_id.nunique())
            base={**identity,'support_scope':scope,'benchmark_version':'V6GL03','rows':rows,'sites':sites,
                'source_annual_rows':len(part),'excluded_annual_rows':len(part)-rows,
                'status':'eligible' if rows else 'ineligible_no_exact_shared_support'}
            y=selected.observed_annual_pm25.to_numpy(float)
            columns={'full':'model_annual_pm25','raw_CAMS':'raw_cams_annual_pm25','SatPM':'satpm_annual_pm25'}
            computed={}
            for arm,column in columns.items():
                values=selected[column].to_numpy(float)
                require(np.isfinite(y).all() and np.isfinite(values).all(),'eligible paired annual vectors nonfinite')
                value=runner._metric(y,values)
                computed[arm]=value
                metrics.append({**base,'arm':arm,**value})
            for control in ('raw_CAMS','SatPM'):
                seed=helper.seed_for(('fair-totalpm-v1',str(identity['alias']),str(identity['country']),
                    str(identity['qa']),str(identity['profile']),str(identity['estimator']),str(identity['task']),
                    str(identity['target_strategy']),str(identity['backtransform']),scope,'V6GL03',control))
                boot=runner._paired_bootstrap(selected.site_id.to_numpy(str),y,
                    selected.model_annual_pm25.to_numpy(float),selected[columns[control]].to_numpy(float),
                    REPS,seed,engine='cluster_stats') if rows and sites>=2 else {}
                row={**base,'contrast':'full minus '+control,'bootstrap_seed':seed,'bootstrap_replicates':REPS,
                    'uncertainty_status':'eligible' if sites>=2 else 'ineligible_fewer_than_two_sites',
                    'inference_scope':protocol['scope']}
                for metric in METRICS:
                    row['delta_'+metric]=computed['full'][metric]-computed[control][metric]
                    row['delta_'+metric+'_ci_low']=boot.get('delta_'+metric+'_lo',float('nan'))
                    row['delta_'+metric+'_ci_high']=boot.get('delta_'+metric+'_hi',float('nan'))
                deltas.append(row)
            roster.append(base)
        print(json.dumps({'cells_scored':len(roster)//2,'total':176}),flush=True)
    require(len(metrics)==1056 and len(deltas)==704,'fixed metric/contrast grid incomplete')
    args.out.mkdir(parents=True)
    outputs={}
    for name,frame in (('matched_annual_metrics.csv',pd.DataFrame(metrics)),('paired_annual_deltas.csv',pd.DataFrame(deltas)),
                       ('annual_support_roster.csv',pd.DataFrame(roster))):
        path=args.out/name;frame.to_csv(path,index=False);outputs[name]=sha(path)
    for path,digest in bound.items():
        require(sha(path)==digest,'artifact changed during scoring: '+path)
    result={'format':'fair-totalpm-fixed-shared-support-metrics-v1','status':'complete','protocol':protocol,
        'protocol_sha256':protocol_sha,'pairing_acceptance_sha256':args.acceptance_sha,
        'metric_rows':len(metrics),'contrast_rows':len(deltas),'bound_sources':bound,'output_sha256':outputs,
        'driver_sha256':sha(Path(__file__)),'model_fitting_performed':False,
        'NA01_ineligible_disposition':report['benchmark_versions']['V6NA01']}
    with (args.out/'completion_manifest.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':'complete','metric_rows':1056,'contrast_rows':704}),flush=True)


if __name__=='__main__':
    main()
