"""Descriptive future support/P90 retention on accepted exact core/smoke pairs."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import pandas as pd
import future_core_gate_v1 as gate_api
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'future28_actual_acceptance'))
import future28_gate_v1 as future_gate

JOINT_SHA='780fd37472e3241334cea697dd688f039f99a8d4f91102e28cb5196e3e7711b4'
SUPPORT_SHA='b7c88d905b5ece5a57f26fae9cbd4236b2da41248b37d70ddd85dce5367353c8'


def load_module(name,path,digest,gate):
    path=gate_api.pinned(path,digest,gate['used'])
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    gate=gate_api.load_gate()
    gate_api.require(not args.out.exists(),'fresh joint-support output required')
    gate_api.pinned(Path(__file__),gate_api.sha(Path(__file__)),gate['used'])
    gate_api.pinned(gate_api.__file__,'93f7f5903ecb73092bca57df9a386e8c6899ce7b2fb0bc57c5b33f1058716d00',gate['used'])
    gate_api.pinned(future_gate.__file__,'2a8fca8fafcf8e15de3b7485ce409ac3009e5b822fe0f544eee37fde2ac1f650',gate['used'])
    future=future_gate.load_gate()
    matrix=future_gate.accepted_json(future,gate_api.V/'future_bound_matrix_v2_20261115/bound_no_fit_matrix.json',future_gate.MATRIX_SHA)
    for source in matrix['source_bindings']['runtime']['sources'].values():
        future_gate.accepted_file(future,source['path'],source['sha256'])
    legacy=load_module('approved_joint_support_helpers',Path(__file__).with_name('historical_joint_helpers.py'),JOINT_SHA,gate)
    validation=gate_api.V/'final_model_runtime_v4_20261114/source_snapshot/validation'
    sys.path.insert(0,str(validation))
    support=load_module('frozen_joint_support_rules',validation.parent/'checks/assess_deployment_support.py',SUPPORT_SHA,gate)
    jobs=[j for j in matrix['jobs'] if j['analysis_role']=='primary' and j['dataset_policy'] in
          {'usa_primary','can_primary_all_published'}]
    gate_api.require(len(jobs)==16 and len({j['job_id'] for j in jobs})==16,'exact16 primary future jobs required')
    smoke={};all_pairs=[];all_summary=[];mappings=[]
    for job in jobs:
        country,qa=job['country'],job['qa_variant']
        if (country,qa) not in smoke:
            path=gate_api.accepted_file(gate,gate_api.ROOT/'results'/f'{country}_{qa}'/'smoke_oof_predictions.parquet')
            smoke[(country,qa)]=pd.read_parquet(path)
        root=Path(job['output_root'])
        original=future_gate.accepted_json(future,root/'external_run_manifest.json')
        predictions=pd.read_parquet(future_gate.accepted_file(future,root/'external_oof_predictions.parquet'))
        flags=pd.read_parquet(future_gate.accepted_file(future,root/'support_row_flags.parquet'))
        paired,_=support.join_support(predictions,flags)
        year=int(job['external_year']);task=f"T3_{job['validation_mode']}_{year}"
        expected_core={f'T4_external_{year}_{p}' for p in ('with_met','no_met')}
        gate_api.require(set(paired.task.astype(str))==expected_core and paired.year.eq(year).all(),
                         'original future core task/year differs')
        paired['source_core_task']=paired.task.astype(str)
        paired['task']=task;paired['qa_variant']=qa
        for key,value in [('source_job_id',job['job_id']),('dataset_policy',job['dataset_policy']),
                          ('validation_mode',job['validation_mode']),('train_through',int(job['train_through']))]:
            paired[key]=value
        event=smoke[(country,qa)]
        event=event.loc[event.task.eq(task)&np.isclose(event.gate_quantile.to_numpy(float),.90)].copy()
        event=legacy.select_supported_target_branches(event)
        paired.fold_id=paired.fold_id.astype(str);event.fold_id=event.fold_id.astype(str)
        matched=legacy.exact_p90_pair(paired,event)
        summary=legacy.summarize_coverage(matched,support.RULES,support.metrics)
        gate_api.require(len(summary)==96,'one future source job must retain96 support/event/output rows')
        for key,value in [('source_job_id',job['job_id']),('dataset_policy',job['dataset_policy']),
                          ('validation_mode',job['validation_mode']),('train_through',int(job['train_through']))]:
            summary[key]=value
        all_pairs.append(matched);all_summary.append(summary)
        mappings.append({'source_job_id':job['job_id'],'original_core_tasks':sorted(expected_core),
            'derived_smoke_task':task,'year':year,'train_through':int(original['train_through']),
            'paired_rows':len(matched),'source_files_modified':False})
        print(json.dumps({'source_jobs_joined':len(mappings),'total':16,'job':job['job_id']}),flush=True)
    paired=pd.concat(all_pairs,ignore_index=True);summary=pd.concat(all_summary,ignore_index=True)
    gate_api.require(len(paired)==461072 and len(summary)==1536,'complete futureP90/support census differs')
    future_gate.assert_unchanged(future);gate['used'].update(future['used']);gate_api.assert_unchanged(gate)
    args.out.mkdir(parents=True)
    paired_path=args.out/'exact_future_core_P90_support_pairs.parquet';paired.to_parquet(paired_path,index=False)
    summary_path=args.out/'coverage_summary.csv';summary.to_csv(summary_path,index=False)
    with (args.out/'audit.json').open('x') as stream:
        json.dump({'format':'future-smoke-joint-support-P90-audit-v1','status':'PASS','findings':[],
            'paired_rows':len(paired),'summary_rows':len(summary),'source_job_mappings':mappings,
            'source_pins':gate['used'],'rules':support.RULES,'fits_performed':False,
            'scope':'descriptive fixed-rule support/retained-excluded error and P90 fire-feature-proxy retention; no identified smoke mass',
            'outputs':{str(paired_path):gate_api.sha(paired_path),str(summary_path):gate_api.sha(summary_path)},
            'auditor_sha256':gate_api.sha(Path(__file__))},stream,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASS','paired_rows':len(paired),'summary_rows':len(summary)}),flush=True)


if __name__=='__main__':
    main()
