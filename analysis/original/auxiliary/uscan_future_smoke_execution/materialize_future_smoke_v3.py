"""Materialize accepted future smoke rows and actual external core bindings, no fit."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','REPAIR_V2_SMOKE_THREADS'):
    os.environ[name]='1'

import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK))
from future28_actual_acceptance import future28_gate_v1 as gate

V=gate.V
BASE=V.parent
SOURCE=BASE/'smoke/final_AUS_core_bound_20261114/runtime'
MATRIX=V/'future_bound_matrix_v2_20261115/bound_no_fit_matrix.json'
COMPAT=WORK/'uscan_future_smoke_binding/core_split_binding_compat_v1.py'
COMPAT_SHA='8ab5d94fabae104764d857d886cee0f02e43d5bc3a96d372007cfbeda0259ad0'
CODE_PINS={
 'smoke/run_smoke_validation.py':'3624d294dde1702e8fa5c8918584a13ebba45c4347683284c1171c69148c1679',
 'smoke/core_split_binding.py':'cef7f8c262304f750c2086bf76089b1a50540c94c4c954356cb53b99549fc43e',
 'validation/run_validation.py':'05f4a584d9f9bfef3cf3de627345d95d4e76558594672e69594dc79d0b3b287c',
 'validation/validation_core.py':'027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157'}
PY='/external/fire_downscaling_20261107/env/bin/python3.11'
KEYS=['country','site_id','year','month']


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def ref(path):
    return {'path':str(Path(path).resolve()),'sha256':gate.sha(path)}


def primary_jobs(matrix):
    selected=[j for j in matrix['jobs'] if j['analysis_role']=='primary' and
        j['dataset_policy'] in {'usa_primary','can_primary_all_published'}]
    wanted={(c,qa,mode,year) for c,years in {'USA':{'fixed':[2023,2024,2025],'rolling':[2024,2025]},
            'CAN':{'fixed':[2023,2024],'rolling':[2024]}}.items()
            for mode,yy in years.items() for year in yy for qa in ('strict','sensitivity')}
    actual={(j['country'],j['qa_variant'],j['validation_mode'],j['external_year']) for j in selected}
    gate.require(len(selected)==16 and actual==wanted,'exact future primary roster differs')
    return selected


def exact_union(frames,columns):
    """Retain first source order, rejecting duplicate predictor/target disagreement."""
    for frame in frames:
        gate.require(set(columns).issubset(frame.columns),'accepted source misses smoke model/QA columns')
        gate.require(not frame.duplicated(KEYS).any(),'source contains duplicate physical keys')
    combined=pd.concat([f[columns] for f in frames],ignore_index=True)
    repeated=combined.loc[combined.duplicated(KEYS,keep=False)]
    value_columns=[c for c in columns if c not in KEYS]
    for key,part in repeated.groupby(KEYS,sort=False):
        for column in value_columns:
            values=part[column]
            first=values.iloc[0]
            same=values.isna() if pd.isna(first) else values.eq(first)
            gate.require(bool(same.all()),f'duplicate source predictor/target differs: {key}/{column}')
    result=combined.drop_duplicates(KEYS,keep='first').reset_index(drop=True)
    gate.require(np.isfinite(result.pm25.to_numpy(float)).all() and result.pm25.ge(0).all(),
                 'union has invalid assigned targets')
    gate.require(np.isfinite(result[['lat','lon']].to_numpy(float)).all(),'union has invalid coordinates')
    return result


def import_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def adapt_smoke_loader(text):
    """Remove the unused raw gate-source loader requirement; keep all math."""
    replacements={
        '        "gfas_emission_r15", *FIRE,':'        *FIRE,',
        '    for col in (GATE_FEATURE, "gfas_emission_r15"):':'    for col in (GATE_FEATURE,):',
    }
    for old,new in replacements.items():
        gate.require(text.count(old)==1,'frozen unused-raw-column loader anchor differs')
        text=text.replace(old,new)
    gate.require('gfas_emission_r15' not in text,'unexpected raw gate column calculation in frozen runner')
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    accepted=gate.load_gate()
    matrix=gate.accepted_json(accepted,MATRIX,gate.MATRIX_SHA)
    selected=primary_jobs(matrix)
    gate.pinned(COMPAT,COMPAT_SHA)
    gate.require(not args.out.exists(),'fresh smoke staging root required')
    for name,digest in CODE_PINS.items():
        gate.pinned(SOURCE/'work/repair_v2'/name,digest)
    # Clone only this runtime. The accepted historical and external producers stay immutable.
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(SOURCE,args.out/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cloned={}
    for path in SOURCE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            dest=args.out/'runtime'/path.relative_to(SOURCE)
            gate.require(gate.sha(path)==gate.sha(dest),'cloned runtime byte mismatch')
            cloned[str(dest)]=gate.sha(dest)
    code=args.out/'runtime/work/repair_v2'
    shutil.copyfile(COMPAT,code/'smoke/core_split_binding.py')
    cloned[str(code/'smoke/core_split_binding.py')]=COMPAT_SHA
    smoke_path=code/'smoke/run_smoke_validation.py'
    smoke_path.write_text(adapt_smoke_loader(smoke_path.read_text()),encoding='utf-8')
    cloned[str(smoke_path)]=gate.sha(smoke_path)
    sys.path.insert(0,str(code/'smoke'))
    smoke=import_module('future_smoke_materializer_runtime',code/'smoke/run_smoke_validation.py')
    # Read original future features/targets only through the accepted full census.
    source_frames={}
    lineage={}
    for job in selected:
        for feature_role,target_role in [('training_features','training_target_source'),
                                        ('external_features','external_target_source')]:
            feature_ref=job['input_bindings'][feature_role]
            target_ref=job['input_bindings'][target_role]
            key=(feature_ref['path'],target_ref['path'])
            if key in source_frames:
                continue
            fp=gate.accepted_file(accepted,feature_ref['path'],feature_ref['sha256'])
            tp=gate.accepted_file(accepted,target_ref['path'],target_ref['sha256'])
            # _load and replace_targets are the original accepted external source contract.
            rr=job['runner']
            rp=gate.accepted_file(accepted,rr['path'],rr['sha256'])
            external=import_module('future_smoke_external_source',rp)
            frame=external._load(fp,job['country'])
            frame,status=external.replace_targets(frame,tp,country=job['country'],target_column='pm25')
            source_frames[key]=frame
            lineage[key]={'features':feature_ref,'assigned_target':target_ref,'assigned_target_status':status,
                          'rows':len(frame)}
    reports=[]
    for country in ('USA','CAN'):
        for qa in ('strict','sensitivity'):
            group=[j for j in selected if j['country']==country and j['qa_variant']==qa]
            keys=[]
            for j in group:
                for f,t in [('training_features','training_target_source'),('external_features','external_target_source')]:
                    key=(j['input_bindings'][f]['path'],j['input_bindings'][t]['path'])
                    if key not in keys:
                        keys.append(key)
            columns=list(dict.fromkeys([*KEYS,'lat','lon','pm25',*smoke.GEO,*smoke.MET,'aod_raw',*smoke.AOD,
                *smoke.FIRE,smoke.GATE_FEATURE,'maiac_qa_definition',
                'maiac_cell_geometry_verified','maiac_qa_cell_month_present']))
            union=exact_union([source_frames[k] for k in keys],columns)
            gate.require(union.country.eq(country).all() and union.maiac_qa_definition.eq(qa).all(),
                         'union country/QA differs')
            gate.require(union.maiac_cell_geometry_verified.eq(True).all()
                         and union.maiac_qa_cell_month_present.eq(True).all(),'source geometry/QA incomplete')
            directory=args.out/f'{country}_{qa}'
            directory.mkdir()
            feature=directory/'features.parquet'
            target=directory/'assigned_targets.parquet'
            union.to_parquet(feature,index=False)
            union[KEYS+['lat','lon','pm25']].to_parquet(target,index=False)
            reread=pd.read_parquet(feature)
            pd.testing.assert_frame_equal(union,reread,check_exact=True)
            identity={'format':'repair-v2-derived-future-smoke-union-identity-v1',
                'purpose':'exact accepted predictor/assigned-target union; original source QA inherited, not rewritten',
                'parent_future_acceptance':accepted['acceptance_ref'],'parent_root_adoption':accepted['adoption_ref'],
                'source_lineage':[lineage[k] for k in keys],
                'geometry_complete':True,'source_coverage_audited':True,'raw_source_gap_audit_pending':False,
                'qa_variants':[qa],'union_audit':{'geometry_complete':True,'source_coverage_audited':True,
                    'raw_source_gap_audit_pending':False,'duplicate_physical_keys_agree':True,
                    'model_predictors_and_assigned_targets_unchanged':True},
                'outputs':{qa:{**ref(feature),'rows':len(union),'qa_variant':qa,
                    'persisted_output_verification':{'source_key_order_preserved':True,'target_unchanged':True,
                        'geometry_verified_rows':len(union),'cell_month_qa_matched_rows':len(union)}}}}
            identity_path=directory/'qa_identity.json'
            save(identity_path,identity)
            bindings=[];slots=[]
            for profile in ('with_met','no_met'):
                external_refs=[]
                for j in group:
                    manifest_path=Path(j['output_root'])/'external_run_manifest.json'
                    manifest=gate.accepted_json(accepted,manifest_path)
                    status_path=Path(accepted['acceptance']['controller_status']['path']).parent/(j['job_id']+'.status.json')
                    gate.accepted_file(accepted,status_path)
                    expected_ref=manifest['final_multiplicity_expected_test_keys']
                    gate.accepted_file(accepted,expected_ref['path'],expected_ref['sha256'])
                    task=f"T3_{j['validation_mode']}_{j['external_year']}"
                    external_refs.append({'smoke_task':task,'smoke_fold_id':str(j['external_year']),
                        'core_task':f"T4_external_{j['external_year']}_{profile}",
                        'manifest':ref(manifest_path),'execution_receipt':ref(status_path),
                        'execution_context':{'controller_status':accepted['acceptance']['controller_status'],
                                             'matrix':{'path':str(MATRIX),'sha256':gate.MATRIX_SHA}},
                        'expected_test_keys':{'path':expected_ref['path'],'sha256':expected_ref['sha256']}})
                    slots.append({'dataset_key':country,'country':country,'qa_variant':qa,
                                  'profile':profile,'task':task,'fold_id':str(j['external_year'])})
                bindings.append({'dataset_key':country,'country':country,'qa_variant':qa,'profile':profile,
                                 'external':external_refs,'source_remaps':{}})
            registry_path=directory/'registry.json'
            save(registry_path,{'format':'repair-v2-smoke-core-binding-registry-v1','mode':'final',
                'bindings':bindings,'requested_slots':slots,'parent_acceptance':accepted['acceptance_ref']})
            fixed=sorted({j['external_year'] for j in group if j['validation_mode']=='fixed'})
            rolling=sorted({j['external_year'] for j in group if j['validation_mode']=='rolling'})
            registry,base,_=smoke.load_binding_registry(registry_path)
            cfg=argparse.Namespace(tasks=['T3_fixed','T3_rolling'],fixed_external_years=fixed,
                rolling_external_years=rolling,profiles=['with_met','no_met'],qa_provenance=[str(identity_path)])
            context=smoke._preflight_final_registry(args=cfg,selected_keys=[country],
                feature_overrides={country:[feature]},target_overrides={country:[target]},
                binding_registry=registry,binding_base=base,config=smoke.ModelConfig(),
                requested_smear_splits=3,smear_splits_defaulted=False)
            partitions=[]
            for profile,splits in context[country]['bound_by_profile'].items():
                gate.require(len(splits)==len(group),'actual core-binding census differs')
                for split in splits:
                    partitions.append({'profile':profile,'task':split['task'],'fold_id':split['fold_id'],
                        'train_rows':len(split['train']),'test_rows':len(split['test']),
                        'core_train_row_key_sha256':split['core_train_row_key_sha256'],
                        'core_test_row_key_sha256':split['core_test_row_key_sha256'],
                        'training_support_fingerprint':split['core_training_support_fingerprint']})
            argv=[PY,'-u',str(code/'smoke/run_smoke_validation.py'),'--out',str(args.out/'results'/f'{country}_{qa}'),
                '--countries',country,'--tasks','T3_fixed,T3_rolling','--fixed-external-years',','.join(map(str,fixed)),
                '--rolling-external-years',','.join(map(str,rolling)),'--profiles','with_met,no_met',
                '--gate-quantiles','0.80,0.90,0.95','--target-strategies','log_smear,original_scale',
                '--max-iter','400','--smear-splits','3','--bootstrap-replicates','500',
                '--feature-input',country+'='+str(feature),'--target-input',country+'='+str(target),
                '--qa-provenance',str(identity_path),'--core-split-binding-registry',str(registry_path),
                '--no-can-exclusion-sensitivity','--skip-oc']
            report={'country':country,'qa':qa,'feature':ref(feature),'target':ref(target),
                'identity':ref(identity_path),'registry':ref(registry_path),'partitions':partitions,
                'cli_argv':argv,'fits_performed':False,'models_and_scores_read':False,
                'comparator_and_OC_runtime_configuration_pending':True}
            save(directory/'actual_no_fit_preflight.json',report)
            reports.append(report)
            print(json.dumps({'phase':'preflight_pass','country':country,'qa':qa,'slots':len(partitions)}),flush=True)
    gate.assert_unchanged(accepted)
    save(args.out/'materialization.json',{'format':'future-smoke-actual-no-fit-materialization-v3',
        'status':'accepted_sources_actual_binding_preflight_passed','fits_performed':False,
        'scores_computed':False,'independent_materialization_review_required':True,
        'compatibility_binder':ref(COMPAT),'cloned_runtime_pins':cloned,
        'smoke_loader_compatibility':{'original_sha256':CODE_PINS['smoke/run_smoke_validation.py'],
            'adapted_runner':ref(smoke_path),'reason':'unused raw gfas_emission_r15 loader-only requirement removed',
            'model_predictors_thresholds_and_calculations_changed':False},
        'accepted_source_pins_used':accepted['used'],'jobs':reports,
        'OC_HMS_comparator_configuration_pending':True})
    print(json.dumps({'status':'prepared_no_fit','jobs':len(reports),'output':str(args.out)}),flush=True)


if __name__=='__main__':
    main()
