"""Bind accepted comparator and OC inputs to the no-fit future smoke stage."""
from pathlib import Path
import copy
import json
import shutil
import sys

import materialize_future_smoke_v3 as stage

V=stage.V
ROOT=V/'uscan_future_smoke_stage_v3_20261115'
MATERIALIZATION_SHA='ed799fb30ee520c0139e5bedb7b655b69fc99383b21f27d28960721879b36b92'
TRANSPORT=V/'uscan_final_comparator_inputs_v1_20261115'
TRANSPORT_SHA='80ac49a3bc6720ccf02294286096773576caa87442baf6f2664952f7ee80a9a4'
OC_REPORT=V/'oc2023_independent_acceptance_v1_20261115/report.json'
OC_SHA='b7bca33e9530b9ece61f2e2eacfe584df0ee7aa8d237b9dd89be68fe59d62884'
GLOBAL=V.parent/'smoke/final_AUS_core_bound_20261114/global_comparators'
OC_SIDECARS={'strict':'b913a5d2e0db7da45af7a427a6285bd3a8d78e3a351c2ce2a67f2b7e70baaafa',
             'sensitivity':'beb7b7751f929b3a57088a45ee3d53df1d99c53a14b8c6026e9ff12f4eca8c57'}


def pinned_json(path,digest,bound):
    path=stage.gate.pinned(path,digest)
    bound[str(path)]=digest
    return json.loads(path.read_text())


def main():
    accepted=stage.gate.load_gate()
    bound={}
    material=pinned_json(ROOT/'materialization.json',MATERIALIZATION_SHA,bound)
    stage.gate.require(material['status']=='accepted_sources_actual_binding_preflight_passed'
        and material['fits_performed'] is False and material['scores_computed'] is False,'actual no-fit stage required')
    for pins in (material['cloned_runtime_pins'],material['accepted_source_pins_used']):
        for path,digest in pins.items():
            stage.gate.pinned(path,digest);bound[path]=digest
    transport=pinned_json(TRANSPORT/'transport_inventory.json',TRANSPORT_SHA,bound)
    oc=pinned_json(OC_REPORT,OC_SHA,bound)
    stage.gate.require(oc['status']=='accepted' and oc['checks_passed']==6090,'OC source QA not accepted')
    original_oc=TRANSPORT/'oc/oc_primary_same_method_matched_2018_2025.parquet'
    oc_source_sha=transport['pins'][str(original_oc)]['sha256']
    stage.gate.require(oc_source_sha=='05a0666dac829bca3755af6ab2d4116f0fc36a7bb1317876d7ed51e73af11117',
                       'OC observation source binding differs')
    stage.gate.pinned(original_oc,oc_source_sha);bound[str(original_oc)]=oc_source_sha
    dest=ROOT/'runtime/outputs/repair_v2/epa/oc_primary_same_method_matched_2018_2025.parquet'
    stage.gate.require(not dest.exists(),'fresh OC runtime destination required')
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original_oc,dest)
    stage.gate.pinned(dest,oc_source_sha);bound[str(dest)]=oc_source_sha
    qfed=GLOBAL/'qfed_station_month.parquet';gfed=GLOBAL/'gfed_station_month.parquet'
    for path,digest in ((qfed,'a710d44f4b00d11c8e606c6151e62de9b5a69f4f66b582242d707e29f4c9ecdf'),
                        (gfed,'38c639d9fd549404fc1d6ceaea711adab1c4ae1dd6ce66a38d9adfcb493f0fa4')):
        stage.gate.pinned(path,digest);bound[str(path)]=digest
    # HMS tables remain separate date-aware postfit diagnostics, with unknown dates retained.
    hms={}
    for path,record in transport['pins'].items():
        if '/hms_2023/' in path or '/hms_2024_2025/' in path:
            stage.gate.pinned(path,record['sha256']);bound[path]=record['sha256'];hms[path]=record['sha256']
    jobs=[]
    for item in material['jobs']:
        job=copy.deepcopy(item)
        for name in ('feature','target','identity','registry'):
            rr=job[name];stage.gate.pinned(rr['path'],rr['sha256']);bound[rr['path']]=rr['sha256']
        argv=job['cli_argv']
        argv+=['--qfed-path',str(qfed),'--gfed-path',str(gfed)]
        if job['country']=='USA':
            path=V/'oc2023_final_qa_v1_20261115/prepared_v1'/f"oc_grid_features_2023_usa_{job['qa']}.parquet"
            digest=oc['evidence'][str(path)]
            stage.gate.pinned(path,digest);bound[str(path)]=digest
            side=path.with_suffix('.provenance.json')
            sidecar=pinned_json(side,OC_SIDECARS[job['qa']],bound)
            stage.gate.require(sidecar['output_feature_sha256']==digest
                and sidecar['maiac_qa_merge_manifest_sha256']==
                    oc['evidence'][str(path.parent/'oc_grid_features_2023_usa_qa_merge_manifest.json')],
                'OC QA-specific sidecar binding differs')
            argv.remove('--skip-oc')
            argv+=['--oc-grid-features',str(path)]
            job['OC_grid_features']={'path':str(path),'sha256':digest}
        else:
            job['OC_grid_features']=None
        job['comparator_and_OC_runtime_configuration_pending']=False
        jobs.append(job)
    stage.gate.assert_unchanged(accepted)
    for path,digest in bound.items():
        stage.gate.pinned(path,digest)
    result={'format':'future-smoke-launch-configuration-v1','status':'configured_no_fit',
        'materialization':{'path':str(ROOT/'materialization.json'),'sha256':MATERIALIZATION_SHA},
        'source_pins':bound,'jobs':jobs,'HMS_postfit_input_pins':hms,
        'HMS_date_availability_diagnostic_pending':True,
        'OC_scope':'USA2023 tracer only; 10km exclusion; historical PM training only; parameter/method series separate',
        'QFED_GFED_scope':'exact source calendar/coordinates; any unavailable year or row retained as unmatched, no zero filling',
        'Childs_scope':'USA2018-2020 only; no future comparator available',
        'fits_performed':False,'independent_actual_configuration_and_launch_review_required':True,
        'configuration_builder_sha256':stage.gate.sha(Path(__file__))}
    stage.save(ROOT/'launch_configuration_v1.json',result)
    print(json.dumps({'status':'configured_no_fit','jobs':4,'configuration_sha256':
        stage.gate.sha(ROOT/'launch_configuration_v1.json')}),flush=True)


if __name__=='__main__':
    main()
