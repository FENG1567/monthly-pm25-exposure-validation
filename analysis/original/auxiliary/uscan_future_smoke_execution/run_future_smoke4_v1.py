"""One-time root-controlled four-job future smoke launcher, two single-thread workers."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import sys
import threading

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'future28_actual_acceptance'))
import future28_gate_v1 as gate

V=gate.V
ROOT=V/'uscan_future_smoke_stage_v3_20261115'
CONFIG=ROOT/'launch_configuration_v1.json'
CONFIG_SHA='e46f1a6276ea8b058cbcdfaf241950acb0d28252f7e226121a1d21f7513a9009'
EXECUTOR=V/'uscan_historical_execution/1c8e49668f72_f4e32b4b9b40/controller/run_reviewed_uscan32.py'
EXECUTOR_SHA='06b48c4cc693018df2ee00f4c6f87701aadb358fe212e697bfb85d666cb4841c'
THREADS={name:'1' for name in ('REPAIR_V2_SMOKE_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}


def read_pinned(path,digest):
    return json.loads(gate.pinned(path,digest).read_text())


def option(argv,name):
    gate.require(argv.count(name)==1,'missing/duplicate launch option: '+name)
    return argv[argv.index(name)+1]


def validate():
    accepted=gate.load_gate()
    config=read_pinned(CONFIG,CONFIG_SHA)
    gate.require(config['format']=='future-smoke-launch-configuration-v1' and
        config['status']=='configured_no_fit' and config['fits_performed'] is False,'configuration gate failed')
    gate.pinned(EXECUTOR,EXECUTOR_SHA)
    for path,digest in config['source_pins'].items():
        gate.pinned(path,digest)
    gate.require(len(config['jobs'])==4 and {(j['country'],j['qa']) for j in config['jobs']}==
        {(c,q) for c in ('USA','CAN') for q in ('strict','sensitivity')},'four-job roster differs')
    gate.require(not (ROOT/'.future_smoke_claimed').exists() and not (ROOT/'future_controller_run').exists(),
                 'permanent claim/controller already exists; automatic restart prohibited')
    for job in config['jobs']:
        country,qa=job['country'],job['qa']
        argv=job['cli_argv']
        gate.require(argv[:3]==['/external/fire_downscaling_20261107/env/bin/python3.11','-u',
            str(ROOT/'runtime/work/repair_v2/smoke/run_smoke_validation.py')],'canonical runtime argv differs')
        required={'--countries':country,'--tasks':'T3_fixed,T3_rolling','--profiles':'with_met,no_met',
            '--gate-quantiles':'0.80,0.90,0.95','--target-strategies':'log_smear,original_scale',
            '--max-iter':'400','--smear-splits':'3','--bootstrap-replicates':'500',
            '--fixed-external-years':'2023,2024,2025' if country=='USA' else '2023,2024',
            '--rolling-external-years':'2024,2025' if country=='USA' else '2024',
            '--feature-input':country+'='+job['feature']['path'],
            '--target-input':country+'='+job['target']['path'],
            '--qa-provenance':job['identity']['path'],'--core-split-binding-registry':job['registry']['path'],
            '--out':str(ROOT/'results'/f'{country}_{qa}')}
        for name,value in required.items():
            gate.require(option(argv,name)==value,'frozen future launch option differs: '+name)
        gate.require('--no-can-exclusion-sensitivity' in argv and
            ('--skip-oc' in argv)==(country=='CAN'),'primary country/OC scope differs')
        gate.require(not Path(required['--out']).exists(),'job output already exists')
        if country=='USA':
            gate.require(option(argv,'--oc-grid-features')==job['OC_grid_features']['path'],'OC input differs')
        for name in ('feature','target','identity','registry'):
            rr=job[name];gate.pinned(rr['path'],rr['sha256'])
    gate.assert_unchanged(accepted)
    return config,accepted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--review',type=Path)
    parser.add_argument('--review-sha')
    parser.add_argument('--root-go',type=Path)
    parser.add_argument('--root-go-sha')
    args=parser.parse_args()
    config,accepted=validate()
    pins={'configuration_sha256':CONFIG_SHA,'executor_sha256':EXECUTOR_SHA,
          'launcher_sha256':gate.sha(Path(__file__))}
    if not args.execute:
        print(json.dumps({'status':'check_only_passed','jobs':4,'workers':2,'pins':pins,
                          'models_fit':False}),flush=True)
        return
    gate.require(args.review and args.review_sha and args.root_go and args.root_go_sha,
                 'separate QA record and release approval required')
    review=read_pinned(args.review,args.review_sha);go=read_pinned(args.root_go,args.root_go_sha)
    gate.require(review.get('decision')=='approve_future_smoke_launch' and review.get('findings')==[]
        and review.get('pins')==pins,'independent future launch review differs')
    gate.require(go.get('format')=='future-smoke-four-job-root-go-v1' and go.get('decision')=='go'
        and go.get('pins')=={**pins,'review_sha256':args.review_sha} and go.get('jobs')==4
        and go.get('workers')==2 and go.get('future_source_adoption_sha256')==gate.ADOPTION_SHA,
        'root future launch GO differs')
    for path,digest in config['source_pins'].items():
        gate.pinned(path,digest)
    gate.assert_unchanged(accepted)
    claim={'pid':os.getpid(),'pins':pins,'review_sha256':args.review_sha,'root_go_sha256':args.root_go_sha}
    with (ROOT/'.future_smoke_claimed').open('x') as stream:
        json.dump(claim,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    fd=os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    spec=importlib.util.spec_from_file_location('reviewed_future_smoke_executor',EXECUTOR)
    executor=importlib.util.module_from_spec(spec);spec.loader.exec_module(executor)
    cancel=threading.Event()
    def stop(signum,frame):
        cancel.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,**THREADS)
    env['PYTHONPATH']=str(ROOT/'runtime/work/repair_v2/validation')
    jobs=[{'job_id':j['country']+'_'+j['qa'],'command_argv':j['cli_argv']} for j in config['jobs']]
    outcome=executor.run_state_machine(jobs,workers=2,run_dir=ROOT/'future_controller_run',
                                      cwd=V,env=env,cancel_event=cancel)
    result={'format':'future-smoke-four-job-controller-status-v1',**outcome,'pins':pins,
        'pid':os.getpid(),'review_sha256':args.review_sha,'root_go_sha256':args.root_go_sha}
    with (ROOT/'future_controller_run/controller.status.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
    if outcome['status']!='succeeded':
        raise SystemExit(1)


if __name__=='__main__':
    main()
