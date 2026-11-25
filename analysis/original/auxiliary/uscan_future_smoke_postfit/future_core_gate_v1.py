"""Immutable accepted future-smoke core gate for downstream derivatives."""
from pathlib import Path
import hashlib
import json

V=Path('/external/project/validation')
ROOT=V/'uscan_future_smoke_stage_v3_20261115'
ACCEPTANCE=V/'uscan_future_smoke_core_independent_acceptance_v1_20261115/independent_acceptance.json'
ACCEPTANCE_SHA='7af554199569cc9c30b5e28abffe2b14c305fa8974c1b6161b8e756ce518dd6b'
REPORT_SHA='84f10fbd35cffd784246296b3d18bb8d3a83652f591092b2978f31f23c126ad5'
CONFIG_SHA='e46f1a6276ea8b058cbcdfaf241950acb0d28252f7e226121a1d21f7513a9009'
CONTROLLER_SHA='2f75c3097b25dd84369e48270c6c6abe07a175d06916b0468ba0d64c0c57f295'
RUNNER_SHA='ae141aa0a7230bc573e483d67d5a280760a7b619e0c534626fdc539625b1b21a'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def pinned(path,digest,used):
    path=Path(path).resolve(strict=True)
    require(sha(path)==digest,'source hash mismatch: '+str(path))
    require(str(path) not in used or used[str(path)]==digest,'conflicting accepted source pin')
    used[str(path)]=digest
    return path


def load_gate():
    used={}
    acceptance=json.loads(pinned(ACCEPTANCE,ACCEPTANCE_SHA,used).read_text())
    require(acceptance.get('format')=='future-smoke-independent-core-postfit-acceptance-v1'
        and acceptance.get('decision')=='accepted_core_postfit_only' and acceptance.get('findings')==[]
        and acceptance.get('report')=={'path':str(ROOT/'postfit_audit_v1/audit_result.json'),'sha256':REPORT_SHA}
        and acceptance.get('configuration_sha256')==CONFIG_SHA,'independent core acceptance differs')
    expected={'jobs':4,'bound_core_slots':32,'model_gate_records':192,
              'ordered_oof_rows':1383216,'unique_saved_bundles_loaded':256}
    require(acceptance.get('census')==expected,'accepted future core census differs')
    report=json.loads(pinned(ROOT/'postfit_audit_v1/audit_result.json',REPORT_SHA,used).read_text())
    require(report['status']=='future4_core_postfit_replay_passed' and report['model_gate_records_total']==192
        and report['bound_core_split_slots_total']==32 and report['oof_rows_replayed_total']==1383216
        and report['models_refitted'] is False and report['runner_sha256']==RUNNER_SHA,'core replay report differs')
    pins=acceptance['accepted_artifact_pins']
    require(isinstance(pins,dict) and len(pins)==731,'complete accepted core artifact census required')
    config=json.loads(pinned(ROOT/'launch_configuration_v1.json',CONFIG_SHA,used).read_text())
    controller=json.loads(pinned(ROOT/'future_controller_run/controller.status.json',CONTROLLER_SHA,used).read_text())
    require(controller['format']=='future-smoke-four-job-controller-status-v1'
        and controller['status']=='succeeded' and controller['completed_jobs']==controller['launched_jobs']==4
        and controller['cancelled'] is False and controller['failures']==[], 'future controller no longer accepted')
    require(config['status']=='configured_no_fit' and config['fits_performed'] is False,'source configuration differs')
    return {'acceptance':acceptance,'report':report,'configuration':config,'pins':pins,'used':used}


def accepted_file(gate,path,expected=None):
    path=Path(path).resolve(strict=True)
    digest=gate['pins'].get(str(path))
    require(isinstance(digest,str) and (expected is None or expected==digest),
             'file outside accepted future core census: '+str(path))
    return pinned(path,digest,gate['used'])


def accepted_json(gate,path,expected=None):
    return json.loads(accepted_file(gate,path,expected).read_text())


def assert_unchanged(gate):
    for path,digest in gate['used'].items():
        require(sha(path)==digest,'accepted artifact changed during derivative: '+path)
