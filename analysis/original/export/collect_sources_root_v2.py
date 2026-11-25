"""Collect accepted metric tables for writing; no model or uncertainty rerun."""
from pathlib import Path
import hashlib,json,os,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import pandas as pd
V=Path('/data/admin/fengshuo/cand1_repair_v2_20261112/validation')
OUT=V/'submission_publication_root_sources_v2_20261115'
USED={};EXPECTED={};FILES=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p,expected=None):
 p=Path(p).resolve();h=sha(p);e=expected or EXPECTED.get(str(p))
 if e and h!=e:raise ValueError('changed accepted source '+str(p))
 if str(p) in USED and USED[str(p)]!=h:raise ValueError('changed during collection '+str(p))
 USED[str(p)]=h;return p
def refs(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and x['path'].startswith('/') and isinstance(x.get('sha256'),str) and len(x['sha256'])==64:EXPECTED[x['path']]=x['sha256']
  for k,v in x.items():
   if k.startswith('/') and isinstance(v,str) and len(v)==64:EXPECTED[k]=v
   refs(v)
 elif isinstance(x,list):
  for y in x:refs(y)
def js(p,expected=None):
 x=json.loads(pin(p,expected).read_text());refs(x);return x
def table(p,group,metadata=None):
 p=Path(p)
 if not p.is_file():return
 expected=EXPECTED.get(str(p.resolve()));pin(p)
 try:d=pd.read_csv(p)
 except pd.errors.EmptyDataError:d=pd.DataFrame()
 if metadata:
  for k,v in metadata.items():
   if k not in d:d[k]=v
 dest=OUT/group/(hashlib.sha256(str(p).encode()).hexdigest()[:12]+'__'+p.parent.name+'__'+p.name);dest.parent.mkdir(parents=True,exist_ok=True)
 d.to_csv(dest,index=False)
 FILES.append({'group':group,'source':str(p),'path':str(dest),'sha256':sha(dest),'source_sha256':USED[str(p.resolve())],
  'basis':'accepted_artifact_pin' if expected else 'fresh_hash_with_accepted_parent_scope','rows':len(d),'columns':list(d)})
 print(json.dumps({'group':group,'file':p.name,'rows':len(d)}),flush=True)
def directory(p,group,metadata=None,recursive=False):
 p=Path(p)
 if not p.is_dir():return
 for f in sorted(p.rglob('*.csv') if recursive else p.glob('*.csv')):
  if any(t in f.name.lower() for t in ('predictions','oof','paired_rows','station_splits','calibration_scores','training_feature_ranges','expected_keys','row_flags','assignment')):continue
  table(f,group,metadata)
def main():
 if OUT.exists():raise ValueError('fresh output required')
 seal=js(V/'technical_completion_v1_20261115/root_completion_seal_v1.json','26cbc04eab2d27f94afff650e5e0d6065033949e59f089f76d3996abd3701384')
 registry=js(V/'technical_completion_v1_20261115/technical_completion_registry_v1.json','7887f43746fbffc7cf650caed1649dc64345b3ca3a56642bd68d561f18bf370e')
 for entry in registry['entries']:
  for ref in entry['refs']:
   if Path(ref['path']).suffix=='.json':js(ref['path'],ref['sha256'])
 js(V/'uscan_historical_derivatives_acceptance_v1_20261115/independent_addendum_v1.json','4c4d60bde163f8b61236ad9d0792c114611bc7d50ced31ac538e3332b7b54601')
 # Read only directly linked scientific audits/manifests, not the raw-source DAG.
 for p,h in list(EXPECTED.items()):
  q=Path(p)
  if q.suffix=='.json' and q.is_file() and q.name in ('audit.json','audit_result.json','completion_manifest.json','materialization.json','provenance.json','job_provenance.json','source_hashes.json'):
   js(q,h)
 OUT.mkdir()
 global_root=V/'global_accepted_integration_20261114'
 directory(global_root,'historical_global')
 directory(V/'uscan32_postfit_audit_v3_20261115','historical_uscan_audited')
 for p in sorted((V/'final_model_runtime_v4_20261114/results').glob('*')):
  if p.is_dir():directory(p,'historical_uscan',{'source_job_id':p.name})
 for p in sorted((V/'bra_in_historical_model_pack_v1_20261115/outputs').glob('*')):
  if p.is_dir():directory(p,'historical_bra_in',{'source_job_id':p.name})
 for p in sorted((V/'bra2023_temporal_model_pack_v1_20261115/outputs').glob('*')):
  if p.is_dir():directory(p,'bra2023',{'source_job_id':p.name})
 audit=V/'future28_postfit_runtime_v2_20261115/audits/20261115T044244Z-249343-f11e7b78'
 directory(audit,'future_audited')
 matrix_path=V/'future_bound_matrix_v2_20261115/bound_no_fit_matrix.json'
 if matrix_path.exists():
  matrix=js(matrix_path)
  for job in matrix['jobs']:
   directory(job['output_root'],'future',{'source_job_id':job['job_id'],**{k:job[k] for k in ('country','qa_variant','dataset_policy','validation_mode','external_year') if k in job}})
 for qa in ('strict','sensitivity'):
  directory(V/f'outputs/repair_v2/validation/loco_preflight/final_loco4_score_{qa}_v1_20261115','loco',{'qa_variant':qa})
 for root,group in [
  ('final_multiplicity_actual_v2_20261115','multiplicity'),
  ('fair_totalpm_metrics_v1_20261115','annual_comparators'),
  ('uscan_cams_remaining_comparison_v3_20261115','cams_remaining'),
  ('uscan_future_cams_comparison_v1_20261115','cams_future'),
  ('usa_cams_matched_comparison_v1_20261115','cams_usa'),
  ('global_current_comparison_20261114','cams_global'),
  ('uscan_future_HMS_actual_v1_20261115','hms'),
  ('uscan_joint_fire_actual_v2_20261115','joint_historical'),
  ('uscan_future_joint_support_v1_20261115','joint_future'),
  ('uscan_smoke_isolated_v2_20261115/results','smoke_historical'),
  ('uscan_future_smoke_stage_v3_20261115/results','smoke_future'),
  ('uscan_smoke_isolated_v2_20261115/comparator_audit_v3','smoke_historical_comparators'),
  ('uscan_future_smoke_global_comparators_v1_20261115','smoke_future_comparators')]:
  directory(V/root,group,recursive=True)
 # Remaining exact accepted CSV references cover tables whose output root is
 # named by a controller rather than the common locations above.
 already={f['source'] for f in FILES}
 for p,h in list(EXPECTED.items()):
  q=Path(p)
  if q.suffix=='.csv' and q.is_file() and str(q) not in already and any(s in p.lower() for s in ('thinning','conformal_summary','support_coverage','cams')):
   if not any(t in q.name.lower() for t in ('predictions','oof','paired_rows','station_splits','calibration_scores','training_feature_ranges','row_flags')):
    table(q,'accepted_extra');already.add(str(q))
 grid=js(V/'final_reference16_grids_v1_20261115/materialization.json')
 (OUT/'grid_materialization.json').write_text(json.dumps(grid,indent=2))
 for path,digest in USED.items():
  if sha(path)!=digest:raise ValueError('source changed during collection '+path)
 (OUT/'source_manifest.json').write_text(json.dumps({'status':'passed','files':FILES,'source_pins':USED,'expected_pin_count':len(EXPECTED),
  'source_receipt_registry_sha256':sha(V/'technical_completion_v1_20261115/technical_completion_registry_v1.json'),
  'fits_performed':False,'bootstrap_or_new_inference':False,'pid':os.getpid()},indent=2))
 print(json.dumps({'status':'passed','tables':len(FILES),'source_pins':len(USED),'output':str(OUT)}),flush=True)
if __name__=='__main__':main()
