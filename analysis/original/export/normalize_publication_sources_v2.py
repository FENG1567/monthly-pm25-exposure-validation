from pathlib import Path
import json,hashlib,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import pandas as pd
ROOT=Path('/data/admin/fengshuo/cand1_repair_v2_20261112/validation')
SRC=ROOT/'submission_publication_root_sources_v2_20261115'
OUT=ROOT/'submission_publication_normalized_v2_20261115'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((SRC/'source_manifest.json').read_text());used={};catalog=[]
def read(entry):
 p=Path(entry['path']);assert sha(p)==entry['sha256'],p
 used[str(p)]=entry['sha256'];d=pd.read_csv(p)
 return d
def identity(d):
 d=d.copy()
 for dest,choices in {'source_job_id':['job_id'],'qa_variant':['qa_label','qa'],'profile':['profile_label'],
                      'estimator_kind':['estimator_label','estimator']}.items():
  if dest not in d:
   for c in choices:
    if c in d:d[dest]=d[c];break
 if 'source_job_id' in d:
  s=d.source_job_id.astype(str)
  for col,expr in [('qa_variant',s.str.extract('(strict|sensitivity)',expand=False)),
    ('profile',s.str.extract('(with_met|no_met)',expand=False)),('estimator_kind',s.str.extract('(hgb|ridge)',expand=False))]:
   if col not in d:d[col]=expr
 if 'qa_variant' in d:d['qa_variant']=d.qa_variant.astype(str).str.replace('sensitive','sensitivity',regex=False)
 return d
def select(group,name):return [e for e in manifest['files'] if e['group']==group and Path(e['source']).name==name]
def save(name,parts):
 if not parts:return
 d=pd.concat(parts,ignore_index=True,sort=False);d.to_csv(OUT/(name+'.csv'),index=False)
 catalog.append({'name':name,'rows':len(d),'columns':list(d),'sha256':sha(OUT/(name+'.csv'))})
 return d
OUT.mkdir(exist_ok=False)
hist=[]
for e in select('historical_global','all_task_metrics.csv'):hist.append(identity(read(e)))
meta_entries=select('historical_uscan_audited','job_audit_summary.csv');meta=read(meta_entries[0])
for e in select('historical_uscan_audited','independent_task_metrics.csv'):
 d=read(e)
 for c in ['country','qa_variant','profile','estimator_kind','dataset_key']:
  if c not in d:d=d.merge(meta[['job_id',c]],on='job_id',validate='many_to_one')
 hist.append(identity(d))
for e in select('historical_bra_in','task_metrics.csv'):hist.append(identity(read(e)))
h=save('historical_metrics',hist)
future=[];fmeta=read(select('future_audited','job_audit_summary.csv')[0])
for e in select('future_audited','independent_external_metrics.csv'):
 d=read(e)
 cols=['job_id']+[c for c in ['qa_variant','estimator_kind','dataset_policy','validation_mode','external_year','train_through'] if c not in d]
 d=d.merge(fmeta[cols],on='job_id',validate='many_to_one');future.append(identity(d))
f=save('future_metrics',future)
groups={'historical_eligibility': [('historical_global','all_eligibility.csv'),('historical_uscan_audited','all_eligibility.csv'),('historical_bra_in','eligibility.csv')],
 'future_paired_deltas':[('future','paired_deltas.csv'),('future','external_paired_deltas.csv')],
 'historical_paired_deltas':[('historical_global','all_paired_deltas.csv'),('historical_uscan','paired_deltas.csv'),('historical_bra_in','paired_deltas.csv')],
 'multiplicity':[('multiplicity','simultaneous_rmse_contrasts.csv')],
 'hms_correlations':[('hms','monthly_HMS_correlations.csv')], 'hms_coverage':[('hms','monthly_HMS_coverage_all_strata.csv')],
 'hms_bins':[('hms','monthly_HMS_day_bins.csv')], 'oc_observed':[('hms','OC2023_observed_contrast_metrics.csv')],
 'smoke_future_comparators':[('smoke_future','smoke_comparator_metrics.csv')],
 'smoke_historical_comparators':[('smoke_historical','smoke_comparator_metrics.csv')],
 'smoke_future_summary':[('smoke_future','smoke_component_summary.csv')],
 'joint_future':[('joint_future','coverage_summary.csv')],'joint_historical':[('joint_historical','coverage_summary.csv')],
 'annual_metrics':[('annual_comparators','matched_annual_metrics.csv')], 'annual_deltas':[('annual_comparators','paired_annual_deltas.csv')],
 'loco_metrics':[('loco','metrics_original_scale.csv')], 'loco_pairs':[('loco','paired_station_bootstrap.csv')]}
for name,pairs in groups.items():
 parts=[]
 for group,file in pairs:
  for e in select(group,file):
   d=read(e)
   if 'qa_variant' not in d and 'strict' in e['source']:d['qa_variant']='strict'
   elif 'qa_variant' not in d and 'sensitivity' in e['source']:d['qa_variant']='sensitivity'
   parts.append(identity(d))
 save(name,parts)
# Preserve each additional summary independently with its audited source identity.
for group in ('accepted_extra','cams_future','cams_remaining','cams_global','cams_usa','bra2023'):
 for e in [e for e in manifest['files'] if e['group']==group]:
  d=identity(read(e));name=group+'__'+Path(e['path']).stem
  d.to_csv(OUT/(name+'.csv'),index=False)
  catalog.append({'name':name,'rows':len(d),'columns':list(d),'sha256':sha(OUT/(name+'.csv'))})
summary={'catalog':catalog,'historical_rows':len(h),'historical_countries':sorted(h.country.unique().tolist()),
 'historical_qa':sorted(h.qa_variant.dropna().unique().tolist()),'historical_profiles':sorted(h.profile.dropna().unique().tolist()),
 'historical_dataset_keys':sorted(h.dataset_key.dropna().unique().tolist()) if 'dataset_key' in h else [],
 'future_policies':sorted(f.dataset_policy.unique().tolist()),'future_rows':len(f),
 'primary_tags':{c:sorted(h[c].dropna().astype(str).unique().tolist()) for c in ('target_strategy','backtransform','task')},
 'historical_primary_records':identity(h).loc[(h.qa_variant=='strict')&(h.estimator_kind=='hgb')&(h.target_strategy=='log_smear')&(h.arm=='full')].fillna('').to_dict('records'),
 'future_primary_records':f.loc[(f.qa_variant=='strict')&(f.target_strategy=='log_smear')&(f.arm=='full')].fillna('').to_dict('records')}
for p,digest in used.items():assert sha(p)==digest,p
(OUT/'catalog.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=str))
print(json.dumps({'status':'passed','files':len(catalog),'historical_rows':len(h),'future_rows':len(f),'countries':summary['historical_countries']}),flush=True)
