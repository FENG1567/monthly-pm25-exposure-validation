import json,hashlib,os,re,shutil
from pathlib import Path
import pandas as pd
V=Path('/data/admin/fengshuo/cand1_repair_v2_20261112/validation')
S=V/'submission_publication_normalized_v2_20261115'
O=V/'submission_editor_data_v2_20261115';O.mkdir(exist_ok=True)
D=O/'Supplementary_Data';D.mkdir(exist_ok=True)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cat=json.loads((S/'catalog.json').read_text())['catalog']; index=[]
def clean(d):
 d=d.drop(columns=[c for c in d if c in ['source_root','source_job_id','job_id','source_binding_sha256','model_HMS_evidence_sha256']],errors='ignore')
 for c in d.select_dtypes(include='object'):
  d[c]=d[c].map(lambda x:'source_quality_controlled_cohort' if isinstance(x,str) and 'pending_independent' in x else (Path(x).name if isinstance(x,str) and x.startswith('/data/') else x))
 return d
for e in cat:
 p=S/(e['name']+'.csv');assert H(p)==e['sha256']
 d=clean(pd.read_csv(p,low_memory=False));name=p.name if '__' not in p.name else f'Additional_{len(index)+1:03d}_'+p.name.split('__')[-1];q=D/name;d.to_csv(q,index=False)
 index.append({'file':q.name,'source_alias':e['name'],'rows':len(d),'fields':list(d),'sha256':H(q)})
# Small accepted supplements absent from the core normalization.
T=V/'station_thinning_final_execution_v1_20261115/final_run/run_20261115T020544Z_32c4673b59'
A=json.loads((V/'station_thinning_postfit_audit_v1_20261115/acceptance.json').read_text())['details']['output_artifact_sha256']
for n in ['metrics.csv','paired_deltas.csv','repeat_ranges.csv','level_status.csv']:
 p=T/n;assert H(p)==A[n]
 try:d=clean(pd.read_csv(p))
 except pd.errors.EmptyDataError:d=pd.DataFrame(columns=['status'])
 q=D/('thinning_'+n);d.to_csv(q,index=False)
 index.append({'file':q.name,'rows':len(d),'fields':list(d),'sha256':H(q)})
(D/'data_catalog.json').write_text(json.dumps({'files':index,'notes':'All rows preserved; private storage paths and execution identifiers removed. n denotes matched held-out pairs unless the file specifies a different denominator.'},indent=2))
b=json.loads((V/'submission_claimbook_v1_20261115/claimbook.json').read_text());h=pd.DataFrame(b['historical']);f=pd.DataFrame(b['future']);l=pd.DataFrame(b['loco']);oc=pd.DataFrame(b['smoke_primary']);cm=pd.DataFrame(b['cams_metrics']);cd=pd.DataFrame(b['cams_deltas'])
F=lambda x:f'{x:.3f}'
P=lambda x:'With MET' if x=='with_met' else 'No MET'
tables=[]
rows=[]
for c in ['AUS','BD','BRA','CAN','CL','ES','GR','IN','IT','MEX','NP','PK','PT','TH','USA','VN','ZA']:
 d=h[(h.country==c)&(h.task=='T1_station')]
 if d.empty:rows.append([c,'Ineligible','—','—','—','—']);continue
 a=d[d.profile=='no_met'].iloc[0];z=d[d.profile=='with_met'].iloc[0]
 nsites=a.n_stations
 if pd.isna(nsites):
  ll=l[l.country==c].iloc[0];assert ll.n==a.n;nsites=ll.n_test_sites
 rows.append([c,f'{int(a.n):,} / {int(nsites)}',F(a.r2),F(z.r2),F(a.rmse),F(z.rmse)])
tables.append({'type':'table','caption':'Table 1. Primary historical station-grouped validation. Strict QA, HGB, log-smear target with smearing retransformation, full feature arm. N/sites denotes held-out station-months/physical stations; periods and task-specific guards are retained in Supplementary Data. AUS represents Queensland. RMSE is in µg m⁻³. No MET and With MET are prespecified profiles.','columns':['Country','N / sites','R² no MET','R² with MET','RMSE no MET','RMSE with MET'],'rows':rows,'widths_mm':[18,33,27,27,30,30]})
ff=f[f.validation_mode=='fixed'].sort_values(['country','external_year','profile']);rows=[]
for _,a in ff.iterrows():rows.append([f'{a.country} {int(a.external_year)}',P(a.profile),f'{int(a.n):,}/{int(a.n_stations)}',F(a.r2),F(a.rmse),F(a.bias)])
tables.append({'type':'table','caption':'Table 2. Fixed-window future validation of the full model. All models trained through 2022; primary source policies, strict QA, HGB and log-smear/smearing. N/sites denotes held-out station-months/stations. The previously consulted 2023 review period is distinguished from the later frozen-protocol evaluations. RMSE and signed bias are in µg m⁻³.','columns':['Country / year','Profile','N / sites','R²','RMSE','Bias'],'rows':rows,'widths_mm':[30,28,35,22,25,25]})
rows=[]
for _,a in oc[oc.comparator=='EPA_OC_tracer'].sort_values(['profile','scope']).iterrows():
 parts=a.scope.split(':');rows.append([parts[-2]+'/'+parts[-1],P(a.profile),f'{int(a.n_pairs):,}/{int(a.n_groups)}',F(a.pearson_r),f'{F(a.pearson_r_ci_low)}–{F(a.pearson_r_ci_high)}'])
tables.append({'type':'table','caption':'Table 3. OC tracer consistency in 2023. Strict QA, P90 fire gate and log-smear branch; separate parameter/method series. PM training ended in 2022 and excluded PM sites within 10 km of any OC site. N/sites denotes matched station-month pairs/stations. Intervals use the saved station resampling procedure with fitted predictions fixed. OC is nonspecific and these correlations do not establish smoke mass.','columns':['Parameter / method','Profile','N / sites','Pearson r','95% interval'],'rows':rows,'widths_mm':[40,30,32,25,38]})
rows=[]
for _,a in cd[(cd.validation_mode=='fixed')&(cd.contrast=='B+CAMS minus full')].sort_values(['country','external_year','profile']).iterrows():
 d=cm[(cm.country==a.country)&(cm.external_year==a.external_year)&(cm.validation_mode=='fixed')&(cm.profile==a.profile)]
 full=d[d.arm=='full'].iloc[0];cams=d[d.arm=='B+CAMS'].iloc[0]
 rows.append([f'{a.country} {int(a.external_year)}',P(a.profile),F(full.rmse),F(cams.rmse),f'{F(a.delta_rmse)} [{F(a.delta_rmse_ci_low)}, {F(a.delta_rmse_ci_high)}]'])
tables.append({'type':'table','caption':'Table 4. Matched future comparison with a fitted B+CAMS model. Primary source policies, strict QA, log-smear/smearing and fixed historical training cutoff. RMSE and differences are in µg m⁻³. Positive B+CAMS minus full differences favor the full model. Brackets give the saved conditional physical-station resampling intervals; these comparator intervals are outside the 78-contrast multiplicity family. All fixed country-years and both profiles are retained.','columns':['Country / year','Profile','Full RMSE','B+CAMS RMSE','Difference [95% interval]'],'rows':rows,'widths_mm':[28,25,26,29,57]})
(O/'main_tables.json').write_text(json.dumps(tables,indent=2,ensure_ascii=False))
for i,t in enumerate(tables,1):pd.DataFrame(t['rows'],columns=t['columns']).to_csv(D/f'Main_Table_{i}.csv',index=False)
print(json.dumps({'status':'passed','csv_files':len(index)+4,'tables':[len(t['rows']) for t in tables],'pid':os.getpid()}),flush=True)
