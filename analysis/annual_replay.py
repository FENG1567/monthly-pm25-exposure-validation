"""Replay paired annual metrics from explicit shared-support station-year inputs.

Self-authored adapter: MIT. The source archive retains the original annual
support and cluster-bootstrap kernels. No model fitting or data fetching.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

IDENTITY=('alias','qa','profile','estimator','task','arm','target_strategy','backtransform','country')
SCOPES={'primary_exact_12_months':12,'sensitivity_at_least_10_months':10}
METRICS=('rmse','mae','bias','r2')
KERNEL_SHA='f8cbbbb53ce94a79b3da6b506788583880d81b817f18a5eba2b0f92fd1eb9332'

def seed_for(labels):
    raw=('cams-matched-station-bootstrap-v1\0'+'\0'.join(map(str,labels))).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)

def annual_seed(identity,scope,control):
    fields=('alias','country','qa','profile','estimator','task','target_strategy','backtransform')
    return seed_for(('fair-totalpm-v1',*(identity[k] for k in fields),scope,'V6GL03',control))

def load_kernel():
    path=Path(__file__).resolve().parent/'original'/'annual'/'fast_cluster_bootstrap.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=KERNEL_SHA:
        raise ValueError('Archived bootstrap kernel hash differs')
    spec=importlib.util.spec_from_file_location('annual_frozen_bootstrap',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.paired_bootstrap

def metric(y,p):
    if not len(y):return {k:float('nan') for k in METRICS}
    e=p-y;sst=float(np.sum((y-y.mean())**2))
    return {'rmse':float(np.sqrt(np.mean(e**2))), 'mae':float(np.mean(abs(e))),
            'bias':float(np.mean(e)), 'r2':float(1-np.sum(e**2)/sst) if sst>0 else float('nan')}

def check_frame(frame):
    required=set(IDENTITY)|{'site_id','year','shared_months','benchmark_version',
        'observed_annual_pm25','model_annual_pm25','raw_cams_annual_pm25','satpm_annual_pm25'}
    if required-set(frame):raise ValueError('Missing columns: '+', '.join(sorted(required-set(frame))))
    d=frame.copy()
    if d[list(IDENTITY)+['site_id','year','shared_months','benchmark_version']].isna().any().any():
        raise ValueError('Missing identity or support value')
    if not d.benchmark_version.eq('V6GL03').all():raise ValueError('Mixed or unsupported benchmark version')
    for field in ('year','shared_months'):
        values=pd.to_numeric(d[field],errors='raise')
        if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all():
            raise ValueError('Noninteger '+field)
        d[field]=values.astype(int)
    if not d.shared_months.between(0,12).all():raise ValueError('Invalid shared month count')
    if (d.year.eq(2025)&d.shared_months.gt(0)).any():raise ValueError('2025 SatPM was unavailable in this analysis')
    if d.duplicated(list(IDENTITY)+['site_id','year']).any():raise ValueError('Duplicate station-year within model cell')
    for col in ('observed_annual_pm25','model_annual_pm25','raw_cams_annual_pm25','satpm_annual_pm25'):
        d[col]=pd.to_numeric(d[col],errors='raise')
        if not np.isfinite(d.loc[d.shared_months.ge(10),col]).all():raise ValueError('Nonfinite eligible '+col)
    return d

def replay(frame):
    d=check_frame(frame);kernel=load_kernel();rows=[]
    for labels,part in d.groupby(list(IDENTITY),sort=True,dropna=False):
        identity=dict(zip(IDENTITY,labels))
        for scope,nmonths in SCOPES.items():
            selected=part.loc[part.shared_months.eq(12) if nmonths==12 else part.shared_months.ge(10)]
            y=selected.observed_annual_pm25.to_numpy(float);p=selected.model_annual_pm25.to_numpy(float)
            sites=int(selected.site_id.nunique());model=metric(y,p)
            for control,column in [('raw_CAMS','raw_cams_annual_pm25'),('SatPM','satpm_annual_pm25')]:
                q=selected[column].to_numpy(float);other=metric(y,q);seed=annual_seed(identity,scope,control)
                ci=kernel(selected.site_id.to_numpy(str),y,p,q,500,seed) if sites>=2 else {}
                row={**identity,'support_scope':scope,'benchmark_version':'V6GL03','rows':len(selected),
                    'sites':sites,'source_annual_rows':len(part),'excluded_annual_rows':len(part)-len(selected),
                    'status':'eligible' if len(selected) else 'ineligible_no_exact_shared_support',
                    'contrast':'full minus '+control,'bootstrap_seed':seed,'bootstrap_replicates':500,
                    'uncertainty_status':'eligible' if sites>=2 else 'ineligible_fewer_than_two_sites'}
                for k in METRICS:
                    row['delta_'+k]=model[k]-other[k]
                    row['delta_'+k+'_ci_low']=ci.get('delta_'+k+'_lo',float('nan'))
                    row['delta_'+k+'_ci_high']=ci.get('delta_'+k+'_hi',float('nan'))
                rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annual-pairs',type=Path,required=True)
    ap.add_argument('--input-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();data=args.annual_pairs.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('A fresh output directory is required')
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    if digest!=args.input_sha256.lower():raise ValueError('Input SHA256 mismatch')
    frame=pd.read_parquet(data) if data.suffix.lower()=='.parquet' else pd.read_csv(data,dtype={'site_id':str})
    result=replay(frame)
    if hashlib.sha256(data.read_bytes()).hexdigest()!=digest:raise ValueError('Input changed during replay')
    out.mkdir(parents=True,exist_ok=False);result.to_csv(out/'paired_annual_deltas.csv',index=False)
    (out/'replay.json').write_text(json.dumps({'status':'complete','input_sha256':digest,
        'contrast_rows':len(result),'bootstrap_replicates':500,'cluster':'physical station; all years retained',
        'quantiles':[2.5,97.5],'model_refitting':False,'numpy':np.__version__,'pandas':pd.__version__,
        'source_stage':'explicit supplied annual pairs; upstream source identity must be established separately'},indent=2)+'\n')
    print('ANNUAL_REPLAY_COMPLETE',len(result))

if __name__=='__main__':main()
