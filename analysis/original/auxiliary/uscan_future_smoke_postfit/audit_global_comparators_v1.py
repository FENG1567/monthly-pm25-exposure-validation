"""Replay frozen QFED/GFED future smoke comparisons; OC/HMS stay separate."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

for name in ('REPAIR_V2_SMOKE_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'

import pandas as pd
import future_core_gate_v1 as gate_api

HELPER_SHA='a655a1d158e8c52e8b54c3c77481e76e2977663d221d9b9f62a90b993f81bb00'
STRATA=['task','fold_id','profile','gate_quantile','gate_role','target_strategy','backtransform','year']


def coverage_rows(primary,matched):
    result=[]
    for labels,part in primary.groupby(STRATA,sort=True,dropna=False):
        matched_rows=matched
        for column,value in zip(STRATA,labels):
            matched_rows=matched_rows.loc[matched_rows[column].eq(value)]
        n=len(matched_rows)
        identity={name:(value.item() if hasattr(value,'item') else value) for name,value in zip(STRATA,labels)}
        result.append({**identity,'source_primary_smoke_rows':len(part),'matched':n,
            'source_stations':int(part.site_id.nunique()),'matched_stations':int(matched_rows.site_id.nunique()),
            'unmatched':len(part)-n,'status':'matched' if n else 'source_unavailable_or_no_exact_match'})
    return result


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
    gate_api.require(not args.out.exists(),'fresh comparator audit output required')
    gate['used'][str(Path(__file__).resolve())]=gate_api.sha(Path(__file__))
    gate['used'][str(Path(gate_api.__file__).resolve())]=gate_api.sha(Path(gate_api.__file__))
    stage=gate['configuration']
    for path,digest in stage['source_pins'].items():
        gate_api.pinned(path,digest,gate['used'])
    helper=load_module('accepted_historical_comparator_helpers',
        Path(__file__).with_name('historical_comparator_helpers.py'),HELPER_SHA,gate)
    code=gate_api.ROOT/'runtime/work/repair_v2'
    sys.path.insert(0,str(code/'smoke'))
    runner=load_module('future_global_comparator_runner',code/'smoke/run_smoke_validation.py',gate_api.RUNNER_SHA,gate)
    def no_fit(*args,**kwargs):
        raise RuntimeError('fitting forbidden in comparator replay')
    runner.fit_model=no_fit;sys.modules['validation_core'].fit_model=no_fit
    provenance=gate_api.V.parent/'smoke/final_AUS_core_bound_20261114/global_comparators/global_comparator_provenance.json'
    gate_api.pinned(provenance,'ece248119f860dd8b064ac241e475f148334c6f03b93aaa34bacaa1a8fdc32c1',gate['used'])
    args.out.mkdir(parents=True)
    results=[]
    absent=set()
    for job in stage['jobs']:
        country,qa=job['country'],job['qa']
        out=gate_api.ROOT/'results'/f'{country}_{qa}'
        manifest=gate_api.accepted_json(gate,out/'smoke_manifest.json')
        for ref in manifest['input_hashes']+manifest['code_hashes']:
            gate_api.accepted_file(gate,ref['path'],ref['sha256'])
        for path in (runner.COMPARATOR_PATHS['CHILDS'],runner.COMPARATOR_PATHS['USA_COUNTY'],
                     runner.COMPARATOR_PATHS['CHILDS'].parent/'comparator_provenance.json'):
            if path.exists():
                gate_api.accepted_file(gate,path)
            else:
                absent.add(str(path))
        oof=pd.read_parquet(gate_api.accepted_file(gate,out/'smoke_oof_predictions.parquet'))
        saved=pd.read_csv(gate_api.accepted_file(gate,out/'smoke_comparator_metrics.csv'),float_precision='round_trip')
        saved=saved.loc[~saved.comparator.eq('EPA_OC_tracer')].copy()
        argv=job['cli_argv']
        paths={name:Path(helper.cli_value(argv,'--'+name.lower()+'-path')) for name in ('QFED','GFED')}
        replay=args.out/f'{country}_{qa}';replay.mkdir()
        metrics,records=runner.run_station_comparators(oof,replay,500,comparator_paths=paths)
        gate_api.require(records.get('CHILDS',{}).get('status')!='matched','future Childs unavailable beyond2018-2020')
        expected=pd.DataFrame(metrics)
        if len(expected):
            gate_api.require(set(expected.columns).issubset(saved.columns),'global comparator score columns missing')
            for extra in set(saved.columns)-set(expected.columns):
                gate_api.require(saved[extra].isna().all(),'unexpected global comparator field: '+extra)
            numeric=helper.compare_table(expected,saved[expected.columns],
                ['scope','comparator','qa_variant','profile','gate_quantile'])
        else:
            gate_api.require(saved.empty,'unmatched source emitted saved global scores')
            numeric=0
        products={}
        primary=oof.loc[oof.target_strategy.eq('log_smear')]
        for product in ('QFED','GFED'):
            record=records.get(product,{'status':'unavailable'})
            actual=primary.iloc[0:0].copy()
            if record.get('status')=='matched':
                saved_path=gate_api.accepted_file(gate,out/f'{product.lower()}_matched_station_pairs.parquet')
                actual=pd.read_parquet(saved_path);reconstructed=pd.read_parquet(replay/saved_path.name)
                cells=helper.compare_pair_table(reconstructed,actual,['cohort','country','task','qa_variant','profile',
                    'gate_quantile','gate_role','target_strategy','site_id','year','month'])
                geometry=helper._validate_global_pair_metadata(reconstructed,runner,product)
                products[product]={'status':'matched','pair_values_verified':cells,
                    'source_record':record.get('verified_source'),**geometry}
            else:
                products[product]={'status':record.get('status'),'matched_pairs':0,
                    'unmatched_primary_smoke_rows':len(primary)}
            products[product]['stratified_coverage']=coverage_rows(primary,actual)
        results.append({'country':country,'qa':qa,'metric_rows':len(expected),'numeric_metric_cells':numeric,
                        'products':products,'Childs':'not_available_for_future_years','HMS':'separate_date_aware_derivative'})
        print(json.dumps({'job_passed':[country,qa],'metric_rows':len(expected)}),flush=True)
    gate_api.assert_unchanged(gate)
    gate_api.require(all(not Path(path).exists() for path in absent),'unbound automatic comparator source appeared')
    result={'format':'future-smoke-global-comparator-replay-v1','status':'PASS','findings':[],
        'jobs':results,'metric_rows':sum(j['metric_rows'] for j in results),'bootstrap_replicates':500,
        'accepted_core_receipt_sha256':gate_api.ACCEPTANCE_SHA,'bound_source_pins':gate['used'],
        'fits_performed':False,'primary_target_only':'log_smear (unchanged frozen comparator protocol)',
        'OC_scope':'separate accepted tracer audit','HMS_scope':'separate date-aware plume diagnostic',
        'unmatched_years_and_rows_retained_in_coverage':True,
        'automatic_comparator_sources_absent':sorted(absent),
        'auditor_sha256':gate_api.sha(Path(__file__))}
    with (args.out/'audit.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASS','jobs':4,'metric_rows':result['metric_rows']}),flush=True)


if __name__=='__main__':
    main()
