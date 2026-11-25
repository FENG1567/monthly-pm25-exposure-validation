"""Bind and materialize all16 accepted station-supported2023 reference grids."""
from pathlib import Path
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'

import joblib
import netCDF4
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'uscan_future_smoke_postfit'))
import future_core_gate_v1 as gate_api

V=gate_api.V
INPUT_ACCEPTANCE=V/'final_grid2023_input_acceptance_v1_20261115/independent_actual_input_acceptance_v1.json'
INPUT_SHA='3e2d46454a8b3c4d4f3efd70bc5ab150e0d2dbda34ee481a6ab0738fdc258a18'
INPUT_MANIFEST=V/'final_grid2023_inputs_v2_20261115/input_materialization_manifest.json'
INPUT_MANIFEST_SHA='11c1736943f192c9a4da5aeb5c63c28461453c1266aeed062664b8c540bd46fe'
WRITER=Path(__file__).with_name('write_prediction_grid_v2.py')
WRITER_SHA='0049035bb798ddeede6100b8391975660d99f2a5b1f7438bf4b687ef1f8038e4'
WRITER_TEST_SHA='def1635d7da8a52ee762ca1ab7f59b1712bc5b44232bcca84c41e2cadec02a1b'
GATE_SHA='93f7f5903ecb73092bca57df9a386e8c6899ce7b2fb0bc57c5b33f1058716d00'
CORE_SHA='027fedd8cc1c77be30ee4dfe78e39291efe2e9ace89548369710814e71f34157'
KEYS=['country','site_id','year','month']


def load_pinned(path,digest,gate):
    return json.loads(gate_api.pinned(path,digest,gate['used']).read_text())


def prepare_plan():
    gate_api.require(gate_api.sha(gate_api.__file__)==GATE_SHA,'accepted shared gate code differs')
    gate=gate_api.load_gate()
    gate_api.pinned(Path(__file__),gate_api.sha(Path(__file__)),gate['used'])
    gate_api.pinned(gate_api.__file__,GATE_SHA,gate['used'])
    accepted=load_pinned(INPUT_ACCEPTANCE,INPUT_SHA,gate)
    gate_api.require(accepted['decision']=='accepted' and accepted['findings']==[], 'grid input acceptance failed')
    pinmap=accepted['artifact_pinmap']
    def grid_file(ref):
        gate_api.require(pinmap.get(ref['path'],{}).get('sha256')==ref['sha256'],
                         'grid input is outside independent accepted census')
        return gate_api.pinned(ref['path'],ref['sha256'],gate['used'])
    grid_file({'path':str(INPUT_MANIFEST),'sha256':INPUT_MANIFEST_SHA})
    inputs=json.loads(INPUT_MANIFEST.read_text())
    gate_api.require(inputs['reference_period']==2023 and inputs['pm_targets_materialized'] is False,
                     'only accepted target-free2023 grid inputs allowed')
    legacy_writer=inputs['source_bindings']['reviewed_linear_memory_writer']
    gate_api.require(legacy_writer['used_for_predictions'] is False and legacy_writer['sha256']==
        '0049035bb798ddee60d10b8391975660d99f2a5b1f7438bf4b687ef1f8038e4',
        'upstream unused-writer typo disposition differs')
    proof_path=gate_api.pinned(Path(__file__).with_name('writer_million_point_test.log'),WRITER_TEST_SHA,gate['used'])
    proof=ast.literal_eval(proof_path.read_text().strip())
    gate_api.require(proof['status']=='PASS' and proof['writer_v2_sha256']==WRITER_SHA
        and proof['points_checked']==1000000 and proof['non_locate_ast_unchanged']=='PASS'
        and proof['ascending_descending_reference']=='PASS' and proof['rejection_cases']=='PASS',
        'original exact-writer million-point verification differs')
    template=inputs['geometry_template'];grid_file(template)
    gate_api.pinned(WRITER,WRITER_SHA,gate['used'])
    products=[]
    for job in gate['configuration']['jobs']:
        country,qa=job['country'],job['qa']
        source=inputs['materialized_inputs'][f'{country}_{qa}']
        rows=source['materialized_station_rows'];domain=source['materialized_support_domain']
        grid_file(rows);grid_file(domain)
        manifest_path=gate_api.ROOT/'results'/f'{country}_{qa}'/'smoke_manifest.json'
        manifest=gate_api.accepted_json(gate,manifest_path)
        refs=gate_api.accepted_file(gate,manifest_path.parent/'future_fixed_train_through_2022_pairs.parquet')
        selected=[r for r in manifest['model_artifacts'] if r['task']=='T3_fixed_2023'
                  and str(r['fold_id'])=='2023' and float(r['gate_quantile'])==.90]
        gate_api.require(len(selected)==4 and {(r['profile'],r['target_strategy']) for r in selected}==
            {(p,t) for p in ('with_met','no_met') for t in ('log_smear','original_scale')},
            'allfour P90 fixed2023 profile/target assets required')
        for record in selected:
            context=record['core_manifest_refs']
            gate_api.require(context['train_through']==2022 and context['external_year']==2023,
                             'grid model is not fixed-through2022')
            threshold_path=gate_api.accepted_file(gate,record['threshold_json'])
            threshold=json.loads(threshold_path.read_text())
            models={}
            for kind in ('full','background'):
                path=gate_api.accepted_file(gate,record[kind+'_model'],threshold[kind+'_model_sha256'])
                gate_api.require('OC_2023' not in str(path),'OC-exclusion models cannot produce these reference grids')
                models[kind]={'path':str(path),'sha256':threshold[kind+'_model_sha256']}
            profile,target=record['profile'],record['target_strategy']
            products.append({'product_id':f'{country}_{qa}_{profile}_{target}','country':country,'qa':qa,
                'profile':profile,'target_strategy':target,'gate_quantile':.90,
                'rows':rows,'support_domain':domain,'input_rows':source['source_rows'],
                'supported_cell_months':source['support_cell_months'],'template':template,'models':models,
                'threshold':{'path':str(threshold_path),'sha256':gate['pins'][str(threshold_path)],
                             'value':threshold['threshold_value_train_only']},
                'saved_station_reference':{'path':str(refs),'sha256':gate['pins'][str(refs)]},
                'manifest':{'path':str(manifest_path),'sha256':gate['pins'][str(manifest_path)]}})
    gate_api.require(len(products)==16 and len({p['product_id'] for p in products})==16,'exact16 grid product census required')
    gate_api.assert_unchanged(gate)
    plan={'format':'final-reference16-bound-grid-plan-v1','status':'bound_no_prediction',
        'accepted_grid_input_sha256':INPUT_SHA,'accepted_future_smoke_core_sha256':gate_api.ACCEPTANCE_SHA,
        'writer_sha256':WRITER_SHA,'products':products,'source_pins':dict(gate['used']),
        'scope':'2023 accepted station-supported cell-months only;16 QA/MET/target products retained',
        'writer_pin_erratum':{'upstream_unused_writer_pin':legacy_writer['sha256'],
            'correct_writer_sha256':WRITER_SHA,'original_million_point_test_log_sha256':
            WRITER_TEST_SHA,'original_input_acceptance_and_producer_metadata_modified':False,
            'disposition':'transcription typo in unused writer reference; actual code matches original million-point test; no algorithm change'},
        'domain_extrapolated':False,'original_producer_flags_modified':False,'predictions_performed':False}
    return gate,plan


def sample_saved_grid(writer,path,rows):
    columns=['pm25_total','pm25_background','pm25_smoke_proxy','prediction_domain','affected']
    values={name:np.full(len(rows),np.nan) for name in columns}
    with netCDF4.Dataset(path) as ds:
        lat=np.asarray(ds['lat'][:]);lon=np.asarray(ds['lon'][:])
        dates=netCDF4.num2date(ds['time'][:],ds['time'].units,calendar=getattr(ds['time'],'calendar','standard'))
        times={(d.year,d.month):i for i,d in enumerate(dates)}
        iy=writer.locate(lat,rows.grid_lat.to_numpy());ix=writer.locate(lon,rows.grid_lon.to_numpy())
        for (year,month),indices in rows.groupby(['year','month'],sort=True).groups.items():
            index=np.asarray(list(indices),dtype=int)
            ti=times[(int(year),int(month))]
            for name in columns:
                layer=np.ma.filled(ds[name][ti,:,:],np.nan if name.startswith('pm25') else 255)
                values[name][index]=layer[iy[index],ix[index]]
    return values


def validate_saved_domain(writer,path,domain):
    columns=['year','month','grid_lat','grid_lon']
    gate_api.require(not domain.duplicated(columns).any(),'accepted support domain has duplicate cells')
    checked=0
    with netCDF4.Dataset(path) as ds:
        lat=np.asarray(ds['lat'][:]);lon=np.asarray(ds['lon'][:])
        dates=netCDF4.num2date(ds['time'][:],ds['time'].units,calendar=getattr(ds['time'],'calendar','standard'))
        times={(d.year,d.month):i for i,d in enumerate(dates)}
        iy=writer.locate(lat,domain.grid_lat.to_numpy());ix=writer.locate(lon,domain.grid_lon.to_numpy())
        ti=np.array([times[(int(y),int(m))] for y,m in domain[['year','month']].itertuples(index=False,name=None)])
        for t in range(len(dates)):
            expected=np.zeros((len(lat),len(lon)),dtype=bool)
            indices=np.flatnonzero(ti==t);expected[iy[indices],ix[indices]]=True
            flags=np.ma.filled(ds['prediction_domain'][t,:,:],255)
            gate_api.require(np.array_equal(flags,np.where(expected,1,255).astype(np.uint8)),
                             'saved prediction-domain mask differs from accepted exact domain')
            affected=np.ma.filled(ds['affected'][t,:,:],255)
            gate_api.require(np.isin(affected[expected],[0,1]).all() and (affected[~expected]==255).all(),
                             'saved affected flags violate domain or0/1contract')
            for name in ('pm25_total','pm25_background','pm25_smoke_proxy'):
                layer=np.ma.filled(ds[name][t,:,:],np.nan)
                gate_api.require(np.isfinite(layer[expected]).all() and np.isnan(layer[~expected]).all(),
                                 'saved concentration nodata/domain mismatch: '+name)
            checked+=int(expected.sum())
    gate_api.require(checked==len(domain),'saved domain calendar coverage differs')
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--go',type=Path)
    parser.add_argument('--go-sha')
    args=parser.parse_args()
    gate_api.require(not args.out.exists(),'fresh grid/plan output root required')
    gate,plan=prepare_plan()
    if not args.execute:
        args.out.mkdir(parents=True)
        with (args.out/'plan.json').open('x') as stream:
            json.dump(plan,stream,indent=2,allow_nan=False)
        print(json.dumps({'status':'bound_no_prediction','products':16,'plan_sha256':gate_api.sha(args.out/'plan.json')}),flush=True)
        return
    gate_api.require(args.plan and args.go and args.go_sha,'separate QA steped grid release approval required')
    go=load_pinned(args.go,args.go_sha,gate)
    gate_api.require(go.get('format')=='final-reference16-grid-root-go-v1' and go.get('decision')=='go'
        and go.get('driver_sha256')==gate_api.sha(Path(__file__)) and go.get('output_root')==str(args.out)
        and go.get('accepted_core_sha256')==gate_api.ACCEPTANCE_SHA and go.get('writer_sha256')==WRITER_SHA,
        'grid release approval differs')
    approved=load_pinned(args.plan,go['plan_sha256'],gate)
    gate_api.require(approved['products']==plan['products'] and approved['source_pins']==plan['source_pins'],
                     'current bound product/source plan changed')
    rr=go['qa_record_ref'];review=load_pinned(rr['path'],rr['sha256'],gate)
    gate_api.require(review.get('decision')=='approve_reference16_grid_execution' and review.get('findings')==[]
        and review.get('driver_sha256')==go['driver_sha256'],'independent grid review differs')
    code=gate_api.ROOT/'runtime/work/repair_v2/validation'
    gate_api.accepted_file(gate,code/'validation_core.py',CORE_SHA)
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('frozen_reference16_grid_writer',WRITER)
    writer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=writer;spec.loader.exec_module(writer)
    expected={'numpy':'2.4.6','pandas':'2.3.3','scikit-learn':'1.9.1'}
    gate_api.require({k:importlib.metadata.version(k) for k in expected}==expected and
        platform.python_version()=='3.11.16','canonical prediction runtime differs')
    gate_api.require(Path(sys.modules['validation_core'].__file__).resolve()==(code/'validation_core.py').resolve(),
                     'grid validation-core import origin differs')
    def no_fit(*args,**kwargs):
        raise RuntimeError('fitting forbidden during saved-model grid materialization')
    sys.modules['validation_core'].fit_model=no_fit
    args.out.mkdir(parents=True)
    with (args.out/'.grid_materialization.claim').open('x') as stream:
        json.dump({'pid':os.getpid(),'root_go_sha256':args.go_sha,'products':16},stream)
        stream.flush();os.fsync(stream.fileno())
    reports=[]
    for product in plan['products']:
        rows=pd.read_parquet(product['rows']['path']).reset_index(drop=True)
        gate_api.require('pm25' not in rows and len(rows)==product['input_rows'],'only accepted target-free grid rows allowed')
        pairs=pd.read_parquet(product['saved_station_reference']['path'])
        selected=pairs.loc[pairs.task.eq('T3_fixed_2023')&pairs.profile.eq(product['profile'])
            &pairs.target_strategy.eq(product['target_strategy'])&np.isclose(pairs.gate_quantile.to_numpy(float),.9)].copy()
        gate_api.require(not selected.duplicated(KEYS).any(),'duplicate saved station reference')
        aligned=rows[KEYS+['grid_lat','grid_lon']].merge(selected[KEYS+[
            'predicted_total','predicted_background','component_prediction','training_threshold','affected']],
            on=KEYS,how='outer',validate='one_to_one',indicator=True,sort=False).reset_index(drop=True)
        gate_api.require(len(aligned)==len(rows)==len(selected) and aligned._merge.eq('both').all(),
                         'accepted station reference and grid input keys differ')
        gate_api.require(aligned.training_threshold.eq(product['threshold']['value']).all(),'P90 station threshold differs')
        full=joblib.load(product['models']['full']['path']);background=joblib.load(product['models']['background']['path'])
        gate_api.require(full['target_strategy']==background['target_strategy']==product['target_strategy'],
                         'saved grid model target strategy differs')
        destination=args.out/(product['product_id']+'.nc')
        cells,writer_record=writer.write_grid(rows,full,background,product['template']['path'],destination,
            product['threshold']['value'],product['country'])
        gate_api.require(len(cells)==product['supported_cell_months'],'declared support-domain count differs')
        domain=pd.read_parquet(product['support_domain']['path'])
        domain_columns=['year','month','grid_lat','grid_lon']
        actual_cells=cells[domain_columns].sort_values(domain_columns).reset_index(drop=True)
        accepted_cells=domain[domain_columns].sort_values(domain_columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_cells,accepted_cells,check_dtype=False,check_exact=True)
        domain_count=validate_saved_domain(writer,destination,domain)
        sampled=sample_saved_grid(writer,destination,aligned)
        for field,column in (('pm25_total','predicted_total'),('pm25_background','predicted_background'),
                             ('pm25_smoke_proxy','component_prediction')):
            gate_api.require(np.isfinite(sampled[field]).all() and
                np.allclose(sampled[field],aligned[column].to_numpy(float),atol=1e-5,rtol=1e-6),
                'actual saved grid differs from accepted independent station reference: '+field)
            aligned['saved_'+field]=sampled[field]
        gate_api.require(np.array_equal(sampled['prediction_domain'],np.ones(len(aligned))) and
            np.array_equal(sampled['affected'],aligned.affected.to_numpy(dtype=np.uint8)),'saved station domain/gate flags differ')
        pair_path=destination.with_suffix('.station_reference.parquet')
        aligned.drop(columns='_merge').to_parquet(pair_path,index=False)
        cell_path=destination.with_suffix('.cell_pairs.parquet');cells.to_parquet(cell_path,index=False)
        reports.append({'product':product,'writer_record':writer_record,'grid':{'path':str(destination),
            'sha256':gate_api.sha(destination)},'station_reference':{'path':str(pair_path),'sha256':gate_api.sha(pair_path)},
            'cell_pairs':{'path':str(cell_path),'sha256':gate_api.sha(cell_path)},
            'accepted_domain_masks_and_all_nodata_layers_verified':True,'verified_domain_cells':domain_count,
            'accepted_saved_station_reference_matched':True,'independent_grid_acceptance_pending':True})
        print(json.dumps({'products_written':len(reports),'total':16,'product':product['product_id']}),flush=True)
    gate_api.assert_unchanged(gate)
    with (args.out/'materialization.json').open('x') as stream:
        json.dump({'format':'final-reference16-saved-grid-materialization-v1','status':'self_checks_passed',
            'products':reports,'source_pins':gate['used'],'fits_performed':False,'driver_sha256':gate_api.sha(Path(__file__)),
            'writer_sha256':WRITER_SHA,'independent_grid_acceptance_pending':True,
            'scope':plan['scope'],'global_or_rectangular_prediction_coverage_claimed':False},stream,indent=2,allow_nan=False)
    print(json.dumps({'status':'self_checks_passed','products':16,'output':str(args.out)}),flush=True)


if __name__=='__main__':
    main()
