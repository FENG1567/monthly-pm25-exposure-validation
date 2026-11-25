"""Write actual total/background/component grids and verify saved-value sampling."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import joblib
import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'validation'))
from validation_core import predict_model


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inverse_for(bundle):
    return 'model_native' if bundle.get('target_strategy') == 'original_scale' else 'smearing'


def locate(coordinates, points):
    coordinates = np.asarray(coordinates)
    points = np.asarray(points)
    if len(points) == 0 or not np.isfinite(coordinates).all() or not np.isfinite(points).all():
        raise ValueError('Grid coordinates and prediction locations must be finite and nonempty')
    if len(coordinates) == 0:
        raise ValueError('Grid coordinate axis must be nonempty and strictly monotonic')
    ascending = np.all(coordinates[1:] > coordinates[:-1])
    descending = np.all(coordinates[1:] < coordinates[:-1])
    if not ascending and not descending:
        raise ValueError('Grid coordinate axis must be strictly monotonic')

    ordered_coordinates = coordinates[::-1] if descending else coordinates
    right = np.searchsorted(ordered_coordinates, points, side='left')
    left = right.copy()
    left -= 1
    np.clip(left, 0, len(coordinates) - 1, out=left)
    np.clip(right, 0, len(coordinates) - 1, out=right)

    distance_dtype = np.result_type(coordinates.dtype, points.dtype, np.float64)
    left_distance = np.empty(points.shape, dtype=distance_dtype)
    right_distance = np.empty_like(left_distance)
    np.subtract(ordered_coordinates[left], points, out=left_distance)
    np.abs(left_distance, out=left_distance)
    np.subtract(ordered_coordinates[right], points, out=right_distance)
    np.abs(right_distance, out=right_distance)

    # Preserve argmin's first-coordinate tie behavior on either axis orientation.
    choose_right = right_distance <= left_distance if descending else right_distance < left_distance
    np.copyto(left, right, where=choose_right)
    if descending:
        np.subtract(len(coordinates) - 1, left, out=left)

    np.subtract(coordinates[left], points, out=right_distance)
    np.abs(right_distance, out=right_distance)
    if np.any(right_distance > 2e-4):
        raise ValueError('Prediction rows must identify exact template cell centres')
    return left


def write_grid(rows, full_bundle, background_bundle, template, destination, threshold, country):
    required = {'year','month','grid_lat','grid_lon','lgfas_r15','country'}
    if not required.issubset(rows.columns):
        raise ValueError(f'Missing geometry/gate columns {required - set(rows.columns)}')
    if set(rows.country.unique()) != {country}:
        raise ValueError('A grid may contain exactly one declared country')
    if not np.isfinite(threshold) or not np.isfinite(rows.lgfas_r15).all():
        raise ValueError('Smoke gating requires a finite training threshold and finite fire predictors')
    dates = rows[['year','month']].to_numpy(dtype=float)
    if not np.isfinite(dates).all() or not np.equal(dates, np.floor(dates)).all():
        raise ValueError('Prediction years/months must be finite integers')
    if full_bundle.get('target_strategy','log_smear') != background_bundle.get('target_strategy','log_smear'):
        raise ValueError('Total and background must target the same concentration estimand')
    with netCDF4.Dataset(template) as source:
        lat,lon=np.asarray(source['lat'][:]),np.asarray(source['lon'][:])
        if lat.ndim != 1 or lon.ndim != 1 or len(lat)<2 or len(lon)<2:
            raise ValueError('Template requires one-dimensional spatial coordinate vectors')
        lat_diff, lon_diff = np.diff(lat), np.diff(lon)
        if (not np.isfinite(lat).all() or not np.isfinite(lon).all() or
                not (np.all(lat_diff>0) or np.all(lat_diff<0)) or
                not np.allclose(np.abs(lat_diff),.1,atol=1e-4) or not np.allclose(lon_diff,.1,atol=1e-4)):
            raise ValueError('Unexpected template grid resolution/orientation')
        time=source['time'][:]
        time_units=source['time'].units
        time_calendar=getattr(source['time'],'calendar','standard')
        dates=netCDF4.num2date(time,time_units,calendar=time_calendar)
        time_map={(d.year,d.month):i for i,d in enumerate(dates)}
        if len(time_map) != len(time):
            raise ValueError('Template contains repeated calendar months')
    d=rows.copy().reset_index(drop=True)
    d['iy']=locate(lat,d.grid_lat.to_numpy())
    d['ix']=locate(lon,d.grid_lon.to_numpy())
    if any((int(y),int(m)) not in time_map for y,m in zip(d.year,d.month)):
        raise ValueError('Prediction month absent from template')
    d['ti']=[time_map[(int(y),int(m))] for y,m in zip(d.year,d.month)]
    # Physically colocated observations must receive the same grid predictors.
    keys=['ti','iy','ix']
    predictor_columns=sorted(set(full_bundle['schema']['base_features']) |
                             set(background_bundle['schema']['base_features']) |
                             set(full_bundle['schema'].get('required_source_columns',[])) |
                             set(background_bundle['schema'].get('required_source_columns',[])))
    for column in predictor_columns:
        if d.groupby(keys)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f'Conflicting grid predictors for colocated rows: {column}')
    cells=d.drop_duplicates(keys).copy()
    cells['expected_total']=predict_model(full_bundle,cells,backtransform=inverse_for(full_bundle))
    cells['expected_background']=predict_model(background_bundle,cells,backtransform=inverse_for(background_bundle))
    cells['affected']=cells.lgfas_r15>threshold
    cells['expected_smoke']=np.where(cells.affected,np.maximum(cells.expected_total-cells.expected_background,0),0)
    if not np.isfinite(cells[['expected_total','expected_background','expected_smoke']]).all().all():
        raise ValueError('Nonfinite prediction cannot be published')
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_suffix('.nc.part')
    with netCDF4.Dataset(partial,'w') as output:
        for name,size in [('time',len(time)),('lat',len(lat)),('lon',len(lon))]:
            output.createDimension(name,size)
        output.createVariable('time','f8',('time',))[:]=time
        output['time'].setncatts({'units':time_units,'calendar':time_calendar})
        output.createVariable('lat','f8',('lat',))[:]=lat
        output.createVariable('lon','f8',('lon',))[:]=lon
        output['lat'].setncatts({'standard_name':'latitude','units':'degrees_north'})
        output['lon'].setncatts({'standard_name':'longitude','units':'degrees_east'})
        for name,long_name in [('pm25_total','Predicted total monthly PM2.5'),
                               ('pm25_background','Predicted low-fire background monthly PM2.5'),
                               ('pm25_smoke_proxy','Nonnegative gated predicted total minus low-fire background')]:
            variable=output.createVariable(name,'f4',('time','lat','lon'),fill_value=np.float32(np.nan),
                                           zlib=True,complevel=4,chunksizes=(1,min(128,len(lat)),min(128,len(lon))))
            variable.setncatts({'units':'ug m-3','long_name':long_name})
        output.createVariable('prediction_domain','u1',('time','lat','lon'),fill_value=255,zlib=True)
        output.createVariable('affected','u1',('time','lat','lon'),fill_value=255,zlib=True)
        for t,frame in cells.groupby('ti',sort=True):
            row_index=frame.iy.to_numpy(dtype=int)
            col_index=frame.ix.to_numpy(dtype=int)
            # NetCDF fancy indexing is orthogonal. Assign through a 2-D layer
            # to preserve paired row/column coordinates.
            for name,column in [('pm25_total','expected_total'),('pm25_background','expected_background'),
                                ('pm25_smoke_proxy','expected_smoke')]:
                layer=np.full((len(lat),len(lon)),np.nan,dtype=np.float32)
                layer[row_index,col_index]=frame[column].to_numpy(dtype=np.float32)
                output[name][int(t),:,:]=layer
            for name,column in [('prediction_domain',None),('affected','affected')]:
                layer=np.full((len(lat),len(lon)),255,dtype=np.uint8)
                layer[row_index,col_index]=1 if column is None else frame[column].to_numpy(dtype=np.uint8)
                output[name][int(t),:,:]=layer
        output.setncatts({'country':country,'fire_threshold_training_only':float(threshold),
                         'prediction_domain_definition':'Only explicitly supplied supported cell-months; outside this domain is missing',
                         'component_interpretation':'Statistical smoke proxy, not established as identified smoke mass',
                         'template_sha256':digest(template),
                         'prediction_code_sha256':digest(__file__),
                         'inverse_total':full_bundle['backtransform_primary'],
                         'inverse_background':background_bundle['backtransform_primary']})
    partial.replace(destination)
    # Verify the materialized file, not an alternative call to the same predictor.
    with netCDF4.Dataset(destination) as saved:
        for name,column in [('pm25_total','expected_total'),('pm25_background','expected_background'),
                            ('pm25_smoke_proxy','expected_smoke')]:
            sampled=np.full(len(cells),np.nan)
            for t,frame in cells.groupby('ti',sort=True):
                layer=np.ma.filled(saved[name][int(t),:,:],np.nan)
                sampled[cells.index.get_indexer(frame.index)]=layer[frame.iy.to_numpy(dtype=int),frame.ix.to_numpy(dtype=int)]
            cells[f'saved_{name}']=sampled
            if not np.allclose(sampled,cells[column],atol=1e-5,rtol=1e-6):
                raise AssertionError(f'Saved grid sampling disagrees with station prediction path: {name}')
    return cells, {'grid_path':str(destination),'grid_sha256':digest(destination),
                   'prediction_code_sha256':digest(__file__),
                   'supported_cell_months':len(cells),'input_rows':len(d),
                   'domain':'Explicit supported cell-months only',
                   'saved_grid_roundtrip_verified':True,
                   'independent_station_reference_verified':False,
                   'country_assignment':'Country declared in supplied rows; polygon/coastal mask acceptance must be verified separately'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--full-model',type=Path,required=True)
    parser.add_argument('--background-model',type=Path,required=True)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threshold',type=float,required=True)
    parser.add_argument('--country',required=True)
    args=parser.parse_args()
    rows=pd.read_parquet(args.rows)
    full=joblib.load(args.full_model)
    background=joblib.load(args.background_model)
    cells,record=write_grid(rows,full,background,args.template,args.out,args.threshold,args.country)
    cells.to_parquet(args.out.with_suffix('.pairs.parquet'),index=False)
    record.update(rows_sha256=digest(args.rows),full_model_sha256=digest(args.full_model),
                  background_model_sha256=digest(args.background_model))
    args.out.with_suffix('.verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
