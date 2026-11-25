"""Resample predictor grids for cleaned US targets and attach comparable ERA5."""
from pathlib import Path
import argparse
import hashlib
import json

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(r'F:\cand1_20261105')
MET = Path(r'F:\cand1_repair_v2_data\era5')
OUT = Path(__file__).resolve().parents[2] / 'outputs/repair_v2/features'
OUT.mkdir(parents=True, exist_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def months(dataset):
    name = 'valid_time' if 'valid_time' in dataset.variables else 'time'
    variable = dataset[name]
    values = netCDF4.num2date(variable[:], variable.units,
                            calendar=getattr(variable, 'calendar', 'standard'))
    return {(value.year, value.month): index for index, value in enumerate(values)}


def nearest(coordinates, points, tolerance):
    indices = np.abs(coordinates[:, None] - points[None, :]).argmin(axis=0)
    valid = np.abs(coordinates[indices] - points) <= tolerance + 1e-6
    return indices, valid


def sample_grid(frame, path):
    d = frame.copy().reset_index(drop=True)
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds['lat'][:])
        lon = np.asarray(ds['lon'][:])
        iy, vy = nearest(lat, d.lat.to_numpy(), .05001)
        ix, vx = nearest(lon, d.lon.to_numpy(), .05001)
        time_map = months(ds)
        ti = np.array([time_map.get((int(y), int(m)), -1) for y, m in zip(d.year, d.month)])
        valid = vy & vx & (ti >= 0)
        rejected = d.loc[~valid, ['site_id', 'year', 'month', 'lat', 'lon']].copy()
        rejected['reason'] = np.where((~vy | ~vx)[~valid], 'outside_grid', 'month_absent')
        d, iy, ix, ti = d.loc[valid].reset_index(drop=True), iy[valid], ix[valid], ti[valid]
        fields = [name for name, var in ds.variables.items()
                  if var.dimensions in [('lat', 'lon'), ('time', 'lat', 'lon')]]
        for field in fields:
            variable = ds[field]
            values = np.full(len(d), np.nan, dtype=np.float32)
            if variable.ndim == 2:
                layer = np.ma.filled(variable[:], np.nan)
                values[:] = layer[iy, ix]
            else:
                # Compressed source chunks span multiple times. Read each field once
                # to avoid decompressing the same temporal chunks for every month.
                cube = np.ma.filled(variable[:], np.nan)
                values[:] = cube[ti, iy, ix]
                del cube
            d[field] = values
        d['aod_missing'] = (~np.isfinite(d['aod_used'])).astype(np.float32)
        d['grid_lat'] = lat[iy]
        d['grid_lon'] = lon[ix]
    return d, rejected


def attach_era5(frame, directory=MET):
    d = frame.copy().reset_index(drop=True)
    arrays = {}
    source_meta = []
    for path in sorted(Path(directory).glob('data_stream*.nc')):
        with netCDF4.Dataset(path) as ds:
            lat, lon = np.asarray(ds['latitude'][:]), np.asarray(ds['longitude'][:])
            # Match the original 0.1-degree grid-centre -> ERA5 nearest-neighbour path.
            point_lat = d['grid_lat'].to_numpy() if 'grid_lat' in d else d.lat.to_numpy()
            point_lon = d['grid_lon'].to_numpy() if 'grid_lon' in d else d.lon.to_numpy()
            iy, vy = nearest(lat, point_lat, .2)
            ix, vx = nearest(lon, point_lon, .2)
            tmap = months(ds)
            ti = np.array([tmap.get((int(y), int(m)), -1) for y, m in zip(d.year, d.month)])
            valid = vy & vx & (ti >= 0)
            fields = [name for name in ['u10', 'v10', 'blh', 't2m', 'd2m', 'tp'] if name in ds.variables]
            for field in fields:
                values = np.full(len(d), np.nan, dtype=np.float32)
                cube = np.ma.filled(ds[field][:], np.nan)
                values[valid] = cube[ti[valid], iy[valid], ix[valid]]
                del cube
                arrays[field] = values
            source_meta.append({'path': str(path), 'sha256': file_hash(path),
                                'variables': fields, 'matched_rows': int(valid.sum()),
                                'units': {field: getattr(ds[field], 'units', '') for field in fields}})
    required = {'u10', 'v10', 'blh', 't2m', 'd2m', 'tp'}
    if set(arrays) != required:
        raise ValueError(f'Incomplete ERA5 schema, missing {required - set(arrays)}')
    d['blh'] = arrays['blh']
    d['wind10'] = np.hypot(arrays['u10'], arrays['v10'])
    d['tp'] = arrays['tp'] * 1000
    d['t2m'] = arrays['t2m'] - 273.15
    dewpoint = arrays['d2m'] - 273.15
    d['rh'] = np.clip(100 * np.exp(17.67 * dewpoint / (dewpoint + 243.5))
                       / np.exp(17.67 * d['t2m'] / (d['t2m'] + 243.5)), 1, 100)
    d['vent'] = d['blh'] * d['wind10']
    return d, source_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monthly', required=True, type=Path)
    args = parser.parse_args()
    monthly = pd.read_parquet(args.monthly)
    if 'country' in monthly:
        monthly = monthly[monthly.country == 'USA'].copy()
    monthly['country'] = 'USA'
    if 'lat' not in monthly:
        monthly = monthly.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    outputs, rejected, metadata = [], [], []
    for label, years, grid in [('2018_2022', range(2018, 2023), ROOT / 'work/features/grid_features.nc'),
                               ('2023', [2023], ROOT / 'work/features/amer2023/grid_features.nc')]:
        sub = monthly[monthly.year.isin(years)].copy()
        features, excluded = sample_grid(sub, grid)
        if label == '2023':
            features, sources = attach_era5(features)
            metadata.extend(sources)
        target = OUT / f'usa_clean_features_{label}.parquet'
        features.to_parquet(target, index=False)
        outputs.append({'path': str(target), 'rows': len(features), 'sites': features.site_id.nunique(),
                        'sha256': file_hash(target), 'grid_source': str(grid)})
        rejected.append(excluded)
        print(json.dumps(outputs[-1]), flush=True)
    pd.concat(rejected, ignore_index=True).to_csv(OUT / 'usa_grid_excluded.csv', index=False)
    report = {'input_monthly': str(args.monthly), 'input_sha256': file_hash(args.monthly),
              'outputs': outputs, 'era5_sources': metadata,
              'tp_definition': '1000 times ERA5 MODA monthly mean daily accumulation, mm/day, same numeric definition as legacy training input',
              'wind10_definition': 'Magnitude of monthly mean u10/v10, not mean of instantaneous wind speed',
              'aod_status': 'Original pre-QA grid features, pending independently processed QA overlay'}
    (OUT / 'usa_feature_provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
