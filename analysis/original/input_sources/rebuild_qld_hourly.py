"""Rebuild Queensland targets from official hourly tables with distinct hours."""
from pathlib import Path
import json
import io
import re
import time

import numpy as np
import pandas as pd
import requests

from prepare_us_features import file_hash

BASE = Path(__file__).resolve().parents[2]
RAW = Path(r'F:\cand1_repair_v2_data\qld')
OUT = BASE / 'outputs/repair_v2/qld/hourly_rebuild'
API = 'https://www.data.qld.gov.au/api/3/action'
RESOURCES = {
    2018: '7e5fec81-5947-4bf8-9e55-4e82979b3727',
    2019: '4dc8e1b2-7a3c-45e0-a0bd-a3459b8638d2',
    2020: '25a15320-1cba-46a9-a559-b5cf39d2ee7e',
    2021: 'a6f86c6b-712d-4fab-a9b8-2cd69de3fef2',
    2022: '705ead06-8ae5-47a2-a797-f2b361a4d42f',
}


def get_json(endpoint, params):
    for attempt in range(3):
        try:
            response = requests.get(API + '/' + endpoint, params=params, timeout=(20, 150))
            response.raise_for_status()
            body = response.json()
            if not body.get('success'):
                raise ValueError('CKAN response did not indicate success')
            return body['result']
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fetch(year, resource):
    path = RAW / f'hourly_pm25_{year}_datastore.parquet'
    metadata_path = path.with_suffix('.json')
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['sha256'] != file_hash(path):
            raise ValueError('Cached hourly table hash changed')
        return pd.read_parquet(path), metadata
    dump_url = f'https://www.data.qld.gov.au/datastore/dump/{resource}?format=csv'
    csv_path = RAW / f'hourly_pm25_{year}_official_datastore.csv'
    if not csv_path.exists():
        print(json.dumps({'year':year,'stage':'download_official_datastore_export'}),flush=True)
        response = requests.get(dump_url,timeout=(20,90))
        response.raise_for_status()
        if response.status_code != 200 or len(response.content)<10000:
            raise ValueError(f'Official DataStore export unavailable for {year}: HTTP{response.status_code}')
        candidate = pd.read_csv(io.BytesIO(response.content))
        if not {'Date','Time'}.issubset(candidate.columns):
            raise ValueError('DataStore export lacks Date/Time schema')
        csv_path.write_bytes(response.content)
    frame = pd.read_csv(csv_path)
    if not {'Date','Time'}.issubset(frame.columns):
        raise ValueError('Cached DataStore export lacks Date/Time schema')
    package_path = RAW / f'official_annual_package_{year}.json'
    cached_resources = []
    if package_path.exists():
        package = json.loads(package_path.read_text(encoding='utf-8'))
        cached_resources = [item for item in package['resources'] if item['id'] == resource]
    if cached_resources:
        metadata = cached_resources[0]
    else:
        try:
            metadata = get_json('resource_show', {'id':resource})
        except (requests.RequestException,ValueError):
            metadata = {'id':resource,'url':dump_url,'metadata_status':'resource_metadata_unavailable'}
    (RAW / f'hourly_pm25_{year}_resource.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    frame.to_parquet(path, index=False)
    record = {'year': year, 'resource_id': resource, 'source_url': metadata.get('url'),
              'retrieval_endpoint': dump_url, 'rows': len(frame),
              'path': str(path), 'sha256': file_hash(path),
              'raw_csv_path':str(csv_path),'raw_csv_sha256':file_hash(csv_path),
              'representation':'official DataStore CSV export; source bytes retained including CKAN _id',
              'description': metadata.get('description', '')}
    metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return frame, record


def aggregate(long, policy):
    values = long.pm25.copy()
    if policy == 'drop_negative_hourly':
        values = values.where(values >= 0)
    elif policy == 'clip_negative_hourly':
        values = values.clip(lower=0)
    elif policy != 'retain_valid_negative_hourly':
        raise ValueError(policy)
    work = long.assign(pm25=values).dropna(subset=['pm25'])
    daily = work.groupby(['station', 'date'], as_index=False).agg(
        pm25=('pm25', 'mean'), nhours=('datetime', 'nunique'))
    daily = daily[daily.nhours >= 18].copy()
    daily['year'] = daily.date.dt.year
    daily['month'] = daily.date.dt.month
    monthly = daily.groupby(['station', 'year', 'month'], as_index=False).agg(
        pm25=('pm25', 'mean'), ndays=('date', 'nunique'))
    monthly = monthly[monthly.ndays >= 20].copy()
    return daily, monthly


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frames, sources, audits = [], [], []
    for year, resource in RESOURCES.items():
        wide, source = fetch(year, resource)
        sources.append(source)
        stations = [c for c in wide.columns if c not in ['_id', 'Date', 'Time']]
        long = wide.melt(id_vars=['Date', 'Time'], value_vars=stations,
                         var_name='station', value_name='pm25')
        long['station'] = long.station.str.replace(r'\s*\(ug/m\^3\)', '', regex=True).str.strip()
        long['pm25'] = pd.to_numeric(long.pm25, errors='coerce')
        date = pd.to_datetime(long.Date, format='mixed', dayfirst=True, errors='raise').dt.normalize()
        hours = pd.to_timedelta(long.Time.astype(str) + ':00', errors='raise')
        if ((hours < pd.Timedelta(seconds=0)) | (hours >= pd.Timedelta(days=1))).any():
            raise ValueError('Unexpected hour labels require source-convention review')
        long['datetime'] = date + hours
        if not long.datetime.dt.year.eq(year).all():
            raise ValueError('Source-year/date disagreement')
        duplicates = long.duplicated(['station', 'datetime'], keep=False)
        if long.loc[duplicates].groupby(['station', 'datetime']).pm25.nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting duplicate station-hour observations')
        audit = {'year': year, 'source_rows': len(wide), 'station_columns': len(stations),
                 'duplicate_station_hours': int(duplicates.sum()),
                 'valid_negative_hours': int(long.pm25.between(-5, 0, inclusive='left').sum()),
                 'below_minus5_hours': int(long.pm25.lt(-5).sum()),
                 'at_or_above_2000_hours': int(long.pm25.ge(2000).sum()),
                 'numeric_min': float(long.pm25.min()), 'numeric_max': float(long.pm25.max())}
        long = long.drop_duplicates(['station', 'datetime'])
        long['pm25'] = long.pm25.where(long.pm25.between(-5, 2000, inclusive='left'))
        long['date'] = long.datetime.dt.normalize()
        frames.append(long[['station', 'datetime', 'date', 'pm25']])
        audits.append(audit)
        print(json.dumps(audit), flush=True)
    hourly = pd.concat(frames, ignore_index=True)
    legacy = pd.read_parquet(r'F:\cand1_20261105\work\features\stations_monthly_queensland.parquet')
    identity = legacy[['station', 'site_id', 'lat', 'lon']].drop_duplicates('station')
    output_records = []
    for policy in ['retain_valid_negative_hourly', 'drop_negative_hourly', 'clip_negative_hourly']:
        daily, monthly = aggregate(hourly, policy)
        negative = monthly[monthly.pm25 < 0].copy()
        negative.to_csv(OUT / f'{policy}_negative_months_excluded.csv', index=False)
        monthly = monthly[monthly.pm25 >= 0].merge(identity, on='station', how='left', validate='many_to_one')
        monthly['country'] = 'AUS'
        monthly['source_policy'] = policy
        path = OUT / f'{policy}_monthly.parquet'
        monthly.to_parquet(path, index=False)
        daily.to_parquet(OUT / f'{policy}_daily.parquet', index=False)
        output_records.append({'policy': policy, 'path': str(path), 'rows': len(monthly),
                               'stations': monthly.station.nunique(), 'negative_months_excluded': len(negative),
                               'sha256': file_hash(path)})
    comparison = pd.read_parquet(OUT / 'retain_valid_negative_hourly_monthly.parquet').merge(
        legacy[['station', 'year', 'month', 'pm25', 'ndays']], on=['station', 'year', 'month'],
        how='outer', suffixes=('_rebuild', '_legacy'), indicator=True, validate='one_to_one')
    comparison.to_parquet(OUT / 'legacy_target_comparison.parquet', index=False)
    report = {'sources': sources, 'hourly_audit': audits, 'outputs': output_records,
              'primary_policy': 'retain source-reported valid hourly noise down to -5 ug/m3',
              'aggregation': 'distinct station-hours, >=18 hours/day, equal daily means, >=20 distinct dates/month',
              'range_policy': '[-5,2000) ug/m3, upper check retained for legacy comparability with counts reported',
              'negative_month_policy': 'negative monthly targets excluded and enumerated before log fitting',
              'coordinate_status': 'pending separate official coordinate repair',
              'legacy_match_counts': comparison['_merge'].value_counts().to_dict()}
    (OUT / 'hourly_rebuild_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output_records, indent=2), flush=True)


if __name__ == '__main__':
    main()
