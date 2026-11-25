"""Sample original predictor grids at corrected monitoring dates and coordinates."""
from pathlib import Path
import argparse
import json
import sys

import netCDF4
import numpy as np
import pandas as pd

from prepare_us_features import sample_grid,file_hash

sys.path.insert(0,str(Path(__file__).resolve().parent/'validation'))
from run_validation import TABLES,MET,GEO,AOD,FIRE

PROJECT = Path(r'F:\cand1_20261105')
COUNTRY_REGION = {country:region for region,countries in TABLES.items() for country in countries}
ALIASES = {'BR':'BRA','US':'USA','CA':'CAN','AU':'AUS','ESP':'ES','PRT':'PT','GRC':'GR',
           'ITA':'IT','CHL':'CL','ZAF':'ZA','IND':'IN','PAK':'PK','NPL':'NP','BGD':'BD',
           'THA':'TH','VNM':'VN','PHL':'PH','IDN':'ID'}


def normalized(frame):
    result = frame.copy()
    if 'country' not in result:
        result['country'] = result['iso']
    result['country'] = result.country.astype(str).str.strip().str.upper().replace(ALIASES)
    if 'ndays' not in result and 'n_valid_dates' in result:
        result['ndays'] = result.n_valid_dates
    required = ['country','site_id','lat','lon','year','month','pm25','ndays']
    if not set(required).issubset(result.columns):
        raise ValueError(f'Corrected target schema is incomplete: {set(required)-set(result.columns)}')
    if result.duplicated(['country','site_id','year','month']).any():
        raise ValueError('Corrected target table has duplicate physical station-month keys')
    if not np.isfinite(result[['year','month','pm25','ndays']]).all().all():
        raise ValueError('Corrected targets must have finite dates, counts and concentrations')
    if result.pm25.lt(0).any() or result.ndays.lt(20).any():
        raise ValueError('Corrected targets fail the nonnegative monthly/20-day contract')
    return result


def grid_path(region,year):
    directory = 'amer2023' if region=='amer' and year==2023 else region
    if directory=='amer':
        return PROJECT/'work/features/grid_features.nc'
    return PROJECT/'work/features'/directory/'grid_features.nc'


def empty_features(target,path):
    result = target.iloc[:0].copy()
    with netCDF4.Dataset(path) as dataset:
        for name,variable in dataset.variables.items():
            if variable.dimensions in [('lat','lon'),('time','lat','lon')]:
                result[name] = pd.Series(dtype='float32')
    for name in ['grid_lat','grid_lon','aod_missing']+MET:
        if name not in result:
            result[name] = pd.Series(dtype='float32')
    return result


def main(args):
    frame = normalized(pd.read_parquet(args.monthly))
    requested = [country.strip() for country in args.countries.split(',')]
    if not set(requested).issubset(COUNTRY_REGION):
        raise ValueError('A requested country has no documented source predictor grid')
    start,end = map(int,args.years.split('-'))
    args.out.mkdir(parents=True,exist_ok=True)
    reject_keys = ['country','site_id','year','month','lat','lon']
    exclusions = []
    outside = frame[~frame.country.isin(requested)|~frame.year.between(start,end)][reject_keys].copy()
    outside['reason'] = 'outside_requested_country_or_year_scope'
    exclusions.append(outside)
    selected = frame[frame.country.isin(requested)&frame.year.between(start,end)].copy()
    if args.eligible_column:
        if args.eligible_column not in selected:
            raise ValueError('Requested source-eligibility column is absent')
        eligible = selected[args.eligible_column].eq(True)
        excluded = selected.loc[~eligible,reject_keys].copy()
        excluded['reason'] = 'source_eligibility_'+args.eligible_column
        exclusions.append(excluded)
        selected = selected.loc[eligible].copy()
    selected['feature_region'] = selected.country.map(COUNTRY_REGION)
    selected['source_grid'] = [str(grid_path(region,int(year)))
                               for region,year in zip(selected.feature_region,selected.year)]
    sampled,grids = [],[]
    for source,group in selected.groupby('source_grid',sort=True):
        path = Path(source)
        features,rejected = sample_grid(group,path)
        rejected = rejected.merge(group[reject_keys],on=['site_id','year','month','lat','lon'],how='left',validate='one_to_one')
        exclusions.append(rejected)
        absent = [name for name in GEO+AOD+FIRE+['aod_raw'] if name not in features]
        if absent:
            raise ValueError(f'Predictor grid {source} lacks required features {absent}')
        unavailable_met = [name for name in MET if name not in features]
        for name in unavailable_met:
            features[name] = np.nan
        sampled.append(features)
        grids.append({'path':str(path),'sha256':file_hash(path),'requested_rows':len(group),
                      'supported_rows':len(features),'missing_meteorology_columns':unavailable_met})
        print(json.dumps(grids[-1]),flush=True)
    combined = pd.concat(sampled,ignore_index=True) if sampled else None
    outputs = []
    for country in requested:
        if combined is None or not combined.country.eq(country).any():
            output = empty_features(frame[frame.country.eq(country)],grid_path(COUNTRY_REGION[country],start))
        else:
            output = combined[combined.country.eq(country)].copy()
        destination = args.out/f'{country}_{args.label}_{start}_{end}.parquet'
        output.to_parquet(destination,index=False)
        outputs.append({'country':country,'path':str(destination),'rows':len(output),
                        'sites':output.site_id.nunique(),'sha256':file_hash(destination),
                        'status':'sampled' if len(output) else 'no_verified_target_months_in_supported_scope'})
    pd.concat(exclusions,ignore_index=True).to_csv(args.out/f'{args.label}_feature_exclusions.csv',index=False)
    report = {'monthly_path':str(args.monthly),'monthly_sha256':file_hash(args.monthly),
              'source_eligibility_column':args.eligible_column,'requested_years':[start,end],
              'requested_countries':requested,'source_grids':grids,'outputs':outputs,
              'AOD_status':'legacy grid sampling, final QA overlay still required',
              'empty_country_policy':'explicit empty feature tables, no fallback to inherited monitoring targets',
              'code_sha256':file_hash(__file__)}
    (args.out/f'{args.label}_feature_provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monthly',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--countries',required=True)
    parser.add_argument('--years',default='2018-2022')
    parser.add_argument('--eligible-column')
    main(parser.parse_args())
