"""Validate smoke-proxy and chemical-tracer consistency against observed HMS plumes."""
from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from prepare_us_features import file_hash

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs/repair_v2/hms/validation_2023_pre_qa'
RAW = Path(r'F:\cand1_repair_v2_data\hms')
SMOKE = BASE / 'outputs/repair_v2/smoke/pilot_2023_server/pilot_2023'
POINT_KEYS = ['country','site_id','lat','lon']
SEED = 610
REPS = 500


def classify(points,plumes,available):
    days = pd.date_range('2023-01-01','2023-12-31',freq='D')
    available = np.array([date.strftime('%Y%m%d') in available for date in days])
    tree = shapely.STRtree(shapely.points(points.lon.to_numpy(),points.lat.to_numpy()))
    matrices = np.zeros((len(days),len(points)),dtype=np.uint8)
    for day_index,day in enumerate(days):
        selected = plumes[(plumes.start_day<=day)&(plumes.end_day>=day)]
        if not selected.empty:
            pairs = tree.query(selected.geometry.to_numpy(),predicate='intersects')
            if pairs.shape[1]:
                density = selected.density_code.to_numpy()[pairs[0]]
                np.maximum.at(matrices[day_index],pairs[1],density)
    rows = pd.DataFrame({'date':np.repeat(days,len(points)),
                         'point_id':np.tile(points.point_id.to_numpy(),len(days)),
                         'density_code':matrices.ravel(),
                         'archive_available':np.repeat(available,len(points))})
    lag_matrix = matrices.copy()
    lag_matrix[1:] = np.maximum(lag_matrix[1:],matrices[:-1])
    lag_matrix[:-1] = np.maximum(lag_matrix[:-1],matrices[1:])
    lag_available = available.copy()
    lag_available[1:] &= available[:-1]
    lag_available[:-1] &= available[1:]
    lag_available[[0,-1]] = False
    rows['density_code_plusminus1day'] = lag_matrix.ravel()
    rows['archive_available_plusminus1day'] = np.repeat(lag_available,len(points))
    return rows


def oc_contrasts(daily):
    records, tables = [], []
    keys = ['site_id','parameter_code','method_code','year','month']
    for alignment,suffix in [('same_named_date_EPA_local_HMS_UTC',''),('plusminus1day_window_expansion','_plusminus1day')]:
        eligible = daily[daily['archive_available'+suffix]].copy()
        for threshold,label in [(1,'any_plume'),(2,'medium_or_heavy')]:
            eligible['plume'] = eligible['density_code'+suffix] >= threshold
            cells = eligible.groupby(keys+['plume'],as_index=False).agg(
                value=('value','mean'), n_dates=('date','nunique'))
            positive = cells[cells.plume].drop(columns='plume').rename(
                columns={'value':'oc_plume','n_dates':'n_plume_dates'})
            negative = cells[~cells.plume].drop(columns='plume').rename(
                columns={'value':'oc_below_density_threshold','n_dates':'n_below_threshold_dates'})
            paired = positive.merge(negative,on=keys,validate='one_to_one')
            paired = paired[(paired.n_plume_dates>=2)&(paired.n_below_threshold_dates>=2)].copy()
            paired['delta_oc'] = paired.oc_plume-paired.oc_below_density_threshold
            paired['plume_definition'],paired['alignment'] = label,alignment
            tables.append(paired)
            for codes, group in paired.groupby(['parameter_code','method_code']):
                station_effects = group.groupby('site_id').delta_oc.mean().to_numpy()
                if len(station_effects)<2:
                    continue
                rng = np.random.default_rng(SEED)
                boot = np.array([rng.choice(station_effects,len(station_effects),replace=True).mean()
                                 for _ in range(REPS)])
                interval = np.quantile(boot,[.025,.975])
                records.append({'parameter_code':codes[0],'method_code':codes[1],
                                'alignment':alignment,'plume_definition':label,
                                'station_month_contrasts':len(group),'sites':len(station_effects),
                                'mean_station_oc_difference':station_effects.mean(),
                                'ci_low':interval[0],'ci_high':interval[1],
                                'interval_scope':'independent-station bootstrap, fixed 2023; shared regional episodes unquantified'})
    return pd.concat(tables,ignore_index=True),pd.DataFrame(records)


def correlation(frame,x,y):
    if len(frame)<3 or np.ptp(frame[x])==0 or np.ptp(frame[y])==0:
        return np.nan
    return float(np.corrcoef(frame[x],frame[y])[0,1])


def station_correlation_interval(frame,x,y,reps=REPS):
    codes,station_ids = pd.factorize(frame.site_id,sort=False)
    values_x,values_y = frame[x].to_numpy(float),frame[y].to_numpy(float)
    moments = np.column_stack([
        np.bincount(codes),np.bincount(codes,weights=values_x),
        np.bincount(codes,weights=values_y),np.bincount(codes,weights=values_x**2),
        np.bincount(codes,weights=values_y**2),np.bincount(codes,weights=values_x*values_y)])
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(reps):
        selected = rng.integers(0,len(station_ids),len(station_ids))
        n,sum_x,sum_y,sum_x2,sum_y2,sum_xy = moments[selected].sum(axis=0)
        variance_x = max(n*sum_x2-sum_x**2,0.)
        variance_y = max(n*sum_y2-sum_y**2,0.)
        denominator = np.sqrt(variance_x*variance_y)
        draws.append((n*sum_xy-sum_x*sum_y)/denominator if denominator>0 else np.nan)
    finite = np.array(draws)[np.isfinite(draws)]
    interval = np.quantile(finite,[.025,.975]) if len(finite)>=.95*reps else [np.nan,np.nan]
    return interval,len(finite)


def monthly_checks(models,monthly):
    d = models.merge(monthly,on=['point_id','year','month'],validate='many_to_one')
    d = d[d.n_available>=np.ceil(d.calendar_days*.8)].copy()
    grouping = ['model_scope','parameter_code','method_code']
    results = []
    for labels,group in d.groupby(grouping,dropna=False):
        support_columns = ['country','site_id','year','month','point_id']
        support = group[support_columns].sort_values(support_columns).reset_index(drop=True)
        support_hash = hashlib.sha256(pd.util.hash_pandas_object(support,index=False).to_numpy().tobytes()).hexdigest()
        for column in ['any_plume_days','medium_heavy_days','any_plume_fraction','medium_heavy_fraction']:
            r = correlation(group,'component_prediction',column)
            station_ids = group.site_id.unique()
            interval,n_finite = station_correlation_interval(group,'component_prediction',column)
            results.append({**dict(zip(grouping,labels)),'HMS_indicator':column,'n_pairs':len(group),
                            'sites':len(station_ids),'pearson_r':r,'ci_low':interval[0],'ci_high':interval[1],
                            'valid_bootstrap_reps':n_finite,'support_sha256':support_hash,
                            'interval_scope':'independent-station bootstrap, fixed 2023; shared regional episodes unquantified'})
    d['hms_day_bin'] = pd.cut(d.any_plume_days,[-.1,.1,5,15,31],labels=['0','1-5','6-15','16+'])
    bins = d.groupby(grouping+['hms_day_bin'],observed=True,dropna=False).agg(
        n=('component_prediction','size'),mean_component=('component_prediction','mean'),
        mean_total=('predicted_total','mean')).reset_index()
    return d,pd.DataFrame(results),bins


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path = BASE / 'outputs/repair_v2/hms/availability_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    index_records = [record for record in manifest['records'] if record['kind']=='index' and '/2023/' in record['url']]
    if len(index_records)!=12 or any(record['status']!='verified' for record in index_records):
        raise ValueError('Complete 2023 availability-index checks are required before classifying absent plumes')
    raw_path = RAW/'hms_smoke2023.zip'
    plumes = gpd.read_file('zip://'+str(raw_path))
    if plumes.crs is None or plumes.crs.to_epsg()!=4326:
        raise ValueError('Unexpected HMS coordinate reference system')
    plumes['start_day'] = pd.to_datetime(plumes.Start,format='%Y%j %H%M',errors='raise').dt.normalize()
    plumes['end_day'] = pd.to_datetime(plumes.End,format='%Y%j %H%M',errors='raise').dt.normalize()
    if (plumes.end_day<plumes.start_day).any() or (plumes.end_day-plumes.start_day).dt.days.gt(2).any():
        raise ValueError('Unexpected HMS temporal span')
    plumes['density_code'] = plumes.Density.map({'Light':1,'Medium':2,'Heavy':3})
    if plumes.density_code.isna().any() or plumes.geometry.isna().any():
        raise ValueError('Unknown density or absent HMS geometry')
    invalid = ~plumes.geometry.is_valid
    plumes.loc[invalid,'geometry'] = shapely.make_valid(plumes.loc[invalid].geometry.to_numpy())
    oc_path = BASE/'outputs/repair_v2/epa/oc_station_day_2018_2025.parquet'
    oc = pd.read_parquet(oc_path)
    oc = oc[oc.year.eq(2023)].copy()
    oc['parameter_code'],oc['method_code'] = oc.parameter_code.astype(str),oc.method_code.astype(str)
    if oc.duplicated(['site_id','parameter_code','method_code','date']).any():
        raise ValueError('Duplicate OC station-method-date keys')
    reference_path = SMOKE/'usa_can_future_2023_reference_pairs.parquet'
    reference = pd.read_parquet(reference_path)
    reference = reference[reference.cohort.isin(['USA','CAN_all_published'])].copy()
    reference['model_scope'] = reference.cohort
    reference['parameter_code'],reference['method_code'] = 'not_applicable','not_applicable'
    oc_model_path = SMOKE/'oc_grid_2023_matched_pairs.parquet'
    oc_models = pd.read_parquet(oc_model_path)
    oc_models['model_scope'] = 'USA_OC_10km_PM_training_exclusion'
    models = pd.concat([reference,oc_models],ignore_index=True)
    points = pd.concat([oc[POINT_KEYS],models[POINT_KEYS]],ignore_index=True).drop_duplicates().reset_index(drop=True)
    if not np.isfinite(points[['lat','lon']]).all().all():
        raise ValueError('Missing station geometry')
    points['point_id'] = np.arange(len(points))
    points.to_parquet(OUT/'station_points.parquet',index=False)
    daily = classify(points,plumes,set(manifest['daily_archive_dates']))
    daily.to_parquet(OUT/'hms_station_day_2023.parquet',index=False)
    matched_oc = oc.merge(points,on=POINT_KEYS,validate='many_to_one').merge(
        daily,on=['point_id','date'],validate='many_to_one')
    matched_oc.to_parquet(OUT/'oc_hms_daily_pairs.parquet',index=False)
    contrasts,contrast_metrics = oc_contrasts(matched_oc)
    contrasts.to_parquet(OUT/'oc_within_station_month_contrasts.parquet',index=False)
    contrast_metrics.to_csv(OUT/'oc_contrast_metrics.csv',index=False)
    daily['year'],daily['month'] = daily.date.dt.year,daily.date.dt.month
    daily['any_plume'] = (daily.density_code>=1)&daily.archive_available
    daily['medium_heavy'] = (daily.density_code>=2)&daily.archive_available
    monthly = daily.groupby(['point_id','year','month'],as_index=False).agg(
        n_available=('archive_available','sum'),any_plume_days=('any_plume','sum'),
        medium_heavy_days=('medium_heavy','sum'),calendar_days=('date','nunique'))
    monthly['any_plume_fraction'] = monthly.any_plume_days/monthly.n_available.replace(0,np.nan)
    monthly['medium_heavy_fraction'] = monthly.medium_heavy_days/monthly.n_available.replace(0,np.nan)
    monthly.to_parquet(OUT/'hms_station_month_2023.parquet',index=False)
    models = models.merge(points,on=POINT_KEYS,validate='many_to_one')
    pairs,metrics,bins = monthly_checks(models,monthly)
    pairs.to_parquet(OUT/'predicted_component_hms_pairs.parquet',index=False)
    metrics.to_csv(OUT/'component_hms_metrics.csv',index=False)
    bins.to_csv(OUT/'component_by_hms_day_bins.csv',index=False)
    report = {'phase':'pre-MAIAC-QA supplementary 2023 consistency analysis',
              'plume_polygons':len(plumes),'invalid_geometries_repaired':int(invalid.sum()),
              'unique_station_coordinates':len(points),'daily_point_rows':len(daily),
              'OC_daily_pairs':len(matched_oc),'OC_days_with_HMS_archive':int(matched_oc.archive_available.sum()),
              'within_station_month_contrasts':len(contrasts),'monthly_model_pairs':len(pairs),
              'bootstrap_reps':REPS,'sources':[{'path':str(path),'sha256':file_hash(path)} for path in
                  [raw_path,manifest_path,oc_path,reference_path,oc_model_path]],
              'protocol_sha256':file_hash(Path(__file__).with_name('hms_2023_protocol.md')),
              'analysis_code_sha256':file_hash(__file__),
              'temporal_alignment':'EPA Date Local matched to the same named HMS UTC date; +/-1 day expands the plume window',
              'control_definitions':{'any_plume':'no delineated plume','medium_or_heavy':'light or no delineated plume'},
              'dependence':'Identical support hashes identify repeated model/HMS comparisons across OC parameter series. Regional-event dependence is unquantified.',
              'interpretation':'Satellite-delineated plumes can be aloft. OC has multiple sources. Associations quantify consistency.'}
    (OUT/'hms_validation_provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='sources'},indent=2),flush=True)


if __name__ == '__main__':
    main()
