from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
SEED=610
REPS=500

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

