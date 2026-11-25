import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from annual_replay import replay,check_frame,annual_seed,load_kernel,metric

def fixture():
    return pd.DataFrame([dict(alias='fixture',country='USA',qa='strict',profile='no_met',estimator='hgb',
        task='T1_station',arm='full',target_strategy='log_smear',backtransform='smearing',
        site_id=s,year=y,shared_months=12,benchmark_version='V6GL03',observed_annual_pm25=o,
        model_annual_pm25=p,raw_cams_annual_pm25=c,satpm_annual_pm25=c)
        for s,y,o,p,c in [('a',2020,3.,4.,3.5),('a',2021,6.,4.,8.),('b',2020,4.,8.,5.),
                          ('b',2021,7.,6.,8.),('c',2020,5.,5.5,4.5),('c',2021,10.,9.,8.)]])

class Tests(unittest.TestCase):
    def test_cluster_bootstrap_direct_row_equivalence(self):
        d=fixture();identity=d.iloc[0].to_dict();seed=annual_seed(identity,'primary_exact_12_months','SatPM')
        y=d.observed_annual_pm25.to_numpy();a=d.model_annual_pm25.to_numpy();b=d.satpm_annual_pm25.to_numpy()
        kernel=load_kernel();got=kernel(d.site_id.to_numpy(),y,a,b,500,seed)
        ids=np.unique(d.site_id);rng=np.random.default_rng(seed);values={k:[] for k in ('rmse','mae','bias','r2')}
        for _ in range(500):
            draw=rng.choice(len(ids),len(ids),replace=True)
            ix=np.concatenate([np.flatnonzero(d.site_id.to_numpy()==ids[i]) for i in draw])
            aa=metric(y[ix],a[ix]);bb=metric(y[ix],b[ix])
            for k in values:values[k].append(aa[k]-bb[k])
        for k in values:
            np.testing.assert_allclose([got['delta_'+k+'_lo'],got['delta_'+k+'_hi']],np.percentile(values[k],[2.5,97.5]),atol=1e-12,rtol=1e-12)
    def test_support_and_no_favorable_filter(self):
        d=fixture();d.loc[0,'shared_months']=10;out=replay(d)
        self.assertEqual(len(out),4)
        self.assertEqual(out.loc[out.support_scope.eq('primary_exact_12_months'),'rows'].tolist(),[5,5])
        self.assertEqual(out.loc[out.support_scope.eq('sensitivity_at_least_10_months'),'rows'].tolist(),[6,6])
        self.assertTrue(out.delta_rmse.notna().all())
    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):check_frame(pd.concat([fixture(),fixture().iloc[:1]],ignore_index=True))
    def test_nonfinite_eligible_rejected(self):
        d=fixture();d.loc[0,'model_annual_pm25']=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):check_frame(d)
    def test_unavailable_2025_rejected(self):
        d=fixture();d.loc[0,'year']=2025
        with self.assertRaisesRegex(ValueError,'2025'):check_frame(d)
    def test_less_than_two_sites_withholds_intervals(self):
        out=replay(fixture().query("site_id=='a'"))
        self.assertTrue(out.delta_rmse_ci_low.isna().all())
        self.assertTrue(out.delta_rmse.notna().all())

if __name__=='__main__':unittest.main()
