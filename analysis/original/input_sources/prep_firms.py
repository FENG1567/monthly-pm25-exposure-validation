"""Aggregate FIRMS VIIRS 375 m detections to 0.1 deg cell-months.

in  : incoming/firms/{bra,mex,usa}/<year>/firms_viirs_snpp_<cc>_<date>.csv
out : work/features/firms_monthly_0p1.nc
      firms_frp (MW, sum) and firms_n (detections), dims (time, lat, lon)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid0p1 as G  # noqa: E402

ROOT = os.environ.get("FIRE_SMOKE_ROOT", os.getcwd())
INC = os.path.join(ROOT, "incoming", "firms")
OUT = os.path.join(ROOT, "work", "features", os.environ.get("FEAT_SUBDIR", ""), "firms_monthly_0p1.nc")
YEARS = range(int(os.environ.get("FIRMS_Y0", 2018)), int(os.environ.get("FIRMS_Y1", 2022)) + 1)


def load_year(year: int) -> pd.DataFrame:
    frames = []
    env_dirs = os.environ.get("FIRMS_DIRS", "").strip()
    if env_dirs:
        dirs = [d.strip() for d in env_dirs.split(",") if d.strip()]
    else:
        dirs = [d for d in os.listdir(INC) if os.path.isdir(os.path.join(INC, d))]
    for cc in sorted(dirs):
        for fp in glob.glob(os.path.join(INC, cc, str(year), "*.csv")):
            try:
                d = pd.read_csv(fp, usecols=["latitude", "longitude", "frp", "confidence",
                                             "acq_date", "acq_time"])
            except Exception as e:  # noqa: BLE001
                print("  skip", fp, e)
                continue
            if len(d):
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "frp"])
    return df


def main() -> None:
    acc_frp = np.zeros((len(YEARS) * 12, G.NY, G.NX), dtype="float64")
    acc_n = np.zeros_like(acc_frp)
    times = []
    for yi, y in enumerate(YEARS):
        df = load_year(y)
        if df.empty:
            print(f"{y}: no rows")
            continue
        iy, ix = G.cell_index(df["latitude"].to_numpy(), df["longitude"].to_numpy())
        ok = iy >= 0
        df = df.loc[ok].copy()
        iy, ix = iy[ok], ix[ok]
        f = G.flat(iy, ix)
        m = (df["acq_date"].dt.month.to_numpy() - 1) + yi * 12
        key = m.astype("int64") * (G.NY * G.NX) + f
        ncell = len(YEARS) * 12 * G.NY * G.NX
        acc_n += np.bincount(key, minlength=ncell).reshape(acc_n.shape)
        acc_frp += np.bincount(key, weights=df["frp"].to_numpy(), minlength=ncell).reshape(acc_frp.shape)
        print(f"{y}: detections={len(df):,} frp_sum={df['frp'].sum():,.0f} MW "
              f"cells={len(np.unique(f)):,}", flush=True)
        times.extend([np.datetime64(f"{y}-{mo:02d}-01") for mo in range(1, 13)])

    ds = xr.Dataset(
        {"firms_frp": (("time", "lat", "lon"), acc_frp.astype("float32"),
                       {"units": "MW", "long_name": "sum of VIIRS FRP in cell-month"}),
         "firms_n": (("time", "lat", "lon"), acc_n.astype("float32"),
                     {"units": "1", "long_name": "VIIRS 375 m detections in cell-month"})},
        coords={"time": np.array(times, dtype="datetime64[ns]"), "lat": G.LAT, "lon": G.LON},
    )
    ds.attrs.update(source="NASA FIRMS VIIRS_SNPP_SP (375 m)",
                    note="country bbox downloads merged and de-duplicated on (lat,lon,date,time,frp)")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(OUT, encoding=enc)
    print("WROTE", OUT, os.path.getsize(OUT) / 1e6, "MB", ds.sizes)


if __name__ == "__main__":
    main()
