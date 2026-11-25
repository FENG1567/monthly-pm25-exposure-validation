"""Assemble the 0.1 deg predictor stack and station-month feature table.

Inputs (work/features):
  gfas_monthly_0p1.nc      gfas_flux, gfas_emission
  firms_monthly_0p1.nc     firms_frp, firms_n
  maiac_aod_monthly_0p1.nc aod, nobs
  era5_monthly_0p1.nc      blh, wind10, tp, t2m, rh        (optional)
  stations_monthly.parquet country, site_id, lat, lon, year, month, pm25

Outputs:
  grid_features.nc            predictors on the 0.1 deg grid, dims (time, lat, lon)
  stations_features.parquet   one row per station-month with predictor columns
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid0p1 as G  # noqa: E402

ROOT = os.environ.get("FIRE_SMOKE_ROOT", os.getcwd())
FEAT = os.path.join(ROOT, "work", "features", os.environ.get("FEAT_SUBDIR", ""))

VARS = ["gfas_flux", "gfas_emission", "firms_frp", "firms_n", "aod", "nobs",
        "blh", "wind10", "tp", "t2m", "rh"]


def load_stack() -> xr.Dataset:
    gfas = xr.open_dataset(os.path.join(FEAT, "gfas_monthly_0p1.nc"))
    firms = xr.open_dataset(os.path.join(FEAT, "firms_monthly_0p1.nc"))
    aod = xr.open_dataset(os.path.join(FEAT, "maiac_aod_monthly_0p1.nc"))
    ds = xr.merge([gfas, firms, aod], join="inner")
    era = os.path.join(FEAT, "era5_monthly_0p1.nc")
    if os.path.exists(era):
        e = xr.open_dataset(era)
        ds = xr.merge([ds, e], join="inner", compat="override")
        print("ERA5 merged:", [v for v in e.data_vars])
    else:
        print("ERA5 absent - meteorology features omitted")
    pop = os.path.join(FEAT, "pop_0p1.nc")
    if os.path.exists(pop):
        ds = xr.merge([ds, xr.open_dataset(pop)], compat="override")
        print("population merged: lpop")
    # restrict to the calibration window
    w0 = os.environ.get("WIN_START", "2018-01-01")
    w1 = os.environ.get("WIN_END", "2022-12-31")
    ds = ds.sel(time=slice(w0, w1))
    ds = ds.sortby("lat", ascending=False)
    print(ds.sizes, list(ds.data_vars))
    return ds


def add_derived(ds: xr.Dataset) -> xr.Dataset:
    m = ds["time"].dt.month
    ds["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    ds["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    ds["month_sin2"] = np.sin(4 * np.pi * (m - 1) / 12)
    ds["month_cos2"] = np.cos(4 * np.pi * (m - 1) / 12)
    if "blh" in ds and "wind10" in ds:
        ds["vent"] = ds["blh"] * ds["wind10"]  # ventilation coefficient proxy
    # broadcast time-only covariates onto the grid so every predictor is (time, lat, lon)
    ref = ds["gfas_flux"] if "gfas_flux" in ds else ds[list(ds.data_vars)[0]]
    for v in ("month_sin", "month_cos", "month_sin2", "month_cos2"):
        ds[v] = ds[v].broadcast_like(ref)

    # AOD gap filling: 0.1 deg monthly MAIAC has many empty cells (clouds).
    # Fill from the mean of valid 0.1 deg cells within a 9x9 (+-0.4 deg) window.
    if "aod" in ds:
        from scipy.ndimage import uniform_filter
        aod = ds["aod"].values.astype("float32")
        valid = np.isfinite(aod) & (aod > -0.05) & (aod < 6)
        filled = np.full_like(aod, np.nan)
        for t in range(aod.shape[0]):
            v = np.where(valid[t], aod[t], 0.0)
            s = uniform_filter(v, size=9, mode="constant")
            c = uniform_filter(valid[t].astype("float32"), size=9, mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(c > 0.02, s / np.maximum(c, 1e-6), np.nan)
            filled[t] = f
        ds["aod_raw"] = ds["aod"]
        ds["aod_fill"] = (("time", "lat", "lon"), filled,
                          {"units": "1", "long_name": "MAIAC AOD filled from a +-0.4 deg window"})
        ds["aod_used"] = ds["aod"].where(np.isfinite(ds["aod"]), ds["aod_fill"])
        print("AOD missing: raw %.1f%% -> after fill %.1f%%"
              % (100 * (~valid).mean(), 100 * float(np.isnan(ds["aod_used"].values).mean())))
    # log transforms used by the model
    for src, dst, scale in [("gfas_emission", "lgfas", 1.0), ("firms_frp", "lfrp", 1.0),
                            ("firms_n", "lfn", 1.0)]:
        if src in ds:
            ds[dst] = np.log10(1.0 + ds[src] / scale)

    # regional fire context: smoke travels hundreds of km, so a cell with no
    # local fire can still be under a smoke plume. Mean emission/FRP within
    # +-2.5 deg (51 cells) and +-7.5 deg (151 cells) boxes (separable filter).
    from scipy.ndimage import uniform_filter
    for src, tag, size in [("gfas_emission", "r5", 51), ("gfas_emission", "r15", 151),
                           ("firms_frp", "r5", 51)]:
        if src not in ds:
            continue
        arr = ds[src].values.astype("float32")
        sm = uniform_filter(arr, size=(1, size, size), mode="constant")
        ds[f"{src}_{tag}"] = (("time", "lat", "lon"), sm.astype("float32"),
                              {"units": ds[src].attrs.get("units", "1"),
                               "long_name": f"{src} smoothed over +-{size//20} deg"})
    for src, dst in [("gfas_emission_r5", "lgfas_r5"), ("gfas_emission_r15", "lgfas_r15"),
                     ("firms_frp_r5", "lfrp_r5")]:
        if src in ds:
            ds[dst] = np.log10(1.0 + ds[src] / 1e-6)
    # coordinates as smooth spatial covariates for the statistical field
    lat2 = np.repeat(G.LAT[:, None], G.NX, axis=1)
    lon2 = np.repeat(G.LON[None, :], G.NY, axis=0)
    ds["lat_f"] = (("lat", "lon"), lat2.astype("float32"))
    ds["lon_f"] = (("lat", "lon"), lon2.astype("float32"))
    return ds


def station_samples(ds: xr.Dataset, st: pd.DataFrame) -> pd.DataFrame:
    iy, ix = G.cell_index(st["lat"].to_numpy(), st["lon"].to_numpy())
    st = st.assign(iy=iy, ix=ix)
    st = st[(st.iy >= 0) & (st.ix >= 0)].copy()
    times = pd.to_datetime(ds["time"].values)
    tidx = {f"{t.year}-{t.month}": i for i, t in enumerate(times)}
    st["tidx"] = [tidx.get(f"{y}-{m}", -1) for y, m in zip(st["year"], st["month"])]
    st = st[st.tidx >= 0].copy()

    out = st[["country", "site_id", "lat", "lon", "year", "month", "pm25", "ndays",
              "iy", "ix", "tidx"]].copy()
    keep = {"gfas_flux", "gfas_emission", "firms_frp", "firms_n", "aod", "aod_raw",
            "aod_fill", "aod_used", "nobs", "blh", "wind10", "tp", "t2m", "rh", "vent",
            "lgfas", "lfrp", "lfn", "month_sin", "month_cos", "month_sin2", "month_cos2",
            "gfas_emission_r5", "gfas_emission_r15", "firms_frp_r5",
            "lgfas_r5", "lgfas_r15", "lfrp_r5", "lat_f", "lon_f", "lpop", "pop"}
    for v in ds.data_vars:
        if v not in keep:
            continue
        arr = ds[v].values
        if arr.ndim == 3:
            out[v] = arr[out["tidx"].to_numpy(), out["iy"].to_numpy(), out["ix"].to_numpy()]
        elif arr.ndim == 2:
            out[v] = arr[out["iy"].to_numpy(), out["ix"].to_numpy()]
    return out


def main() -> None:
    ds = add_derived(load_stack())
    gp = os.path.join(FEAT, "grid_features.nc")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(gp, encoding=enc)
    print("WROTE", gp, os.path.getsize(gp) / 1e6, "MB")

    env_st = os.environ.get("STATIONS_FILE")
    allp = os.path.join(FEAT, "stations_monthly_all.parquet")
    stp = env_st if env_st else (allp if os.path.exists(allp)
                                 else os.path.join(FEAT, "stations_monthly.parquet"))
    st = pd.read_parquet(stp)
    sf = station_samples(ds, st)
    sp = os.environ.get("FEATURES_OUT") or os.path.join(
        FEAT, "stations_features_all.parquet" if os.path.exists(allp)
        else "stations_features.parquet")
    sf.to_parquet(sp, index=False)
    print("WROTE", sp, sf.shape)
    print(sf.groupby("country").agg(
        n=("pm25", "size"), sites=("site_id", "nunique"),
        aod_missing=("aod", lambda s: float(s.isna().mean())),
        pm25=("pm25", "mean")).to_string())


if __name__ == "__main__":
    main()
