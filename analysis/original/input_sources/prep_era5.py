"""ERA5 monthly means -> 0.1 deg features (wind, boundary layer, RH, precip, T).

The CDS result for a monthly request comes as two NetCDF files:
  data_stream-moda_stepType-avgad.nc  -> tp (accumulated)
  data_stream-moda_stepType-avgua.nc  -> u10, v10, blh, t2m, d2m (averaged)

out: work/features/era5_monthly_0p1.nc
     blh (m), wind10 (m/s), tp (mm/month), t2m (degC), rh (%), vent (blh*wind10)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid0p1 as G  # noqa: E402

ROOT = os.environ.get("FIRE_SMOKE_ROOT", os.getcwd())
INC = os.path.join(ROOT, "incoming", "era5")
OUT = os.path.join(ROOT, "work", "features", os.environ.get("FEAT_SUBDIR", ""),
                   "era5_monthly_0p1.nc")


def rh_from_dewpoint(t2m_k: xr.DataArray, d2m_k: xr.DataArray) -> xr.DataArray:
    t = t2m_k - 273.15
    td = d2m_k - 273.15
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    return (100.0 * e / es).clip(1, 100)


def main() -> None:
    env_file = os.environ.get("ERA5_FILE", "").strip()
    env_dir = os.environ.get("ERA5_DIR", "").strip()
    if env_file:
        files = [env_file]
    elif env_dir:
        files = sorted(glob.glob(os.path.join(env_dir, "*.nc")))
    else:
        files = sorted(glob.glob(os.path.join(INC, "*.nc")))
    if not files:
        raise SystemExit(f"no ERA5 files in {INC}")
    parts = []
    avgad = None
    for f in files:
        if os.path.getsize(f) < 1e5:
            print("skip small file", f)
            continue
        if "era5_monthly_americas" in os.path.basename(f):
            continue
        print("reading", os.path.basename(f))
        d = xr.open_dataset(f)
        if "stepType-avgad" in os.path.basename(f):
            avgad = d          # accumulated field, stamped at 06:00
        else:
            parts.append(d)
    ds = parts[0]
    if avgad is not None:
        # align the accumulation file onto the same monthly stamps
        avgad = avgad.assign_coords(valid_time=ds.valid_time.values)
        ds = xr.merge([ds, avgad], compat="override")
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    ds = ds.sortby("latitude", ascending=False).sortby("longitude")

    out = xr.Dataset()
    out["blh"] = ds["blh"]
    out["wind10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    out["tp"] = ds["tp"] * 1000.0
    out["t2m"] = ds["t2m"] - 273.15
    out["rh"] = rh_from_dewpoint(ds["t2m"], ds["d2m"])
    out["vent"] = out["blh"] * out["wind10"]

    out = out.reindex(latitude=G.LAT, longitude=G.LON, method="nearest", tolerance=0.2)
    out = out.rename({"latitude": "lat", "longitude": "lon"})
    out = out.sel(time=slice("2018-01-01", "2022-12-31"))
    for v in out.data_vars:
        out[v] = out[v].astype("float32")
        out[v].attrs["source"] = "ERA5 monthly means (CDS)"
    out.attrs["note"] = ("wind10 = sqrt(u10^2+v10^2); rh from 2 m dewpoint (Magnus); "
                         "tp in mm/month; vent = blh*wind10")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars}
    out.to_netcdf(OUT, encoding=enc)
    print("WROTE", OUT, dict(out.sizes))
    for v in out.data_vars:
        print(f"  {v}: mean={float(out[v].mean()):.2f} max={float(out[v].max()):.2f}")


if __name__ == "__main__":
    main()
