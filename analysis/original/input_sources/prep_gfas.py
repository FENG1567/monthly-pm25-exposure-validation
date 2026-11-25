"""Extract the Americas box from GFAS monthly-mean PM2.5 fire flux.

in  : incoming/gfas_pm25_monthly/<year>/gfas_pm25_monthly_<year>.nc  (pm2p5fire, kg m-2 s-1)
out : work/features/gfas_monthly_0p1.nc
      variables gfas_flux (kg m-2 s-1) and gfas_emission (kg m-2 month-1), dims (time, lat, lon)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid0p1 as G  # noqa: E402

ROOT = os.environ.get("FIRE_SMOKE_ROOT", os.getcwd())
INC = os.path.join(ROOT, "incoming", "gfas_pm25_monthly")
OUT = os.path.join(ROOT, "work", "features", os.environ.get("FEAT_SUBDIR", ""), "gfas_monthly_0p1.nc")
YEARS = range(int(os.environ.get("GFAS_Y0", 2003)), int(os.environ.get("GFAS_Y1", 2023)) + 1)
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype="float32")


def year_cube(y: int) -> np.ndarray:
    fp = os.path.join(INC, str(y), f"gfas_pm25_monthly_{y}.nc")
    with xr.open_dataset(fp) as ds:
        lon = ((ds["longitude"] + 180) % 360) - 180
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
        reg = ds["pm2p5fire"].reindex(latitude=G.LAT, longitude=G.LON,
                                      method="nearest", tolerance=0.011)
    vals = reg.values.astype("float32")
    if vals.shape != (12, G.NY, G.NX):
        raise SystemExit(f"{y}: unexpected shape {vals.shape}")
    return vals


def main() -> None:
    cubes, emis, times = [], [], []
    for y in YEARS:
        vals = year_cube(y)
        days = DAYS.copy()
        if y % 4 == 0:
            days[1] = 29
        cubes.append(vals)
        emis.append(vals * (days * 86400.0)[:, None, None])
        times.extend([np.datetime64(f"{y}-{m:02d}-01") for m in range(1, 13)])
        print(f"{y}: {vals.shape} max={np.nanmax(vals):.3e} sum={np.nansum(vals):.3e}",
              flush=True)

    flux = np.concatenate(cubes, axis=0)
    em = np.concatenate(emis, axis=0)
    ds = xr.Dataset(
        {"gfas_flux": (("time", "lat", "lon"), flux,
                       {"units": "kg m-2 s-1",
                        "long_name": "GFAS v1.2 wildfire PM2.5 flux, monthly mean"}),
         "gfas_emission": (("time", "lat", "lon"), em,
                           {"units": "kg m-2 month-1",
                            "long_name": "GFAS v1.2 wildfire PM2.5 emission, monthly total"})},
        coords={"time": np.array(times, dtype="datetime64[ns]"), "lat": G.LAT, "lon": G.LON},
    )
    ds.attrs.update(source="CAMS GFAS v1.2 pm2p5fire", grid="0.1 deg Americas box",
                    note="flux is an emission flux, not a surface concentration")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(OUT, encoding=enc)
    print("WROTE", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB", dict(ds.sizes))


if __name__ == "__main__":
    main()
