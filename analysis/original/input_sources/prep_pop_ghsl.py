"""GHSL 2020 population (30 arcsec) -> 0.1 deg grid, as a fallback to WorldPop.

Downloads the 10x10 degree GHSL tiles covering the current grid, accumulates
population per 0.1 deg cell and writes pop_0p1.nc (same layout as prep_pop.py).

Tile naming: GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_R{r}_C{c}.zip
  c = floor((lon + 180) / 10) + 1,  r = floor((90 - lat) / 10) + 1
"""
from __future__ import annotations

import os
import sys
import zipfile

import numpy as np
import rasterio
import urllib.request
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid0p1 as G  # noqa: E402

ROOT = os.environ.get("FIRE_SMOKE_ROOT", os.getcwd())
CACHE = os.path.join(ROOT, "incoming", "pop", "ghsl")
OUT = os.path.join(ROOT, "work", "features", os.environ.get("FEAT_SUBDIR", ""), "pop_0p1.nc")
BASE = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
        "GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/tiles/"
        "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_R{r}_C{c}.zip")
NC = G.NY * G.NX


def ensure_tile(r: int, c: int) -> str | None:
    os.makedirs(CACHE, exist_ok=True)
    name = f"GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_R{r}_C{c}"
    tif = os.path.join(CACHE, name + ".tif")
    if os.path.exists(tif):
        return tif
    z = os.path.join(CACHE, name + ".zip")
    if not os.path.exists(z):
        try:
            urllib.request.urlretrieve(BASE.format(r=r, c=c), z)
        except Exception as e:  # noqa: BLE001
            print("tile missing", r, c, e)
            return None
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(CACHE)
    except Exception as e:  # noqa: BLE001
        print("unzip fail", z, e)
        return None
    return tif if os.path.exists(tif) else None


def main() -> None:
    acc = np.zeros(NC, dtype="float64")
    r0 = int(np.floor((90 - G.LAT_N) / 10)) + 1
    r1 = int(np.floor((90 - G.LAT_S) / 10)) + 1
    c0 = int(np.floor((G.LON_W + 180) / 10)) + 1
    c1 = int(np.floor((G.LON_E + 180) / 10)) + 1
    print(f"grid rows R{r0}-R{r1}, cols C{c0}-C{c1}")
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            tif = ensure_tile(r, c)
            if tif is None:
                continue
            with rasterio.open(tif) as src:
                tr = src.transform
                a, cc, e, f = tr.a, tr.c, tr.e, tr.f
                for rr in range(0, src.height, 2048):
                    h = min(2048, src.height - rr)
                    arr = src.read(1, window=((rr, rr + h), (0, src.width)))
                    rows = np.arange(rr, rr + h)[:, None]
                    cols = np.arange(0, src.width)[None, :]
                    xs = cc + (cols + 0.5) * a
                    ys = f + (rows + 0.5) * e
                    iy, ix = G.cell_index(ys, xs)
                    ok = iy >= 0
                    if not ok.any():
                        continue
                    fk = G.flat(iy[ok], ix[ok])
                    vals = np.nan_to_num(arr[ok].astype("float64"))
                    acc += np.bincount(fk, weights=vals, minlength=NC)
            print("  done", r, c, f"pop={acc.sum()/1e6:.1f}M", flush=True)
    pop = acc.reshape(G.NY, G.NX)
    ds = xr.Dataset({
        "pop": (("lat", "lon"), pop.astype("float32"),
                {"units": "people per 0.1 deg cell", "long_name": "GHSL 2020 population"}),
        "lpop": (("lat", "lon"), np.log10(1.0 + pop).astype("float32"),
                 {"units": "1", "long_name": "log10(1+population)"})},
        coords={"lat": G.LAT, "lon": G.LON})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds.to_netcdf(OUT)
    print("WROTE", OUT, "total", round(float(pop.sum()) / 1e6, 1), "M")


if __name__ == "__main__":
    main()
