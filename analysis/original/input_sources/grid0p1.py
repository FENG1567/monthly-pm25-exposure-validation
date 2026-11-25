"""Shared 0.1-degree analysis grid (cell centres).

Bounds are configurable so the same code serves the Americas box and the
Australia box:
    GRID_LAT_N=72 (or -9), GRID_LAT_S=-34 (or -45),
    GRID_LON_W=-125 (or 138), GRID_LON_E=-34 (or 155)
"""
from __future__ import annotations

import os

import numpy as np

LAT_N = float(os.environ.get("GRID_LAT_N", "50.0"))
LAT_S = float(os.environ.get("GRID_LAT_S", "-34.0"))
LON_W = float(os.environ.get("GRID_LON_W", "-125.0"))
LON_E = float(os.environ.get("GRID_LON_E", "-34.0"))
RES = 0.1

# cell centres, descending latitude / ascending longitude
LAT = np.round(np.arange(LAT_N - RES / 2, LAT_S, -RES), 4)
LON = np.round(np.arange(LON_W + RES / 2, LON_E, RES), 4)
NY, NX = LAT.size, LON.size


def cell_index(lat, lon):
    """Return (iy, ix) integer arrays for coordinates; -1 where outside the box."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    iy = np.floor((LAT_N - lat) / RES).astype("int32")
    ix = np.floor((lon - LON_W) / RES).astype("int32")
    bad = (~np.isfinite(lat)) | (~np.isfinite(lon)) | (iy < 0) | (iy >= NY) | (ix < 0) | (ix >= NX)
    iy = np.where(bad, -1, iy)
    ix = np.where(bad, -1, ix)
    return iy, ix


def cell_center(iy, ix):
    return LAT[iy], LON[ix]


def flat(iy, ix):
    return iy.astype("int64") * NX + ix.astype("int64")
