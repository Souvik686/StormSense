"""Load SRTM DEM tiles and regrid to the ERA5 0.25 deg grid.

SRTM tiles are 1 arc-second (~30 m), EPSG:4326, named n{lat}_e{lon}_1arc_v3.tif
and tile the domain 20-29N, 84-92E with no gaps (verified in inventory).
Regridding averages the ~900x900 SRTM pixels falling in each 0.25 deg ERA5
cell (block-mean), which is the correct operation for a static orography
feature at this target resolution -- nearest-neighbour would alias ridge
lines and bilinear would blur them less accurately than an areal mean.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import rasterio
from rasterio.merge import merge

from src.utils.config import Config


def _mosaic_srtm(srtm_dir: str):
    files = sorted(glob.glob(os.path.join(srtm_dir, "*.tif")))
    if not files:
        raise FileNotFoundError(f"No SRTM tiles under {srtm_dir}")
    srcs = [rasterio.open(f) for f in files]
    mosaic, transform = merge(srcs, nodata=srcs[0].nodata)
    for s in srcs:
        s.close()
    return mosaic[0], transform, srcs[0].nodata


def load_dem_on_era5_grid(cfg: Config) -> np.ndarray:
    """Return DEM elevation (meters) block-averaged onto the ERA5 grid.

    Shape matches cfg.domain.grid_shape = (n_lat, n_lon), lat descending
    (north to south) to match the ERA5 array orientation used elsewhere.
    """
    srtm_dir = cfg.path("dem", "srtm_dir")
    mosaic, transform, nodata = _mosaic_srtm(srtm_dir)
    mosaic = mosaic.astype(np.float32)
    mosaic[mosaic == nodata] = np.nan

    res = cfg.get("domain", "resolution_deg")
    lat_max, lat_min = cfg.get("domain", "lat_max"), cfg.get("domain", "lat_min")
    lon_min, lon_max = cfg.get("domain", "lon_min"), cfg.get("domain", "lon_max")
    n_lat, n_lon = cfg.get("domain", "grid_shape")

    lats = np.linspace(lat_max, lat_min, n_lat)   # descending, cell centers
    lons = np.linspace(lon_min, lon_max, n_lon)    # ascending, cell centers
    half = res / 2.0

    out = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    inv = ~transform
    for i, la in enumerate(lats):
        row_top, col_left = inv * (lon_min, la + half)
        row_bot, col_right = inv * (lon_min, la - half)
        for j, lo in enumerate(lons):
            c0, _ = inv * (lo - half, la)
            c1, _ = inv * (lo + half, la)
            r0 = int(round((~transform * (lo, la + half))[1]))
            r1 = int(round((~transform * (lo, la - half))[1]))
            c0 = int(round((~transform * (lo - half, la))[0]))
            c1 = int(round((~transform * (lo + half, la))[0]))
            r0, r1 = sorted((max(r0, 0), min(r1, mosaic.shape[0])))
            c0, c1 = sorted((max(c0, 0), min(c1, mosaic.shape[1])))
            block = mosaic[r0:r1, c0:c1]
            if block.size:
                out[i, j] = np.nanmean(block)
    return out, lats, lons


def fill_ocean_cells(elev: np.ndarray, sea_level: float = 0.0) -> np.ndarray:
    """Fill grid cells with no SRTM coverage (ocean / outside mosaic) with
    sea level. SRTM has no ocean bathymetry, so NaN cells here are always
    open water (Bay of Bengal, south of the domain) rather than missing land data."""
    out = elev.copy()
    out[~np.isfinite(out)] = sea_level
    return out


def validate_dem(elev: np.ndarray) -> dict:
    finite = elev[np.isfinite(elev)]
    assert finite.size > 0, "DEM regrid produced no valid cells"
    assert finite.min() >= -50, f"implausible DEM min {finite.min()}"
    assert finite.max() < 9000, f"implausible DEM max {finite.max()}"
    return {"n_valid": int(finite.size), "n_total": int(elev.size),
            "min_m": float(finite.min()), "max_m": float(finite.max()),
            "mean_m": float(finite.mean())}
