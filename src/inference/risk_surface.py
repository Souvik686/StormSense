"""Continuous West Bengal Risk Surface Generator.

Generates smooth, continuous meteorological risk maps interpolated from
SevereWeatherNet V2 predictions, strictly clipped to the official West Bengal
administrative boundary.
"""
from __future__ import annotations

import io
import json
import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.path as mpath

# Official West Bengal coordinate bounding box (degrees North, degrees East)
WB_BOUNDS_LEAFLET = [[21.5394, 86.6103], [26.9960, 89.8828]]
WB_MIN_LAT = 21.5394
WB_MAX_LAT = 26.9960
WB_MIN_LON = 86.6103
WB_MAX_LON = 89.8828

DEFAULT_H = 300
DEFAULT_W = 200


def get_or_create_wb_mask(
    geojson_path: str = "Data/BOUNDARIES/west_bengal.geojson",
    cache_path: str = "Data/BOUNDARIES/wb_mask_300x200.npy",
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
) -> np.ndarray:
    """Retrieve or precompute the 2D boolean mask for the West Bengal boundary."""
    if os.path.exists(cache_path):
        try:
            mask = np.load(cache_path)
            if mask.shape == (h, w):
                return mask
        except Exception:
            pass

    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"West Bengal GeoJSON not found at: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        wb_data = json.load(f)

    coords = wb_data["features"][0]["geometry"]["coordinates"]
    paths = [mpath.Path(poly[0]) for poly in coords]

    lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, h)  # North to South
    lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, w)  # West to East
    lon_grid, lat_grid = np.meshgrid(lons_fine, lats_fine)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    mask_flat = np.zeros(len(pts), dtype=bool)
    for p in paths:
        bb = p.get_extents()
        in_bb = (
            (pts[:, 0] >= bb.x0)
            & (pts[:, 0] <= bb.x1)
            & (pts[:, 1] >= bb.y0)
            & (pts[:, 1] <= bb.y1)
        )
        if not np.any(in_bb):
            continue
        mask_flat[in_bb] |= p.contains_points(pts[in_bb])

    mask = mask_flat.reshape(h, w)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, mask)
    except Exception:
        pass
    return mask


def colormap_risk_surface(smoothed_grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply model 4-stage risk color ramp to continuous probability values (0.0 to 1.0).

    Green (Normal/Low Risk: <0.25), Yellow (Watch/Moderate: 0.25-0.50),
    Orange (Alert/High: 0.50-0.75), Red (Warning/Severe: >=0.75).
    Every valid pixel inside the West Bengal polygon receives a visible color.
    Alpha is strictly transparent (0) outside the West Bengal boundary.
    """
    h, w = smoothed_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Level 1: < 0.25 (Soft Emerald/Green - Normal / Routine Watch)
    # Solid visible emerald so low-risk areas (e.g. Northern WB) are NEVER transparent!
    m1 = smoothed_grid < 0.25
    t1 = np.clip(smoothed_grid[m1] / 0.25, 0.0, 1.0)
    rgba[m1, 0] = (16 + 20 * t1).astype(np.uint8)
    rgba[m1, 1] = (185 - 10 * t1).astype(np.uint8)
    rgba[m1, 2] = (129 - 40 * t1).astype(np.uint8)
    rgba[m1, 3] = (150 + 25 * t1).astype(np.uint8)

    # Level 2: 0.25 to 0.50 (Yellow/Amber - Watch)
    m2 = (smoothed_grid >= 0.25) & (smoothed_grid < 0.50)
    t2 = (smoothed_grid[m2] - 0.25) / 0.25
    rgba[m2, 0] = (234 + 11 * t2).astype(np.uint8)
    rgba[m2, 1] = (179 - 15 * t2).astype(np.uint8)
    rgba[m2, 2] = (8 + 12 * t2).astype(np.uint8)
    rgba[m2, 3] = (175 + 25 * t2).astype(np.uint8)

    # Level 3: 0.50 to 0.75 (Orange - Alert)
    m3 = (smoothed_grid >= 0.50) & (smoothed_grid < 0.75)
    t3 = (smoothed_grid[m3] - 0.50) / 0.25
    rgba[m3, 0] = (249 - 10 * t3).astype(np.uint8)
    rgba[m3, 1] = (115 - 47 * t3).astype(np.uint8)
    rgba[m3, 2] = (22 + 46 * t3).astype(np.uint8)
    rgba[m3, 3] = (200 + 25 * t3).astype(np.uint8)

    # Level 4: >= 0.75 (Severe Red - Warning)
    m4 = smoothed_grid >= 0.75
    t4 = np.clip((smoothed_grid[m4] - 0.75) / 0.25, 0.0, 1.0)
    rgba[m4, 0] = 239
    rgba[m4, 1] = (68 * (1.0 - 0.4 * t4)).astype(np.uint8)
    rgba[m4, 2] = (68 * (1.0 - 0.4 * t4)).astype(np.uint8)
    rgba[m4, 3] = (225 + 25 * t4).astype(np.uint8)

    # Strictly clip outside West Bengal
    rgba[~mask] = [0, 0, 0, 0]
    return rgba


def generate_risk_surface_png(
    pred_grid: np.ndarray,
    lats_coarse: np.ndarray,
    lons_coarse: np.ndarray,
    mask: Optional[np.ndarray] = None,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    sigma: float = 1.2,
) -> bytes:
    """Interpolate coarse grid to fine resolution, clip to WB boundary, return PNG bytes."""
    if mask is None:
        mask = get_or_create_wb_mask(h=h, w=w)

    # Ensure coordinates are strictly ascending for RegularGridInterpolator
    if lats_coarse[0] > lats_coarse[-1]:
        lat_asc = lats_coarse[::-1]
        grid_asc = pred_grid[::-1, :]
    else:
        lat_asc = lats_coarse
        grid_asc = pred_grid

    if lons_coarse[0] > lons_coarse[-1]:
        lon_asc = lons_coarse[::-1]
        grid_asc = grid_asc[:, ::-1]
    else:
        lon_asc = lons_coarse

    interp = RegularGridInterpolator(
        (lat_asc, lon_asc),
        grid_asc,
        method="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, h)
    lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, w)
    lat_mesh, lon_mesh = np.meshgrid(lats_fine, lons_fine, indexing="ij")

    fine_vals = interp((lat_mesh, lon_mesh))
    smoothed = np.clip(gaussian_filter(fine_vals, sigma=sigma), 0.0, 1.0)

    rgba = colormap_risk_surface(smoothed, mask)
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

