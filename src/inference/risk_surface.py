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

# Official West Bengal bounding box (degrees North, degrees East), derived from
# the COMPLETE 23-district state outline (see scripts/build_wb_boundaries.py).
# The previously used box stopped at 26.996N / 86.6103E, which cut off Darjeeling
# and Kalimpong in the north and Purulia in the west.
WB_BOUNDS_LEAFLET = [[21.5394, 85.8325], [27.2206, 89.8828]]
WB_MIN_LAT = 21.5394
WB_MAX_LAT = 27.2206
WB_MIN_LON = 85.8325
WB_MAX_LON = 89.8828

# Authoritative full-state geometry. west_bengal.geojson (the original file) only
# contained 13 of 23 districts and is retained untouched for reference.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WB_STATE_GEOJSON = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "west_bengal_full.geojson")
WB_DISTRICTS_GEOJSON = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "west_bengal_districts_full.geojson")

DEFAULT_H = 600
DEFAULT_W = 400

# ---------------------------------------------------------------------------
# THREE DISTINCT GRIDS. Keeping them separate is what stops the visualization
# from implying predictive skill the model does not have:
#
#   1. MODEL GRID          33 x 25 at 0.25 deg (~28 km). What the network
#                          actually predicts. Never changed here.
#   2. VISUALIZATION GRID  DEFAULT_H x DEFAULT_W = 66 x 50 (~10 km sampling).
#                          The interpolated field. This is a RESAMPLING of the
#                          model grid for display -- it adds no information and
#                          is not a finer forecast.
#   3. RENDER GRID         RENDER_H x RENDER_W below. Purely the raster the
#                          browser draws. Rendering the 66x50 field directly
#                          produced a ~15x browser upscale of a 50-pixel-wide
#                          image, which is why the map looked like rectangular
#                          blocks. Drawing the SAME interpolated field at a
#                          higher pixel count gives smooth, contour-like
#                          boundaries without inventing any new values.
#
# The render grid is an image-resolution choice, exactly like exporting a chart
# at a higher DPI. It does not change the science, the values, or the ~10 km
# visualization sampling that the UI reports.
# ---------------------------------------------------------------------------
# Raised from 660x500: at high map zoom the band boundaries are magnified, and
# a coarser raster made those boundaries look like stair-stepped rectangles.
# This is image resolution only -- the field and its values are unchanged.
RENDER_H = 1320
RENDER_W = 1000


def get_or_create_wb_mask(
    geojson_path: str = WB_STATE_GEOJSON,
    cache_path: str = None,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
) -> np.ndarray:
    """Retrieve or precompute the 2D boolean mask for the West Bengal boundary."""
    if cache_path is None:
        cache_path = os.path.join(PROJECT_ROOT, "Data", "BOUNDARIES", "wb_mask_full_66x50.npy")
    
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

    geom = wb_data["features"][0]["geometry"]
    # Accept either Polygon or MultiPolygon; a dissolved state outline can be
    # either depending on whether offshore islands stay separate.
    polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]

    lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, h)  # North to South
    lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, w)  # West to East
    lon_grid, lat_grid = np.meshgrid(lons_fine, lats_fine)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    mask_flat = np.zeros(len(pts), dtype=bool)
    for rings in polygons:
        if not rings:
            continue
        outer = mpath.Path(rings[0])
        bb = outer.get_extents()
        in_bb = (
            (pts[:, 0] >= bb.x0)
            & (pts[:, 0] <= bb.x1)
            & (pts[:, 1] >= bb.y0)
            & (pts[:, 1] <= bb.y1)
        )
        if not np.any(in_bb):
            continue
        inside = outer.contains_points(pts[in_bb])
        # Subtract interior rings (holes) so enclaves are not painted as land.
        for hole in rings[1:]:
            inside &= ~mpath.Path(hole).contains_points(pts[in_bb])
        mask_flat[in_bb] |= inside

    mask = mask_flat.reshape(h, w)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, mask)
    except Exception:
        pass
    return mask


# ── Spatial risk field colour bands ──────────────────────────────────────────
# WHICH MODEL VARIABLE THIS COLOURS (traced, per requirement):
#   `severe_weather_prob` -- the temperature-scaled calibrated sigmoid of the
#   model's `severe_weather_logit` head. Its training label is defined in
#   src/features/targets.py as  (rolling 3h tp > heavy_rain_3h_mm) OR
#   (CAPE > cape_severe_jkg AND CIN < cin_weak_jkg), i.e. a severe-weather
#   PROXY probability in [0, 1].
#
#   It is deliberately NOT any of these, which are different quantities:
#     * rain_3h_mm_pred  -- a regression in millimetres, not a probability;
#                           colouring it on a 0-1 ramp would be meaningless.
#     * radar reflectivity -- this model has no reflectivity head, and none is
#                           ingested; there is nothing to colour.
#     * lightning        -- not modelled and not available in ERA5/GFS input.
#   Because the bands below are probability cut-points, the variable they are
#   applied to MUST be a probability. severe_weather_prob is the only such
#   spatial field the model produces.
#
# Band edges are an internal implementation detail and are intentionally not
# surfaced as numbers anywhere in the public UI -- the interface communicates
# only the resulting category/colour.
# Band edges come from THE single source of truth. Four private `_BAND_*`
# constants used to sit here (0.20/0.25/0.75); nothing ever read them, and their
# 0.20 "clear" edge contradicted the 0.25 the painter below actually applies.
# A future reader adjusting them would have changed nothing and believed they
# had. Removed in favour of the shared definition.
from src.inference.risk_thresholds import WATCH_MIN, ALERT_MIN, WARNING_MIN


def colormap_risk_surface(smoothed_grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the 4-stage risk colour ramp as FLAT, SINGLE-COLOUR bands.

    Each risk category is painted in ONE solid colour with no intra-band fade:

        < 0.25  Normal   emerald   #10b981
        < 0.50  Watch    amber     #f59e0b
        < 0.75  Alert    orange    #f97316
       >= 0.75  Warning  red       #ef4444

    These are exactly the four colours in the map legend, so a pixel's colour
    now maps one-to-one onto the legend entry rather than sitting somewhere on a
    continuous gradient between two of them.

    Why flat rather than graded: the previous ramp interpolated within each band
    (alpha and RGB both varied with the value), which made neighbouring
    interpolation cells differ slightly and read as visible rectangular tiles at
    high zoom. Constant colour inside a band removes that cell-to-cell variation
    entirely, so the only visible edges are the BAND boundaries -- and because
    the underlying field is smooth and rendered at the high-resolution render
    grid, those boundaries are smooth curves, like contour fills on an
    operational meteorological chart.

    The risk VALUES are untouched; this changes only how a value is coloured.
    Band edges match NowcastService._level_for_prob (0.25 / 0.50 / 0.75).
    """
    h, w = smoothed_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Single opacity for every band: a value-dependent alpha would reintroduce
    # exactly the per-cell variation this change removes.
    ALPHA = 200

    bands = (
        (smoothed_grid < WATCH_MIN,                                     (16, 185, 129)),   # emerald
        ((smoothed_grid >= WATCH_MIN) & (smoothed_grid < ALERT_MIN),    (245, 158, 11)),   # amber
        ((smoothed_grid >= ALERT_MIN) & (smoothed_grid < WARNING_MIN),  (249, 115, 22)),   # orange
        (smoothed_grid >= WARNING_MIN,                                  (239, 68, 68)),    # red
    )
    for sel, (r, g, b) in bands:
        rgba[sel, 0] = r
        rgba[sel, 1] = g
        rgba[sel, 2] = b
        rgba[sel, 3] = ALPHA

    # Strictly clip outside West Bengal.
    rgba[~mask] = [0, 0, 0, 0]
    return rgba


def generate_risk_surface_png(
    pred_grid: np.ndarray,
    lats_coarse: np.ndarray,
    lons_coarse: np.ndarray,
    mask: Optional[np.ndarray] = None,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    sigma: float = 3.0,
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

    # STEP 1 -- VISUALIZATION GRID (~10 km sampling, h x w).
    # This is the field the UI describes as the ~10 km visualization grid, and
    # it is what the smoothing acts on. Unchanged from before.
    lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, h)
    lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, w)
    lat_mesh, lon_mesh = np.meshgrid(lats_fine, lons_fine, indexing="ij")

    fine_vals = interp((lat_mesh, lon_mesh))
    smoothed = np.clip(gaussian_filter(fine_vals, sigma=sigma), 0.0, 1.0)

    # STEP 2 -- RENDER GRID (image pixels only).
    # Resample the ALREADY-SMOOTHED ~10 km field up to the raster the browser
    # displays. This is a pure image-resolution step: every value still comes
    # from `smoothed`, so no new information is introduced. Doing it here rather
    # than letting the browser upscale a 50px-wide PNG is what turns the blocky
    # rectangles into smooth, contour-like boundaries.
    render_h, render_w = RENDER_H, RENDER_W
    band_interp = RegularGridInterpolator(
        (lats_fine[::-1], lons_fine),
        smoothed[::-1, :],
        # CUBIC, not linear. Bilinear resampling is piecewise-planar, so each
        # visualization cell became a flat facet and the band boundaries ran
        # along straight cell edges -- the "box" look at high zoom. Cubic gives
        # a continuously curved surface, so an iso-level through it is a smooth
        # curve. It resamples the existing field; it adds no new information.
        method="cubic",
        bounds_error=False,
        fill_value=0.0,
    )
    lat_r = np.linspace(WB_MAX_LAT, WB_MIN_LAT, render_h)
    lon_r = np.linspace(WB_MIN_LON, WB_MAX_LON, render_w)
    lat_rm, lon_rm = np.meshgrid(lat_r, lon_r, indexing="ij")
    render_vals = np.clip(band_interp((lat_rm, lon_rm)), 0.0, 1.0)

    # A light final smoothing removes the faceting that bilinear resampling of a
    # coarse source leaves along band edges. Sigma is deliberately small (about
    # one visualization cell) so the risk PATTERN is untouched -- it softens
    # edges, it does not move or flatten maxima.
    # Smoothing is scaled to the RENDER grid (about one visualization cell wide)
    # so band boundaries are curved at the resolution actually displayed. It
    # softens edges only: maxima are not moved or flattened, and the value at
    # any point still comes from the model field.
    render_vals = np.clip(
        gaussian_filter(render_vals, sigma=max(4.0, render_h / float(h) * 1.5)),
        0.0, 1.0,
    )

    render_mask = _resample_mask(mask, render_h, render_w)
    rgba = colormap_risk_surface(render_vals, render_mask)
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _resample_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resample of the West Bengal boolean mask.

    The clip must stay a hard boundary: interpolating the mask itself would
    produce semi-transparent fringes outside the state outline, i.e. risk colour
    painted over territory the product does not cover.
    """
    if mask.shape == (h, w):
        return mask
    src_h, src_w = mask.shape
    row_idx = np.clip((np.arange(h) * src_h // h), 0, src_h - 1)
    col_idx = np.clip((np.arange(w) * src_w // w), 0, src_w - 1)
    return mask[np.ix_(row_idx, col_idx)]



# Physical ceiling used to normalize the historical ERA5 rainfall analysis for
# display. This is a RENDERING scale only -- it never clips or alters the stored
# data (see nowcast_service.era5_surface_t0, which keeps true mm/hour). Chosen to
# match the observation surface's rain_1h_mm ramp so the historical t=0 field and
# the live NOW field are read on comparable intensity scales.
HISTORICAL_RAIN_RENDER_MAX_MM_H = 30.0


def generate_analysis_surface_png(
    field: np.ndarray,
    lats_coarse: np.ndarray,
    lons_coarse: np.ndarray,
    vmax: float = HISTORICAL_RAIN_RENDER_MAX_MM_H,
    mask: Optional[np.ndarray] = None,
    h: int = DEFAULT_H,
    w: int = DEFAULT_W,
    sigma: float = 1.2,
) -> bytes:
    """Render a gridded ANALYSIS (observed/reanalysis) field as a PNG.

    This exists so the historical case study's t=0 state can be shown on the map
    as what it actually is -- an observed reanalysis field at the analysis time --
    instead of either (a) leaving the map blank or (b) painting a model FORECAST
    under a "NOW" label, which would misrepresent future model output as a
    present-tense observation.

    The palette is the BLUE/TEAL observation ramp, deliberately distinct from the
    green->amber->orange->red forecast risk ramp, so an analysis field can never
    be misread as a severe-weather risk forecast. Interpolation, smoothing and
    the West Bengal clip are identical to the forecast surface, so the two are
    geographically comparable.
    """
    if mask is None:
        mask = get_or_create_wb_mask(h=h, w=w)

    # Ensure strictly ascending coordinates for RegularGridInterpolator.
    if lats_coarse[0] > lats_coarse[-1]:
        lat_asc = lats_coarse[::-1]
        grid_asc = field[::-1, :]
    else:
        lat_asc = lats_coarse
        grid_asc = field

    if lons_coarse[0] > lons_coarse[-1]:
        lon_asc = lons_coarse[::-1]
        grid_asc = grid_asc[:, ::-1]
    else:
        lon_asc = lons_coarse

    interp = RegularGridInterpolator(
        (lat_asc, lon_asc),
        np.nan_to_num(grid_asc, nan=0.0),
        method="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, h)
    lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, w)
    lat_mesh, lon_mesh = np.meshgrid(lats_fine, lons_fine, indexing="ij")

    fine_vals = gaussian_filter(interp((lat_mesh, lon_mesh)), sigma=sigma)
    # Cubic interpolation can undershoot below zero around sharp gradients;
    # rainfall cannot be negative.
    fine_vals = np.clip(fine_vals, 0.0, None)

    # RENDER GRID (image pixels only) -- same rationale as the forecast surface:
    # resample the already-smoothed visualization field up to the raster the
    # browser draws, so the observed field gets smooth contour-like boundaries
    # instead of a browser upscale of a 50px-wide PNG. No new values are created.
    render_h, render_w = RENDER_H, RENDER_W
    up = RegularGridInterpolator(
        (lats_fine[::-1], lons_fine),
        fine_vals[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    lat_r = np.linspace(WB_MAX_LAT, WB_MIN_LAT, render_h)
    lon_r = np.linspace(WB_MIN_LON, WB_MAX_LON, render_w)
    lat_rm, lon_rm = np.meshgrid(lat_r, lon_r, indexing="ij")
    fine_vals = np.clip(up((lat_rm, lon_rm)), 0.0, None)
    fine_vals = np.clip(
        gaussian_filter(fine_vals, sigma=max(1.0, render_h / float(h) * 0.5)),
        0.0, None,
    )
    mask = _resample_mask(mask, render_h, render_w)
    h, w = render_h, render_w

    norm = np.clip(fine_vals / max(float(vmax), 1e-9), 0.0, 1.0)

    # Blue -> teal -> cyan -> white, deliberately NOT the green->red forecast
    # risk ramp, so an observed field can never be misread as predicted risk.
    # The extra stops give light and moderate rain visible separation instead of
    # collapsing them into one flat blue.
    stops = np.array([
        [12, 74, 110],    # deep blue (light rain)
        [14, 165, 190],   # teal
        [34, 211, 238],   # cyan
        [125, 240, 250],  # bright cyan
        [224, 252, 255],  # near-white (heaviest)
    ], dtype=np.float64)

    pos = norm * (len(stops) - 1)
    idx = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - idx)[..., None]
    colours = stops[idx] * (1 - frac) + stops[idx + 1] * frac

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = mask.astype(bool)
    rgba[..., 0] = np.where(valid, colours[..., 0], 0)
    rgba[..., 1] = np.where(valid, colours[..., 1], 0)
    rgba[..., 2] = np.where(valid, colours[..., 2], 0)
    # Dry cells fade out so the field reads as "no rain here", not a painted
    # sheet. Rainfall is strongly right-skewed -- most wet cells sit far below
    # the domain peak -- so a linear ramp left almost the whole field at single
    # digit alpha and the storm was effectively invisible. A square-root curve
    # lifts light and moderate rain into view while keeping genuinely dry cells
    # transparent; it changes only opacity, never the mapped value or its colour.
    # Only ~20% of the domain is wet at all and most of that is light rain, so a
    # curve alone still rendered the storm nearly invisible. Give every cell that
    # carries real rain a solid visible floor and ramp up from there; dry cells
    # stay fully transparent so the shape of the rain area remains honest.
    alpha = np.where(
        norm < 0.02,
        0.0,
        140.0 + np.power(np.clip(norm, 0.0, 1.0), 0.45) * 115.0,
    )
    rgba[..., 3] = np.where(valid, alpha.astype(np.uint8), 0)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
