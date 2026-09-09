"""Geographic validation of the rendered forecast surface.

Requirement: every non-transparent pixel of a forecast raster must fall inside
West Bengal, and the whole state must be covered. These are checked
programmatically against the authoritative boundary geometry rather than by
looking at the map.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point, shape
from shapely.prepared import prep

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.risk_surface import (  # noqa: E402
    WB_MAX_LAT,
    WB_MAX_LON,
    WB_MIN_LAT,
    WB_MIN_LON,
    WB_STATE_GEOJSON,
    generate_risk_surface_png,
    get_or_create_wb_mask,
)

# Present-day West Bengal districts with a representative interior point each.
# All 23 must be inside the forecast domain -- the previously shipped boundary
# file omitted ten of them, which silently clipped the map.
WB_DISTRICT_POINTS = {
    "Darjeeling": (27.04, 88.26),
    "Kalimpong": (27.07, 88.47),
    "Jalpaiguri": (26.52, 88.72),
    "Alipurduar": (26.49, 89.53),
    "Cooch Behar": (26.32, 89.45),
    "Uttar Dinajpur": (25.62, 88.13),
    "Dakshin Dinajpur": (25.22, 88.77),
    "Malda": (25.01, 88.14),
    "Murshidabad": (24.18, 88.27),
    "Birbhum": (23.84, 87.62),
    "Nadia": (23.40, 88.50),
    "North 24 Parganas": (22.72, 88.48),
    "South 24 Parganas": (22.10, 88.40),
    "Kolkata": (22.57, 88.36),
    "Howrah": (22.59, 88.26),
    "Hooghly": (22.90, 88.39),
    "Purba Bardhaman": (23.24, 87.86),
    "Paschim Bardhaman": (23.68, 87.10),
    "Bankura": (23.23, 87.07),
    "Purulia": (23.33, 86.36),
    "Jhargram": (22.45, 86.99),
    "Paschim Medinipur": (22.42, 87.32),
    "Purba Medinipur": (22.09, 87.79),
}

# Points that must never receive forecast colouring.
OUTSIDE_POINTS = {
    "Nepal (Kathmandu)": (27.70, 85.32),
    "Bangladesh (Dhaka)": (23.80, 90.40),
    "Bihar (Patna)": (25.59, 85.13),
    "Jharkhand (Ranchi)": (23.36, 85.33),
    "Odisha (Bhubaneswar)": (20.27, 85.84),
}


@pytest.fixture(scope="module")
def wb_geom():
    with open(WB_STATE_GEOJSON, "r", encoding="utf-8") as f:
        return shape(json.load(f)["features"][0]["geometry"])


@pytest.fixture(scope="module")
def mask():
    return get_or_create_wb_mask()


def _pixel_latlon(row: int, col: int, h: int, w: int) -> tuple[float, float]:
    """Geographic centre of a raster pixel, matching the linspace mapping used by
    generate_risk_surface_png (lat north->south, lon west->east)."""
    lat = WB_MAX_LAT + (WB_MIN_LAT - WB_MAX_LAT) * row / (h - 1)
    lon = WB_MIN_LON + (WB_MAX_LON - WB_MIN_LON) * col / (w - 1)
    return lat, lon


def test_all_23_districts_inside_boundary(wb_geom):
    outside = [
        name for name, (lat, lon) in WB_DISTRICT_POINTS.items()
        if not wb_geom.contains(Point(lon, lat))
    ]
    assert not outside, f"West Bengal districts missing from the boundary geometry: {outside}"


def test_neighbouring_regions_excluded(wb_geom):
    leaked = [
        name for name, (lat, lon) in OUTSIDE_POINTS.items()
        if wb_geom.contains(Point(lon, lat))
    ]
    assert not leaked, f"Non-West-Bengal locations inside the boundary: {leaked}"


def test_every_district_has_mask_coverage(mask):
    """The rasterized mask -- not just the vector geometry -- must cover each
    district, otherwise that district renders as a hole in the forecast layer."""
    h, w = mask.shape
    uncovered = []
    for name, (lat, lon) in WB_DISTRICT_POINTS.items():
        row = int(round((WB_MAX_LAT - lat) / (WB_MAX_LAT - WB_MIN_LAT) * (h - 1)))
        col = int(round((lon - WB_MIN_LON) / (WB_MAX_LON - WB_MIN_LON) * (w - 1)))
        row = min(max(row, 0), h - 1)
        col = min(max(col, 0), w - 1)
        # Allow a 1-pixel search radius: a district centroid can land just off a
        # coarse raster cell without the district being genuinely unmasked.
        window = mask[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
        if not window.any():
            uncovered.append(name)
    assert not uncovered, f"Districts with no rasterized forecast coverage: {uncovered}"


def test_no_forecast_pixels_outside_west_bengal(wb_geom, mask):
    """Render a full-domain, maximum-risk surface and assert that every visible
    pixel lies inside the state. A uniform 1.0 grid is the worst case: it would
    paint the entire bounding box if masking were broken."""
    lats = np.linspace(28.0, 20.0, 33)
    lons = np.linspace(84.0, 90.0, 25)
    png = generate_risk_surface_png(np.ones((33, 25), dtype=np.float32), lats, lons, mask=mask)

    rgba = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    visible = np.argwhere(alpha > 0)
    assert visible.size > 0, "forecast surface rendered completely transparent"

    # Half-pixel tolerance: a pixel centre just outside the polygon can still
    # legitimately overlap the boundary. Buffer and prepare ONCE -- this geometry
    # has ~76k vertices, so re-buffering per point is prohibitively slow.
    tol = 0.5 * max((WB_MAX_LAT - WB_MIN_LAT) / (h - 1), (WB_MAX_LON - WB_MIN_LON) / (w - 1))
    allowed = prep(wb_geom.buffer(tol))

    # Checking every visible pixel is still slow; sample densely and deterministically.
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(visible), size=min(3000, len(visible)), replace=False)

    outside = []
    for row, col in visible[sample_idx]:
        lat, lon = _pixel_latlon(int(row), int(col), h, w)
        if not allowed.contains(Point(lon, lat)):
            outside.append((round(lat, 3), round(lon, 3)))

    assert not outside, (
        f"{len(outside)} of {len(sample_idx)} sampled forecast pixels fall outside "
        f"West Bengal, e.g. {outside[:5]}"
    )


def test_forecast_surface_covers_state_interior(mask):
    """Guard against the opposite failure: a mask so eroded that the forecast
    covers only a sliver of the state."""
    assert mask.mean() > 0.25, (
        f"forecast mask covers only {100 * mask.mean():.1f}% of the bounding box, "
        "which suggests the state geometry is incomplete"
    )
