"""Map coordinate / model-domain acceptance tests (spec section X).

Traces the full path a Leaflet click takes:
    clicked lat/lon -> state-boundary containment -> nearest grid cell ->
    backend response -> forecast values

and asserts that valid West Bengal coordinates from every part of the state are
accepted and answered with real forecast values, while a genuinely outside
coordinate is rejected honestly.

These exist because a valid West Bengal click must never be refused because of
a coordinate-conversion bug (reversed lat/lon, descending latitude arrays,
wrong grid-shape assumptions, or stale metadata).
"""
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))
from main import app  # noqa: E402

from src.inference.nowcast_service import get_nowcast_service  # noqa: E402

client = TestClient(app)

# Real settlements spanning every region of West Bengal.
WB_POINTS = [
    # (name, lat, lon, region)
    ("Kolkata",      22.5726, 88.3639, "south"),
    ("Howrah",       22.5958, 88.2636, "south"),
    ("Digha",        21.6270, 87.5090, "south"),
    ("Bardhaman",    23.2324, 87.8615, "central"),
    ("Krishnanagar", 23.4058, 88.4917, "central"),
    ("Berhampore",   24.1024, 88.2518, "central"),
    ("Malda",        25.0119, 88.1433, "north"),
    ("Siliguri",     26.7271, 88.3953, "north"),
    ("Darjeeling",   27.0360, 88.2627, "north"),
    ("Jalpaiguri",   26.5435, 88.7190, "north"),
    ("Alipurduar",   26.4835, 89.5270, "east"),
    ("Cooch Behar",  26.3242, 89.4482, "east"),
    ("Basirhat",     22.6570, 88.8640, "east"),
    ("Bangaon",      23.0690, 88.8290, "east"),
    ("Purulia",      23.3320, 86.3650, "west"),
    ("Bankura",      23.2324, 87.0750, "west"),
    ("Jhargram",     22.4500, 86.9970, "west"),
    ("Asansol",      23.6850, 86.9700, "west"),
]

OUTSIDE_POINTS = [
    ("Patna, Bihar",      25.5941, 85.1376),
    ("Dhaka, Bangladesh", 23.8103, 90.4125),
    ("Kathmandu, Nepal",  27.7172, 85.3240),
    ("Bay of Bengal",     20.0000, 88.0000),
]

REGIONS = {"south", "central", "north", "east", "west"}


def test_every_region_is_represented():
    """Guards the test data itself: the spec requires all five regions."""
    assert {r for *_, r in WB_POINTS} == REGIONS


@pytest.mark.parametrize("name,lat,lon,region", WB_POINTS)
def test_valid_west_bengal_point_is_accepted(name, lat, lon, region):
    r = client.get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=2&mode=historical")
    assert r.status_code == 200, f"{name}: HTTP {r.status_code}"
    d = r.json()
    assert d.get("inside_monitored_region") is True, (
        f"{name} ({lat}, {lon}) in {region} West Bengal was rejected as outside "
        f"the monitored region: {d.get('status')}"
    )
    assert "predictions" in d, f"{name}: no predictions returned"


@pytest.mark.parametrize("name,lat,lon,region", WB_POINTS)
def test_valid_point_maps_to_a_sane_grid_cell(name, lat, lon, region):
    """The selected cell must be the NEAREST one, within half a grid step."""
    r = client.get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=2&mode=historical")
    d = r.json()
    cell = d["grid_cell"]
    # 0.25 deg grid -> nearest centre is at most 0.125 deg away (plus rounding
    # of the reported value to 2dp).
    assert abs(cell["lat"] - lat) <= 0.125 + 0.005, f"{name}: lat cell {cell['lat']} too far from {lat}"
    assert abs(cell["lon"] - lon) <= 0.125 + 0.005, f"{name}: lon cell {cell['lon']} too far from {lon}"


@pytest.mark.parametrize("name,lat,lon,region", WB_POINTS)
@pytest.mark.parametrize("lead", [2, 4, 6])
def test_forecast_values_are_real_numbers_at_every_horizon(name, lat, lon, region, lead):
    r = client.get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead={lead}&mode=historical")
    assert r.status_code == 200
    p = r.json()["predictions"]
    prob = p["thunderstorm_prob_pct"]
    rain = p["heavy_rain_mm"]
    assert isinstance(prob, (int, float)) and 0.0 <= prob <= 100.0, f"{name} +{lead}h: bad prob {prob}"
    assert isinstance(rain, (int, float)) and rain >= 0.0, f"{name} +{lead}h: bad rain {rain}"
    assert not np.isnan(prob) and not np.isnan(rain)


@pytest.mark.parametrize("name,lat,lon", OUTSIDE_POINTS)
def test_genuinely_outside_point_is_rejected_honestly(name, lat, lon):
    r = client.get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=2&mode=historical")
    assert r.status_code == 200
    d = r.json()
    assert d.get("inside_monitored_region") is False, f"{name} should be outside West Bengal"
    assert "predictions" not in d, f"{name}: a forecast was returned for an outside point"


def test_latitude_ordering_and_domain_metadata_are_consistent():
    """The lat vector is DESCENDING (28->20). Nearest-cell lookup must handle
    that; a naive ascending assumption would mirror the grid north-to-south."""
    svc = get_nowcast_service()
    lats, lons = svc.lats, svc.lons
    assert lats[0] > lats[-1], "latitude vector is expected to be descending"
    assert lons[0] < lons[-1], "longitude vector is expected to be ascending"
    assert (len(lats), len(lons)) == (33, 25)
    assert lats.max() == pytest.approx(28.0) and lats.min() == pytest.approx(20.0)
    assert lons.min() == pytest.approx(84.0) and lons.max() == pytest.approx(90.0)

    # A northern point must map to a LOW row index, a southern point to a high
    # one. This is the assertion that actually catches a flipped lat axis.
    i_north = int(np.argmin(np.abs(lats - 27.0)))
    i_south = int(np.argmin(np.abs(lats - 21.0)))
    assert i_north < i_south, "latitude axis appears flipped"


def test_lat_lon_are_not_swapped():
    """Swapping lat/lon must be detectable: (88.36, 22.57) is not in India's
    latitude range at all and must be rejected rather than silently accepted."""
    r = client.get("/api/nowcast/point?lat=88.3639&lon=22.5726&lead=2&mode=historical")
    assert r.status_code == 200
    assert r.json().get("inside_monitored_region") is False, (
        "a swapped lat/lon pair was accepted as a valid location"
    )
