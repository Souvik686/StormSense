"""Multiple geographically separated West Bengal locations must resolve to
distinct grid cells with independently varying risk.

Guards two failure modes that look fine if you only ever check Kolkata:
  * every location collapsing onto one cell (a lat/lon indexing bug), and
  * one scalar risk broadcast across the map (a rendering/aggregation bug).
"""
import numpy as np
import pytest

from src.inference.nowcast_service import get_nowcast_service

# Spread deliberately across the state: far north (Darjeeling/Siliguri), the
# western plateau (Purulia), the coast (Digha), the delta (Basirhat), the
# centre-north (Malda) and the metro (Kolkata).
LOCATIONS = [
    ("Kolkata", 22.57, 88.36),
    ("Siliguri", 26.71, 88.43),
    ("Digha", 21.62, 87.52),
    ("Malda", 25.01, 88.14),
    ("Purulia", 23.33, 86.36),
    ("Basirhat", 22.66, 88.89),
    ("Darjeeling", 27.04, 88.26),
]

GRID_SPACING_DEG = 0.25
# Worst-case distance from a point to its nearest 0.25-deg cell centre is half a
# diagonal ~= 0.177 deg ~= 20 km. Allow a little slack for the cos(lat) factor.
MAX_SNAP_KM = 25.0


@pytest.fixture(scope="module")
def svc():
    return get_nowcast_service()


def _cell(svc, lat, lon):
    return (int(np.argmin(np.abs(svc.lats - lat))),
            int(np.argmin(np.abs(svc.lons - lon))))


def test_grid_orientation(svc):
    """Latitude descending, longitude ascending -- the ordering every consumer
    (risk map, PNG render, point query) indexes by."""
    assert svc.lats[0] > svc.lats[-1], "latitude must be descending"
    assert svc.lons[0] < svc.lons[-1], "longitude must be ascending"
    assert len(svc.lats) == 33 and len(svc.lons) == 25


def test_each_location_gets_its_own_cell(svc):
    cells = {}
    for name, lat, lon in LOCATIONS:
        cells.setdefault(_cell(svc, lat, lon), []).append(name)
    collisions = {k: v for k, v in cells.items() if len(v) > 1}
    assert not collisions, f"locations collapsed onto shared cells: {collisions}"


@pytest.mark.parametrize("name,lat,lon", LOCATIONS)
def test_snapped_cell_is_actually_near_the_request(svc, name, lat, lon):
    i, j = _cell(svc, lat, lon)
    dlat = (lat - float(svc.lats[i])) * 111.0
    dlon = (lon - float(svc.lons[j])) * 111.0 * np.cos(np.radians(lat))
    dist = float(np.hypot(dlat, dlon))
    assert dist <= MAX_SNAP_KM, f"{name} snapped {dist:.1f} km away"


def test_risk_varies_across_locations_and_is_not_broadcast(svc):
    if svc.live_pred is None:
        pytest.skip("no live prediction available")
    field = np.asarray(svc.live_pred["severe_weather_prob"])
    # Spatial structure, not one value painted everywhere.
    for li in range(field.shape[0]):
        assert field[li].std() > 0.0, f"lead index {li} is spatially constant"
    # And the sampled locations do not all read the same number.
    li = svc.lead_times.index(2) if 2 in svc.lead_times else 0
    vals = [float(field[li][_cell(svc, la, lo)]) for _, la, lo in LOCATIONS]
    assert len(set(np.round(vals, 6))) > 1, "every location returned the same risk"


@pytest.mark.parametrize("name,lat,lon", LOCATIONS)
def test_xai_is_location_specific(svc, name, lat, lon):
    """The explanation must describe the requested cell, not a fixed default."""
    out = svc.get_xai_attribution(mode="live", lat=lat, lon=lon, lead_hours=4)
    if out.get("status") == "unavailable":
        pytest.skip("live XAI unavailable")
    gc = out.get("grid_cell")
    assert gc, "XAI returned no grid cell"
    i, j = _cell(svc, lat, lon)
    assert gc["lat"] == pytest.approx(round(float(svc.lats[i]), 2), abs=0.01)
    assert gc["lon"] == pytest.approx(round(float(svc.lons[j]), 2), abs=0.01)
