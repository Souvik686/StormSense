# -*- coding: utf-8 -*-
"""Tests for the NOW current-observation surface.

The NOW layer is the one place where a smooth continuous image is produced from
scattered point data, so these tests exist mainly to pin down what it must NOT
do: invent values, fill gaps, swallow missing readings as zero, or present
itself as a forecast.
"""
from __future__ import annotations

import io
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.observation_surface import (  # noqa: E402
    MAX_INFLUENCE_DEG,
    OBSERVATION_SITES,
    RENDERABLE_VARIABLES,
    StationObservation,
    generate_observation_surface_png,
    idw_interpolate,
    parse_owm_current,
    summarise_observations,
)
from src.inference.risk_surface import (  # noqa: E402
    WB_MAX_LAT,
    WB_MAX_LON,
    WB_MIN_LAT,
    WB_MIN_LON,
)


def _station(name, lat, lon, **kw):
    return StationObservation(name=name, lat=lat, lon=lon, **kw)


@pytest.fixture
def stations():
    return [
        _station("North", 26.32, 89.45, rain_1h_mm=1.23, temperature_c=27.8,
                 humidity_pct=85, observed_unix=1789000000),
        _station("Central", 23.40, 88.50, rain_1h_mm=0.0, temperature_c=28.0,
                 humidity_pct=90, observed_unix=1789000060),
        _station("South", 22.10, 88.40, rain_1h_mm=0.5, temperature_c=27.0,
                 humidity_pct=94, observed_unix=1789000030),
    ]


# ---------------------------------------------------------------------------
# Station geometry
# ---------------------------------------------------------------------------

def test_sampling_sites_lie_inside_the_domain():
    for name, lat, lon in OBSERVATION_SITES:
        assert WB_MIN_LAT <= lat <= WB_MAX_LAT, f"{name} latitude outside domain"
        assert WB_MIN_LON <= lon <= WB_MAX_LON, f"{name} longitude outside domain"


def test_sampling_sites_are_distinct():
    coords = [(round(la, 3), round(lo, 3)) for _, la, lo in OBSERVATION_SITES]
    assert len(set(coords)) == len(coords), "duplicate sampling sites"


def test_enough_sites_for_a_defensible_field():
    """A handful of points cannot describe a state-sized field."""
    assert len(OBSERVATION_SITES) >= 20


# ---------------------------------------------------------------------------
# The honesty contract
# ---------------------------------------------------------------------------

def test_missing_values_are_dropped_not_zero_filled(stations):
    """A station that did not report a variable must be EXCLUDED.

    Zero-filling would invent a measurement -- for rainfall especially, "no
    data" and "no rain" are different facts.
    """
    stations.append(_station("Silent", 24.0, 87.5, rain_1h_mm=None,
                             temperature_c=25.0, observed_unix=1789000000))
    _, _, n_used = idw_interpolate(stations, "rain_1h_mm")
    assert n_used == 3, "station with no rainfall reading must not contribute"

    # It still contributes to a variable it DID report.
    _, _, n_temp = idw_interpolate(stations, "temperature_c")
    assert n_temp == 4


def test_pixels_far_from_every_station_are_not_invented(stations):
    """Beyond the influence radius the field must be NaN, not extrapolated."""
    values, coverage, _ = idw_interpolate(stations, "rain_1h_mm")
    assert np.isnan(values[~coverage]).all(), (
        "uncovered pixels must stay NaN rather than receive a guessed value"
    )
    assert coverage.any(), "no coverage at all from valid stations"
    assert not coverage.all(), (
        "three stations should not cover the whole state; influence radius "
        "is not being enforced"
    )


def test_no_stations_yields_empty_field():
    values, coverage, n = idw_interpolate([], "rain_1h_mm")
    assert n == 0
    assert not coverage.any()
    assert np.isnan(values).all()


def test_interpolation_reproduces_station_values(stations):
    """At a station's own location the field must equal its measurement."""
    values, _, _ = idw_interpolate(stations, "rain_1h_mm")
    h, w = values.shape
    for st in stations:
        row = int(round((WB_MAX_LAT - st.lat) / (WB_MAX_LAT - WB_MIN_LAT) * (h - 1)))
        col = int(round((st.lon - WB_MIN_LON) / (WB_MAX_LON - WB_MIN_LON) * (w - 1)))
        row = min(max(row, 0), h - 1)
        col = min(max(col, 0), w - 1)
        assert values[row, col] == pytest.approx(st.rain_1h_mm, abs=0.25), (
            f"field near {st.name} does not match its measured value"
        )


def test_interpolated_values_stay_within_observed_range(stations):
    """IDW must not overshoot: it is a weighted mean, so every value lies
    between the minimum and maximum observation."""
    values, coverage, _ = idw_interpolate(stations, "rain_1h_mm")
    inside = values[coverage & np.isfinite(values)]
    obs = [s.rain_1h_mm for s in stations]
    assert inside.min() >= min(obs) - 1e-6
    assert inside.max() <= max(obs) + 1e-6


def test_summary_declares_interpolation_and_non_forecast(stations):
    s = summarise_observations(stations, "rain_1h_mm")
    assert s["is_forecast"] is False
    assert s["is_interpolated"] is True
    assert "inverse-distance" in s["interpolation_method"].lower()
    assert s["stations_reporting"] == 3
    assert "interpolated" in s["provenance_note"].lower()
    assert s["max_influence_deg"] == MAX_INFLUENCE_DEG


def test_summary_reports_real_observation_times(stations):
    s = summarise_observations(stations, "rain_1h_mm")
    assert s["oldest_observation_utc"] is not None
    assert s["newest_observation_utc"] is not None
    assert s["oldest_observation_utc"] <= s["newest_observation_utc"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_png_renders_without_numeric_warnings(stations):
    """NaN must never reach an integer cast (it is undefined behaviour and
    previously produced garbage colour indices)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        png = generate_observation_surface_png(stations, "rain_1h_mm")
    assert png[:4] == b"\x89PNG"


def test_png_is_transparent_outside_coverage(stations):
    png = generate_observation_surface_png(stations, "rain_1h_mm")
    rgba = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert (rgba[..., 3] == 0).any(), "nothing left transparent; gaps were filled"
    assert (rgba[..., 3] > 0).any(), "nothing rendered at all"


def test_rendered_surface_uses_a_non_risk_palette(stations):
    """The observation ramp is blue/teal; the forecast risk ramp is
    green/amber/orange/red. A viewer must never confuse the two."""
    png = generate_observation_surface_png(stations, "rain_1h_mm")
    rgba = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    vis = rgba[rgba[..., 3] > 0]
    assert len(vis) > 0
    # Blue channel dominates red across the observation ramp.
    assert (vis[:, 2].astype(int) >= vis[:, 0].astype(int)).mean() > 0.95


def test_unsupported_variable_is_rejected(stations):
    with pytest.raises(ValueError):
        generate_observation_surface_png(stations, "severe_weather_prob")


def test_only_observed_variables_are_renderable():
    """No forecast/model quantity may be offered by this module."""
    for name in RENDERABLE_VARIABLES:
        assert name in {"rain_1h_mm", "temperature_c", "humidity_pct", "cloud_pct"}


# ---------------------------------------------------------------------------
# Provider payload parsing
# ---------------------------------------------------------------------------

def test_parse_owm_uses_provider_resolved_coordinates():
    payload = {
        "coord": {"lat": 22.57, "lon": 88.36},
        "main": {"temp": 28.0, "humidity": 94, "pressure": 1009},
        "wind": {"speed": 2.5},
        "clouds": {"all": 65},
        "dt": 1789000000,
        "name": "Kolkata",
        "weather": [{"description": "broken clouds"}],
    }
    st = parse_owm_current("Req", 22.60, 88.40, payload)
    assert st.lat == 22.57 and st.lon == 88.36
    assert st.temperature_c == 28.0
    assert st.wind_kmh == pytest.approx(9.0, abs=0.1)
    assert st.provider_station == "Kolkata"
    assert st.observed_at_utc is not None


def test_parse_owm_absent_rain_block_means_zero_rain():
    """OWM omits `rain` when there is no precipitation. That is an observed
    zero, distinct from a station failing to report (which raises and drops
    the station entirely)."""
    payload = {
        "coord": {"lat": 23.0, "lon": 88.0},
        "main": {"temp": 30.0, "humidity": 50},
        "dt": 1789000000,
        "weather": [],
    }
    st = parse_owm_current("Dry", 23.0, 88.0, payload)
    assert st.rain_1h_mm == 0.0


def test_parse_owm_missing_fields_stay_none():
    payload = {"coord": {"lat": 23.0, "lon": 88.0}, "main": {}, "dt": 1789000000}
    st = parse_owm_current("Sparse", 23.0, 88.0, payload)
    assert st.temperature_c is None
    assert st.humidity_pct is None
    assert st.wind_kmh is None


# ---------------------------------------------------------------------------
# Temporal contract (NOW vs +2/+4/+6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lead,secs", [(2, 7200), (4, 14400), (6, 21600)])
@pytest.mark.parametrize("now", [
    datetime(2026, 9, 11, 22, 2, 37, 123456, tzinfo=timezone.utc),
    datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc),
    datetime(2026, 12, 31, 23, 30, 0, tzinfo=timezone.utc),
])
def test_forecast_valid_times_are_exact_offsets(lead, secs, now):
    """+2/+4/+6 must be exact second offsets from the reference instant,
    preserving sub-hour precision across hour, date and year rollovers."""
    from src.inference.risk_map import compute_valid_time

    vt = datetime.fromisoformat(
        str(compute_valid_time(now.isoformat(), lead)).replace("Z", "+00:00")
    )
    assert (vt - now).total_seconds() == secs
    assert vt.minute == now.minute and vt.second == now.second
    assert vt.microsecond == now.microsecond


def test_ist_offset_is_exactly_5_30():
    utc = datetime(2026, 9, 11, 18, 39, 12, tzinfo=timezone.utc)
    ist = utc + timedelta(hours=5, minutes=30)
    assert (ist - utc).total_seconds() == 19800
    # Date rollover into the next IST day.
    assert ist.strftime("%d %b %H:%M") == "12 Sep 00:09"
