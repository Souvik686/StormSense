"""Temporal-integrity tests for the live GFS input pipeline.

These guard the one rule that cannot be relaxed (see src/inference/gfs_live.py):

  1. Only GFS f000 ANALYSES are ever used as the model's input timesteps.
     Forecast hours f001+ are future forecast steps; feeding them into the
     model's supposed observed-past input leaks forecast information into an
     input the model was trained to read as analysis.
  2. No GFS cycle is ever used that was not genuinely published as of the
     reference instant, accounting for real production latency. This is what
     makes a historical backtest honest.

An earlier version of this file caught neither: it wrapped the call in a bare
`except: pass` and only inspected URL substrings, so a pipeline fetching f003
from a permitted cycle passed. These assert on the actual requested forecast
hours and cycle times.
"""
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.inference import gfs_live


# ── URL parsing helpers ──────────────────────────────────────────────────────
_URL_RE = re.compile(r"gfs\.(\d{8})/(\d{2})/atmos/gfs\.t(\d{2})z\.pgrb2\.0p25\.f(\d{3})")


def _parse_gfs_url(url: str):
    """Return (cycle_datetime, forecast_hour) for a GFS product URL."""
    m = _URL_RE.search(url)
    if not m:
        return None
    ymd, hh_dir, hh_file, fhr = m.groups()
    assert hh_dir == hh_file, f"cycle hour mismatch in URL: {url}"
    cycle = datetime.strptime(ymd + hh_dir, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return cycle, int(fhr)


class _RecordingClient:
    """Stands in for httpx.Client, recording every URL requested and serving a
    minimal .idx so the fetch path runs far enough to reveal its intent."""

    def __init__(self, recorder):
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass

    def get(self, url, *args, **kwargs):
        self.recorder.append(str(url))
        return _Resp(200, "0:0:d=2026091100:TMP:surface:anl:\n")

    def head(self, url, *args, **kwargs):
        self.recorder.append(str(url))
        return _Resp(200, "")


class _Resp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text
        self.content = b""


@pytest.fixture
def requested_urls(monkeypatch):
    urls = []
    monkeypatch.setattr(gfs_live, "_http_client", lambda: _RecordingClient(urls))
    monkeypatch.setattr(gfs_live, "_HARMONIZED_CACHE", None, raising=False)
    return urls


def _run_harmonize(target_t0):
    lats = np.array([22.0, 21.75])
    lons = np.array([88.0, 88.25])
    # The fake .idx carries no real GRIB messages, so harmonization is EXPECTED
    # to fail partway. We are asserting on what it tried to fetch before failing,
    # which is exactly where a leak would show up.
    with pytest.raises(Exception):
        gfs_live.fetch_and_harmonize(lats, lons, target_t0=target_t0, use_cache=False)


def test_only_analysis_forecast_hour_zero_is_ever_requested(requested_urls):
    """Every GFS product request must be f000. A single f001+ request means the
    model's observed-past input is being fed forecast data."""
    _run_harmonize(datetime(2026, 9, 11, 13, 6, 10, tzinfo=timezone.utc))

    parsed = [_parse_gfs_url(u) for u in requested_urls]
    parsed = [p for p in parsed if p]
    assert parsed, f"No GFS product URLs were requested; got: {requested_urls}"

    offending = sorted({fh for _, fh in parsed if fh != 0})
    assert not offending, (
        f"Forecast hours {offending} were requested as model input timesteps. "
        "Only f000 analyses may be used -- f001+ are future forecast steps."
    )


def test_no_unpublished_future_cycle_is_requested(requested_urls):
    """A cycle may only be used once production latency says it was actually on
    the wire. At 13:06Z the 12Z cycle does not yet exist operationally."""
    target = datetime(2026, 9, 11, 13, 6, 10, tzinfo=timezone.utc)
    _run_harmonize(target)

    parsed = [p for p in (_parse_gfs_url(u) for u in requested_urls) if p]
    assert parsed

    lag = timedelta(hours=gfs_live.GFS_PRODUCTION_LAG_HOURS)
    for cycle, fh in parsed:
        assert cycle <= target, f"Cycle {cycle.isoformat()} is AFTER the reference instant {target.isoformat()}"
        assert cycle + lag <= target, (
            f"Cycle {cycle.isoformat()} would not have been published by "
            f"{target.isoformat()} (production lag {gfs_live.GFS_PRODUCTION_LAG_HOURS}h)"
        )


def test_backtest_at_past_instant_cannot_see_later_cycles(requested_urls):
    """The historical-backtest path uses the same selection rule, so a simulated
    past NOW must never reach a cycle from that instant's future."""
    past = datetime(2026, 9, 10, 4, 30, 0, tzinfo=timezone.utc)
    _run_harmonize(past)

    parsed = [p for p in (_parse_gfs_url(u) for u in requested_urls) if p]
    assert parsed

    for cycle, fh in parsed:
        assert fh == 0
        assert cycle + timedelta(hours=gfs_live.GFS_PRODUCTION_LAG_HOURS) <= past, (
            f"Backtest at {past.isoformat()} reached cycle {cycle.isoformat()}, "
            "which had not been published at that simulated moment."
        )


@pytest.mark.parametrize(
    "wallclock,expected_newest",
    [
        # 13:06Z with 5h lag -> 12Z not yet out, 06Z is newest genuinely available
        ("2026-09-11T13:06:10+00:00", "2026-09-11T06:00:00+00:00"),
        # 17:30Z -> 12Z has had 5.5h, so it is now available
        ("2026-09-11T17:30:00+00:00", "2026-09-11T12:00:00+00:00"),
        # exactly at the lag boundary the cycle counts as available
        ("2026-09-11T17:00:00+00:00", "2026-09-11T12:00:00+00:00"),
        # one second before the boundary it does not
        ("2026-09-11T16:59:59+00:00", "2026-09-11T06:00:00+00:00"),
        # just after midnight -> previous day's 18Z
        ("2026-09-12T00:30:00+00:00", "2026-09-11T18:00:00+00:00"),
    ],
)
def test_cycle_availability_boundary(wallclock, expected_newest):
    wc = datetime.fromisoformat(wallclock)
    cycles = gfs_live._cycles_available_at(wc, gfs_live.GFS_PRODUCTION_LAG_HOURS)
    assert cycles, "no cycles considered available"
    assert cycles[0].isoformat() == expected_newest
    # strictly descending, 6-hourly, and never in the future
    for a, b in zip(cycles, cycles[1:]):
        assert (a - b) == timedelta(hours=6)
        assert a > b
    assert all(c <= wc for c in cycles)


def test_input_slots_are_hourly_and_end_at_the_analysis_time():
    """The six slots the model reads must be consecutive hourly steps ending at
    the analysis time -- the shape the model was trained on."""
    older = gfs_live.GfsCycle(
        cycle_time=datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc),
        surface={v: np.zeros((2, 2), dtype=np.float32) for v in gfs_live.SINGLE_VARS},
        pressure={v: {lvl: np.zeros((2, 2), dtype=np.float32)
                      for lvl in gfs_live.PRESSURE_LEVELS_HPA}
                  for v in gfs_live.PRESSURE_VARS},
    )
    newer = gfs_live.GfsCycle(
        cycle_time=datetime(2026, 9, 11, 6, 0, tzinfo=timezone.utc),
        surface={v: np.ones((2, 2), dtype=np.float32) for v in gfs_live.SINGLE_VARS},
        pressure={v: {lvl: np.ones((2, 2), dtype=np.float32)
                      for lvl in gfs_live.PRESSURE_LEVELS_HPA}
                  for v in gfs_live.PRESSURE_VARS},
    )

    analysis_t0 = newer.cycle_time
    slots = [analysis_t0 - timedelta(hours=h) for h in range(5, -1, -1)]
    assert len(slots) == 6
    assert slots[-1] == analysis_t0
    for a, b in zip(slots, slots[1:]):
        assert (b - a) == timedelta(hours=1)

    # The final slot must be the real analysis, used unmodified (weight 1.0).
    _, _, w_last = gfs_live._interpolate_slot(older, newer, slots[-1])
    assert w_last == pytest.approx(1.0)

    # Interpolation is strictly bounded by the two real analyses -- it can never
    # extrapolate beyond observed values.
    for st in slots:
        surf, _, w = gfs_live._interpolate_slot(older, newer, st)
        assert 0.0 <= w <= 1.0
        for v in gfs_live.SINGLE_VARS:
            assert float(surf[v].min()) >= 0.0
            assert float(surf[v].max()) <= 1.0
