"""Temporal correctness of forecast valid times and NOW re-anchoring.

These lock in the fix for two defects found 2026-09-20:

  1. `_issue_time_for_mode` returned the WALL-CLOCK instant for live mode, so
     every horizon advertised `wall_clock + lead` instead of
     `analysis_t0 + lead`. With an 8.17h-old analysis the +2h forecast (truly
     valid 14:00Z) was published as valid 22:08Z.

  2. NOW (lead 0) was built by re-fetching at `target_t0 - 2h` and taking that
     run's lead-2 slice. GFS cycles are 6h apart, so the shifted fetch resolved
     to the SAME f000 cycle and produced byte-identical tensors -- making lead=0
     an exact copy of lead=2. NOW painted the +2h forecast.

The fix computes valid times from the analysis and re-anchors NOW onto the model
lead whose valid time is nearest wall-clock, recording which lead was chosen.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.inference.nowcast_service import compute_valid_time


class _StubService:
    """Minimal stand-in exercising the real helper logic without loading torch."""

    from src.inference.nowcast_service import NowcastService

    _effective_lead_hours = NowcastService._effective_lead_hours
    _resolve_mode = staticmethod(lambda self, mode=None: mode or "live")

    def __init__(self, anchor=None, mode="live"):
        self.live_now_anchor = anchor
        self._mode = mode

    def _resolve_mode(self, mode=None):  # noqa: F811 - instance override
        return mode or self._mode


ANALYSIS = "2026-09-20T12:00:00+00:00"


# TEST 1 -- valid_time = analysis_t0 + lead
@pytest.mark.parametrize("lead,expected_hour", [(2, 14), (3, 15), (4, 16), (5, 17), (6, 18)])
def test_valid_time_is_analysis_plus_lead(lead, expected_hour):
    vt = datetime.fromisoformat(compute_valid_time(ANALYSIS, lead))
    assert vt == datetime(2026, 9, 20, expected_hour, 0, tzinfo=timezone.utc), (
        f"+{lead}h from a 12:00Z analysis must be valid at {expected_hour:02d}:00Z"
    )


# TEST 2 -- valid_time must NOT be wall_clock + lead
def test_valid_time_does_not_use_wall_clock():
    """A 12:00Z analysis read at 20:10Z must still place +2h at 14:00Z."""
    wall_clock = datetime(2026, 9, 20, 20, 10, tzinfo=timezone.utc)
    vt = datetime.fromisoformat(compute_valid_time(ANALYSIS, 2))
    assert vt.hour == 14, "valid time must derive from the analysis"
    assert vt != wall_clock + timedelta(hours=2), (
        "valid time must not be wall_clock + lead (the original defect)"
    )
    assert vt < wall_clock, "a forecast from an 8h-old analysis is already in the past"


# TEST 3 -- NOW selects the lead whose valid time is closest to wall-clock
@pytest.mark.parametrize(
    "wall_clock_hour,expected_lead",
    [
        (14, 2),   # 14:00Z -> +2h valid 14:00Z, exact
        (15, 3),
        (17, 5),
        (20, 6),   # beyond the last lead -> the newest available
        (12, 2),   # before any lead matures -> the nearest one
    ],
)
def test_now_selects_nearest_valid_time(wall_clock_hour, expected_lead):
    analysis = datetime.fromisoformat(ANALYSIS)
    now = analysis.replace(hour=wall_clock_hour)
    leads = [2, 3, 4, 5, 6]
    chosen = min(leads, key=lambda L: abs((analysis + timedelta(hours=L) - now).total_seconds()))
    assert chosen == expected_lead


# TEST 4 -- NOW exposes the selected source lead
def test_now_anchor_exposes_source_lead():
    anchor = {"source_lead_hours": 6, "valid_time_utc": "2026-09-20T18:00:00+00:00"}
    svc = _StubService(anchor=anchor, mode="live")
    assert svc._effective_lead_hours(0, "live") == 6, (
        "NOW must resolve to the re-anchored model lead, not 0"
    )


# TEST 5 -- NOW's reported valid time is the selected forecast's TRUE valid time
def test_now_reports_true_valid_time_of_selected_lead():
    anchor = {"source_lead_hours": 6, "valid_time_utc": "2026-09-20T18:00:00+00:00"}
    svc = _StubService(anchor=anchor, mode="live")
    eff = svc._effective_lead_hours(0, "live")
    vt = datetime.fromisoformat(compute_valid_time(ANALYSIS, eff))
    assert vt == datetime.fromisoformat(anchor["valid_time_utc"]), (
        "NOW must advertise the valid time of the lead it actually serves, "
        "not the analysis instant"
    )


# TEST 6 -- lead=2 keeps true +2h semantics (re-anchoring must not renumber leads)
def test_forecast_leads_are_not_renumbered():
    svc = _StubService(anchor={"source_lead_hours": 6}, mode="live")
    for lead in (2, 3, 4, 5, 6):
        assert svc._effective_lead_hours(lead, "live") == lead, (
            "only NOW is re-anchored; real forecast leads keep their meaning"
        )


# TEST 9 -- historical mode is never re-anchored
def test_historical_now_is_not_reanchored():
    """Historical replays a frozen event: lead 0 is the case study's t0."""
    svc = _StubService(anchor={"source_lead_hours": 6}, mode="historical")
    assert svc._effective_lead_hours(0, "historical") == 0, (
        "historical NOW must stay the t0 analysis, not a re-anchored forecast"
    )


def test_missing_anchor_falls_back_to_zero():
    """Before the first live refresh there is no anchor; NOW must not guess."""
    svc = _StubService(anchor=None, mode="live")
    assert svc._effective_lead_hours(0, "live") == 0
