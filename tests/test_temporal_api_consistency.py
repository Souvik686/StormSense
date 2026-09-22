"""Cross-endpoint temporal agreement, against a running server.

Skipped when no server is listening, so the offline suite is unaffected.

Guards the defect where `/api/nowcast/point` computed its own valid time from
`issue_time + requested_lead`, ignoring the NOW re-anchor -- so lead=0 reported
the analysis instant (12:00Z) while serving the +6h field (valid 18:00Z).
"""
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

BASE = "http://127.0.0.1:8000"
KOLKATA = (22.5726, 88.3639)


def _get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        pytest.skip(f"StormSense server not reachable: {e}")


@pytest.fixture(scope="module")
def summary_now():
    d = _get("/api/nowcast/summary?lead=0&mode=live")
    if not d.get("now_anchor"):
        pytest.skip("live state not yet refreshed (no now_anchor)")
    return d


def _parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# TEST 7 -- point and summary agree on NOW's valid time and source lead
def test_point_and_summary_agree_on_now(summary_now):
    anchor = summary_now["now_anchor"]
    lat, lon = KOLKATA
    p = _get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=0&mode=live")

    assert p["effective_model_lead_hours"] == anchor["source_lead_hours"], (
        "point and summary must name the same re-anchored lead"
    )
    expected = _parse(anchor["valid_time_utc"]).strftime("%d %b %Y %H:%M UTC")
    assert p["forecast_valid_utc"] == expected, (
        f"point NOW valid time {p['forecast_valid_utc']} != anchor {expected}"
    )


# TEST 1/2 applied to the live API -- every lead is analysis + lead
def test_every_lead_valid_time_is_analysis_plus_lead(summary_now):
    """valid_time == analysis + the lead ACTUALLY SERVED.

    This previously asserted `analysis + the REQUESTED lead`, which held while
    the requested lead was always the model lead. Under the demo wall-clock
    mapping it no longer is: the label "+2h" is served by whichever real model
    lead lands nearest `now + 2h` (leads 5-8 at a ~6h-old analysis), precisely
    so the label means what it says. Asserting the old identity would be
    asserting that the mapping does not happen.

    The invariant that actually matters is unchanged and is asserted here: a
    forecast's valid time is its ANALYSIS time plus the lead the model really
    produced -- never wall_clock + lead, and never the requested label treated
    as if it were a model lead. `effective_model_lead_hours` names that lead, so
    the relationship stays checkable rather than becoming a matter of trust.
    """
    analysis = _parse(summary_now["analysis_time_iso"])
    lat, lon = KOLKATA
    leads = _get("/api/health")["lead_times_hours"]
    for lead in [L for L in leads if L != 0][:5]:
        p = _get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead={lead}&mode=live")
        served = p.get("effective_model_lead_hours", lead)
        expected = (analysis + timedelta(hours=served)).strftime("%d %b %Y %H:%M UTC")
        assert p["forecast_valid_utc"] == expected, (
            f"lead {lead} (served as +{served}h) must be valid at {expected}, "
            f"got {p['forecast_valid_utc']}"
        )


def test_valid_times_are_not_wall_clock_derived(summary_now):
    """The whole point of the fix: valid times track the analysis, not now()."""
    analysis = _parse(summary_now["analysis_time_iso"])
    now = datetime.now(analysis.tzinfo)
    if (now - analysis).total_seconds() / 3600.0 < 1.0:
        pytest.skip("analysis is younger than 1h; the two bases are indistinguishable")
    lat, lon = KOLKATA
    p = _get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=2&mode=live")
    wall_clock_based = (now + timedelta(hours=2)).strftime("%d %b %Y %H:%M UTC")
    assert p["forecast_valid_utc"] != wall_clock_based


# TEST 10 -- NOW is a distinct field, not a copy of +2h
def test_now_is_not_a_copy_of_lead_two(summary_now):
    anchor = summary_now["now_anchor"]
    if anchor["source_lead_hours"] == 2:
        pytest.skip("anchor legitimately selected +2h; identity is expected here")
    lat, lon = KOLKATA
    p0 = _get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=0&mode=live")["predictions"]
    p2 = _get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=2&mode=live")["predictions"]
    assert p0 != p2, (
        "NOW returned values identical to +2h while anchored to lead "
        f"{anchor['source_lead_hours']} -- the original re-anchoring defect"
    )


def test_now_anchor_declares_it_is_a_forecast(summary_now):
    """NOW must never be presented as an observation of the present instant."""
    anchor = summary_now["now_anchor"]
    assert "not an observation" in anchor["note"].lower()
    assert isinstance(anchor["reaches_wall_clock"], bool)
    assert set(anchor) >= {
        "source_lead_hours", "valid_time_utc", "analysis_time_utc",
        "wall_clock_utc", "offset_from_wall_clock_hours",
    }


# TEST 9 (live) -- historical mode keeps t0 semantics
def test_historical_now_is_not_reanchored():
    d = _get("/api/nowcast/summary?lead=0&mode=historical")
    assert d.get("now_anchor") is None, (
        "historical NOW must not carry a live re-anchor (cross-mode leak)"
    )
