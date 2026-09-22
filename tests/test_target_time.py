"""Contract tests for the wall-clock target-time resolver.

These pin the property the UI depends on: a horizon labelled "+Nh" is either
served by a lead whose TRUE valid time is within tolerance of
`wall_clock + N`, or it is reported unavailable. There is no third outcome in
which a label is attached to a forecast valid at some other time.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.inference.target_time import (
    DEFAULT_TOLERANCE_HOURS,
    required_lead_hours,
    resolve,
    resolve_all,
)

UTC = timezone.utc


def test_required_lead_is_age_plus_horizon():
    wall = datetime(2026, 9, 21, 17, 16, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    # analysis is 5.2667h old; "+2h from now" therefore needs a ~7.27h lead
    assert required_lead_hours(wall, anal, 0) == pytest.approx(5.2667, abs=1e-3)
    assert required_lead_hours(wall, anal, 2) == pytest.approx(7.2667, abs=1e-3)
    assert required_lead_hours(wall, anal, 6) == pytest.approx(11.2667, abs=1e-3)


def test_served_valid_time_matches_the_label():
    """The whole point: what we serve must be valid at the advertised instant."""
    wall = datetime(2026, 9, 21, 18, 0, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)  # 6h old
    leads = list(range(4, 17))                        # v4's hourly 4..16h
    r = resolve(wall, anal, 2, leads)
    assert r.available
    assert r.source_lead_hours == 8          # 6h age + 2h horizon
    target = datetime.fromisoformat(r.target_time_utc)
    served = datetime.fromisoformat(r.served_valid_time_utc)
    assert target == datetime(2026, 9, 21, 20, 0, tzinfo=UTC)
    assert abs((served - target).total_seconds()) / 3600.0 <= DEFAULT_TOLERANCE_HOURS


@pytest.mark.parametrize("horizon,expect_lead", [(0, 6), (2, 8), (4, 10), (6, 12)])
def test_all_four_ui_horizons_resolve_with_hourly_leads(horizon, expect_lead):
    wall = datetime(2026, 9, 21, 18, 0, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    r = resolve(wall, anal, horizon, list(range(4, 17)))
    assert r.available, r.reason
    assert r.source_lead_hours == expect_lead


def test_five_lead_v3_model_cannot_serve_most_horizons():
    """v3's coarse {8,10,12,14,16} leaves gaps at the odd required leads, which
    must surface as UNAVAILABLE rather than as a mislabelled neighbour."""
    wall = datetime(2026, 9, 21, 17, 0, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)  # 5h old -> needs 5,7,9,11
    res = resolve_all(wall, anal, [0, 2, 4, 6], [8, 10, 12, 14, 16])
    # none of 5/7/9/11 is within 0.5h of an available even lead
    assert not any(v.available for v in res.values())
    for v in res.values():
        assert v.reason


def test_unavailable_when_requirement_exceeds_model_range():
    wall = datetime(2026, 9, 21, 21, 30, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)  # 9.5h old -> +6h needs 15.5h
    r = resolve(wall, anal, 6, [4, 6, 8])
    assert not r.available
    assert "forecasts out to" in r.reason


def test_no_leads_is_unavailable_not_a_crash():
    r = resolve(datetime(2026, 9, 21, 12, tzinfo=UTC),
                datetime(2026, 9, 21, 12, tzinfo=UTC), 2, [])
    assert not r.available


def test_interpolation_is_opt_in_and_brackets_the_target():
    wall = datetime(2026, 9, 21, 17, 0, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)  # +2h needs a 7h lead
    leads = [6, 8]
    assert not resolve(wall, anal, 2, leads).available
    r = resolve(wall, anal, 2, leads, allow_interpolation=True)
    assert r.available and r.method == "interpolated"
    assert r.source_leads_hours == [6.0, 8.0]
    assert r.interpolation_weight == pytest.approx(0.5)


def test_naive_datetimes_are_treated_as_utc():
    r = resolve(datetime(2026, 9, 21, 18, 0), datetime(2026, 9, 21, 12, 0),
                2, list(range(4, 17)))
    assert r.available and r.source_lead_hours == 8


def test_label_error_is_reported_when_not_exact():
    """A within-tolerance-but-inexact match must still disclose its error."""
    wall = datetime(2026, 9, 21, 18, 20, tzinfo=UTC)   # age 6.333h
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    r = resolve(wall, anal, 2, list(range(4, 17)))     # needs 8.333 -> serves 8
    assert r.available and r.method == "nearest"
    assert r.label_error_hours == pytest.approx(-0.333, abs=1e-2)


def test_hourly_leads_cover_the_whole_cycle_at_the_measured_lag():
    """The v4 lead set exists to make the four UI horizons servable at ANY point
    in the 6-hourly GFS cycle. Pinned so a future lead-set change cannot quietly
    reintroduce holes where a horizon would have to be hidden."""
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    leads = list(range(4, 17))
    lag = 3.6  # measured NCEP publish lag
    t, end = anal + timedelta(hours=lag), anal + timedelta(hours=lag + 6)
    gaps = []
    while t <= end:
        for h in (0, 2, 4, 6):
            if not resolve(t, anal, h, leads).available:
                gaps.append((t.isoformat(), h))
        t += timedelta(minutes=5)
    assert not gaps, f"horizons unservable at {len(gaps)} instants, e.g. {gaps[:3]}"


def test_five_coarse_leads_cannot_cover_the_cycle():
    """Contrast case: v3's {8,10,12,14,16} leaves most instants unservable, which
    is precisely why the hourly set was trained."""
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    leads = [8, 10, 12, 14, 16]
    t, end = anal + timedelta(hours=3.6), anal + timedelta(hours=9.6)
    total = served = 0
    while t <= end:
        for h in (0, 2, 4, 6):
            total += 1
            if resolve(t, anal, h, leads).available:
                served += 1
        t += timedelta(minutes=5)
    assert served / total < 0.5, "coarse leads unexpectedly covered the cycle"


def test_required_lead_grows_as_the_analysis_ages():
    """GFS publishes only at 00/06/12/18Z, so between cycles the newest analysis
    ages from ~3.6h to ~9.6h -- the data is alternately fresh and stale. The lead
    needed to answer a FIXED user horizon must grow with that age, otherwise the
    same button would silently start pointing into the past."""
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    leads = list(range(4, 17))
    seen = []
    for minutes in range(0, 6 * 60 + 1, 30):
        wall = anal + timedelta(hours=3.6) + timedelta(minutes=minutes)
        r = resolve(wall, anal, 2, leads)
        seen.append(r.required_lead_hours)
        if r.available:
            # whatever lead is chosen, it must be valid at now+2h
            served = datetime.fromisoformat(r.served_valid_time_utc)
            target = datetime.fromisoformat(r.target_time_utc)
            assert abs((served - target).total_seconds()) <= DEFAULT_TOLERANCE_HOURS * 3600
    assert seen == sorted(seen), "required lead must increase monotonically with age"
    assert seen[-1] - seen[0] == pytest.approx(6.0, abs=0.01), (
        "across one 6h cycle the required lead should grow by 6h"
    )


def test_a_stale_analysis_eventually_exhausts_a_short_lead_model():
    """v2 (leads 2-6h) can serve NOW only while the analysis is young enough.
    Once it ages past 6h + tolerance, NOW must become unavailable rather than
    being answered with a forecast valid in the past."""
    anal = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    v2 = [2, 3, 4, 5, 6]
    young = resolve(anal + timedelta(hours=6.0), anal, 0, v2)
    assert young.available, "NOW should be servable when the analysis is ~6h old"
    old = resolve(anal + timedelta(hours=9.0), anal, 0, v2)
    assert not old.available
    assert "6" in old.reason


def test_midnight_and_date_crossing():
    wall = datetime(2026, 9, 21, 22, 0, tzinfo=UTC)
    anal = datetime(2026, 9, 21, 18, 0, tzinfo=UTC)    # 4h old
    r = resolve(wall, anal, 4, list(range(4, 17)))     # needs 8 -> valid next day 02Z
    assert r.available
    assert datetime.fromisoformat(r.target_time_utc) == datetime(2026, 9, 22, 2, 0, tzinfo=UTC)
    assert datetime.fromisoformat(r.served_valid_time_utc) == datetime(2026, 9, 22, 2, 0, tzinfo=UTC)
