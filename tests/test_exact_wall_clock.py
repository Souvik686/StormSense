"""Exact wall-clock temporal acceptance tests.

The core StormSense requirement: NOW is the actual current instant, and
+2h/+4h/+6h are NOW plus exactly 7200/14400/21600 seconds. NOW must never be
floored to the hour, and an hourly model/analysis time must never be relabelled
as the exact current time.

Every assertion below compares PARSED datetimes and second-differences, never
formatted strings, so a cosmetically-correct-but-wrong timestamp cannot pass.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.inference.nowcast_service import compute_valid_time

IST = ZoneInfo("Asia/Kolkata")

# Reference instants deliberately carrying non-zero minutes, non-zero seconds
# and (where noted) fractional seconds.
CASES = [
    "2026-09-11T13:06:10.449910+00:00",  # the specified acceptance instant
    "2026-09-11T12:47:31+00:00",         # non-zero minute and second
    "2026-09-11T17:11:23+05:30",         # IST, non-zero minute/second
    "2026-09-11T20:15:45.123456+00:00",  # +6h rolls past midnight UTC
    "2026-09-11T23:59:59+00:00",         # extreme rollover
    "2026-12-31T22:30:00+00:00",         # year rollover at +2h
    "2026-02-28T23:45:12+00:00",         # month rollover
]


def _parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@pytest.mark.parametrize("now_iso", CASES)
@pytest.mark.parametrize("lead,expected_seconds", [(2, 7200), (4, 14400), (6, 21600)])
def test_horizon_is_exactly_n_hours_after_now(now_iso, lead, expected_seconds):
    """The difference must be mathematically exact, to the microsecond."""
    now = _parse(now_iso)
    vt = _parse(compute_valid_time(now_iso, lead))
    delta = (vt - now).total_seconds()
    assert delta == expected_seconds, (
        f"NOW={now.isoformat()} +{lead}h produced {vt.isoformat()} "
        f"({delta}s, expected exactly {expected_seconds}s)"
    )


@pytest.mark.parametrize("now_iso", CASES)
@pytest.mark.parametrize("lead", [2, 4, 6])
def test_sub_hour_precision_is_preserved_not_floored(now_iso, lead):
    """NOW must not be floored: minutes, seconds and microseconds must survive."""
    now = _parse(now_iso)
    vt = _parse(compute_valid_time(now_iso, lead))
    assert vt.minute == now.minute, f"minute was altered: {now.minute} -> {vt.minute}"
    assert vt.second == now.second, f"second was altered: {now.second} -> {vt.second}"
    assert vt.microsecond == now.microsecond, (
        f"microsecond was altered: {now.microsecond} -> {vt.microsecond}"
    )


def test_specified_acceptance_instant_exactly():
    """The exact instant named in the acceptance criteria."""
    now_iso = "2026-09-11T13:06:10.449910+00:00"
    assert _parse(compute_valid_time(now_iso, 2)) == _parse("2026-09-11T15:06:10.449910+00:00")
    assert _parse(compute_valid_time(now_iso, 4)) == _parse("2026-09-11T17:06:10.449910+00:00")
    assert _parse(compute_valid_time(now_iso, 6)) == _parse("2026-09-11T19:06:10.449910+00:00")


def test_ist_example_from_specification():
    """The IST worked example, verified as parsed instants."""
    now_iso = "2026-09-11T17:11:23+05:30"
    for lead, expected in [
        (2, "2026-09-11T19:11:23+05:30"),
        (4, "2026-09-11T21:11:23+05:30"),
        (6, "2026-09-11T23:11:23+05:30"),
    ]:
        assert _parse(compute_valid_time(now_iso, lead)) == _parse(expected)


def test_midnight_rollover_at_plus_six():
    """+6h from 20:15:45.123456Z must land on the NEXT day, same wall-clock
    minute/second, not be clamped to the end of the day."""
    now_iso = "2026-09-11T20:15:45.123456+00:00"
    vt = _parse(compute_valid_time(now_iso, 6))
    assert vt.date() == datetime(2026, 9, 12).date()
    assert (vt - _parse(now_iso)).total_seconds() == 21600
    assert (vt.hour, vt.minute, vt.second, vt.microsecond) == (2, 15, 45, 123456)


def test_timezone_conversion_to_ist_preserves_the_instant():
    """Converting to IST must change the label, never the underlying instant."""
    now_iso = "2026-09-11T13:06:10.449910+00:00"
    vt_utc = _parse(compute_valid_time(now_iso, 6))
    vt_ist = vt_utc.astimezone(IST)
    assert vt_ist == vt_utc, "timezone conversion altered the instant"
    assert vt_ist.utcoffset() == timedelta(hours=5, minutes=30)
    # 19:06:10.449910Z is 00:36:10.449910 IST the following day
    assert (vt_ist.hour, vt_ist.minute, vt_ist.second) == (0, 36, 10)
    assert vt_ist.date() == datetime(2026, 9, 12).date()


def test_horizons_are_mutually_consistent():
    """+4h must be exactly 2h after +2h, and +6h exactly 2h after +4h, all from
    the SAME reference instant."""
    now_iso = "2026-09-11T12:47:31+00:00"
    v2 = _parse(compute_valid_time(now_iso, 2))
    v4 = _parse(compute_valid_time(now_iso, 4))
    v6 = _parse(compute_valid_time(now_iso, 6))
    assert (v4 - v2).total_seconds() == 7200
    assert (v6 - v4).total_seconds() == 7200
    assert (v6 - v2).total_seconds() == 14400


def test_a_floored_reference_would_fail_these_assertions():
    """Negative control: prove these tests actually detect hour-flooring.

    If NOW were silently floored to the hour, the sub-hour components would be
    lost. This asserts that such a value is genuinely different from the exact
    one, so the tests above are meaningful rather than vacuous."""
    exact = _parse("2026-09-11T13:06:10.449910+00:00")
    floored = exact.replace(minute=0, second=0, microsecond=0)
    assert exact != floored
    exact_vt = _parse(compute_valid_time(exact.isoformat(), 2))
    floored_vt = _parse(compute_valid_time(floored.isoformat(), 2))
    assert exact_vt != floored_vt
    # Both are internally consistent (+7200s), which is exactly why the
    # difference must be checked against the REFERENCE, not just the delta.
    assert (exact_vt - exact).total_seconds() == 7200
    assert (floored_vt - floored).total_seconds() == 7200
    assert (exact_vt - floored_vt).total_seconds() == 6 * 60 + 10.449910
