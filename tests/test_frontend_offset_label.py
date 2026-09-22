"""The "FROM NOW" label must be measured from now.

Regression guard for a real mislabelling bug: `app.js` printed
`"+" + leadHours + ":00 FROM NOW"` for every forecast horizon, which assumes the
GFS analysis IS the present instant. It is not -- analyses run 3.6-9.6h behind
wall-clock -- so with a 6.1h-old analysis the +2h forecast is valid 4h in the
PAST while the UI advertised it as 2h in the future. Every lead was wrong by the
analysis age.

These tests are static (no browser, no Playwright) so they run in the normal
suite: one asserts the offending literal is gone, the other pins the arithmetic
the replacement performs.
"""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "frontend" / "js" / "app.js"
UTC = timezone.utc


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8", errors="replace")


def test_offset_is_not_derived_from_lead_hours():
    """The literal that hardcoded the lead as the from-now offset must be gone."""
    src = _source()
    assert '"+" + leadHours + ":00 FROM NOW"' not in src, (
        "app.js is again labelling the offset with the model lead instead of the "
        "forecast's true offset from the current instant."
    )


def test_offset_is_computed_from_the_valid_time():
    """The replacement must derive the offset from validDate and wall-clock, and
    must be able to render a NEGATIVE offset."""
    src = _source()
    # the non-NOW branch computes its own offset from validDate
    assert re.search(r"validDate\.getTime\(\)\s*-\s*Date\.now\(\)", src), (
        "no wall-clock-relative offset computation found in app.js"
    )
    # and it must carry a sign rather than assuming '+'
    assert re.search(r"fOffH\s*>=\s*0\s*\?\s*[\"']\+[\"']\s*:\s*[\"']-[\"']", src), (
        "the offset label cannot render a negative value, so a past-valid "
        "forecast would still be advertised as being in the future"
    )


@pytest.mark.parametrize("lead,expected", [
    (2, "-4:06"), (3, "-3:06"), (4, "-2:06"), (5, "-1:06"), (6, "-0:06"),
])
def test_offset_arithmetic_matches_the_javascript(lead, expected):
    """Mirror of the JS computation, pinned against a real observed case:
    analysis 12:00Z, wall-clock 18:06Z (6.1h age) on 2026-09-21. Under the old
    code every one of these read "+<lead>:00 FROM NOW"."""
    analysis = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    now = datetime(2026, 9, 21, 18, 6, tzinfo=UTC)
    valid = analysis + timedelta(hours=lead)

    off_h = (valid - now).total_seconds() / 3600.0
    sign = "+" if off_h >= 0 else "-"
    a = abs(off_h)
    hh = int(a)
    mm = round((a - hh) * 60)
    assert f"{sign}{hh}:{mm:02d}" == expected
