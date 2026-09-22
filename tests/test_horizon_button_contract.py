"""The horizon buttons and the availability endpoint must agree.

`app.js::refreshWallclockHorizonAvailability` reads each button's horizon out of
its `onclick` with a regex, then looks that horizon up in the
`/api/nowcast/wallclock-horizons` payload. If the markup, the regex or the
payload keys drift apart, the lookup silently misses and every button keeps its
default (available-looking) appearance -- which is the failure mode the endpoint
exists to prevent.

These tests are static plus one API call; no browser required.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "frontend" / "index.html"
APP_JS = ROOT / "frontend" / "js" / "app.js"

# The exact pattern app.js uses to pull the horizon out of a button's onclick.
ONCLICK_RX = re.compile(r"window\.setForecastHorizon\(this,\s*'([^']+)'\s*,\s*(\d+)\)")
BUTTON_RX = re.compile(
    r'<button class="horizon-btn[^>]*onclick="([^"]+)"[^>]*>([^<]+)</button>'
)

client = TestClient(app)


@pytest.fixture(scope="module")
def buttons():
    html = INDEX_HTML.read_text(encoding="utf-8", errors="replace")
    found = BUTTON_RX.findall(html)
    assert found, "no .horizon-btn buttons found in index.html"
    return found


def test_the_four_wallclock_horizons_are_present(buttons):
    hs = sorted(int(ONCLICK_RX.search(oc).group(2)) for oc, _ in buttons
                if ONCLICK_RX.search(oc))
    assert hs == [0, 2, 4, 6], f"expected NOW/+2/+4/+6, got {hs}"


def test_every_button_is_parsable_by_the_app_js_regex(buttons):
    for onclick, label in buttons:
        assert ONCLICK_RX.search(onclick), (
            f"button {label!r} has an onclick app.js cannot parse: {onclick!r}"
        )


def test_app_js_still_uses_this_pattern():
    """If the regex in app.js changes, this file's assumption is stale."""
    src = APP_JS.read_text(encoding="utf-8", errors="replace")
    assert "setForecastHorizon" in src
    assert "wallclock-horizons" in src, (
        "app.js no longer queries the horizon-availability endpoint"
    )


def test_endpoint_has_a_key_for_every_button(buttons):
    payload = client.get("/api/nowcast/wallclock-horizons").json()
    horizons = payload.get("horizons", {})
    assert horizons, "endpoint returned no horizons"
    for onclick, label in buttons:
        m = ONCLICK_RX.search(onclick)
        key = str(int(m.group(2)))
        assert key in horizons, (
            f"button {label!r} maps to horizon key {key!r}, absent from the "
            f"payload keys {sorted(horizons)}"
        )


def test_unavailable_horizons_carry_a_reason():
    payload = client.get("/api/nowcast/wallclock-horizons").json()
    for key, info in payload.get("horizons", {}).items():
        if not info.get("available"):
            assert info.get("reason"), f"horizon {key} unavailable with no reason"
        else:
            # An available horizon must name the lead backing it and its valid time.
            assert info.get("source_lead_hours") is not None or \
                   info.get("source_leads_hours"), f"horizon {key} names no source lead"
            assert info.get("served_valid_time_utc"), f"horizon {key} has no valid time"
