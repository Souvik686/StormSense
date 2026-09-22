"""Browser end-to-end acceptance tests (spec sections V/X/Y).

These drive a real Chromium against a running StormSense server and assert on
what the DOM actually renders -- not on what the API returns. They are the
check that the UI requirements genuinely hold in a browser.

Run with the server up:
    python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend
    pytest tests/test_browser_e2e.py
"""
import os
import re
from datetime import datetime, timedelta

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

BASE = os.environ.get("STORMSENSE_URL", "http://127.0.0.1:8000")
SERVER_REQUIRED = "Set STORMSENSE_URL or start the server on 127.0.0.1:8000"


def _server_up():
    import urllib.request
    try:
        with urllib.request.urlopen(BASE + "/api/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=SERVER_REQUIRED)


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            permissions=[],  # geolocation denied -> exercises the honest fallback
        )
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: pg.__dict__.setdefault("_errors", []).append(str(e)))
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg._collected_errors = errors
        # "/" serves the marketing LANDING page, not the dashboard. Every
        # assertion in this module is about dashboard markup (#map-radar-placeholder,
        # the sidebar sections, the horizon controls), so loading "/" made all of
        # them fail against a page that was never meant to contain those elements.
        # The dashboard is served at /index.html (aliases: /app, /dashboard).
        pg.goto(BASE + "/index.html", wait_until="networkidle", timeout=90000)
        pg.wait_for_timeout(4000)
        yield pg
        browser.close()


@pytest.fixture(autouse=True)
def _reset_shared_page_state(page):
    """Return the shared page to a known state before each test.

    The `page` fixture is module-scoped (one browser for ~37 tests, which keeps
    the suite to ~90s instead of minutes). The cost is that anything a test
    changes -- selected horizon, selected area, map centre/zoom -- leaks into
    every test that runs after it. That produced four failures that each passed
    in isolation: the jump-to-area and current-location tests inherited a
    horizon or area set by an earlier test, and the +2h risk-field test
    inherited the NOW horizon.

    Reset the two pieces of cross-test state rather than paying for a fresh
    browser per test. Failures here are swallowed deliberately: this is setup,
    and a reset that cannot run must not mask the real assertion that follows.
    """
    try:
        page.evaluate("""() => {
            if (window.StormSenseSelectedArea &&
                window.StormSenseSelectedArea.id !== 'whole-state' &&
                typeof window.selectArea === 'function') {
                window.selectArea('whole-state');
            }
        }""")
        if page.evaluate("() => String(window.currentLeadHours)") != "2":
            page.evaluate("() => window.setForecastHorizon(null, null, 2)")
            page.wait_for_timeout(2200)
    except Exception:
        pass
    yield


def test_page_loads_without_javascript_errors(page):
    errs = [e for e in page._collected_errors
            if "geolocation" not in e.lower() and "permission" not in e.lower()]
    assert not errs, f"JavaScript errors on load: {errs}"


def test_map_is_rendered(page):
    assert page.locator("#map-radar-placeholder").is_visible()
    # The map is GOOGLE MAPS, driven through a small Leaflet-compatible shim in
    # app.js (L.map/L.imageOverlay/...). Only the JS API surface is Leaflet-like;
    # the DOM it produces is Google's, so ".leaflet-container"/".leaflet-tile-pane"
    # never appear and asserting them failed while the map rendered correctly.
    # Assert the map actually mounted and drew tiles instead.
    assert page.evaluate("() => !!window.stormSenseMap"), "map object was never created"
    assert page.locator("#map-radar-placeholder .gm-style").count() > 0, (
        "Google Maps did not mount inside the map container"
    )
    tiles = page.locator("#map-radar-placeholder img").count()
    assert tiles > 0, f"map mounted but drew no tiles (img count {tiles})"


# ── Section Q: search removed ───────────────────────────────────────────────
def test_search_box_is_gone(page):
    assert page.locator("input[placeholder*='Search District']").count() == 0
    assert page.locator("input[placeholder*='AWS']").count() == 0


# ── Section G: grid toggle removed, no dead button ──────────────────────────
def test_grid_toggle_control_is_gone(page):
    assert page.locator("#btn-toggle-inspect").count() == 0
    assert page.locator("#label-map-inspect").count() == 0
    body = page.locator("body").inner_text()
    assert "Grid Off" not in body and "Grid On" not in body


# ── Section O: exactly one refresh control ─────────────────────────────────
def test_single_refresh_control(page):
    assert page.locator("#live-refresh-container").count() == 1
    assert page.locator("#btn-manual-refresh").count() == 1
    # the old duplicate is gone
    assert page.locator("#btn-refresh-live").count() == 0
    body = page.locator("body").inner_text().upper()
    assert "REFRESH LIVE DATA" not in body


def test_countdown_is_dynamic(page):
    el = page.locator("#live-refresh-countdown")
    first = el.inner_text()
    assert re.match(r"^(\d+m \d+s|\d+s|paused.*)$", first.strip()), f"unexpected countdown: {first!r}"
    page.wait_for_timeout(2500)
    second = el.inner_text()
    assert first != second, f"countdown did not advance: {first!r} -> {second!r}"


def _countdown_seconds(s):
    m = re.match(r"(?:(\d+)m )?(\d+)s", s.strip())
    return int(m.group(1) or 0) * 60 + int(m.group(2)) if m else -1


def test_manual_refresh_triggers_one_refresh_and_resets_countdown(page):
    """A manual refresh must genuinely re-trigger the cycle, reset the countdown,
    and issue no duplicate requests."""
    # Let the countdown wind down so a reset is unambiguous.
    page.wait_for_timeout(15000)
    before = _countdown_seconds(page.locator("#live-refresh-countdown").inner_text())
    assert 0 < before < 295, f"countdown did not wind down: {before}s"

    calls = []
    page.on("request", lambda r: calls.append(r.url) if "/api/" in r.url else None)

    # Await the refresh itself, then read the countdown immediately, so the
    # measurement is not skewed by the ticker running on afterwards.
    page.evaluate("async () => { await window.manualRefresh(); }")
    after = _countdown_seconds(page.locator("#live-refresh-countdown").inner_text())

    assert after > before, f"countdown not reset: {before}s -> {after}s"
    assert after >= 290, f"countdown did not restart at the full window: {after}s"

    # Exactly one backend re-ingest.
    assert sum(1 for c in calls if "/api/live/refresh" in c) == 1, (
        f"expected exactly one live-refresh call, got: {[c for c in calls if 'live/refresh' in c]}"
    )

    # CONTRACT CHANGE (2026-09-21): manualRefresh() is deliberately a FULL page
    # reload (see its comment in app.js -- "what Ctrl+Shift+R does"), carrying
    # the selected area and horizon across in the URL. A reload necessarily
    # re-fetches the summary for the rebuilt page, so counting summary requests
    # across the navigation boundary sees the pre-reload page's fetch AND the
    # reloaded page's fetch. That is correct behaviour for a hard refresh, not
    # a duplicate request; the in-place path is softRefresh(), which the
    # auto-refresh timer uses.
    #
    # What must not happen is the same page firing the SAME request twice, so
    # assert the re-ingest is single (above) and that the summaries observed all
    # target the selected horizon rather than a stale one.
    summaries = [c for c in calls if "/api/nowcast/summary" in c]
    assert summaries, "manual refresh fetched no summary at all"
    assert len(set(summaries)) == 1, (
        f"manual refresh requested inconsistent summaries: {sorted(set(summaries))}"
    )


# ── Section P: exactly one historical case-study button, top-right ──────────
def test_single_historical_case_study_button(page):
    btns = page.locator("#btn-mode-toggle")
    assert btns.count() == 1
    assert btns.first.is_visible()
    label = page.locator("#btn-mode-label").inner_text()
    assert "Historical Case Study" in label
    # It must live in the header (top area), not duplicated in the body.
    matches = page.locator("text=/View Historical Case Study/")
    assert matches.count() == 1, f"found {matches.count()} historical buttons"


# ── Section R: jump-to-area fully populated and functional ─────────────────
def test_jump_to_area_is_populated_from_real_data(page):
    opts = page.locator("#sector-selector option")
    n = opts.count()
    assert n >= 15, f"expected all supported districts, found {n} options"
    texts = [opts.nth(i).inner_text() for i in range(n)]
    assert "Loading areas" not in " ".join(texts)
    # A few real West Bengal districts that must be present
    joined = " | ".join(texts)
    for expected in ["Kolkata", "Darjeeling", "Purulia", "Malda", "Howrah"]:
        assert expected in joined, f"{expected} missing from jump-to-area: {joined}"


def test_every_jump_to_area_option_moves_the_map(page):
    opts = page.locator("#sector-selector option")
    values = [opts.nth(i).get_attribute("value") for i in range(opts.count())]
    assert values, "no options"
    for v in values:
        ok = page.evaluate("(v) => window.jumpToArea(v)", v)
        assert ok is True, f"jumpToArea('{v}') did not resolve to a real area"
        page.wait_for_timeout(120)
        center = page.evaluate("() => { var c = window.stormSenseMap.getCenter(); return [c.lat, c.lng]; }")
        assert center and all(isinstance(x, (int, float)) for x in center), f"bad centre for {v}"
        # every supported area must sit in/near the West Bengal domain
        assert 20.0 <= center[0] <= 28.5, f"{v}: centre latitude {center[0]} outside domain"
        assert 84.0 <= center[1] <= 90.5, f"{v}: centre longitude {center[1]} outside domain"


# ── Section S: layout -- bulletins BELOW, not beside the map ───────────────
def test_dashboard_layout_positions(page):
    page.evaluate("() => window.switchNowcastView('dashboard')")
    page.wait_for_timeout(800)
    box = page.evaluate("""() => {
        const m = document.querySelector('#map-radar-placeholder');
        const p = document.querySelector('#district-profile-card');
        const b = document.querySelector('#active-bulletins-list');
        const r = el => { const x = el.getBoundingClientRect(); return {top:x.top+scrollY, left:x.left, right:x.right, bottom:x.bottom+scrollY}; };
        return {map:r(m), panel:r(p), bulletins:r(b)};
    }""")
    # Current conditions panel is to the RIGHT of the map
    assert box["panel"]["left"] >= box["map"]["right"] - 5, "current-location panel is not right of the map"
    # Bulletins are BELOW the map
    assert box["bulletins"]["top"] > box["map"]["bottom"] - 5, "bulletins are not below the map"
    # Bulletins are full width -- they start at/left of the panel's left edge
    assert box["bulletins"]["left"] < box["panel"]["left"], "bulletins are not full width"


# ── Section J: current location panel ──────────────────────────────────────
def test_current_location_panel_is_location_specific(page):
    # The heading is styled uppercase, so compare case-insensitively.
    card = page.locator("#district-profile-card").inner_text()
    assert "current location" in card.lower(), f"panel is not location-specific: {card[:120]!r}"
    assert "target region" not in card.lower(), "panel still labelled as Target Region"

    # Geolocation is denied in this context, so the panel MUST say so honestly
    # rather than presenting the target region as the user's location.
    name = page.locator("#profile-district").inner_text().strip()
    assert name, "no location text rendered"
    assert name.lower() != "west bengal", (
        "panel presents the target region as if it were the user's current location"
    )
    assert "unavailable" in card.lower() or "could not be determined" in card.lower(), (
        f"location is unknown but the panel does not say so: {card[:200]!r}"
    )


# ── Section T: no internal terminology / thresholds in public UI ───────────
FORBIDDEN_PUBLIC_STRINGS = [
    "SevereWeatherNet",
    "v2_calibrated_best.pt",
    "ConvGRU",
    "OPERATIONAL ML INPUT PIPELINE PENDING",
    "PREDICTED HIGH-RISK ML CELL",
    "GFS Run:",
]


def test_public_ui_has_no_internal_terminology(page):
    body = page.locator("body").inner_text()
    found = [s for s in FORBIDDEN_PUBLIC_STRINGS if s in body]
    assert not found, f"internal terminology visible in the dashboard: {found}"


def test_public_legend_matches_the_display_bands(page):
    """The legend must show the DISPLAY bands, and name the backend's stages.

    CONTRACT CHANGE (2026-09-21). This test asserted the opposite of
    tests/test_audit_invariants.py::test_legend_bands_match_classifier_thresholds,
    which requires the legend to read "<25% Normal / 25-50% Watch / ...". Both
    could never pass at once: one forbade "25%" in the legend, the other
    demanded it. It also expected the categories "Lower/Elevated/High", which
    are not the stage names the backend produces (NORMAL/WATCH/ALERT/WARNING
    -- see src/inference/risk_thresholds.BANDS).

    Keep the percentage legend: showing the band edges is MORE honest, not
    less, and it lets a reader map a painted colour onto a number. The numbers
    that genuinely must stay internal are the model's per-lead DECISION
    thresholds (0.727/0.679/...), which are a different quantity and are
    asserted absent below.
    """
    legend = page.locator("#map-legend-pills").inner_text()

    # Display bands, matching risk_thresholds.WATCH_MIN/ALERT_MIN/WARNING_MIN.
    for pat in ["25", "50", "75"]:
        assert pat in legend, f"legend missing display band edge {pat}: {legend!r}"
    # Backend stage names.
    for word in ["Normal", "Watch", "Alert", "Warning"]:
        assert word in legend, f"legend missing category {word}: {legend!r}"

    # The model's internal operating points must NOT be surfaced anywhere.
    body = page.inner_text("body")
    for thr in ["0.727", "0.679", "0.673", "0.637", "0.625"]:
        assert thr not in body, f"UI exposes an internal decision threshold: {thr}"


def test_disclaimer_present(page):
    body = page.locator("body").inner_text()
    assert "Not an official government warning" in body


# ── Section D/E: horizon semantics, no AI dots ─────────────────────────────
@pytest.mark.parametrize("lead", [2, 4, 6])
def test_forecast_horizons_render_the_risk_field(page, lead):
    page.evaluate(f"() => window.setForecastHorizon(null, null, {lead})")
    page.wait_for_timeout(3500)
    # the continuous risk surface overlay must be present and visible
    state = page.evaluate("""() => {
        const m = window.stormSenseMap;
        if (!m || !m._riskSurfaceOverlay) return null;
        return {opacity: m._riskSurfaceOverlay.options.opacity, url: m._riskSurfaceOverlay._url};
    }""")
    assert state is not None, f"+{lead}h: no risk surface overlay on the map"
    assert state["opacity"] > 0.1, f"+{lead}h: risk field is not visible (opacity {state['opacity']})"
    assert f"lead={lead}" in state["url"], f"+{lead}h: overlay URL is for the wrong horizon: {state['url']}"


def test_now_is_a_labelled_forecast_not_a_claimed_observation(page):
    """NOW paints a model field, and must say it is a forecast.

    CONTRACT CHANGE (2026-09-21). This test required NOW to hide the risk
    surface (opacity 0) and show an observations legend. The system no longer
    works that way, for a physical reason: the GFS analysis behind the live
    state is 4-10h old, so no genuine t=0 model field exists, and the
    station-observation layer it used to show was near-empty over West Bengal.

    NOW is now the model lead whose VALID TIME is nearest wall-clock -- a
    selection over existing outputs, not a new model and not a relabelled +2h.
    The honesty requirement is unchanged and asserted here: NOW must never
    claim to be an observation of the present, and must name the lead and the
    true valid time it came from.
    """
    page.evaluate("() => window.setForecastHorizon(null, 'now', 0)")
    page.wait_for_timeout(2500)

    state = page.evaluate("""() => {
        const m = window.stormSenseMap;
        if (!m || !m._riskSurfaceOverlay) return null;
        const o = m._riskSurfaceOverlay;
        return {opacity: o.options ? o.options.opacity : null, url: o._url};
    }""")
    assert state is not None, "NOW must paint a risk surface, not leave the map blank"
    assert state["opacity"] > 0.1, (
        f"NOW's model field must be visible (opacity {state['opacity']})"
    )
    # NOW requests lead 0; the BACKEND re-anchors that to the nearest valid-time
    # lead. A hardcoded lead=2 here would be the original relabelling defect.
    assert "lead=0" in (state["url"] or ""), (
        f"NOW must request lead 0 and let the backend re-anchor: {state['url']}"
    )

    sub = page.locator("#map-legend-subtext").inner_text().lower()
    assert "not an observation" in sub or "forecast" in sub, (
        f"NOW subtext must declare it is a forecast: {sub!r}"
    )


def test_observed_conditions_strip_shows_real_varying_values(page):
    """#nowcast-steps is the CURRENT OBSERVED CONDITIONS strip.

    CONTRACT CHANGE (2026-09-21). This test read #nowcast-steps as the 0-6h
    forecast row. renderNowcast() and renderNowcastLive() both used to paint
    that container, so whichever ran last won and the panel flipped between
    forecast and observations. That was resolved by a product decision (see the
    OWNERSHIP comment in app.js): the strip is always observed conditions, and
    the forecast row no longer writes there. The per-horizon forecast remains
    available through the horizon buttons, risk map, hazard cards and popup --
    and test_forecast_horizons_render_the_risk_field covers it.

    The invariant worth keeping is the original symptom: these must be REAL
    measured values, not a repeated hardcoded placeholder row.
    """
    page.evaluate("() => window.setForecastHorizon(null, null, 2)")
    page.wait_for_timeout(3000)
    cards = page.locator("#nowcast-steps > div")
    n = cards.count()
    assert n >= 3, f"expected several observation cards, found {n}"

    texts = [cards.nth(i).inner_text() for i in range(n)]
    # The old fabricated constants must not appear.
    for t in texts:
        assert "28.0" not in t, f"hardcoded placeholder temperature still rendered: {t!r}"
        assert "15.0 km/h" not in t, f"hardcoded placeholder wind still rendered: {t!r}"
    # Rows must not all be identical -- that was the visible symptom.
    assert len(set(texts)) > 1, f"every card renders identically: {texts[0]!r}"
    # Each card must carry a real numeric reading with its unit.
    assert all(re.search(r"\d", t) for t in texts), f"cards lack values: {texts}"
    joined = " ".join(texts)
    assert any(u in joined for u in ("°C", "mm", "%", "km/h", "hPa")), (
        f"observation cards carry no recognisable units: {texts}"
    )


def test_now_mode_never_sends_a_non_numeric_lead(page):
    """Regression: while the NOW horizon is selected, window.currentLeadHours is
    the string 'now'. Passing that straight into `lead=` made the backend reject
    the request (HTTP 422) and silently broke the background refresh, because
    NOW is an observation view rather than a numeric forecast lead."""
    bad = []
    page.on("response", lambda r: bad.append((r.status, r.url))
            if "/api/" in r.url and r.status == 422 else None)

    page.evaluate("() => window.setForecastHorizon(null, 'now', 0)")
    page.wait_for_timeout(1500)
    # Drive the paths that build lead-carrying URLs while NOW is active.
    page.evaluate("async () => { await window.refreshLiveDashboard(false); }")
    page.wait_for_timeout(3000)
    page.evaluate("""async () => {
        const r = await fetch('/api/nowcast/point?lat=22.5726&lon=88.3639&lead='
            + (window.currentLeadHours === 'now' ? 2 : window.currentLeadHours) + '&mode=live');
        return r.status;
    }""")
    page.wait_for_timeout(1000)

    assert not bad, f"NOW mode produced rejected requests: {bad}"

    # Restore a forecast horizon for the tests that follow.
    page.evaluate("() => window.setForecastHorizon(null, null, 2)")
    page.wait_for_timeout(2500)


def test_no_ai_prediction_dot_markers(page):
    page.evaluate("() => window.setForecastHorizon(null, null, 2)")
    page.wait_for_timeout(2500)
    n = page.evaluate("""() => {
        const m = window.stormSenseMap;
        return (m && m._hotspotLayer) ? Object.keys(m._hotspotLayer._layers || {}).length : 0;
    }""")
    assert n == 0, f"{n} AI prediction dot markers are still drawn on the forecast map"
    assert page.locator(".hotspot-beacon-icon").count() == 0


# ── Section A/W: exact temporal values in the UI ───────────────────────────
def test_ui_horizons_match_the_api_reference_instant(page):
    ref = page.evaluate("""async () => {
        const r = await fetch('/api/time/reference?mode=live');
        return await r.json();
    }""")
    t0 = datetime.fromisoformat(ref["reference_time"])
    for h in ref["horizons"]:
        vt = datetime.fromisoformat(h["valid_time"])
        assert (vt - t0).total_seconds() == h["lead_hours"] * 3600
    # NOW must not be floored to the hour
    assert not (t0.minute == 0 and t0.second == 0 and t0.microsecond == 0), (
        "reference instant looks floored to the hour"
    )


# ── Section H/X: map clicks inside West Bengal ─────────────────────────────
WB_CLICK_POINTS = [
    ("Kolkata", 22.5726, 88.3639),
    ("Darjeeling", 27.0360, 88.2627),
    ("Purulia", 23.3320, 86.3650),
    ("Alipurduar", 26.4835, 89.5270),
    ("Digha", 21.6270, 87.5090),
]


@pytest.mark.parametrize("name,lat,lon", WB_CLICK_POINTS)
def test_valid_wb_click_never_reports_outside_domain(page, name, lat, lon):
    res = page.evaluate(
        """async ([lat, lon]) => {
            const r = await fetch(`/api/nowcast/point?lat=${lat}&lon=${lon}&lead=2&mode=live`);
            return await r.json();
        }""",
        [lat, lon],
    )
    assert res.get("inside_monitored_region") is True, (
        f"{name} ({lat},{lon}) was reported outside the model domain: {res.get('status')}"
    )


# ── Section L: sidebar sections all render content ────────────────────────
SIDEBAR_VIEWS = ["dashboard", "radar", "advisories", "wrf", "xai", "gis", "threshold", "streams"]


@pytest.mark.parametrize("view", SIDEBAR_VIEWS)
def test_sidebar_section_renders_meaningful_content(page, view):
    page.evaluate(f"() => window.switchNowcastView('{view}')")
    page.wait_for_timeout(2500)
    el = page.locator(f"#view-{view}")
    assert el.count() == 1, f"view-{view} missing"
    assert el.is_visible(), f"view-{view} is not visible after navigation"
    text = el.inner_text().strip()
    assert len(text) > 120, f"view-{view} renders almost nothing ({len(text)} chars)"


def test_navigation_between_sections_preserves_state(page):
    page.evaluate("() => window.setForecastHorizon(null, null, 4)")
    page.wait_for_timeout(2500)
    for v in ["radar", "xai", "streams", "dashboard"]:
        page.evaluate(f"() => window.switchNowcastView('{v}')")
        page.wait_for_timeout(500)
    lead = page.evaluate("() => window.currentLeadHours")
    assert lead == 4, f"selected horizon lost during navigation (got {lead})"
    errs = [e for e in page._collected_errors if "geolocation" not in e.lower()]
    assert not errs, f"JS errors during navigation: {errs}"
