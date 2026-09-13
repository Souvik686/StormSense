# -*- coding: utf-8 -*-
"""Regression tests for the invariants fixed in the final end-to-end audit.

Each test here protects a defect that was actually found in the running
application, so a regression re-breaks a test rather than only the UI.

Groups:
  1. Source encoding integrity (no U+FFFD / mojibake anywhere).
  2. Temporal semantics (+2/+4/+6 are exact offsets from NOW).
  3. Live thermodynamic provenance (real values, honest unavailability).
  4. Map integrity (no RainViewer on the nowcasting map, no fake markers).
  5. Horizon label binding (no hardcoded "+2h" on dynamic elements).
  6. No fabricated weather fallbacks in the frontend.
"""
from __future__ import annotations

import io
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_JS = os.path.join(PROJECT_ROOT, "frontend", "js", "app.js")
INDEX_HTML = os.path.join(PROJECT_ROOT, "frontend", "index.html")


def _read(path: str) -> str:
    with io.open(path, encoding="utf-8-sig") as f:
        return f.read()


def _strip_js_comments(src: str) -> str:
    """Remove // and /* */ comments.

    The audit fixes are documented in comments that necessarily quote the old
    bad values ("Station Anemometer", the risk hex codes). Assertions must run
    against executable code only, or the documentation trips its own test.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    src = re.sub(r"^(\s*\*.*)$", "", src, flags=re.M)
    return src


def _strip_html_comments(src: str) -> str:
    return re.sub(r"<!--.*?-->", "", src, flags=re.S)


# ---------------------------------------------------------------------------
# 1. Encoding integrity
# ---------------------------------------------------------------------------

FRONTEND_SOURCES = [
    os.path.join(PROJECT_ROOT, "frontend", "js", "app.js"),
    os.path.join(PROJECT_ROOT, "frontend", "index.html"),
    os.path.join(PROJECT_ROOT, "frontend", "css", "styles.css"),
]


@pytest.mark.parametrize("path", FRONTEND_SOURCES)
def test_no_replacement_characters(path):
    """No U+FFFD anywhere.

    A batch edit once rewrote app.js through a cp1252 assumption, destroying
    every non-ASCII character (90 sites). The damage reached data-bearing
    strings -- degree signs in "degC" and the em-dash no-data placeholder --
    not just decorative separators.
    """
    raw = open(path, "rb").read()
    assert b"\xef\xbf\xbd" not in raw, f"{path} contains U+FFFD replacement characters"


@pytest.mark.parametrize("path", FRONTEND_SOURCES)
def test_sources_are_valid_utf8(path):
    raw = open(path, "rb").read()
    raw.decode("utf-8")  # raises on invalid UTF-8


def test_expected_unicode_glyphs_present():
    """The characters that were destroyed are actually back."""
    js = _read(APP_JS)
    assert "°C" in js, "degree sign missing from temperature formatting"
    assert "·" in js, "middle dot separator missing"
    assert "—" in js, "em dash (no-data placeholder) missing"
    assert "→" in js, "arrow glyph missing (was mojibake '?')"


def test_no_mojibake_arrow_in_labels():
    """'Case Study ?' was a destroyed arrow, not a real question."""
    js = _read(APP_JS)
    assert "Case Study ?" not in js
    assert "Attributions ?" not in js


# ---------------------------------------------------------------------------
# 2. Temporal semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 9, 11, 22, 2, 37, tzinfo=timezone.utc),   # non-zero min/sec
        datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc),  # hour+date rollover
        datetime(2026, 12, 31, 23, 30, 0, tzinfo=timezone.utc),  # year rollover
        datetime(2026, 2, 28, 18, 45, 12, tzinfo=timezone.utc),  # month rollover
    ],
)
@pytest.mark.parametrize("lead_hours,expected_seconds", [(2, 7200), (4, 14400), (6, 21600)])
def test_valid_time_is_exact_offset_from_now(now, lead_hours, expected_seconds):
    """+2h/+4h/+6h must be EXACTLY now + 7200/14400/21600 seconds.

    Never floored to the hour, never snapped to a GFS cycle.
    """
    from src.inference.risk_map import compute_valid_time

    valid = compute_valid_time(now.isoformat(), lead_hours)
    valid_dt = datetime.fromisoformat(str(valid).replace("Z", "+00:00"))

    delta = (valid_dt - now).total_seconds()
    assert delta == expected_seconds, (
        f"+{lead_hours}h resolved to {delta}s after NOW, expected {expected_seconds}s"
    )
    # Sub-hour precision must survive: minutes/seconds are preserved.
    assert valid_dt.minute == now.minute
    assert valid_dt.second == now.second


def test_ist_conversion_is_utc_plus_5_30():
    """IST is the primary user-facing zone and is exactly UTC+5:30."""
    utc = datetime(2026, 9, 11, 18, 32, 0, tzinfo=timezone.utc)
    ist = utc + timedelta(hours=5, minutes=30)
    assert (ist - utc).total_seconds() == 19800
    assert ist.strftime("%d %b %Y %H:%M") == "12 Sep 2026 00:02"


# ---------------------------------------------------------------------------
# 3. Live thermodynamic provenance
# ---------------------------------------------------------------------------

def test_live_thermo_never_borrows_historical_values():
    """Live and historical thermodynamic state are separate attributes.

    Reporting the frozen ERA5 case-study sounding as a live observation would
    be a data-integrity violation, so the two must not share storage.
    """
    src = _read(os.path.join(PROJECT_ROOT, "src", "inference", "nowcast_service.py"))
    assert "self.live_thermo" in src
    assert "self.current_thermo" in src
    # The live computation must not read the historical attribute.
    start = src.index("def _compute_live_thermo")
    end = src.index("def ", src.index("return {", start))
    body = src[start:end]
    assert "current_thermo" not in body, (
        "_compute_live_thermo must not read the historical case-study state"
    )


def test_live_thermo_reports_shear_layer_truthfully():
    """Live GFS carries wind at 1000/850/700 hPa only.

    So a live shear value is a 1000-700 hPa (~0-3 km) bulk shear and must not
    be presented as 0-6 km shear.
    """
    src = _read(os.path.join(PROJECT_ROOT, "src", "inference", "nowcast_service.py"))
    assert "bulk_shear_1000_700hpa_mps" in src
    assert "bulk_shear_0_6km_status" in src, "must state why 0-6 km shear is absent"


def test_frontend_does_not_claim_station_anemometer():
    """The UI printed 'Station Anemometer' for shear.

    StormSense reads no station anemometer; the value came from gridded
    analysis. A source label must name the feed the number actually came from.
    """
    js = _strip_js_comments(_read(APP_JS))
    assert "Station Anemometer" not in js


def test_thermo_source_label_is_runtime_bound():
    """The 'Source:' line must be written from the payload that supplied the
    values, not hardcoded in markup."""
    html = _read(INDEX_HTML)
    assert 'id="thermo-source-label"' in html
    assert "Source: 0.25° atmospheric analysis" not in html, (
        "source label must not be hardcoded independently of the data"
    )


# ---------------------------------------------------------------------------
# 4. Map integrity
# ---------------------------------------------------------------------------

def test_rainviewer_not_loaded_on_nowcasting_map():
    """initMap() builds the Interactive Nowcasting Map and must not attach the
    RainViewer mosaic. Radar belongs to the Radar & Satellite Feeds view."""
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("function initMap(")
    end = js.index("function loadRainViewerRadar(")
    init_map_body = js[start:end]
    assert "loadRainViewerRadar" not in init_map_body, (
        "initMap must not load the RainViewer radar layer"
    )


def test_rainviewer_guarded_to_radar_container():
    """A structural guard, so no future call site can reattach radar to the
    nowcasting map."""
    js = _read(APP_JS)
    assert "function isRadarFeedsMap(" in js
    assert 'getContainer().id === "radar-map-container"' in js
    guard_pos = js.index("async function loadRainViewerRadar(")
    body = js[guard_pos:guard_pos + 600]
    assert "isRadarFeedsMap(map)" in body, "radar loader must refuse foreign maps"


def test_no_fabricated_station_or_radar_markers():
    """The red 'IMD Kolkata DWR' marker and the green 'Live Telemetry Station'
    marker were both hardcoded points with no backing data source; the station
    popup additionally invented weather values."""
    js = _strip_js_comments(_read(APP_JS))
    assert "function renderLiveStationMarker" not in js
    assert "LIVE STATION TELEMETRY" not in js
    assert "Doppler Weather Radar" not in js
    assert "radar-station-icon" not in js


def test_live_location_marker_colour_is_not_a_risk_colour():
    """The live location marker must not reuse any risk-ramp colour, or it
    reads as a severity level."""
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("function renderLiveLocationMarker(")
    body = js[start:start + 2000]
    risk_colours = ["#10b981", "#f59e0b", "#f97316", "#ef4444", "#22c55e"]
    for colour in risk_colours:
        assert colour not in body, f"live location marker must not use risk colour {colour}"
    assert "#a855f7" in body, "live location marker should use the dedicated violet"


# ---------------------------------------------------------------------------
# 5. Horizon label binding
# ---------------------------------------------------------------------------

def test_benchmark_horizon_labels_are_dynamically_bound():
    """Selecting +4h/+6h changed the metrics but left the headings reading
    '+2h Horizon' and 'Lead Time: +2 Hours', so the desk misreported which
    horizon the numbers belonged to."""
    js = _read(APP_JS)
    start = js.index("function renderBenchmarkCards(")
    end = js.index("function renderEvolutionTimeline(")
    body = js[start:end]
    assert 'setText("benchmark-selected-lead-label"' in body
    assert 'setText("benchmark-lead-badge"' in body
    assert '"+" + lead + "h Horizon' in body, "label must interpolate the selected lead"


def test_no_dead_benchmark_element_write():
    """renderBenchmarkCards wrote to 'bm-lead-display-tag', an element that
    does not exist -- a dead write that masked the stale-label bug."""
    js = _read(APP_JS)
    html = _read(INDEX_HTML)
    if "bm-lead-display-tag" in js:
        assert 'id="bm-lead-display-tag"' in html, (
            "app.js writes to bm-lead-display-tag but no such element exists"
        )


def test_static_verification_table_keeps_per_horizon_rows():
    """The +2h..+6h verification table rows are genuinely static descriptions of
    their own horizon and must NOT be rebound to the selection."""
    html = _read(INDEX_HTML)
    for label in ("+2h Horizon", "+3h Horizon", "+4h Horizon", "+5h Horizon", "+6h Horizon"):
        assert label in html, f"static verification table lost its {label} row"


# ---------------------------------------------------------------------------
# 6. No fabricated weather fallbacks
# ---------------------------------------------------------------------------

FABRICATED_FALLBACKS = [
    '"32.0"', "'32.0'", '"36.0"', '"16.0"', '"6.5"',
    '"1006"', '"70"',
]


def test_no_fabricated_observation_fallbacks():
    """Missing observations must render as an explicit no-data marker.

    These literals were fallback values that appeared identical to real
    measurements when the feed was absent.
    """
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("function renderNowcastLive(")
    end = js.index("function renderXaiLive(")
    body = js[start:end]
    for literal in FABRICATED_FALLBACKS:
        assert literal not in body, (
            f"renderNowcastLive still falls back to fabricated value {literal}"
        )


def test_rainfall_never_defaults_to_zero():
    """Absent rainfall data is not a measurement of zero rain."""
    js = _read(APP_JS)
    start = js.index("function paintDashboardLive(")
    end = js.index("function renderNowcastLive(")
    body = js[start:end]
    assert 'rainfall_1h_mm != null ? Number(obs.rainfall_1h_mm).toFixed(1) : "0.0"' not in body


# ---------------------------------------------------------------------------
# 7. Risk legend consistency
# ---------------------------------------------------------------------------

def test_legend_bands_match_classifier_thresholds():
    """The legend's percentage bands must be the classifier's own edges.

    NowcastService._level_for_prob uses 0.25 / 0.50 / 0.75, so the public
    legend must read <25 / 25-50 / 50-75 / >=75 or the map and legend disagree.
    """
    svc_src = _read(os.path.join(PROJECT_ROOT, "src", "inference", "nowcast_service.py"))
    start = svc_src.index("def _level_for_prob")
    body = svc_src[start:start + 400]
    assert "0.75" in body and "0.50" in body and "0.25" in body

    for source in (_read(APP_JS), _read(INDEX_HTML)):
        assert "25% Normal" in source or "&lt;25% Normal" in source
        assert "25–50% Watch" in source
        assert "50–75% Alert" in source
        assert "≥75% Warning" in source


# ---------------------------------------------------------------------------
# 8. Current location semantics
# ---------------------------------------------------------------------------

def test_location_meters_bound_to_point_forecast():
    """The Current Location risk meters must describe the user's own
    coordinates, not the state-wide aggregate (which attributed the peak
    district's risk to the user's position)."""
    js = _read(APP_JS)
    assert "function applyPointRiskMeters(" in js
    assert "applyPointRiskMeters(d)" in js, "point forecast must feed the meters"

    # paintForecastCards is state-wide and must no longer write the meters.
    start = js.index("function paintForecastCards(")
    end = js.index("function applyPointRiskMeters(")
    body = js[start:end]
    assert 'setText("meter-ts-label"' not in body, (
        "state-wide painter must not write the per-location meters"
    )


def test_bulletins_and_xai_are_side_by_side():
    """Layout requirement: bulletins LEFT, XAI RIGHT, stacking on small
    screens."""
    html = _strip_html_comments(_read(INDEX_HTML))
    assert "lg:grid-cols-2" in html
    bulletins = html.index("Active Meteorological Bulletins")
    xai = html.index("Explainable AI (XAI) Factor Attribution")
    grid = html.rindex("lg:grid-cols-2", 0, bulletins)
    assert grid < bulletins < xai, "bulletins must precede XAI inside a 2-column grid"


def test_now_horizon_paints_no_forecast_surface():
    """NOW must not render the +2h predicted risk field.

    renderContinuousRiskSurface() used to coerce the 'now' horizon to lead=2
    through apiLeadHours(), so selecting NOW painted the +2h PREDICTION under a
    legend reading "CURRENT OBSERVATIONS" -- future model output presented as a
    present-tense observation.
    """
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("function renderContinuousRiskSurface(")
    end = js.index("function renderHotspotBeacons(")
    body = js[start:end]
    assert "leadHours === 'now'" in body, (
        "renderContinuousRiskSurface must special-case the 'now' horizon"
    )
    assert "removeRiskSurface" in body, "NOW must remove the forecast surface"
    # NOW draws the OBSERVATION layer instead of the model field.
    assert "renderObservationSurface" in body, (
        "NOW must render the current-observation layer, not the forecast field"
    )
    # The 'now' guard must come BEFORE any fallback that coerces to a lead.
    assert body.index("leadHours === 'now'") < body.index("apiLeadHours()"), (
        "the 'now' guard must precede the numeric-lead fallback"
    )
    # And a forecast horizon must drop the observation layer, so a current field
    # can never sit underneath a prediction.
    assert "removeObservationSurface" in body


def test_now_legend_discloses_interpolation():
    """NOW renders a real observation field, and must say how it was made.

    The field comes from scattered station readings, so values BETWEEN stations
    are interpolated rather than measured. Showing a smooth continuous surface
    without disclosing that would imply a gridded observed product the provider
    does not supply. (An earlier revision of this test asserted NOW had no field
    at all; that was correct only while no genuine observation source was
    wired -- the honesty requirement is disclosure, not absence.)
    """
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("function applyHorizonLegend(")
    end = js.index("function pointErrorHtml(")
    body = js[start:end]
    assert "interpolated" in body.lower(), (
        "NOW legend must state that between-station values are interpolated"
    )
    assert "not measured" in body.lower(), (
        "NOW legend must distinguish interpolated values from measurements"
    )
    # It must still refuse to call itself a forecast.
    assert "not an AI forecast" in body or "not a forecast" in body.lower()


def test_now_removes_surface_rather_than_hiding_it():
    """An opacity-0 overlay is still attached and can be repainted.

    The NOW branch of setForecastHorizon() now delegates layer choice to
    renderContinuousRiskSurface(), which is mode-aware (live NOW -> observations,
    historical NOW -> the case study's t=0 analysis field). The INVARIANT is
    unchanged and asserted end-to-end below: selecting NOW must REMOVE the
    forecast overlay, never merely hide it.
    """
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("window.setForecastHorizon = function")
    body = js[start:start + 1800]
    assert "_riskSurfaceOverlay.setOpacity(0)" not in body, (
        "NOW must remove the forecast overlay, not merely make it transparent"
    )
    # NOW must route through the single mode-aware renderer, which removes the
    # forecast layer, rather than picking a layer itself.
    assert "renderContinuousRiskSurface(window.stormSenseMap, 'now')" in body, (
        "The NOW branch must delegate to renderContinuousRiskSurface so that "
        "historical NOW gets the t=0 analysis field instead of a blank map."
    )

    # ...and that renderer must REMOVE the overlay on the NOW path.
    rstart = js.index("function renderContinuousRiskSurface(")
    rbody = js[rstart:rstart + 1600]
    now_branch = rbody[rbody.index("leadHours === 'now'"):]
    assert "removeRiskSurface(mapInstance)" in now_branch, (
        "renderContinuousRiskSurface's NOW branch must remove the forecast layer"
    )
    assert "setOpacity(0)" not in now_branch

    # removeRiskSurface must detach the layer, not hide it.
    dstart = js.index("function removeRiskSurface(")
    dbody = js[dstart:dstart + 300]
    assert "removeLayer" in dbody and "= null" in dbody, (
        "removeRiskSurface must detach the overlay and clear the reference"
    )


def test_refresh_does_not_relabel_now_as_a_forecast():
    """updateDynamicTimes() runs on every live refresh with a coerced numeric
    lead. It used to stamp "+2h" over the map legend while NOW was selected, so
    the map correctly showed no forecast field while the badge claimed one."""
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("function updateDynamicTimes(")
    end = js.index("function formatIstClock(")
    body = js[start:end]
    assert 'window.currentLeadHours !== "now"' in body, (
        "updateDynamicTimes must not overwrite the legend tag while NOW is selected"
    )


def test_forecast_strip_badge_is_owned_by_its_renderer():
    """The 0-6h strip is painted by two different renderers into one container.

    renderNowcast() paints MODEL PREDICTIONS with future valid times;
    renderNowcastLive() paints OBSERVED surface values. The badge used to be
    set by the caller, so the live path left forecast cards sitting under a
    "LIVE OBSERVATION" badge. Each renderer must name its own data.
    """
    js = _strip_js_comments(_read(APP_JS))

    fc_start = js.index("function renderNowcast(timeline)")
    fc_body = js[fc_start:fc_start + 900]
    assert 'setText("nowcast-timeline-badge"' in fc_body, (
        "renderNowcast must set the strip badge itself"
    )
    assert "LIVE OBSERVATION" not in fc_body, (
        "forecast renderer must never badge its cards as a live observation"
    )

    lv_start = js.index("function renderNowcastLive(")
    lv_body = js[lv_start:lv_start + 700]
    assert 'setText("nowcast-timeline-badge"' in lv_body
    assert "LIVE OBSERVATION" in lv_body


def test_baseline_brier_elements_exist_for_their_writes():
    """app.js writes bm-v1-brier / bm-pers-brier. If those elements do not
    exist the write is a silent no-op and the hardcoded markup value freezes --
    the same defect class as the old bm-lead-display-tag."""
    html = _read(INDEX_HTML)
    for el in ("bm-v1-brier", "bm-pers-brier"):
        assert f'id="{el}"' in html, f"{el} is written by app.js but has no element"


def test_public_ui_hides_internal_model_version_labels():
    """Section 17: internal model version terminology must not be public."""
    html = _strip_html_comments(_read(INDEX_HTML))
    assert not re.search(r"\bV2\b", html), "public UI exposes internal 'V2' label"
    assert not re.search(r"\bV1\b", html), "public UI exposes internal 'V1' label"
    assert "SevereWeatherNet" not in html


def test_high_risk_area_labelled_state_wide():
    """The state-wide peak district figure must be labelled as state-wide so it
    is not read as the user's local conditions."""
    html = _read(INDEX_HTML)
    idx = html.index('id="active-high-risk-area"')
    context = html[max(0, idx - 400):idx]
    assert "state-wide" in context.lower()


# ---------------------------------------------------------------------------
# 7. Historical case-study data integrity.
#
# The historical ERA5 `tp` field is stored in the cache in mm/hour --
# src/data/era5_loader.py converts raw ECMWF meters/hour to mm/hour before the
# memmap is written. _init_operational_state() additionally multiplied the
# denormalized value by 1000, overflowing every cell past a 150 mm clip
# ceiling, so 100% of the historical grid reported a constant 150.0 mm instead
# of the real rainfall field. Every historical rainfall value in the UI
# (dashboard, map popup, point inspection) came from that constant.
# ---------------------------------------------------------------------------

NOWCAST_SERVICE_PY = os.path.join(
    PROJECT_ROOT, "src", "inference", "nowcast_service.py"
)


def _strip_py_comments(src: str) -> str:
    """Drop full-line and trailing # comments so assertions run against code.

    The fix is documented in comments that necessarily quote the old bad
    expression, which would otherwise trip its own test.
    """
    out = []
    for line in src.splitlines():
        stripped = line.split("#", 1)[0]
        out.append(stripped)
    return "\n".join(out)


def test_historical_rainfall_is_not_rescaled_by_1000():
    """ERA5 tp is already mm/hour in the cache; re-scaling saturates the clip."""
    code = _strip_py_comments(_read(NOWCAST_SERVICE_PY))
    assert "tp_m * 1000.0" not in code, (
        "Historical rainfall re-multiplied by 1000. The cache already holds "
        "mm/hour (see src/data/era5_loader.py), so this pins every cell to the "
        "clip ceiling and fabricates a constant rainfall field."
    )


def test_historical_rainfall_has_no_upper_clip_ceiling():
    """A real extreme must be allowed to read as extreme, not be capped."""
    code = _strip_py_comments(_read(NOWCAST_SERVICE_PY))
    assert "0.0, 150.0" not in code, (
        "A 150 mm upper clip on historical rainfall caps genuine extremes and "
        "previously masked the unit-conversion bug by making every cell equal."
    )


def test_era5_surface_t0_initialized_before_use():
    """get_summary() reads self.era5_surface_t0 directly, so a cache-load
    failure must leave a defined None rather than an undefined attribute."""
    code = _read(NOWCAST_SERVICE_PY)
    assert "self.era5_surface_t0: Optional[Dict[str, np.ndarray]] = None" in code, (
        "era5_surface_t0 must be initialized in __init__ before "
        "_init_operational_state() runs, or a cache failure raises "
        "AttributeError instead of degrading to 'no data'."
    )
    init_pos = code.index("self.era5_surface_t0: Optional")
    call_pos = code.index("self._init_operational_state()")
    assert init_pos < call_pos, (
        "era5_surface_t0 must be initialized BEFORE _init_operational_state()."
    )


def test_historical_xai_shear_label_matches_the_field_it_reads():
    """The historical XAI factor reads bulk_shear_0_6km_mps, so it must not be
    labelled as the 1000-700 hPa layer -- that label belongs to the live GFS
    path, which carries no wind above 700 hPa (see _compute_live_thermo)."""
    code = _read(NOWCAST_SERVICE_PY)
    idx = code.index("bulk_shear_0_6km_mps', 'N/A')} m/s")
    factor_block = code[max(0, idx - 600):idx]
    name_line = factor_block[factor_block.rindex('"name":'):]
    assert "1000" not in name_line and "700" not in name_line, (
        "Historical XAI labels a 0-6 km shear value as a 1000-700 hPa shear. "
        "The two describe different layer depths and must not be conflated."
    )


@pytest.mark.parametrize("field", ["rain_mm", "temp_c", "rh_pct", "wind_kmh", "pres_hpa"])
def test_historical_surface_state_is_physically_plausible(field):
    """End-to-end numeric guard on the denormalized historical ERA5 state.

    Skips when the ERA5 cache is absent (the service degrades to None there);
    runs for real when it is present, which is the case that regressed.
    """
    import numpy as np

    try:
        from src.inference.nowcast_service import get_nowcast_service
        svc = get_nowcast_service()
    except Exception as e:  # pragma: no cover - environment without model/cache
        pytest.skip(f"NowcastService unavailable: {e}")

    state = getattr(svc, "era5_surface_t0", None)
    if not state:
        pytest.skip("ERA5 historical cache not available in this environment")

    arr = np.asarray(state[field])
    bounds = {
        # mm/hour. West Bengal's extreme hourly rates are well under 200; the
        # bug produced a constant 150.0, so a degenerate field is also rejected.
        "rain_mm": (0.0, 200.0),
        "temp_c": (-10.0, 60.0),
        "rh_pct": (0.0, 100.0),
        "wind_kmh": (0.0, 300.0),
        # SURFACE pressure, and the model domain reaches 28N / ~5,900 m in the
        # Himalayas, where genuine surface pressure is ~500 hPa (verified: the
        # grid minimum sits at 28.00N 86.75E, DEM 5,424 m, barometric
        # expectation 510 hPa). A sea-level-only bound would reject real data.
        "pres_hpa": (450.0, 1100.0),
    }[field]
    assert np.isfinite(arr).all(), f"{field} contains non-finite values"
    assert float(arr.min()) >= bounds[0], f"{field} below physical bound {bounds}"
    assert float(arr.max()) <= bounds[1], f"{field} above physical bound {bounds}"
    # A constant field over 825 cells is not a real meteorological analysis.
    assert float(arr.max()) > float(arr.min()), (
        f"{field} is spatially constant across the whole grid -- that is a "
        f"saturated/fabricated field, not an analysis."
    )


# ---------------------------------------------------------------------------
# 8. Historical case-study identity and mode isolation.
#
# The case study is Cyclone Remal (2024-05-26T12:00:00Z). It was previously
# possible for the app to load Remal data and label it "Kalbaishakhi", and for
# a missing Remal window to silently fall back to "sample 100" -- serving one
# event's data under another event's name.
# ---------------------------------------------------------------------------


def test_historical_case_has_single_source_of_truth():
    """Event identity is defined once, not duplicated as literals."""
    code = _read(NOWCAST_SERVICE_PY)
    for const in (
        "HISTORICAL_EVENT_NAME",
        "HISTORICAL_EVENT_DETAIL",
        "HISTORICAL_ANALYSIS_TIME",
        "HISTORICAL_ANALYSIS_SOURCE",
    ):
        assert f"{const} = " in code, f"{const} must be defined once at module level"
    assert 'HISTORICAL_ANALYSIS_TIME = "2024-05-26T12:00:00Z"' in code


def test_historical_load_failure_does_not_substitute_another_event():
    """A missing Remal window must raise, never fall back to a sample index."""
    code = _strip_py_comments(_read(NOWCAST_SERVICE_PY))
    assert "sample_idx = 100" not in code, (
        "Silent fallback to sample 100 reintroduced: a missing case-study "
        "window would serve a different event under the Remal label."
    )
    assert "raise RuntimeError(" in code
    assert "Refusing to substitute a different event." in _read(NOWCAST_SERVICE_PY)


def test_live_dashboard_painter_refuses_historical_mode():
    """Live station observations must never paint the historical case study."""
    js = _strip_js_comments(_read(APP_JS))
    idx = js.index("function paintDashboardLive(")
    body = js[idx:idx + 500]
    assert 'stormSenseMode === "historical"' in body and "return" in body, (
        "paintDashboardLive() must short-circuit in historical mode, or live "
        "current weather is painted into a frozen 2024 event replay."
    )


def test_historical_now_renders_analysis_not_forecast():
    """Historical NOW paints the t=0 ANALYSIS layer, never a forecast field."""
    js = _strip_js_comments(_read(APP_JS))
    assert "renderHistoricalAnalysisSurface" in js
    assert "/api/historical/analysis-surface" in js
    idx = js.index("function renderContinuousRiskSurface(")
    body = js[idx:idx + 1500]
    now_branch = body[body.index("leadHours === 'now'"):]
    assert "renderHistoricalAnalysisSurface" in now_branch, (
        "Historical NOW must paint the analysis surface; otherwise both the "
        "forecast and observation layers are removed and the map goes blank."
    )


def test_xai_fetches_are_bound_to_the_selected_horizon():
    """XAI must not show +2h attribution while +4h/+6h is selected."""
    js = _strip_js_comments(_read(APP_JS))
    for chunk in js.split('"/api/nowcast/xai"')[1:]:
        head = chunk[:200]
        assert "lead=" in head, (
            "An XAI fetch omits the lead parameter, so attribution stays pinned "
            "to the default horizon while another horizon is displayed."
        )


def test_radar_scan_time_uses_ist_not_viewer_locale():
    """Radar timestamps must agree with the rest of the app's IST clock."""
    js = _strip_js_comments(_read(APP_JS))
    idx = js.index("Last Scan: ")
    line = js[idx - 200:idx + 260]
    assert "formatIstClock" in line, (
        "Radar scan time rendered with toLocaleTimeString() uses the VIEWER's "
        "timezone, disagreeing with the app's IST-primary timestamps."
    )


# ---------------------------------------------------------------------------
# 9. Cross-mode state isolation in the API payload, and honest labelling of
#    model output vs observations on the radar/satellite page.
# ---------------------------------------------------------------------------


def test_summary_analysis_time_is_mode_scoped():
    """Historical mode must not report the LIVE GFS analysis time.

    get_summary() previously read analysis_time/last_refreshed/is_stale
    unconditionally from the live pipeline, so the Cyclone Remal case study
    displayed an "Input analysis" of the current 2026 GFS cycle beside a
    correct 2024-05-26 valid time -- a live-state leak into a frozen replay.
    """
    code = _read(NOWCAST_SERVICE_PY)
    start = code.index('"model": "StormSense AI Forecast"')
    block = code[start:start + 2200]
    assert 'resolved_mode == "live"' in block, (
        "analysis_time/last_refreshed/is_stale must be gated on the resolved "
        "mode, or historical mode reports live pipeline state."
    )
    assert "analysis_source" in block


def test_historical_map_does_not_label_model_output_as_radar():
    """StormSense holds no archived Remal radar; model output must not be
    presented as a radar archive."""
    js = _strip_js_comments(_read(APP_JS))
    assert "CYCLONE REMAL RADAR ARCHIVE" not in js, (
        "The historical spatial field is the StormSense model forecast, not "
        "archived radar imagery."
    )
    assert "STORMSENSE MODEL FORECAST" in js


def test_peak_probability_is_labelled_as_a_grid_cell_maximum():
    """'PEAK PROB: 100%' reads as a state-wide certainty; it is one cell."""
    js = _strip_js_comments(_read(APP_JS))
    assert "PEAK CELL: " in js, (
        "The peak badge must identify itself as a single grid-cell maximum."
    )


def test_t0_inputs_are_labelled_as_model_initial_conditions():
    """t=0 physical inputs are identical across horizons by design."""
    js = _strip_js_comments(_read(APP_JS))
    assert "Model Initial Conditions (t=0)" in js, (
        "Constant t=0 inputs must be labelled as initial conditions, not as "
        "observations valid at the selected forecast time."
    )


def test_no_fabricated_current_condition_constants():
    """Invented fallback weather must not reappear in any viewmodel path."""
    js = _strip_js_comments(_read(APP_JS))
    for bad in ("|| 28.5", "|| 32.0", "|| 16.0", "|| 1006", "rainfall_1h || 0.0"):
        assert bad not in js, (
            f"Fabricated fallback {bad!r} reintroduced: a failed observation "
            f"fetch would render invented weather as a real measurement."
        )


def test_current_location_follows_selected_horizon():
    """The local forecast meters must re-query on horizon change."""
    js = _strip_js_comments(_read(APP_JS))
    start = js.index("window.setForecastHorizon = function")
    # The handler grew when the lower physical-condition cards became
    # horizon-aware; scan to the end of the function rather than a fixed window.
    nxt = js.find(chr(10) + "  window.", start + 10)
    body = js[start:(nxt if nxt > 0 else start + 12000)]
    assert "applyLiveLocation" in body, (
        "Changing the horizon must refresh the Current Location panel, or it "
        "keeps the forecast from whichever horizon was active at page load."
    )


# ---------------------------------------------------------------------------
# 10. Current Location lower cards follow the selected horizon, and the
#     historical radar panel never claims a radar product that does not exist.
# ---------------------------------------------------------------------------


def test_lower_condition_cards_follow_the_selected_horizon():
    """The physical-conditions cards must not stay on NOW at +2/+4/+6."""
    js = _strip_js_comments(_read(APP_JS))
    assert "function paintHorizonConditions(" in js, (
        "A horizon-aware painter must own the lower cards; otherwise only "
        "observation/t0 painters write them and they never change."
    )
    # The observation painters must yield at forecast horizons.
    for fn_name in ("paintDashboardLive(", "paintHistoricalConditions("):
        i = js.index("function " + fn_name)
        body = js[i:i + 700]
        assert 'currentLeadHours !== "now"' in body, (
            f"{fn_name} must not paint the lower cards at a forecast horizon."
        )


def test_unforecast_fields_are_not_carried_into_forecast_horizons():
    """The model predicts only severe probability and 3 h rain.

    Temperature/humidity/wind exist only at t=0, so at a forecast horizon they
    must read as not-forecast rather than silently repeating the analysis state
    under a "+4h" label.
    """
    js = _strip_js_comments(_read(APP_JS))
    i = js.index("function paintHorizonConditions(")
    body = js[i:i + 3000]
    assert "Not forecast by the model" in body
    assert "mm/3h" in body, "forecast rainfall must be labelled as a 3 h accumulation"


def test_historical_radar_panel_does_not_claim_a_radar_product():
    """Switching modes must re-apply radar semantics on an existing map."""
    js = _strip_js_comments(_read(APP_JS))
    assert "function applyRadarModeSemantics(" in js, (
        "initRadarMap() short-circuits when the map exists, so a live-created "
        "radar map kept 'RADAR READY' and live tiles over the case study."
    )
    i = js.index("function applyRadarModeSemantics(")
    body = js[i:i + 1800]
    assert "rainviewer" in body.lower() and "removeLayer" in body, (
        "Historical mode must detach live radar tiles."
    )
    assert "HISTORICAL ANALYSIS" in body


def test_remal_hub_reports_radar_unavailability_without_substitution():
    """The case-study hub must state why radar is absent, not fill the gap."""
    main_py = _read(os.path.join(PROJECT_ROOT, "backend", "main.py"))
    assert '"/api/historical/case-study"' in main_py
    i = main_py.index('"radar": {')
    block = main_py[i:i + 600]
    assert '"status": "unavailable"' in block
    assert "never labelled radar" in block or "not synthesized" in block
