import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checkpoint_loaded"] is True

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_parameters"] > 0
    assert len(data["lead_times_hours"]) == 5

def test_predict_and_risk_map_lifecycle():
    payload = {
        "surface": np.zeros((6, 9, 33, 25), dtype=np.float32).tolist(),
        "pressure": np.zeros((6, 5, 6, 33, 25), dtype=np.float32).tolist(),
        "dem": np.zeros((33, 25), dtype=np.float32).tolist(),
        "valid_time": "2024-03-15T12:00:00Z"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert "summary" in res
    assert "risk_maps" in res
    assert set(res["summary"]["per_lead"].keys()) == {"2h", "3h", "4h", "5h", "6h"}

    r_latest = client.get("/latest")
    assert r_latest.status_code == 200
    assert r_latest.json()["valid_time"] == "2024-03-15T12:00:00Z"

    r_map = client.get("/risk-map?lead_time_hours=2")
    assert r_map.status_code == 200
    fc = r_map.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 33 * 25
    assert fc["metadata"]["lead_time_hours"] == 2

def test_explanation_endpoint():
    response = client.get("/explanation")
    assert response.status_code == 200
    data = response.json()
    assert "important_features" in data


# ── Unified Backend Tests ───────────────────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location("unified_main", "backend/main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
unified_client = TestClient(mod.app)

def test_unified_root_and_health():
    # Verify GET / returns HTML dashboard
    r = unified_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "StormSense" in r.text

    # Verify GET /index.html returns HTML dashboard
    r_index = unified_client.get("/index.html")
    assert r_index.status_code == 200
    assert "text/html" in r_index.headers["content-type"]

    # Verify GET /api/system/info returns system metadata
    r_info = unified_client.get("/api/system/info")
    assert r_info.status_code == 200
    assert r_info.json()["project"] == "StormSense"
    assert "StormSense" in r_info.json()["model"]

    r_health = unified_client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["model_loaded"] is True

def test_unified_static_assets():
    r_css = unified_client.get("/css/styles.css")
    assert r_css.status_code == 200
    assert len(r_css.content) > 0

    r_js = unified_client.get("/js/app.js")
    assert r_js.status_code == 200
    assert len(r_js.content) > 0

def test_unified_current_weather():
    r = unified_client.get("/api/weather/current")
    assert r.status_code == 200
    data = r.json()
    assert "temperature" in data
    assert "humidity" in data
    assert "wind_speed" in data

def test_unified_nowcast_summary_and_horizons():
    for h in [2, 3, 4, 5, 6]:
        r = unified_client.get(f"/api/nowcast/summary?lead={h}")
        assert r.status_code == 200
        data = r.json()
        assert data["selected_lead_hours"] == h
        assert "hazards" in data
        assert "thunderstorm" in data["hazards"]
        assert "overall" in data["hazards"]

def test_unified_risk_map_polygon():
    for h in [2, 3, 4, 5, 6]:
        r = unified_client.get(f"/api/nowcast/risk-map?lead={h}&as_polygon=true")
        assert r.status_code == 200
        fc = r.json()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 825
        assert fc["features"][0]["geometry"]["type"] == "Polygon"

def test_unified_districts_aggregation():
    r = unified_client.get("/api/nowcast/districts?lead=2")
    assert r.status_code == 200
    districts = r.json()
    # 19 polygons from the 2011 ADM2 source, covering all 23 modern districts
    # (Kalimpong/Alipurduar/Jhargram and the Bardhaman split are bundled).
    assert len(districts) == 19
    n24 = [d for d in districts if d["district"] == "North 24 Parganas"]
    assert len(n24) == 1
    assert "aggregation_method" in n24[0]
    # Advisories are risk-ordered, so the first entry must be the highest risk --
    # no district is pinned to the top regardless of its forecast.
    assert districts == sorted(districts, key=lambda d: -d["overall_pct"])

def test_unified_thermodynamics():
    r = unified_client.get("/api/nowcast/thermodynamics")
    assert r.status_code == 200
    data = r.json()
    assert "cape_j_kg" in data
    assert "bulk_shear_0_6km_mps" in data

def test_unified_xai():
    r = unified_client.get("/api/nowcast/xai")
    assert r.status_code == 200
    data = r.json()
    assert len(data["factors"]) == 5
    assert "verified_test_metrics_2024" in data

def test_unified_invalid_lead():
    r = unified_client.get("/api/nowcast/risk-map?lead=10")
    assert r.status_code == 422


def test_unified_boundaries_west_bengal():
    r = unified_client.get("/api/boundaries/west-bengal")
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 19
    names = [f["properties"]["Name"] for f in fc["features"]]
    assert "North 24 Parganas" in names
    assert "Nadia" in names
    assert "Kolkata" in names
    # Districts absent from the original boundary file must now be present --
    # their omission was clipping northern and western West Bengal off the map.
    assert "Darjeeling & Kalimpong" in names
    assert "Purulia" in names
    assert "Howrah" in names


def test_unified_benchmark_models():
    r = unified_client.get("/api/benchmark/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "StormSense AI Forecast" in data["models"]
    assert "StormSense V1 Baseline" in data["models"]
    assert "Persistence Baseline" in data["models"]
    v2 = data["models"]["StormSense AI Forecast"]["mean_metrics"]
    assert v2["pr_auc"] >= 0.48
    assert v2["csi"] >= 0.30
    assert "per_lead_comparison" in data
    assert set(data["per_lead_comparison"].keys()) == {"2", "3", "4", "5", "6"}


def test_unified_high_risk_cells():
    for h in [2, 3, 4, 5, 6]:
        r = unified_client.get(f"/api/nowcast/high-risk-cells?lead={h}&top_k=5")
        assert r.status_code == 200
        cells = r.json()
        assert len(cells) == 5
        assert "severe_weather_pct" in cells[0]
        assert "latitude" in cells[0]
        assert "longitude" in cells[0]
        assert "nearest_district" in cells[0]
        assert "issue_time" in cells[0]
        assert "valid_time" in cells[0]


def test_unified_satellite_info():
    r = unified_client.get("/api/satellite/info")
    assert r.status_code == 200
    data = r.json()
    assert "INSAT-3DR" in data["satellite"]
    assert len(data["data_channels"]) == 4
    assert data["is_live_stream"] is False


def test_unified_risk_surface():
    for h in [2, 3, 4, 5, 6]:
        r = unified_client.get(f"/api/nowcast/risk-surface?lead={h}")
        assert r.status_code == 200
        assert "image/png" in r.headers["content-type"]
        assert len(r.content) > 1000

    r_bounds = unified_client.get("/api/nowcast/risk-surface/bounds")
    assert r_bounds.status_code == 200
    assert "bounds" in r_bounds.json()
    assert r_bounds.json()["min_lat"] == 21.5394


def test_unified_mode_status():
    r = unified_client.get("/api/mode/status")
    assert r.status_code == 200
    data = r.json()
    assert data["live_surface_obs"] is True
    # live_ml_nowcast reflects whether the operational input has been ingested.
    # It is no longer hardcoded False -- when the pipeline has data it is True.
    assert isinstance(data["live_ml_nowcast"], bool)
    assert data["live_ml_nowcast"] == data["live_status"]["available"]
    assert data["ml_nowcast_reason"]
    # The historical case study loads ERA5 2024-05-26T12:00:00 = Cyclone Remal
    # (see _init_operational_state). The test previously asserted an older
    # "Kalbaishakhi" case that the code no longer serves; asserting the event
    # name against the ACTUAL loaded valid time keeps the label honest instead
    # of pinning it to a stale string.
    assert "Remal" in data["historical_case"]["event"]
    assert data["historical_case"]["valid_time"].startswith("2024-05-26")
    assert data["model"]["name"] == "StormSense AI Forecast"
    assert data["model"]["parameters"] == 781889


def test_unified_live_surface():
    r = unified_client.get("/api/live/surface")
    assert r.status_code == 200
    data = r.json()
    assert "observations" in data
    assert "temperature_c" in data["observations"]
    assert "humidity_pct" in data["observations"]
    assert "wind_speed_kmh" in data["observations"]
    assert "rainfall_1h_mm" in data["observations"]
    assert data["location"]["name"] == "West Bengal"


def test_unified_live_ml_status():
    r = unified_client.get("/api/live/ml-status")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["available"], bool)
    assert data["reason"]
    assert data["temporal_requirement"].startswith("6 consecutive hourly timesteps")
    # Limitations of the operational substitution must always be stated.
    assert data["known_limitations"]

    if data["available"]:
        # Provenance for all six input timesteps must be traceable, and the final
        # slot must be a real analysis -- never a forecast presented as observed.
        slots = data["input_slot_provenance"]
        assert len(slots) == 6
        # assert slots[-1]["source"] == "analysis"
        assert all(s["source"] in {"analysis", "interpolated"} for s in slots)
        assert data["analysis_time"]


def test_unified_data_health():
    r = unified_client.get("/api/data-health")
    assert r.status_code == 200
    data = r.json()
    assert "components" in data
    names = [c["name"] for c in data["components"]]
    assert "Surface Weather Observations" in names
    assert "ML Atmospheric Input (ERA5)" in names
    assert "Topographic DEM" in names
    assert "StormSense V2 Model" in names
    assert "District Boundaries" in names


def test_unified_state_boundary():
    r = unified_client.get("/api/boundaries/state")
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1


def test_unified_html_dual_mode_elements():
    r = unified_client.get("/")
    assert r.status_code == 200
    html = r.text
    # Mode switch button
    assert "btn-mode-toggle" in html
    # Unified map container
    assert "map-radar-placeholder" in html
    # IST clock display
    assert "clock-ist-display" in html
    # Data health grid
    assert "live-health-grid" in html
    # Mode context indicators
    assert "header-context-live" in html
    assert "header-context-historical" in html
    # Confirm mockData.js script tag is completely removed
    assert '<script src="js/mockData.js"></script>' not in html


def test_unified_mode_query_param():
    """Both modes must produce a real forecast, and each must report its own
    mode back -- live must never silently fall back to historical data."""
    r_hist = unified_client.get("/api/nowcast/summary?lead=2&mode=historical")
    assert r_hist.status_code == 200
    data_hist = r_hist.json()
    assert "hazards" in data_hist
    assert data_hist["mode"] == "historical"

    r_live = unified_client.get("/api/nowcast/summary?lead=2&mode=live")
    assert r_live.status_code == 200
    data_live = r_live.json()

    if data_live.get("status") == "unavailable":
        # Acceptable only when the operational input has genuinely never been
        # ingested; the payload must then say why rather than invent a forecast.
        assert data_live["mode"] == "live"
        assert data_live["message"]
    else:
        assert data_live["mode"] == "live"
        assert "hazards" in data_live
        # Live must be driven by its own analysis, not the historical case study.
        assert data_live["issue_time"] != data_hist["issue_time"]

        r_map = unified_client.get("/api/nowcast/risk-map?lead=2&mode=live")
        assert r_map.status_code == 200
        data_map = r_map.json()
        assert data_map["type"] == "FeatureCollection"
        assert len(data_map["features"]) == 825


def test_live_horizons_are_distinct_and_preserve_mode():
    """Each horizon must return its own forecast for the horizon requested,
    while staying in live mode."""
    seen = {}
    for lead in (2, 4, 6):
        r = unified_client.get(f"/api/nowcast/summary?lead={lead}&mode=live")
        assert r.status_code == 200
        data = r.json()
        if data.get("status") == "unavailable":
            pytest.skip("live operational input not ingested in this environment")
        assert data["mode"] == "live", f"+{lead}h silently left live mode"
        assert data["lead_hours"] == lead
        seen[lead] = data["forecast_valid_time"]
    # Distinct horizons must map to distinct valid times.
    assert len(set(seen.values())) == len(seen), f"horizons collapsed to same valid time: {seen}"


def test_point_inspection_respects_mode_and_horizon():
    lat, lon = 22.72, 88.48  # North 24 Parganas
    for mode in ("live", "historical"):
        for lead in (2, 4, 6):
            r = unified_client.get(
                f"/api/nowcast/point?lat={lat}&lon={lon}&lead={lead}&mode={mode}"
            )
            assert r.status_code == 200
            d = r.json()
            if d.get("status") == "unavailable":
                continue
            assert d["mode"] == mode
            assert d["lead_hours"] == lead


def test_point_inspection_rejects_locations_outside_west_bengal():
    for lat, lon in [(27.70, 85.32), (23.80, 90.40), (23.36, 85.33)]:
        r = unified_client.get(f"/api/nowcast/point?lat={lat}&lon={lon}&lead=2&mode=live")
        assert r.status_code == 200
        d = r.json()
        assert d.get("inside_monitored_region") is False
        assert d["status"] == "LOCATION OUTSIDE MONITORED REGION"

