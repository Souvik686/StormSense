import importlib.util
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location('main_mod', 'weather-app/backend/main.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
client = TestClient(mod.app)

print('=== 1. VERIFYING DUAL MODE HTML (GET /) ===')
r_root = client.get('/')
assert r_root.status_code == 200
html = r_root.text
assert 'btn-mode-toggle' in html, 'Mode switch button missing'
assert 'section-live-mode' in html, 'Live mode section missing'
assert 'section-historical-mode' in html, 'Historical mode section missing'
assert 'global-clock-footer' in html, 'Clock footer missing'
assert 'clock-ist-display' in html, 'Clock display missing'
assert 'map-live-container' in html, 'Live map container missing'
assert 'live-health-grid' in html, 'Data health grid missing'
assert 'mockData.js' not in html, 'mockData.js script tag was not removed'
assert 'LIVE MONITORING' in html
assert 'HISTORICAL CASE STUDY' in html
print('   -> HTML elements for both LIVE and HISTORICAL modes confirmed present!')

print('=== 2. VERIFYING MODE STATUS (/api/mode/status) ===')
r_mode = client.get('/api/mode/status')
assert r_mode.status_code == 200
m = r_mode.json()
print('   live_surface_obs:', m['live_surface_obs'])
print('   live_ml_nowcast:', m['live_ml_nowcast'])
print('   ml_nowcast_reason:', m['ml_nowcast_reason'][:60] + '...')
print('   model:', m['model']['name'], 'parameters:', m['model']['parameters'])
assert m['live_surface_obs'] is True
assert m['live_ml_nowcast'] is False
assert m['model']['parameters'] == 781889

print('=== 3. VERIFYING LIVE SURFACE TELEMETRY (/api/live/surface) ===')
r_surf = client.get('/api/live/surface')
assert r_surf.status_code == 200
s = r_surf.json()
print('   is_genuinely_live:', s['is_genuinely_live'])
print('   data_source:', s['data_source'])
print('   location:', s['location']['name'])
print('   temperature_c:', s['observations']['temperature_c'])
print('   humidity_pct:', s['observations']['humidity_pct'])
print('   rainfall_1h_mm:', s['observations']['rainfall_1h_mm'])
print('   wind_speed_kmh:', s['observations']['wind_speed_kmh'])

print('=== 4. VERIFYING LIVE ML STATUS (/api/live/ml-status) ===')
r_ml = client.get('/api/live/ml-status')
assert r_ml.status_code == 200
ml = r_ml.json()
print('   available:', ml['available'])
print('   reason:', ml['reason'][:60] + '...')
print('   required_inputs count:', len(ml['required_inputs']))
for inp in ml['required_inputs']:
    print('     -', inp['name'], '->', inp['status'])

print('=== 5. VERIFYING DATA PIPELINE HEALTH (/api/data-health) ===')
r_health = client.get('/api/data-health')
assert r_health.status_code == 200
h = r_health.json()
print('   Components count:', len(h['components']))
for c in h['components']:
    print('     *', c['name'], ':', c['status'], '(' + (c.get('detail') or '')[:40] + ')')

print('=== 6. VERIFYING STATE & DISTRICT BOUNDARIES ===')
r_state = client.get('/api/boundaries/state')
assert r_state.status_code == 200
print('   State boundary features:', len(r_state.json()['features']))
r_dist = client.get('/api/boundaries/west-bengal')
assert r_dist.status_code == 200
print('   District boundaries features:', len(r_dist.json()['features']))

print('=== 7. VERIFYING HISTORICAL ML CASE STUDY (/api/nowcast/*) ===')
r_summary = client.get('/api/nowcast/summary?lead=2')
assert r_summary.status_code == 200
sum_data = r_summary.json()
print('   Historical issue_time:', sum_data['issue_time'])
print('   Historical valid_time:', sum_data['forecast_valid_time'])
print('   Thunderstorm prob:', sum_data['hazards']['thunderstorm']['probability'])
print('   Overall alert level:', sum_data.get('civil_defense_alert_level'))

r_surface = client.get('/api/nowcast/risk-surface?lead=2')
assert r_surface.status_code == 200
print('   Risk surface PNG bytes:', len(r_surface.content))

r_bench = client.get('/api/benchmark/models')
assert r_bench.status_code == 200
bm = r_bench.json()['models']['SevereWeatherNet V2 Calibrated']['mean_metrics']
print('   SevereWeatherNet V2 Mean CSI:', bm['csi'], 'PR-AUC:', bm['pr_auc'])

print('\nALL DUAL-MODE VERIFICATIONS COMPLETED SUCCESSFULLY WITH 100% PASS RATE!')
