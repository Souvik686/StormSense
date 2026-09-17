import urllib.request
import json
import time

print("=== COMPLETE FORENSIC VERIFICATION ===")
print()

# 1. API endpoints
endpoints = [
    ('lead=0 live', '/api/nowcast/summary?lead=0&mode=live'),
    ('lead=2 live', '/api/nowcast/summary?lead=2&mode=live'),
    ('lead=4 live', '/api/nowcast/summary?lead=4&mode=live'),
    ('lead=6 live', '/api/nowcast/summary?lead=6&mode=live'),
    ('lead=0 historical', '/api/nowcast/summary?lead=0&mode=historical'),
    ('lead=2 historical', '/api/nowcast/summary?lead=2&mode=historical'),
    ('lead=4 historical', '/api/nowcast/summary?lead=4&mode=historical'),
    ('lead=6 historical', '/api/nowcast/summary?lead=6&mode=historical'),
]

print("--- API ENDPOINT TESTS ---")
for name, path in endpoints:
    try:
        start = time.time()
        with urllib.request.urlopen(f'http://localhost:8000{path}') as r:
            data = json.loads(r.read())
            dur = time.time() - start
            it = data.get('issue_time', 'MISSING')
            vt = data.get('forecast_valid_time', 'MISSING')
            mode = data.get('mode', 'MISSING')
            risk = data.get('hazards', {}).get('overall_risk', {}).get('percentage', 'MISSING') if 'hazards' in data else 'N/A'
            print(f'  {name}: HTTP 200 | issue={it} valid={vt} mode={mode} risk={risk} | {dur:.2f}s')
    except Exception as e:
        print(f'  {name}: ERROR {e}')

print()

# 2. Risk surface PNG
print("--- RISK SURFACE PNG TESTS ---")
for lead in [0, 2, 4, 6]:
    try:
        start = time.time()
        with urllib.request.urlopen(f'http://localhost:8000/api/nowcast/risk-surface?lead={lead}&mode=live') as r:
            data = r.read()
            dur = time.time() - start
            print(f'  lead={lead}: {len(data)} bytes, content-type={r.headers.get("content-type", "unknown")} | {dur:.2f}s')
    except Exception as e:
        print(f'  lead={lead}: ERROR {e}')

print()

# 3. /api/config
print("--- CONFIG ENDPOINT ---")
try:
    with urllib.request.urlopen('http://localhost:8000/api/config') as r:
        data = json.loads(r.read())
        key = data.get('google_maps_api_key', 'MISSING')
        print(f'  google_maps_api_key: {"EMPTY" if key == "" else key[:8]+"..." if len(key) > 8 else key}')
except Exception as e:
    print(f'  ERROR: {e}')

print()

# 4. Districts
print("--- DISTRICT/AREA TESTS ---")
try:
    with urllib.request.urlopen('http://localhost:8000/api/nowcast/summary?lead=2&mode=live&district=Kolkata') as r:
        data = json.loads(r.read())
        print(f'  Kolkata lead=2: risk={data.get("hazards", {}).get("overall_risk", {}).get("percentage", "N/A")}')
except Exception as e:
    print(f'  Kolkata: ERROR {e}')

try:
    with urllib.request.urlopen('http://localhost:8000/api/nowcast/summary?lead=2&mode=live&district=West+Bengal') as r:
        data = json.loads(r.read())
        print(f'  West Bengal lead=2: risk={data.get("hazards", {}).get("overall_risk", {}).get("percentage", "N/A")}')
except Exception as e:
    print(f'  West Bengal: ERROR {e}')

print()
print("=== DIAGNOSIS COMPLETE ===")

