import urllib.request
import json

def fetch_lead(lead):
    url = f"http://127.0.0.1:8000/api/nowcast/summary?lead={lead}&mode=live"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode())
    return data

try:
    print("=== LIVE FORECAST HORIZONS EVIDENCE ===")
    for lead in [2, 4, 6]:
        data = fetch_lead(lead)
        print(f"\n--- LEAD +{lead}h ---")
        print(f"issue_time (NOW):          {data.get('issue_time')}")
        print(f"forecast_valid_time (+{lead}h): {data.get('forecast_valid_time')}")
        if lead == 2:
            print("\n--- GFS PROVENANCE (from +2h) ---")
            prov = data.get('input_slot_provenance', [])
            for p in prov:
                print(f"  Slot T{p.get('relative_t')}: type={p.get('type')}, cycle={p.get('cycle')}, offset={p.get('offset_hours')}h -> effective={p.get('effective_valid_time')}")
except Exception as e:
    print("Error fetching data:", e)
