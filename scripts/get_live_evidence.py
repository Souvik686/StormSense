import urllib.request
import json

def fetch_lead(lead):
    url = f"http://127.0.0.1:8000/api/nowcast/summary?lead={lead}&mode=live"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode())
    return data

try:
    print("=== EXACT WALL-CLOCK HORIZON PROOF ===")
    for lead in [2, 4, 6]:
        data = fetch_lead(lead)
        print(f"\n--- LEAD +{lead}h ---")
        print(f"reference_now:             {data.get('issue_time')}")
        print(f"forecast_valid_time (+{lead}h): {data.get('forecast_valid_time')}")
        if lead == 2:
            print("\n--- EXACT GFS PROVENANCE ---")
            prov = data.get('input_slot_provenance', [])
            for i, p in enumerate(prov):
                print(f"  Slot T-{5-i}: {p.get('source')} -> {p.get('details')}")
except Exception as e:
    print("Error fetching data:", e)
