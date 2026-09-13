import http.client
import urllib.parse
import json
import time

with open("reports/baseline_behavior.json", "r", encoding="utf-8") as f:
    baseline = json.load(f)

endpoints = list(baseline.keys())
print(f"Beginning post-reorganization verification for {len(endpoints)} endpoints...", flush=True)

passed = 0
failed = 0
results = []

for idx, ep in enumerate(endpoints):
    base_info = baseline[ep]
    expected_code = base_info["status"]
    
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=120)
    try:
        conn.request("GET", ep, headers={"User-Agent": "PostReorgAudit"})
        resp = conn.getresponse()
        code = resp.status
        body = resp.read()
        
        status_match = (code == expected_code)
        if status_match:
            passed += 1
            print(f"[{idx+1:02d}/{len(endpoints):02d}] PASS [HTTP {code}]: {ep}", flush=True)
            results.append((ep, code, expected_code, "PASS"))
        else:
            failed += 1
            print(f"[{idx+1:02d}/{len(endpoints):02d}] FAIL [HTTP {code} != {expected_code}]: {ep}", flush=True)
            results.append((ep, code, expected_code, "FAIL"))
    except Exception as e:
        failed += 1
        print(f"[{idx+1:02d}/{len(endpoints):02d}] ERROR: {ep} -> {e}", flush=True)
        results.append((ep, "ERR", expected_code, "ERROR"))
    finally:
        conn.close()

print("=" * 80, flush=True)
print(f"VERIFICATION COMPLETE: {passed}/{len(endpoints)} PASSED ({(passed/len(endpoints))*100:.1f}%)", flush=True)
print(f"TOTAL MATCHING BASELINE: {passed}", flush=True)
print(f"TOTAL DIFFERENCES: {failed}", flush=True)
print("=" * 80, flush=True)
