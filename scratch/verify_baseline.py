import urllib.request
import urllib.error
import json
import os
import sys

with open("reports/baseline_behavior.json", "r", encoding="utf-8") as f:
    baseline = json.load(f)

endpoints = list(baseline.keys())
print(f"Comparing {len(endpoints)} endpoints against Phase 7 baseline...")

passed = 0
failed = 0
results = []

for ep in endpoints:
    url = f"http://127.0.0.1:8000{ep}"
    base_info = baseline[ep]
    base_status = base_info["status"]
    base_data = base_info.get("data")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PostReorgVerifier"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.status
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                data = json.loads(resp.read().decode())
            else:
                raw = resp.read()
                data = f"binary/non-json: {len(raw)} bytes"
            
            status_match = (code == base_status)
            if status_match:
                passed += 1
                results.append((ep, code, base_status, "PASS", ""))
            else:
                failed += 1
                results.append((ep, code, base_status, "FAIL", f"Status mismatch: expected {base_status}, got {code}"))
    except urllib.error.HTTPError as e:
        if e.code == base_status:
            passed += 1
            results.append((ep, e.code, base_status, "PASS", "Expected HTTP status"))
        else:
            failed += 1
            results.append((ep, e.code, base_status, "FAIL", f"Status mismatch: expected {base_status}, got {e.code}"))
    except Exception as e:
        failed += 1
        results.append((ep, "ERR", base_status, "ERROR", str(e)))

print("=" * 80)
print(f"POST-REORGANIZATION API VERIFICATION: {passed}/{len(endpoints)} PASSED ({(passed/len(endpoints))*100:.1f}%)")
print("=" * 80)
for ep, code, base_status, outcome, note in results:
    note_str = f" ({note})" if note else ""
    print(f"[{outcome:5}] {ep:60} -> HTTP {code} vs baseline {base_status}{note_str}")

print("=" * 80)
if failed == 0:
    print("SUCCESS: 100% OF ENDPOINTS MATCHED PRE-REORGANIZATION BASELINE!")
else:
    print(f"WARNING: {failed} endpoints differed.")
