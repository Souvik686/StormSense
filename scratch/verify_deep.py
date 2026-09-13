import urllib.request
import urllib.error
import json
import os
import sys

with open("reports/baseline_behavior.json", "r", encoding="utf-8") as f:
    baseline = json.load(f)

endpoints = list(baseline.keys())
print(f"Deep verifying {len(endpoints)} endpoints...")

def compare_historical(base, new, path="root"):
    diffs = []
    if isinstance(base, dict) and isinstance(new, dict):
        for k in base:
            if k not in new:
                diffs.append(f"{path}: Key '{k}' missing in new.")
            else:
                diffs.extend(compare_historical(base[k], new[k], path=f"{path}.{k}"))
        for k in new:
            if k not in base:
                diffs.append(f"{path}: Key '{k}' extra in new.")
    elif isinstance(base, list) and isinstance(new, list):
        if len(base) != len(new):
            diffs.append(f"{path}: List length mismatch {len(base)} vs {len(new)}.")
        for i, (b, n) in enumerate(zip(base, new)):
            diffs.extend(compare_historical(b, n, path=f"{path}[{i}]"))
    elif isinstance(base, float) and isinstance(new, float):
        if abs(base - new) > 1e-5:
            diffs.append(f"{path}: Value mismatch {base} vs {new}.")
    else:
        if base != new:
            diffs.append(f"{path}: Value mismatch {base} vs {new}.")
    return diffs

def compare_live(base, new, path="root"):
    diffs = []
    if isinstance(base, dict) and isinstance(new, dict):
        for k in base:
            if k not in new:
                diffs.append(f"{path}: Key '{k}' missing in new.")
            else:
                diffs.extend(compare_live(base[k], new[k], path=f"{path}.{k}"))
        for k in new:
            if k not in base:
                diffs.append(f"{path}: Key '{k}' extra in new.")
    elif isinstance(base, list) and isinstance(new, list):
        if len(base) > 0 and len(new) > 0:
            diffs.extend(compare_live(base[0], new[0], path=f"{path}[0]"))
    elif type(base) != type(new) and base is not None and new is not None:
        diffs.append(f"{path}: Type mismatch {type(base)} vs {type(new)}.")
    return diffs

diff_report = []

for ep in endpoints:
    url = f"http://127.0.0.1:8000{ep}"
    base_info = baseline[ep]
    base_status = base_info["status"]
    base_data = base_info.get("data")
    
    is_historical = "mode=historical" in ep or "/api/historical/" in ep
    mode_name = "HISTORICAL" if is_historical else "LIVE"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PostReorgVerifier"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.status
            content_type = resp.headers.get("content-type", "")
            
            if base_status == "error":
                diff_report.append({"endpoint": ep, "mode": mode_name, "before": "error", "after": code, "diff": "Expected error, got 200", "class": "REORGANIZATION REGRESSION"})
                continue
                
            if "application/json" in content_type:
                new_data = json.loads(resp.read().decode())
                
                if base_data is None:
                    diff_report.append({"endpoint": ep, "mode": mode_name, "before": "No data", "after": "JSON", "diff": "Unexpected JSON data", "class": "REORGANIZATION REGRESSION"})
                    continue
                    
                if is_historical:
                    diffs = compare_historical(base_data, new_data)
                else:
                    diffs = compare_live(base_data, new_data)
                
                if diffs:
                    diff_report.append({"endpoint": ep, "mode": mode_name, "before": "Baseline structure", "after": "New structure", "diff": "; ".join(diffs), "class": "REORGANIZATION REGRESSION"})
                else:
                    diff_report.append({"endpoint": ep, "mode": mode_name, "before": "Baseline", "after": "Baseline", "diff": "None", "class": "MATCH"})
            else:
                if base_data is not None:
                     diff_report.append({"endpoint": ep, "mode": mode_name, "before": "JSON", "after": "non-JSON", "diff": "Expected JSON data", "class": "REORGANIZATION REGRESSION"})
                else:
                     diff_report.append({"endpoint": ep, "mode": mode_name, "before": "non-JSON", "after": "non-JSON", "diff": "None", "class": "MATCH"})
    except urllib.error.HTTPError as e:
        if base_status == "error":
             diff_report.append({"endpoint": ep, "mode": mode_name, "before": "error", "after": f"HTTP {e.code}", "diff": "None", "class": "PRE-EXISTING ISSUE"})
        else:
             diff_report.append({"endpoint": ep, "mode": mode_name, "before": str(base_status), "after": f"HTTP {e.code}", "diff": f"HTTP {e.code}", "class": "REORGANIZATION REGRESSION"})
    except Exception as e:
        diff_report.append({"endpoint": ep, "mode": mode_name, "before": str(base_status), "after": "Connection Error", "diff": str(e), "class": "TEST/ENVIRONMENT ISSUE"})

print("\nFINAL COMPARISON TABLE")
print(f"{'Component':<65} | {'Before':<10} | {'After':<10} | {'Difference':<40} | {'Classification':<30}")
print("-" * 165)
for res in diff_report:
    diff_str = res['diff'][:37] + "..." if len(res['diff']) > 40 else res['diff']
    print(f"{res['endpoint']:<65} | {str(res['before']):<10} | {str(res['after']):<10} | {diff_str:<40} | {res['class']:<30}")

