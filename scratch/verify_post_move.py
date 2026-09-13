import json
import sys

def verify():
    with open("reports/baseline_behavior.json") as f:
        baseline = json.load(f)
    try:
        with open("reports/new_behavior.json") as f:
            new = json.load(f)
    except Exception as e:
        print(f"Error loading new_behavior.json: {e}")
        return False
        
    diffs = []
    
    # Check historical values exactly
    for key, base_val in baseline.items():
        if key not in new:
            diffs.append(f"Missing key in new behavior: {key}")
            continue
            
        new_val = new[key]
        if "historical" in key.lower():
            if base_val != new_val:
                diffs.append(f"Historical mismatch for {key}: expected {base_val}, got {new_val}")
        else:
            # For live values, we just verify they exist and have the right type/structure
            if type(base_val) != type(new_val):
                diffs.append(f"Type mismatch for {key}: expected {type(base_val)}, got {type(new_val)}")
                
    if diffs:
        print("Differences found:")
        for d in diffs:
            print(" -", d)
        return False
    print("Verification passed! No historical differences, live structure matches.")
    return True

if __name__ == "__main__":
    if not verify():
        sys.exit(1)

