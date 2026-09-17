import sys, os
sys.path.append('.')
sys.stdout.reconfigure(encoding='utf-8')

with open('src/inference/nowcast_service.py', encoding='utf-8') as f:
    text = f.read()

# Find the refresh_live_state method which generates T0
start = text.find('def refresh_live_state')
if start == -1:
    # Try alternative names
    for name in ['def _refresh_live', 'def refresh_live', 'refresh_live_state']:
        start = text.find(name)
        if start != -1:
            break

if start != -1:
    print("=== T0/NOW DATA LINEAGE ===")
    print()
    # Show the T0 generation code
    end = min(len(text), start + 3000)
    code = text[start:end]
    
    print("Key code section from nowcast_service.py:")
    print()
    
    # Look for the T-2h logic
    if 'timedelta(hours=2)' in code:
        print("✓ Found T-2h offset logic (timedelta(hours=2))")
    else:
        print("✗ No T-2h offset found!")
    
    if 'preds_dict_now' in code:
        print("✓ Found separate NOW prediction variable (preds_dict_now)")
    else:
        print("✗ No separate NOW prediction!")
    
    if 'concatenate' in code:
        print("✓ Found concatenation of NOW + forecast predictions")
    else:
        print("✗ No concatenation!")
    
    if 'fetch_and_harmonize' in code:
        print("✓ Found fetch_and_harmonize call")
    else:
        print("✗ No fetch_and_harmonize!")
    
    print()
    print("--- Relevant code excerpt ---")
    # Find the specific T-2h section
    t2h_start = code.find('target_t0 - timedelta')
    if t2h_start == -1:
        t2h_start = code.find('timedelta(hours=2)')
    if t2h_start != -1:
        excerpt_start = max(0, t2h_start - 200)
        excerpt_end = min(len(code), t2h_start + 600)
        print(code[excerpt_start:excerpt_end])
else:
    print("Could not find refresh_live_state method!")
    
print()
print("=== Checking lead_times ===")
lt_start = text.find('self.lead_times')
if lt_start != -1:
    print(text[lt_start:lt_start+80])

