import re
from datetime import datetime, timezone

with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace the instantiation of HarmonizedLiveInput
old_res = '''    res = HarmonizedLiveInput(
        t0_timestamp=target_t0,
        provenance_log=provenance,
        inputs=inputs_dict,
        data_age_seconds=max(0.0, age_seconds)
    )'''

new_res = '''    
    slot_prov = []
    for prov in provenance:
        slot_prov.append({"source": "interpolated", "details": prov})
        
    res = HarmonizedLiveInput()
    res.surface = surf_tensor
    res.pressure = pres_tensor
    res.t0 = target_t0
    res.slot_timestamps = [th.isoformat() + "Z" for th in target_hours]
    res.slot_provenance = slot_prov
    res.fetched_at = datetime.now(timezone.utc)
    res.wallclock_age_hours = age_seconds / 3600.0
'''
code = code.replace(old_res, new_res)
with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
