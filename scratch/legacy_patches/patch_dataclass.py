import re
with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_res = '''    res = HarmonizedLiveInput()
    res.surface = surf_tensor
    res.pressure = pres_tensor
    res.t0 = target_t0
    res.slot_timestamps = [th.isoformat() + "Z" for th in target_hours]
    res.slot_provenance = slot_prov
    res.fetched_at = datetime.now(timezone.utc)
    res.wallclock_age_hours = age_seconds / 3600.0'''

new_res = '''    res = HarmonizedLiveInput(
        surface=surf_tensor,
        pressure=pres_tensor,
        t0=target_t0,
        slot_timestamps=[th.isoformat() + "Z" for th in target_hours],
        slot_provenance=slot_prov,
        fetched_at=datetime.now(timezone.utc),
        wallclock_age_hours=age_seconds / 3600.0
    )'''

code = code.replace(old_res, new_res)
with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
