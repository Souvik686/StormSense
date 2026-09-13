import re
with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_fh = code.split('def fetch_and_harmonize')[1]
old_fh = 'def fetch_and_harmonize' + old_fh

new_fh = '''def fetch_and_harmonize(
    lats: np.ndarray, lons: np.ndarray, target_t0: datetime, use_cache: bool = True
) -> HarmonizedLiveInput:
    """Full pipeline: find the latest GFS cycle, then explicitly fetch the required 
    forecast hours to bracket the 6-hour sequence ending exactly at 	arget_t0.
    """
    global _HARMONIZED_CACHE
    H, W = len(lats), len(lons)

    client = _http_client()
    latest_cycle = _find_latest_cycle(client)

    # Cache check
    if use_cache and _HARMONIZED_CACHE is not None:
        if _HARMONIZED_CACHE.t0_timestamp == target_t0:
            return _HARMONIZED_CACHE

    print(f"[gfs_live] target_t0={target_t0}, latest_cycle={latest_cycle}")

    # The model expects t-5, t-4, t-3, t-2, t-1, t0 (6 slots)
    target_hours = [target_t0 - timedelta(hours=h) for h in range(5, -1, -1)]
    
    # We need GFS slices that bracket these hours.
    # GFS output is 3-hourly: 0, 3, 6, 9...
    # For each target hour, find the exact float offset from the latest cycle
    needed_f_hours = set()
    for th in target_hours:
        delta_h = (th - latest_cycle).total_seconds() / 3600.0
        # If delta_h is negative, it's before the latest cycle. We could use previous cycles, 
        # but to keep it simple and robust, we just use the nearest available cycle.
        # Wait, if delta_h < 0, we MUST use a previous cycle because f-3 doesn't exist.
        pass

    # A better, highly robust approach: find the "anchor cycle" which is <= target_t0 - 5h.
    # Then all targets are >= f000 for that cycle.
    anchor_cycle = latest_cycle
    while anchor_cycle > target_hours[0]:
        anchor_cycle -= timedelta(hours=6)
    
    print(f"[gfs_live] Using anchor cycle {anchor_cycle} to ensure all f_hours >= 0")

    needed_f_hours = set()
    for th in target_hours:
        delta_h = (th - anchor_cycle).total_seconds() / 3600.0
        lower_f = int(np.floor(delta_h / 3.0) * 3)
        upper_f = int(np.ceil(delta_h / 3.0) * 3)
        needed_f_hours.add(lower_f)
        needed_f_hours.add(upper_f)

    needed_f_hours = sorted(list(needed_f_hours))
    slices = {}
    for fh in needed_f_hours:
        slices[fh] = fetch_gfs_slice(anchor_cycle, fh, lats, lons)

    # Now interpolate onto the 6 hourly slots
    out_surface = []
    out_pressure = []

    for th in target_hours:
        delta_h = (th - anchor_cycle).total_seconds() / 3600.0
        lower_f = int(np.floor(delta_h / 3.0) * 3)
        upper_f = int(np.ceil(delta_h / 3.0) * 3)

        s_low = slices[lower_f]
        s_up = slices[upper_f]

        if upper_f == lower_f:
            w_up = 0.0
        else:
            w_up = (delta_h - lower_f) / 3.0
        w_low = 1.0 - w_up

        # Interpolate surface
        surf_interp = {}
        for k in SINGLE_VARS:
            surf_interp[k] = s_low.surface[k] * w_low + s_up.surface[k] * w_up
        out_surface.append(surf_interp)

        # Interpolate pressure
        pres_interp = {}
        for k in PRESSURE_VARS:
            pres_interp[k] = {}
            for lvl in s_low.pressure[k].keys():
                pres_interp[k][lvl] = s_low.pressure[k][lvl] * w_low + s_up.pressure[k][lvl] * w_up
        out_pressure.append(pres_interp)

    # Build tensors
    surf_tensor = _build_surface_tensor(out_surface, H, W)
    pres_tensor = _build_pressure_tensor(out_pressure, H, W)

    # We track provenance
    provenance = []
    for th in target_hours:
        delta_h = (th - anchor_cycle).total_seconds() / 3600.0
        provenance.append(f"cycle={anchor_cycle.isoformat()}Z offset={delta_h:.2f}h")

    # Final inputs
    inputs_dict = {
        "surface": surf_tensor,
        "pressure": pres_tensor
    }

    # Determine data age relative to wall clock
    age_seconds = (datetime.now(timezone.utc) - target_t0).total_seconds()

    res = HarmonizedLiveInput(
        t0_timestamp=target_t0,
        provenance_log=provenance,
        inputs=inputs_dict,
        data_age_seconds=max(0.0, age_seconds)
    )
    _HARMONIZED_CACHE = res
    return res
'''
code = code.replace(old_fh, new_fh)
with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Updated fetch_and_harmonize")
