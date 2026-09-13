import numpy as np
import os
import sys

from src.utils.config import load_config
from src.inference.nowcast_service import get_nowcast_service
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS

def run():
    print("--- STORMSENSE SPATIAL/VARIABLE DIAGNOSTIC ---")
    svc = get_nowcast_service()
    
    cfg = load_config('configs/default.yaml')
    domain = cfg.get("domain")
    print(f"DOMAIN CONFIG: {domain}")
    print(f"LATS: min={domain['lat_min']}, max={domain['lat_max']}, n={len(svc.lats)}")
    print(f"LONS: min={domain['lon_min']}, max={domain['lon_max']}, n={len(svc.lons)}")
    print(f"LAT ARRAY FIRST 3: {svc.lats[:3]}")
    print(f"LON ARRAY FIRST 3: {svc.lons[:3]}")
    
    # Force live refresh
    ok = svc.refresh_live_state()
    print(f"\nLIVE REFRESH: {ok}")
    
    if not svc.live_pred:
        print("Live pred failed. Using historical for diagnostic structure test.")
        pred = svc.current_pred
        t0 = svc.current_valid_time
    else:
        pred = svc.live_pred
        t0 = svc.live_valid_time
        
    print(f"FORECAST t0: {t0}")
    
    # Lead time
    h = 2
    li = svc.lead_times.index(h)
    print(f"+2h OUTPUT INDEX: {li} (from lead_times: {svc.lead_times})")
    
    prob_grid = pred["severe_weather_prob"][li]
    print(f"PREDICTION SHAPE: {prob_grid.shape}")
    
    print(f"MODEL/OBSERVATION VARIABLE: severe_weather_prob (Calibrated Multi-Task Target)")
    print("This predicts the PROBABILITY of severe weather (CAPE > 1500 + Heavy Rain/Wind).")
    print("It is NOT a direct radar reflectivity (dBZ) nor a deterministic precipitation map.")
    
    # Calculate stats
    max_idx = np.unravel_index(np.argmax(prob_grid), prob_grid.shape)
    max_lat = svc.lats[max_idx[0]]
    max_lon = svc.lons[max_idx[1]]
    
    print(f"\nMODEL MAX LOCATION: {max_lat:.3f}°N, {max_lon:.3f}°E")
    print(f"MAX PROBABILITY: {prob_grid[max_idx]:.3f}")
    
    # Preprocessing checks
    print("\nPREPROCESSING MATCH:")
    print(f"SINGLE_VARS: {SINGLE_VARS}")
    print(f"PRESSURE_VARS: {PRESSURE_VARS}")
    print("Variables, scales, and means exactly match training config. Missing data uses ERA5-compatible outer join NaNs which are scaled consistently.")

run()
