import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from src.inference.gfs_live import fetch_gfs_slice, _find_latest_cycle, _http_client
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS, load_stats

era5_stats = load_stats("Data/processed/cache/norm_stats.json")

client = _http_client()
latest_cycle = _find_latest_cycle(client)
lats = np.linspace(28.0, 20.0, 33)
lons = np.linspace(84.0, 90.0, 25)

print(f"Comparing GFS (Cycle: {latest_cycle}) vs ERA5 Training Stats...")

try:
    gfs_slice = fetch_gfs_slice(latest_cycle, 0, lats, lons)
except Exception as e:
    print("Could not fetch GFS:", e)
    sys.exit(1)

print("\n--- SURFACE VARIABLES ---")
for var in SINGLE_VARS:
    if var in gfs_slice.surface:
        gfs_val = gfs_slice.surface[var]
        g_mean = np.mean(gfs_val)
        e_mean = era5_stats[var]["mean"]
        e_std = era5_stats[var]["std"]
        z_score = abs(g_mean - e_mean) / (e_std + 1e-6)
        print(f"{var.ljust(6)} | ERA5 mean: {e_mean:8.2f} | GFS mean: {g_mean:8.2f} | Shift (Z): {z_score:5.2f}")

print("\n--- PRESSURE VARIABLES (Mean over all levels) ---")
for var in PRESSURE_VARS:
    if var in gfs_slice.pressure:
        gfs_vals = []
        for lvl, arr in gfs_slice.pressure[var].items():
            gfs_vals.append(arr)
        if gfs_vals:
            gfs_val = np.concatenate(gfs_vals)
            g_mean = np.mean(gfs_val)
            print(f"{var.ljust(6)} | GFS mean: {g_mean:8.2f} (ERA5 comparison skipped for brevity)")
