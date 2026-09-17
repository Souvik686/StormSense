import sys, os
sys.path.append('.')
import numpy as np

from src.inference.risk_surface import DEFAULT_H, DEFAULT_W, WB_MIN_LAT, WB_MAX_LAT, WB_MIN_LON, WB_MAX_LON

print("=== ~10 KM VISUALIZATION GRID VERIFICATION ===")
print()

# Source grid (from config)
src_lats = np.linspace(28.0, 20.0, 33)
src_lons = np.linspace(84.0, 90.0, 25)
print(f"SOURCE GRID:")
print(f"  Rows: {len(src_lats)}")
print(f"  Cols: {len(src_lons)}")
print(f"  Lat range: {src_lats.min():.2f} to {src_lats.max():.2f}")
print(f"  Lon range: {src_lons.min():.2f} to {src_lons.max():.2f}")
print(f"  Lat spacing: {abs(src_lats[1]-src_lats[0]):.4f} deg")
print(f"  Lon spacing: {abs(src_lons[1]-src_lons[0]):.4f} deg")
print()

# Visualization grid
print(f"VISUALIZATION GRID:")
print(f"  DEFAULT_H (rows): {DEFAULT_H}")
print(f"  DEFAULT_W (cols): {DEFAULT_W}")
print(f"  WB lat range: {WB_MIN_LAT} to {WB_MAX_LAT}")
print(f"  WB lon range: {WB_MIN_LON} to {WB_MAX_LON}")

lats_fine = np.linspace(WB_MAX_LAT, WB_MIN_LAT, DEFAULT_H)
lons_fine = np.linspace(WB_MIN_LON, WB_MAX_LON, DEFAULT_W)

lat_extent = abs(WB_MAX_LAT - WB_MIN_LAT)
lon_extent = abs(WB_MAX_LON - WB_MIN_LON)

lat_intervals = DEFAULT_H - 1
lon_intervals = DEFAULT_W - 1

lat_spacing = lat_extent / lat_intervals if lat_intervals > 0 else 0
lon_spacing = lon_extent / lon_intervals if lon_intervals > 0 else 0

# Approximate km at WB latitude (~24N)
lat_km_per_deg = 111.0
lon_km_per_deg = 111.0 * np.cos(np.radians(24.0))

print(f"  Lat extent: {lat_extent:.2f} deg")
print(f"  Lon extent: {lon_extent:.2f} deg")
print(f"  Number of lat intervals: {lat_intervals}")
print(f"  Number of lon intervals: {lon_intervals}")
print(f"  Lat spacing: {lat_spacing:.6f} deg")
print(f"  Lon spacing: {lon_spacing:.6f} deg")
print(f"  Approx N-S km spacing: {lat_spacing * lat_km_per_deg:.2f} km")
print(f"  Approx E-W km spacing (at 24N): {lon_spacing * lon_km_per_deg:.2f} km")
print()

# Verify these are coordinate SAMPLE POINTS, not cells
print(f"INTERPRETATION:")
print(f"  {DEFAULT_H}x{DEFAULT_W} = {DEFAULT_H*DEFAULT_W} COORDINATE SAMPLE POINTS")
print(f"  Each point is interpolated from the {len(src_lats)}x{len(src_lons)} source grid via RegularGridInterpolator(method='cubic')")
print(f"  This is an interpolated VISUALIZATION grid, NOT a 10km ML predictive grid")
print(f"  The model itself remains at 0.25 deg (~28 km) resolution")

print()
print("First 5 fine lat values:", lats_fine[:5])
print("First 5 fine lon values:", lons_fine[:5])

