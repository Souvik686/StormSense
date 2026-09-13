import re
with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace definition
old_def = 'def fetch_gfs_cycle(cycle_time: datetime, lats: np.ndarray, lons: np.ndarray) -> GfsCycle:'
new_def = 'def fetch_gfs_slice(cycle_time: datetime, f_hour: int, lats: np.ndarray, lons: np.ndarray) -> GfsCycle:'
code = code.replace(old_def, new_def)

# Update internal urls
code = code.replace('idx_url = _idx_url(cycle_time, base=base)', 'idx_url = _idx_url(cycle_time, f_hour, base=base)')
code = code.replace('grib_url = _grib_url(cycle_time, base=base)', 'grib_url = _grib_url(cycle_time, f_hour, base=base)')

with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
