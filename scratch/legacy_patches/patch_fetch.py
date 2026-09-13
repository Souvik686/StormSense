import re
with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_fetch = r'''def fetch_gfs_cycle\(cycle_time: datetime, lats: np.ndarray, lons: np.ndarray\) -> GfsCycle:'''
new_fetch = '''def fetch_gfs_slice(cycle_time: datetime, f_hour: int, lats: np.ndarray, lons: np.ndarray) -> GfsCycle:'''
code = code.replace(old_fetch, new_fetch)

# Update internal calls to _idx_url, _grib_url, and the printing
code = code.replace('idx_url = _idx_url(cycle_time, base=base)', 'idx_url = _idx_url(cycle_time, f_hour, base=base)')
code = code.replace('grib_url = _grib_url(cycle_time, base=base)', 'grib_url = _grib_url(cycle_time, f_hour, base=base)')
code = code.replace('print(f"[gfs_live] Fetching cycle {cycle_time} from {base} ...")', 'print(f"[gfs_live] Fetching cycle {cycle_time} f{f_hour:03d} from {base} ...")')

with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Updated fetch")
