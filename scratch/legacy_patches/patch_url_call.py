import re
with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = re.sub(r'idx_url = _idx_url\(cycle_time, base=(.*?)\)', r'idx_url = _idx_url(cycle_time, f_hour, base=\1)', code)
code = re.sub(r'grib_url = _grib_url\(cycle_time, base=(.*?)\)', r'grib_url = _grib_url(cycle_time, f_hour, base=\1)', code)
code = re.sub(r'idx_url = _idx_url\(cycle_time, base\)', r'idx_url = _idx_url(cycle_time, f_hour, base)', code)
code = re.sub(r'grib_url = _grib_url\(cycle_time, base\)', r'grib_url = _grib_url(cycle_time, f_hour, base)', code)

with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
