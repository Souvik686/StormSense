import re
with open('src/inference/gfs_live.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Make url functions accept f_hour
old_idx = r'''def _idx_url\(cycle_time: datetime, base: str = AWS_BASE\) -> str:
    ymd = cycle_time.strftime\("%Y%m%d"\)
    hh = cycle_time.strftime\("%H"\)
    if base == AWS_BASE:
        return f"{AWS_BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000.idx"
    return f"{NOMADS_BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000.idx"'''

new_idx = '''def _idx_url(cycle_time: datetime, f_hour: int = 0, base: str = AWS_BASE) -> str:
    ymd = cycle_time.strftime("%Y%m%d")
    hh = cycle_time.strftime("%H")
    fh = f"{f_hour:03d}"
    if base == AWS_BASE:
        return f"{AWS_BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fh}.idx"
    return f"{NOMADS_BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fh}.idx"'''
code = code.replace(old_idx, new_idx)

old_grib = r'''def _grib_url\(cycle_time: datetime, base: str = AWS_BASE\) -> str:
    return _idx_url\(cycle_time, base\)\[:-4\]  # strip "\.idx"'''

new_grib = '''def _grib_url(cycle_time: datetime, f_hour: int = 0, base: str = AWS_BASE) -> str:
    return _idx_url(cycle_time, f_hour, base)[:-4]  # strip ".idx"'''
code = re.sub(old_grib, new_grib, code)

with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Updated URLs")
