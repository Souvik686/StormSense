import os

filepath = 'src/inference/gfs_live.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('_HARMONIZED_CACHE.t0_timestamp == target_t0', '_HARMONIZED_CACHE.t0 == target_t0')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed t0_timestamp to t0 in gfs_live.py")
