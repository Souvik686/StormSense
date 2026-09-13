import os

filepath = 'src/inference/nowcast_service.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Make target_t0 exact wall-clock time
content = content.replace('target_t0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)', 'target_t0 = datetime.now(timezone.utc)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed target_t0 to be exact wall-clock time.")
