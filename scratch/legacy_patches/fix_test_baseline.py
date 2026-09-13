import os

filepath = 'tests/test_api.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"SevereWeatherNet V1 Baseline"', '"StormSense V1 Baseline"')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed test_api.py Baseline string")
