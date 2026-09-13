import os

filepath = 'tests/test_api.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"StormSense V2 Calibrated"', '"StormSense AI Forecast"')
content = content.replace('"SevereWeatherNet V2 Calibrated"', '"StormSense AI Forecast"')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed test_api.py")
