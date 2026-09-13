import os

def replace_in_file(filepath, old, new):
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(filepath, 'w', encoding='latin-1') as f:
            f.write(content)
        print(f"Updated {filepath}")

replace_in_file('weather-app/index.html', 'V2 Calibrated', 'AI Forecast')
replace_in_file('weather-app/index.html', 'SevereWeatherNet', 'StormSense')
replace_in_file('weather-app/js/app.js', 'V2 Calibrated', 'AI Forecast')
replace_in_file('weather-app/js/app.js', 'SevereWeatherNet', 'StormSense')
replace_in_file('weather-app/backend/main.py', 'V2 Calibrated', 'AI Forecast')
replace_in_file('weather-app/backend/main.py', 'SevereWeatherNet', 'StormSense')
replace_in_file('src/inference/risk_map.py', 'V2 Calibrated', 'AI Forecast')
replace_in_file('src/inference/risk_map.py', 'SevereWeatherNet', 'StormSense')
replace_in_file('src/inference/nowcast_service.py', 'V2 Calibrated', 'AI Forecast')
replace_in_file('src/inference/nowcast_service.py', 'SevereWeatherNet', 'StormSense')

