import sys
filepath = 'weather-app/backend/main.py'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

old_code = '''    dist_to_station = ((lat - LATITUDE)**2 + (lon - LONGITUDE)**2)**0.5
    if dist_to_station < 0.25:
        weather = await current_weather(lat, lon)'''

new_code = '''    dist_to_station = ((lat - LATITUDE)**2 + (lon - LONGITUDE)**2)**0.5
    if dist_to_station < 0.25 and mode == "live":
        weather = await current_weather(lat, lon)'''

if old_code in code:
    code = code.replace(old_code, new_code)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched main.py successfully.")
else:
    print("Could not find old_code in main.py")
