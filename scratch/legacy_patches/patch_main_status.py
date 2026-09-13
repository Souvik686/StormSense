# -*- coding: utf-8 -*-
import sys

filepath = 'weather-app/backend/main.py'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('"event": "Kalbaishakhi Pre-Monsoon Convective Squall"', '"event": "Cyclone Remal (Landfall Approach)"')
code = code.replace('Kalbaishakhi Pre-Monsoon Event', 'Cyclone Remal (Landfall Approach)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched main.py mode status successfully.")
