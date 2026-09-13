# -*- coding: utf-8 -*-
import sys

filepath = 'weather-app/index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('KALBAISHAKHI', 'CYCLONE REMAL')
code = code.replace('05 May 2024', '26 May 2024')
code = code.replace('15:00 UTC', '12:00 UTC')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched index.html successfully.")
