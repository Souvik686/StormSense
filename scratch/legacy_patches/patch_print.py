# -*- coding: utf-8 -*-
import sys

filepath = 'src/inference/nowcast_service.py'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('Historical (2024-05-05) operational state loaded.', 'Historical (Cyclone Remal - 2024-05-26) operational state loaded.')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched print statement in nowcast_service.py successfully.")
