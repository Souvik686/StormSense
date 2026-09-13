import os
import re

with open('src/inference/risk_surface.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add PROJECT_ROOT definition
if 'PROJECT_ROOT' not in code:
    code = code.replace(
        'WB_STATE_GEOJSON = \"Data/BOUNDARIES/west_bengal_full.geojson\"',
        '''PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"..\"))
WB_STATE_GEOJSON = os.path.join(PROJECT_ROOT, \"Data\", \"BOUNDARIES\", \"west_bengal_full.geojson\")'''
    )
    code = code.replace(
        'WB_DISTRICTS_GEOJSON = \"Data/BOUNDARIES/west_bengal_districts_full.geojson\"',
        'WB_DISTRICTS_GEOJSON = os.path.join(PROJECT_ROOT, \"Data\", \"BOUNDARIES\", \"west_bengal_districts_full.geojson\")'
    )
    code = code.replace(
        'cache_path: str = \"Data/BOUNDARIES/wb_mask_full_300x200.npy\"',
        'cache_path: str = None'
    )
    
    # Fix the method to use the default cache path from PROJECT_ROOT
    cache_logic = '''    if cache_path is None:
        cache_path = os.path.join(PROJECT_ROOT, \"Data\", \"BOUNDARIES\", \"wb_mask_full_300x200.npy\")
    
    if os.path.exists(cache_path):'''
    code = code.replace('    if os.path.exists(cache_path):', cache_logic, 1)

with open('src/inference/risk_surface.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Updated risk_surface.py')
