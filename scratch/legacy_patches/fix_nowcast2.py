import os
with open('src/inference/nowcast_service.py', 'r', encoding='utf-8') as f:
    code = f.read()

imports = 'import os\\n'
new_imports = 'import os\\nPROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"..\"))\\n'

if 'PROJECT_ROOT = ' not in code:
    code = code.replace(imports, new_imports, 1)

with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Fixed nowcast_service.py')
