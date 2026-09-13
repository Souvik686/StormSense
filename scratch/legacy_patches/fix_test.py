import os

filepath = 'tests/test_browser_live.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('#nv-valid-time', '#fv-valid-time')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed test_browser_live.py")
