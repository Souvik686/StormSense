import os

filepath = 'tests/test_browser_live.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

import re
content = re.sub(r'# Check map image opacity or presence.*?style_now = await page.locator\("#ai-risk-overlay"\).get_attribute\("style"\)\s*print\(f"Overlay style at NOW: \{style_now\}"\)', '', content, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed test_browser_live.py to remove ai-risk-overlay")
