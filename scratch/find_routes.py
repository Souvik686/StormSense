import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('backend/main.py', encoding='utf-8') as f:
    text = f.read()

# Find root route
import re
for m in re.finditer(r'@app\.get\("(/[^"]*)"', text):
    route = m.group(1)
    start = m.start()
    end = min(len(text), start + 300)
    print(f"ROUTE: {route}")
    print(text[start:end])
    print("---")

