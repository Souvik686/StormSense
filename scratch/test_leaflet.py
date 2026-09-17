import urllib.request
import re

with urllib.request.urlopen('http://localhost:8000/app') as r:
    html = r.read().decode('utf-8')

# Find ALL occurrences of 'leaflet' (case insensitive)
for m in re.finditer(r'leaflet', html, re.IGNORECASE):
    start = max(0, m.start()-50)
    end = min(len(html), m.end()+50)
    print(html[start:end])
    print('---')

