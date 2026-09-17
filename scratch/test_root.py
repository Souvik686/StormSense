import urllib.request
import re

# What does the user see at /
try:
    with urllib.request.urlopen('http://localhost:8000/') as r:
        html = r.read().decode('utf-8')
        print('ROOT / STATUS:', r.status)
        print('ROOT is landing.html:', 'StormSense' in html[:500] and 'app.js' not in html[:2000])
        print('ROOT has link to /app:', '/app' in html)
        print('ROOT has Google Maps:', 'maps.googleapis.com' in html)
        print('ROOT has Leaflet:', 'leaflet' in html.lower())
        print()
except Exception as e:
    print('ROOT ERROR:', e)

# What does the user see at /app
try:
    with urllib.request.urlopen('http://localhost:8000/app') as r:
        html = r.read().decode('utf-8')
        print('/app STATUS:', r.status)
        print('/app has Google Maps:', 'maps.googleapis.com' in html)
        print('/app has Leaflet:', 'leaflet' in html.lower())
        m = re.search(r'maps\.googleapis\.com[^"]+', html)
        if m:
            print('/app Google key src:', m.group())
except Exception as e:
    print('/app ERROR:', e)

