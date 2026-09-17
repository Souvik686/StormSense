import urllib.request
import re

try:
    with urllib.request.urlopen('http://localhost:8000/dashboard') as r:
        html = r.read().decode('utf-8')
        print('STATUS:', r.status)
        print('Has Google Maps:', 'maps.googleapis.com' in html)
        print('Has Leaflet CSS:', 'leaflet.css' in html)
        print('Has Leaflet JS:', 'leaflet.js' in html)
        print('Google key injected:', 'GOOGLE_MAPS_API_KEY_PLACEHOLDER' not in html)
        print('Key placeholder still present:', 'GOOGLE_MAPS_API_KEY_PLACEHOLDER' in html)
        # Extract the actual Google Maps script src
        m = re.search(r'maps\.googleapis\.com[^"]+', html)
        if m:
            print('Actual Google src:', m.group())
except Exception as e:
    print('ERROR:', e)

