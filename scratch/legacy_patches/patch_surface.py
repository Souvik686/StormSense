with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

import re

# Fix fetchLiveSurfaceData to not throw error
js = js.replace('window.fetchLiveSurfaceData(false),', 'window.fetchLiveSurfaceData(false).catch(function() { return null; }),')

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
