with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

import re
js = js.replace('fetch(API_BASE + "/api/nowcast/xai" + "?mode=live")', 'fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live"))')
js = js.replace('fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=live")', 'fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live"))')

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
