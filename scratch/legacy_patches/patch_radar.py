# -*- coding: utf-8 -*-
import sys

filepath = 'weather-app/js/app.js'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

old_radar_init = '''    if (!radarLayer) {
        loadRainViewerRadar(map);
    }
    return map;'''

new_radar_init = '''    if (window.stormSenseMode === "historical") {
        var msg = L.divIcon({
            className: 'radar-archive-msg',
            html: '<div style="color:#00e5ff; background:rgba(0,0,0,0.8); padding:10px; border:1px solid #00e5ff; text-align:center; font-family:monospace;"><b>CYCLONE REMAL RADAR ARCHIVE</b><br/>26 May 2024 12:00 UTC<br/>(Static Reference Snapshot)</div>',
            iconSize: [300, 60]
        });
        L.marker([24.0, 87.5], {icon: msg}).addTo(map);
    } else {
        if (!radarLayer) {
            loadRainViewerRadar(map);
        }
    }
    return map;'''

if old_radar_init in code:
    code = code.replace(old_radar_init, new_radar_init)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched app.js initRadarMap successfully.")
else:
    print("Could not find old_radar_init in app.js")
