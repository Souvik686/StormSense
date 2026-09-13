import re

with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Add loadRainViewerRadar function and `var radarLayer = null;` before initRadarMap
radar_code = """
  var radarLayer = null;
  async function loadRainViewerRadar(map) {
    var statusEl = document.getElementById("radar-status");
    var infoEl = document.getElementById("radar-info");
    var updateEl = document.getElementById("radar-last-update");

    try {
        if (statusEl) statusEl.textContent = "RADAR LOADING";

        var response = await fetch("https://api.rainviewer.com/public/weather-maps.json");
        if (!response.ok) throw new Error("RainViewer API returned HTTP " + response.status);
        
        var data = await response.json();
        if (!data.radar || !data.radar.past || !data.radar.past.length) throw new Error("No radar frames available.");
        
        var latestFrame = data.radar.past[data.radar.past.length - 1];
        
        if (radarLayer && map) {
            map.removeLayer(radarLayer);
            radarLayer = null;
        }

        var tileUrl = data.host + latestFrame.path + "/256/{z}/{x}/{y}/2/1_1.png";
        
        if (map) {
            radarLayer = L.tileLayer(tileUrl, {
                opacity: 0.70,
                maxNativeZoom: 7,
                maxZoom: 16,
                attribution: 'Radar data by <a href="https://www.rainviewer.com/" target="_blank" rel="noopener">RainViewer</a>'
            });
            radarLayer.addTo(map);
        }

        if (statusEl) statusEl.textContent = "RADAR READY";
        if (infoEl) infoEl.textContent = "RainViewer precipitation radar connected";
        if (updateEl) {
            var scanTime = new Date(latestFrame.time * 1000);
            updateEl.textContent = "Last Scan: " + scanTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }
    } catch (error) {
        console.error("RainViewer radar error:", error);
        if (statusEl) statusEl.textContent = "RADAR OFFLINE";
        if (infoEl) infoEl.textContent = "Radar feed unavailable";
        if (updateEl) updateEl.textContent = "Connection failed";
    }
  }

  function initRadarMap(data) {
"""

js = js.replace('  function initRadarMap(data) {', radar_code)

# Call loadRainViewerRadar(map) right before `return map;` in `initRadarMap`
call_code = """
    if (!radarLayer) {
        loadRainViewerRadar(map);
    }
    return map;
  }
"""

js = js.replace('    return map;\n  }', call_code)

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)

