import re

with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

# 1. Update initial fetches to include modeQS for thermodynamics and xai
old_fetches = """fetch(API_BASE + "/api/nowcast/thermodynamics").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai").then(parseJson).catch(function () { return null; }),"""

new_fetches = """fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),"""

js = js.replace(old_fetches, new_fetches)

# 2. Add XAI to refreshLiveDashboard
old_refresh = """      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {"""

new_refresh = """      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=live").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=live").then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {"""

js = js.replace(old_refresh, new_refresh)

# 3. Handle XAI in refreshLiveDashboard
old_handle_refresh = """          if (cells) renderHotspotBeacons(window.stormSenseMap, cells);
        }

        if (window.stormSenseMap) renderContinuousRiskSurface(window.stormSenseMap, lead);"""

new_handle_refresh = """          if (cells) renderHotspotBeacons(window.stormSenseMap, cells);
        }
        
        var xai = results[4];
        if (xai) {
            if (typeof renderXai === 'function') renderXai(xai);
        }
        var thermo = results[5];
        if (thermo) {
            if (typeof renderThermodynamics === 'function') renderThermodynamics(thermo);
        }

        if (window.stormSenseMap) renderContinuousRiskSurface(window.stormSenseMap, lead);"""

js = js.replace(old_handle_refresh, new_handle_refresh)

# 4. Same for setForecastHorizon
old_horizon = """      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {"""

new_horizon = """      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {"""

js = js.replace(old_horizon, new_horizon)

old_handle_horizon = """          if (riskMap && window.gridInspectionMode) renderInspectionGrid(window.stormSenseMap, riskMap);
        }

        window.showToast("StormSense Nowcast updated: +" + lead + "h lead time", "check_circle");"""

new_handle_horizon = """          if (riskMap && window.gridInspectionMode) renderInspectionGrid(window.stormSenseMap, riskMap);
        }
        var xai = results[4];
        if (xai) {
            if (typeof renderXai === 'function') renderXai(xai);
        }
        var thermo = results[5];
        if (thermo) {
            if (typeof renderThermodynamics === 'function') renderThermodynamics(thermo);
        }

        window.showToast("StormSense Nowcast updated: +" + lead + "h lead time", "check_circle");"""

js = js.replace(old_handle_horizon, new_handle_horizon)

# 5. Disable renderXaiLive and renderThermodynamicsLive since they are now handled uniformly
old_xailive = "renderXaiLive();"
new_xailive = "// renderXaiLive();"
js = js.replace(old_xailive, new_xailive)

old_thermolive = "renderThermodynamicsLive();"
new_thermolive = "// renderThermodynamicsLive();"
js = js.replace(old_thermolive, new_thermolive)

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)

