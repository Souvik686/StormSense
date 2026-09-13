# -*- coding: utf-8 -*-
import sys

filepath = 'weather-app/js/app.js'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Fix loadRainViewerRadar
old_radar = '''  async function loadRainViewerRadar(map) {
    if (!isRadarFeedsMap(map)) {'''

new_radar = '''  async function loadRainViewerRadar(map) {
    if (window.stormSenseMode === "historical") {
      console.warn("loadRainViewerRadar: blocked in historical mode to prevent data contamination.");
      return;
    }
    if (!isRadarFeedsMap(map)) {'''
code = code.replace(old_radar, new_radar)

# 2. Prevent fetching current weather on dashboard load for historical
old_dash = '''  function loadDashboardData() {
    var lead = apiLeadHours();
    return Promise.all([
      fetch(API_BASE + "/api/weather/current").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + "&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),'''

new_dash = '''  function loadDashboardData() {
    var lead = apiLeadHours();
    var fetchCurrent = (window.stormSenseMode === "historical") 
        ? Promise.resolve(null) 
        : fetch(API_BASE + "/api/weather/current").then(parseJson).catch(function () { return null; });
    var fetchSurface = (window.stormSenseMode === "historical") 
        ? Promise.resolve(null) 
        : fetch(API_BASE + "/api/live/surface").then(parseJson).catch(function () { return null; });

    return Promise.all([
      fetchCurrent,
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + "&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),'''
code = code.replace(old_dash, new_dash)

# 3. Prevent fetching live surface on dashboard load
old_surface = '''      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/live/surface").then(parseJson).catch(function () { return null; }),'''

new_surface = '''      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetchSurface,'''
code = code.replace(old_surface, new_surface)

# 4. Stop acquireLiveLocation if historical
old_loc = '''  function acquireLiveLocation() {
    if ("geolocation" in navigator) {'''

new_loc = '''  function acquireLiveLocation() {
    if (window.stormSenseMode === "historical") return;
    if ("geolocation" in navigator) {'''
code = code.replace(old_loc, new_loc)

# 5. Fix KALBAISHAKHI -> CYCLONE REMAL
code = code.replace('KALBAISHAKHI', 'CYCLONE REMAL')

# 6. Fix parseUtcIso default fallback to 2024-05-26T12:00:00Z
code = code.replace('2024-05-05T15:00:00Z', '2024-05-26T12:00:00Z')
code = code.replace('05 May 2024', '26 May 2024')
code = code.replace('15:00 UTC', '12:00 UTC')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched app.js successfully.")
