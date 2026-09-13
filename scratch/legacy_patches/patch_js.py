import re

with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Replace setForecastHorizon
old_sfh = r'''window\.setForecastHorizon = function \(btn, label, leadHours\) \{[\s\S]*?\}\);'''
new_sfh = '''window.setForecastHorizon = function (btn, label, leadHours) {
    if (label === 'now' || leadHours === 0 || label === 'NOW') {
      window.currentLeadHours = 'now';
      document.getElementById('now-validity-box').classList.remove('hidden');
      document.getElementById('forecast-validity-box').classList.add('hidden');
      
      if (window.stormSenseMap && window.stormSenseMap._riskSurfaceOverlay) {
        window.stormSenseMap._riskSurfaceOverlay.setOpacity(0);
      }
      
      document.querySelectorAll(".horizon-btn").forEach(function (b) {
        b.classList.remove("bg-cyan-600", "bg-red-600", "text-white", "font-bold");
        b.classList.add("text-slate-400", "font-medium");
        if (b.textContent.trim() === "NOW") {
          b.classList.add("bg-cyan-600", "text-white", "font-bold");
          b.classList.remove("text-slate-400", "font-medium");
        }
      });
      return;
    }

    var lead = leadHours;
    if (!lead) {
      if (label === "+2h") lead = 2;
      else if (label === "+4h") lead = 4;
      else if (label === "+6h") lead = 6;
      else lead = 2;
    }
    if (lead !== 2 && lead !== 4 && lead !== 6) {
      lead = lead < 3 ? 2 : (lead < 5 ? 4 : 6);
    }

    window.currentLeadHours = lead;
    document.getElementById('now-validity-box').classList.add('hidden');
    document.getElementById('forecast-validity-box').classList.remove('hidden');
    
    if (window.stormSenseMap && window.stormSenseMap._riskSurfaceOverlay) {
      window.stormSenseMap._riskSurfaceOverlay.setOpacity(0.85);
    }
    
    if (window.stormSenseMap && window.stormSenseMap.closePopup) window.stormSenseMap.closePopup();

    document.querySelectorAll(".horizon-btn").forEach(function (b) {
      b.classList.remove("bg-cyan-600", "bg-red-600", "text-white", "font-bold");
      b.classList.add("text-slate-400", "font-medium");
      var txt = b.textContent.trim();
      if (txt === "+" + lead + "h") {
        b.classList.add("bg-cyan-600", "text-white", "font-bold");
        b.classList.remove("text-slate-400", "font-medium");
      }
    });'''

js = re.sub(old_sfh, new_sfh, js)

# Inject dynamic time updates into updateDynamicTimes
old_udt = r'''document\.querySelectorAll\("#bind-valid-time, \.bind-valid-time"\)\.forEach\(function \(el\) \{ el\.textContent = validFormatted; \}\);'''
new_udt = '''document.querySelectorAll("#bind-valid-time, .bind-valid-time").forEach(function (el) { el.textContent = validFormatted; });
    var fvValid = document.getElementById("fv-valid-time");
    var fvGfs = document.getElementById("fv-gfs-time");
    if (fvValid) fvValid.textContent = validFormatted;
    if (fvGfs) fvGfs.textContent = issueFormatted;'''
js = re.sub(old_udt, new_udt, js)

# Inject into paintDashboardLive
old_pdl = r'''function paintDashboardLive\(obs, data\) \{'''
new_pdl = '''function paintDashboardLive(obs, data) {
    var nvObs = document.getElementById("nv-obs-time");
    var nvAge = document.getElementById("nv-age");
    if (nvObs && data.observed_at_utc) {
        var d = new Date(data.observed_at_utc);
        nvObs.textContent = d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) + " UTC";
    }
    if (nvAge && data.data_age_seconds !== undefined) {
        nvAge.textContent = Math.floor(data.data_age_seconds / 60) + " min";
    }'''
js = js.replace(old_pdl, new_pdl)

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
print("Updated JS!")
