with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

import re

rs_fix = '''  function applyRegionState(summary) {
    setText("profile-district", "West Bengal");

    var el = document.getElementById("active-high-risk-area");
    if (!el) return;

    if (!summary || summary.status === "unavailable") {
      el.textContent = "UNAVAILABLE";
      el.className = "text-sm font-bold text-slate-400";
      return;
    }

    var active = summary && summary.active_high_risk_district;
    if (active && active.district) {
      el.textContent = active.district + " (" + active.overall_pct + "%)";
      el.className = "text-sm font-bold text-amber-300";
    } else {
      el.textContent = "NO ACTIVE HIGH-RISK DISTRICT";
      el.className = "text-sm font-bold text-emerald-300";
    }
  }'''

js = re.sub(r'  function applyRegionState\(summary\) \{[\s\S]*?\}\n  \}', rs_fix, js)

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
