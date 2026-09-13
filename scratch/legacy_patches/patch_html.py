with open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re

# Just replace horizon-controls
old1 = r'<div id="horizon-controls" class="flex items-center bg-slate-900 p-1 rounded-xl border border-slate-800">.*?</div>'
new1 = '''<div class="flex items-center gap-3">
<div id="horizon-controls" class="flex items-center bg-slate-900 p-1 rounded-xl border border-slate-800">
<button class="horizon-btn px-2.5 py-1 text-xs font-medium text-slate-400 hover:text-white transition-colors cursor-pointer" onclick="window.setForecastHorizon(this, 'now', 0)">NOW</button>
<button class="horizon-btn px-2.5 py-1 text-xs font-bold rounded-lg bg-cyan-600 text-white shadow-sm transition-colors cursor-pointer" onclick="window.setForecastHorizon(this, '+2h', 2)">+2h</button>
<button class="horizon-btn px-2.5 py-1 text-xs font-medium text-slate-400 hover:text-white transition-colors cursor-pointer" onclick="window.setForecastHorizon(this, '+4h', 4)">+4h</button>
<button class="horizon-btn px-2.5 py-1 text-xs font-medium text-slate-400 hover:text-white transition-colors cursor-pointer" onclick="window.setForecastHorizon(this, '+6h', 6)">+6h</button>
</div>
<div id="time-context-container" class="flex items-center bg-slate-900/50 px-3 py-1 rounded-xl border border-slate-800 min-w-[180px] h-[38px]">
<div id="now-validity-box" class="hidden flex-col text-[10px] font-mono text-slate-400 leading-tight">
<span class="font-bold text-emerald-400">CURRENT CONDITIONS</span>
<span>Observed: <span id="nv-obs-time" class="text-slate-300">--:--</span> | Age: <span id="nv-age" class="text-slate-300">--</span></span>
</div>
<div id="forecast-validity-box" class="flex flex-col text-[10px] font-mono text-slate-400 leading-tight">
<span class="font-bold text-cyan-400">AI FORECAST</span>
<span>Valid: <span id="fv-valid-time" class="text-slate-300">--:--</span><span id="fv-gfs-run-container"> | GFS Run: <span id="fv-gfs-time" class="text-slate-300">--:--</span></span></span>
</div>
</div>'''
html = re.sub(old1, new1, html, flags=re.DOTALL)

with open('weather-app/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
