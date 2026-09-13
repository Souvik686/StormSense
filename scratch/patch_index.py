import re

with open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Radar status in the top bar
old_top = """<div class="p-4 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-red-400">radar</span>
<span class="font-bold text-sm text-white">AI Spatial Forecast</span>
</div>
<span id="desk-peak-prob-badge" class="text-[10px] font-mono bg-red-500/20 text-red-400 px-2 py-0.5 rounded border border-red-500/40 font-bold">PEAK PROB: 88.4%</span>
</div>"""

new_top = """<div class="p-4 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-red-400">radar</span>
<span class="font-bold text-sm text-white">AI Spatial Forecast &amp; Live Radar</span>
</div>
<div class="flex gap-2">
<span id="radar-status" class="text-[10px] font-mono bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/40 font-bold">RADAR LOADING</span>
<span id="desk-peak-prob-badge" class="text-[10px] font-mono bg-red-500/20 text-red-400 px-2 py-0.5 rounded border border-red-500/40 font-bold">PEAK PROB: 88.4%</span>
</div>
</div>"""

html = html.replace(old_top, new_top)

# Radar info in the bottom bar
old_bottom = """<div class="p-3 bg-slate-950/90 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 font-mono">
<span class="">Domain: 20.0-28.0°N, 84.0-90.0°E (West Bengal)</span>
<span class="text-cyan-400 font-bold" id="desk-lead-indicator">+2h Calibrated Horizon</span>
</div>"""

new_bottom = """<div class="p-3 bg-slate-950/90 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 font-mono">
<span id="radar-info">Domain: 20.0-28.0°N, 84.0-90.0°E (West Bengal)</span>
<div class="flex gap-4">
<span id="radar-last-update" class="text-emerald-400"></span>
<span class="text-cyan-400 font-bold" id="desk-lead-indicator">+2h Calibrated Horizon</span>
</div>
</div>"""

html = html.replace(old_bottom, new_bottom)

with open('weather-app/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

