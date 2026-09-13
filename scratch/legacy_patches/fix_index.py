import re

with open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Fix backtick n
html = html.replace('\
', '\n')

# Move refresh button
btn_html = '''      <button id="btn-refresh-live" class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 border border-cyan-400/50 shadow-sm text-xs font-bold font-mono text-white transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed" onclick="window.forceLiveRefresh()">
    <span class="material-symbols-outlined text-[14px]">refresh</span>
    <span>REFRESH LIVE DATA</span>
  </button>'''

# Replace button from its old location
html = re.sub(r'\s*<button id="btn-refresh-live".*?REFRESH LIVE DATA</span>\s*</button>', '', html, flags=re.DOTALL)

target = '''<!-- Right IMD 4-Stage Severity Legend & Quick Actions -->
<div class="flex items-center gap-3 overflow-x-auto">'''

# I'm going to just replace the broken double-comment too
target_broken = '''<!-- Right IMD 4-Stage Severity Legend<!-- Right IMD 4-Stage Severity Legend & Quick Actions -->
<div class="flex items-center gap-3 overflow-x-auto">'''
html = html.replace(target_broken, target)

html = html.replace(target, target + '\n' + btn_html)

# Change placeholders from - to ...
html = re.sub(r'id="([^"]+)">-</span>', r'id="\1">...</span>', html)
html = re.sub(r'id="([^"]+)">-</div>', r'id="\1">...</div>', html)
html = re.sub(r'id="([^"]+)">-</', r'id="\1">...</', html) # Also catch h3 etc

with open('weather-app/index.html', 'w', encoding='utf-8', newline='') as f:
    f.write(html)
print('Cleaned index.html')
