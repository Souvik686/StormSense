with open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

btn_html = '''      <button id="btn-refresh-live" class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 border border-cyan-400/50 shadow-sm text-xs font-bold font-mono text-white transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed" onclick="window.forceLiveRefresh()">
    <span class="material-symbols-outlined text-[14px]">refresh</span>
    <span>REFRESH LIVE DATA</span>
  </button>'''

html = html.replace(btn_html, '')
html = html.replace('\n\n  <div class="flex items-center gap-2">', '\n  <div class="flex items-center gap-2">')

target = '''<!-- Right IMD 4-Stage Severity Legend<!-- Right IMD 4-Stage Severity Legend & Quick Actions -->
<div class="flex items-center gap-3 overflow-x-auto">'''

html = html.replace(target, target + '\n' + btn_html)

with open('weather-app/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Moved button')
