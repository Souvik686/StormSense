with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

xai_fix = '''  function renderXai(xai) {
    var root1 = document.getElementById('xai-factors-dashboard');
    var root2 = document.getElementById('xai-factors');
    if (!xai || !xai.factors || xai.status === 'unavailable') {
        var msg = '<div class="p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs font-mono text-slate-400">Attributions UNAVAILABLE for this configuration.</div>';
        if (root1) root1.innerHTML = msg;
        if (root2) root2.innerHTML = msg;
        return;
    }
    
    function buildRows(targetId) {'''
    
js = js.replace('  function renderXai(xai) {\\n    if (!xai || !xai.factors) return;\\n    function buildRows(targetId) {', xai_fix)

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
