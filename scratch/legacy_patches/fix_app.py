import os
import re

filepath = 'weather-app/js/app.js'
with open(filepath, 'r', encoding='latin-1') as f:
    content = f.read()

pattern = re.compile(r'updateModeUI\(targetMode\);')
injection = '''updateModeUI(targetMode);
    if (targetMode === "historical") {
        var lead = window.currentLeadHours || 2;
        if (lead === "now" || lead === "NOW") lead = 2; // Historical mode does not support NOW properly, default to 2
        window.currentLeadHours = lead;
        loadDashboardData().then(function(data) {
            window.refreshLiveDashboard();
            updateMapMode("historical");
        });
    } else {
        window.fetchLiveSurfaceData(false);
        window.refreshLiveDashboard();
        updateMapMode("live");
    }'''
content = pattern.sub(injection, content)

with open(filepath, 'w', encoding='latin-1') as f:
    f.write(content)

print("Fixed mode switching data reload.")
