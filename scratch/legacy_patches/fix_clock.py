import codecs
with codecs.open("weather-app/js/app.js", "r", encoding="utf-8") as f:
    js = f.read()
old_fetch = "        window.StormSenseLiveSurface = data;"
new_fetch = "        window.StormSenseLiveSurface = data;\\n        window.currentObsTime = data.observed_at_utc;"
js = js.replace(old_fetch, new_fetch)
old_clock = "    var clockHeader = document.getElementById(\"header-clock-ist\");\\n    if (clockHeader) clockHeader.textContent = shortTimeStr;"
new_clock = old_clock + """\n    if (window.stormSenseMode === \"live\" && window.currentObsTime) {\n      var obsDate = new Date(window.currentObsTime);\n      var ageSeconds = Math.floor((now.getTime() - obsDate.getTime()) / 1000);\n      if (ageSeconds < 0) ageSeconds = 0;\n      var h = Math.floor(ageSeconds / 3600);\n      var m = Math.floor((ageSeconds % 3600) / 60);\n      var obsText = \"\";\n      if (h > 0) obsText = h + \"h \" + m + \"m ago\";\n      else if (m <= 1) obsText = \"Just now\";\n      else obsText = m + \"m ago\";\n      var obsBadge = document.getElementById(\"header-obs-time-text\");\n      if (obsBadge) obsBadge.textContent = obsText;\n    }"""
js = js.replace(old_clock, new_clock)
with codecs.open("weather-app/js/app.js", "w", encoding="utf-8") as f:
    f.write(js)
print("Updated clock logic")
