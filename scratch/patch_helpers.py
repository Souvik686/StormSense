import re

with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

def replace_function(js_code, func_name, new_body):
    pattern = re.compile(r'function\s+' + func_name + r'\s*\([^\)]*\)\s*\{[^}]*\}', re.MULTILINE)
    return pattern.sub(new_body, js_code)

js = replace_function(js, 'getHazardTrend', """function getHazardTrend(prob) {
    if (prob >= 75) return "HIGH RISK";
    if (prob >= 50) return "ELEVATED";
    if (prob >= 25) return "MONITOR";
    return "LOW";
  }""")

js = replace_function(js, 'getHazardDetail', """function getHazardDetail(hazardName, prob) {
    if (prob >= 75) return hazardName + " conditions indicate high short-term risk.";
    if (prob >= 50) return hazardName + " risk is elevated within the nowcast window.";
    if (prob >= 25) return hazardName + " conditions require continued monitoring.";
    return hazardName + " risk currently remains low.";
  }""")

js = replace_function(js, 'getOverallAction', """function getOverallAction(level) {
    if (level === "red") return "WARNING - Activate Emergency Response Protocol";
    if (level === "orange") return "ALERT - Prepare Civil Defense & Field Units";
    if (level === "yellow") return "WATCH - Maintain Vigilance";
    return "NORMAL - Nominal Observation";
  }""")

js = replace_function(js, 'getOverallStage', """function getOverallStage(level) {
    if (level === "red") return "WARNING";
    if (level === "orange") return "ALERT";
    if (level === "yellow") return "WATCH";
    return "NORMAL";
  }""")

# Let's also check for remaining ERA5 etc
js = js.replace("ERA5 ~5-day latency", "historical latency")

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)

