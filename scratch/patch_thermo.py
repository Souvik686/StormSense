import re

with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

old_renderThermo = """  function renderThermodynamics(thermo) {
    if (!thermo) return;
    var cape = thermo.cape_surface != null ? thermo.cape_surface : thermo.cape_j_kg;"""

new_renderThermo = """  function renderThermodynamics(thermo) {
    if (!thermo) return;
    if (thermo.status === "unavailable") {
        setText("thermo-cape", "Unavailable (Live)");
        setText("thermo-cin", "Unavailable (Live)");
        setText("thermo-shear", "Unavailable (Live)");
        setText("thermo-li", "Unavailable");
        setText("thermo-shear-label", "Ambient Surface Wind");
        setText("thermo-risk-badge", "LIVE AMBIENT");
        setText("thermo-diagnostic", thermo.message || "Thermodynamic profile is not yet computed for live mode.");
        return;
    }
    var cape = thermo.cape_surface != null ? thermo.cape_surface : thermo.cape_j_kg;"""

js = js.replace(old_renderThermo, new_renderThermo)

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)

