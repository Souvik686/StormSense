"""Force all four verdict branches through the REAL assess() function, then
confirm the frontend renderer has a style for each."""
import sys,os
sys.path.insert(0,os.path.abspath("."))
from src.inference import now_evidence as ne
cases=[
 ("AGREE (both quiet)",        dict(model_prob=0.01, echo_fraction=0.0,  echo_coverage=True, station_rain_mm=0.0)),
 ("AGREE (both active)",       dict(model_prob=0.60, echo_fraction=0.50, echo_coverage=True, station_rain_mm=1.2)),
 ("OBSERVED_NOT_MODELLED",     dict(model_prob=0.02, echo_fraction=0.45, echo_coverage=True, station_rain_mm=0.0)),
 ("OBSERVED_NOT_MODELLED(stn)",dict(model_prob=0.02, echo_fraction=0.0,  echo_coverage=True, station_rain_mm=0.9)),
 ("MODELLED_NOT_OBSERVED",     dict(model_prob=0.66, echo_fraction=0.0,  echo_coverage=True, station_rain_mm=0.0)),
 ("NO_OBSERVATION",            dict(model_prob=0.30, echo_fraction=None, echo_coverage=False, station_rain_mm=None)),
]
seen=set()
for name,kw in cases:
    ev=ne.assess(radar_time_utc="2026-09-18T18:10:00+00:00",
                 station_time_utc="2026-09-18T18:12:00+00:00",
                 analysis_time_utc="2026-09-18T12:00:00+00:00",
                 reference_time_utc="2026-09-18T18:15:00+00:00", **kw)
    d=ev.to_dict(); seen.add(d["verdict"])
    print(f"{name:28s} -> {d['verdict']:22s} prob={d['model_probability_pct']}%")
    assert d["observations_modify_probability"] is False
    assert "adjusted" not in str(d).lower()
print("\ndistinct verdicts produced:",sorted(seen))
# confirm frontend has a style for every verdict the backend can emit
js=open("frontend/js/app.js",encoding="utf-8").read()
import re
blk=js[js.index("var NOW_VERDICT_STYLE"):js.index("function nowEvidenceHtml")]
styled=set(re.findall(r"(\w+):\s*\{ color", blk))
print("frontend styles      :",sorted(styled))
missing=seen-styled
print("MISSING STYLES       :", missing if missing else "none -- all covered")
assert not missing, missing
print("\nAll branches OK; no branch returns an adjusted probability.")
