import os
import re

filepath = 'weather-app/js/app.js'
with open(filepath, 'r', encoding='latin-1') as f:
    content = f.read()

pattern = re.compile(r'setText\("bind-wind-source", liveSourceLabel\);')
injection = '''setText("bind-wind-source", liveSourceLabel);
      if (rawLive && rawLive.observed_at_utc) {
          var obsDate = new Date(rawLive.observed_at_utc);
          setText("nv-obs-time", obsDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
          setText("nv-age", Math.floor((rawLive.data_age_seconds || 0) / 60) + "m");
      } else {
          setText("nv-obs-time", "UNAVAILABLE");
          setText("nv-age", "--");
      }
'''

content = pattern.sub(injection, content)

with open(filepath, 'w', encoding='latin-1') as f:
    f.write(content)

print("Injected nv-obs-time updates.")
