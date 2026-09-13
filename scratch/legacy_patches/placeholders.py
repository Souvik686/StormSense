import re
import codecs

with codecs.open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

html = re.sub(r'id="([^"]+)">-<', r'id="\1">...<', html)

with codecs.open('weather-app/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Updated placeholders')
