import codecs
with codecs.open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# The backtick character is chr(96)
html = html.replace(chr(96) + 'n', '\n')

with codecs.open('weather-app/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
