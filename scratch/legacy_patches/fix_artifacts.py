with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

js = js.replace(';\\n        window.StormSense', ';\n        window.StormSense')

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)

with open('weather-app/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

html = html.replace('\
', '\n')

with open('weather-app/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Fixed literal newline artifacts')
