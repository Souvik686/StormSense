with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

js = js.replace('var summary = results[1];', 'var summary = results[1];\\n        window.StormSenseCurrentSummary = summary;')
js = js.replace('var summary = results[0];', 'var summary = results[0];\\n        window.StormSenseCurrentSummary = summary;')

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
