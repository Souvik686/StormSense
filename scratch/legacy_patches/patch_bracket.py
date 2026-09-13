with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

js = js.replace('''        liveCountdownSeconds = LIVE_REFRESH_MS / 1000;
        setText("live-refresh-countdown", formatCountdown(liveCountdownSeconds));

        
        }

        if (isManual) {''', '''        liveCountdownSeconds = LIVE_REFRESH_MS / 1000;
        setText("live-refresh-countdown", formatCountdown(liveCountdownSeconds));

        if (isManual) {''')

with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
