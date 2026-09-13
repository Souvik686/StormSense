import urllib.request
for path in ("/css/styles.css", "/js/app.js", "/static/risk_surfaces/lead_2h.png"):
    with urllib.request.urlopen("http://127.0.0.1:8000" + path) as resp:
        content = resp.read()
        print(path, "-> HTTP", resp.status, "| Bytes:", len(content))
