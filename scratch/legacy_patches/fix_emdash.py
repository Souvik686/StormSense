with open("weather-app/index.html", "r", encoding="utf-8") as f:
    html = f.read()
html = html.replace(">—</span>", ">...</span>")
html = html.replace(">—</div>", ">...</div>")
with open("weather-app/index.html", "w", encoding="utf-8", newline="") as f:
    f.write(html)
print("Done")
