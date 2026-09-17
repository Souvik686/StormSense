import os
import re

with open('backend/main.py', encoding='utf-8') as f:
    text = f.read()

replacement = """
from fastapi.responses import HTMLResponse

@app.get("/index.html", tags=["Frontend"], response_class=HTMLResponse)
@app.get("/app", tags=["Frontend"], response_class=HTMLResponse)
@app.get("/dashboard", tags=["Frontend"], response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f_idx:
            content = f_idx.read()
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        content = content.replace("GOOGLE_MAPS_API_KEY_PLACEHOLDER", google_api_key)
        return HTMLResponse(content=content)
    raise HTTPException(status_code=404, detail="index.html not found")
"""

start = text.find("@app.get(\"/index.html\", tags=[\"Frontend\"], response_class=FileResponse)")
end = text.find("# ── System & Health Endpoints", start)
if start != -1 and end != -1:
    text = text[:start] + replacement.strip() + "\n\n\n" + text[end:]
    with open('backend/main.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Patched serve_dashboard")
else:
    print("Could not find serve_dashboard")

