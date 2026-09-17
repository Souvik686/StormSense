import os
with open('backend/main.py', encoding='utf-8') as f:
    text = f.read()

config_api = """
@app.get("/api/config", tags=["System"])
async def get_config():
    return {"google_maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY", "")}

# ── Live Weather Observations (t=0) ───────────────────────────────────────────
"""

text = text.replace("# ── Live Weather Observations (t=0) ───────────────────────────────────────────", config_api)

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.write(text)

