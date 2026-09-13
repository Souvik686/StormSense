import urllib.request
import json
import time

def main():
    # 1. Break gfs_live.py to simulate failure
    with open('src/inference/gfs_live.py', 'r') as f:
        code = f.read()
    broken_code = code.replace('def _find_latest_cycle(client: httpx.Client, max_lookback_cycles: int = 8) -> datetime:', 'def _find_latest_cycle(client: httpx.Client, max_lookback_cycles: int = 8) -> datetime:\n    raise GfsFetchError("INTENTIONAL_FAILURE")\n')
    with open('src/inference/gfs_live.py', 'w') as f:
        f.write(broken_code)
    
    time.sleep(1) # wait for reload or just hit endpoint, uvicorn might auto-reload?
    # No, Uvicorn in production mode without --reload won't reload.
    # I should restart the server briefly or do it programmatically.
