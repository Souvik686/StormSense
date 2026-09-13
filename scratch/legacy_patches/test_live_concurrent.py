import concurrent.futures
import urllib.request
import json
import time

def fetch(url, method='GET'):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read().decode()
    except Exception as e:
        return 500, str(e)

def main():
    print("Triggering 3 concurrent Live Refreshes...")
    urls = ['http://127.0.0.1:8000/api/live/refresh'] * 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch, u, 'POST') for u in urls]
        results = [f.result() for f in futures]
        
    for i, (status, text) in enumerate(results):
        print(f"Refresh {i+1}: HTTP {status}")
        try:
            print(json.dumps(json.loads(text)['live_status'], indent=2))
        except:
            print("ERROR parsing JSON or key missing:", text[:200])

    print("\nFetching Live Surface...")
    s_status, s_text = fetch('http://127.0.0.1:8000/api/live/surface')
    print(f"Surface HTTP {s_status}")
    try:
        print(json.dumps(json.loads(s_text), indent=2))
    except: pass
    
    print("\nFetching Summaries...")
    l_status, l_text = fetch('http://127.0.0.1:8000/api/nowcast/summary?mode=live')
    print(f"Live Summary HTTP {l_status}")
    h_status, h_text = fetch('http://127.0.0.1:8000/api/nowcast/summary?mode=historical')
    print(f"Historical Summary HTTP {h_status}")
    
    try:
        l_data = json.loads(l_text)
        h_data = json.loads(h_text)
        print(f"Live Issue Time: {l_data.get('timestamp_utc')}")
        print(f"Hist Issue Time: {h_data.get('timestamp_utc')}")
    except: pass

main()
