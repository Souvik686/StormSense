import urllib.request
import json

# Test /api/config
try:
    with urllib.request.urlopen('http://localhost:8000/api/config') as r:
        data = json.loads(r.read())
        print('API CONFIG:', json.dumps(data, indent=2))
except Exception as e:
    print('API CONFIG ERROR:', e)

# Test /api/health
try:
    with urllib.request.urlopen('http://localhost:8000/api/health') as r:
        data = json.loads(r.read())
        print('HEALTH:', data.get('status', 'unknown'))
except Exception as e:
    print('HEALTH ERROR:', e)

# Test lead=0
try:
    with urllib.request.urlopen('http://localhost:8000/api/nowcast/summary?lead=0&mode=live') as r:
        data = json.loads(r.read())
        print('LEAD=0:', data.get('issue_time'), data.get('forecast_valid_time'))
except Exception as e:
    print('LEAD=0 ERROR:', e)

# Test lead=2
try:
    with urllib.request.urlopen('http://localhost:8000/api/nowcast/summary?lead=2&mode=live') as r:
        data = json.loads(r.read())
        print('LEAD=2:', data.get('issue_time'), data.get('forecast_valid_time'))
except Exception as e:
    print('LEAD=2 ERROR:', e)

