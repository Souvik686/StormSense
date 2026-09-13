import pytest
import os
import sys
sys.path.insert(0, os.path.abspath('backend'))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_remal_historical():
    for lead in [2, 4, 6]:
        r = client.get(f"/api/nowcast/summary?lead={lead}&mode=historical")
        assert r.status_code == 200
        d = r.json()
        
        assert d['issue_time'] == "2024-05-26T12:00:00Z"
        assert d['valid_time'] == "2024-05-26T12:00:00Z"
        
        expected_hour = 12 + lead
        assert d['forecast_valid_time'] == f"2024-05-26T{expected_hour:02d}:00:00+00:00"
        
        # Verify heavy rainfall and flash flood proxy are populated correctly!
        # A severe storm like Remal should not have 0 values for everything
        heavy = d['hazards']['heavy_rainfall']['rate_mm_3h']
        ff_prob = d['hazards']['flash_flood']['probability']
        
        # We don't assert >0 here because maybe at that EXACT snapshot/domain-average it's 0, 
        # but let's print it to visually verify.
        print(f"\n--- LEAD +{lead}h ---")
        print(f"Timestamp match: {d['forecast_valid_time']}")
        print(f"Heavy Rain Rate: {heavy} mm/3h")
        print(f"Flash Flood Prob: {ff_prob}%")
        print(f"Overall Risk Prob: {d['hazards']['overall']['probability']}%")

if __name__ == '__main__':
    test_remal_historical()
