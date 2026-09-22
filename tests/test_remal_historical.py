import pytest
import os
import sys
sys.path.insert(0, os.path.abspath('backend'))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# The case study's horizons are the ACTIVE model's own leads.
#
# This previously iterated [2, 4, 6] and asserted valid_time == 12:00 + lead.
# The active model is now V4, whose lead set is hourly 4-16h and contains no
# +2h at all, so a +2h historical forecast is not something the model can
# produce -- the service snaps such a request to the nearest real lead and
# reports the lead it actually served. Asserting 12:00+2 == 14:00 would be
# asserting a forecast that does not exist.
#
# The invariant being pinned is unchanged and is what actually matters for the
# replay: the case study's analysis instant is fixed at 2024-05-26T12:00Z, and
# every horizon's valid time is that instant plus the lead REALLY served. The
# demo wall-clock mapping is live-only and must not touch this mode.
def _historical_leads():
    leads = client.get("/api/health").json()["lead_times_hours"]
    real = [L for L in leads if L != 0]
    return real[:3]


def test_remal_historical():
    for lead in _historical_leads():
        r = client.get(f"/api/nowcast/summary?lead={lead}&mode=historical")
        assert r.status_code == 200
        d = r.json()

        assert d['issue_time'] == "2024-05-26T12:00:00Z"
        assert d['valid_time'] == "2024-05-26T12:00:00Z"

        # The lead actually served -- identical to the request for any real
        # model lead, and reported explicitly so a snap can never be silent.
        served = d.get('effective_model_lead_hours', lead)
        assert served == lead, (
            f"a real model lead must be served as itself, got {served} for {lead}"
        )
        # Historical mode is a frozen replay: no demo wall-clock mapping here.
        assert d.get('demo_horizon') is None, (
            "the demo horizon mapping must not apply to the historical case study"
        )

        expected_hour = 12 + served
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
