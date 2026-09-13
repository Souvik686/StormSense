import json

with open('reports/baseline_behavior.json', 'r') as f:
    d = json.load(f)

for mode in ('live', 'historical'):
    print('=' * 80)
    print(f'POINT INSPECTION (22.5N, 88.3E): {mode.upper()} MODE')
    print('=' * 80)
    for lead in (2, 4, 6):
        k = f'/api/nowcast/point?lat=22.5&lon=88.3&lead={lead}&mode={mode}'
        data = d[k]['data']
        preds = data.get('predictions', {})
        obs = data.get('observed_inputs', {})
        print(f'LEAD +{lead}h:')
        print('  district:            ', data.get('district'))
        print('  issue_time_utc:      ', data.get('issue_time_utc'))
        print('  forecast_valid_utc:  ', data.get('forecast_valid_utc'))
        print('  thunderstorm_prob_pct:', preds.get('thunderstorm_prob_pct'))
        print('  heavy_rain_mm:       ', preds.get('heavy_rain_mm'))
        print('  flash_flood_proxy_pct:', preds.get('flash_flood_proxy_pct'))
        print('  overall_risk_pct:    ', preds.get('overall_risk_pct'))
        print('  risk_level:          ', preds.get('risk_level'))
        print('  obs temp/humidity:   ', obs.get('temperature_c'), 'C,', obs.get('humidity_pct'), '%')
        print('  obs rainfall_mm:     ', obs.get('rainfall_mm'), 'mm')
        print('  obs wind_kmh:        ', obs.get('wind_kmh'), 'km/h')
