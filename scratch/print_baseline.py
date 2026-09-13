import json

with open('reports/baseline_behavior.json', 'r') as f:
    d = json.load(f)

for mode in ('live', 'historical'):
    print('=' * 80)
    print(f'BASELINE SUMMARY: {mode.upper()} MODE')
    print('=' * 80)
    for lead in (2, 4, 6):
        k = f'/api/nowcast/summary?lead={lead}&mode={mode}'
        data = d[k]['data']
        print(f'LEAD +{lead}h:')
        print('  issue_time:         ', data.get('issue_time'))
        print('  forecast_valid_time:', data.get('forecast_valid_time'))
        print('  analysis_time_iso:  ', data.get('analysis_time_iso'))
        hz = data.get('hazards', {})
        print('  thunderstorm:       ', hz.get('thunderstorm', {}).get('probability'), '%')
        hr_rate = hz.get('heavy_rainfall', {}).get('rate_mm_3h')
        hr_prob = hz.get('heavy_rainfall', {}).get('probability')
        print(f'  heavy_rainfall:      {hr_rate} mm/3h ({hr_prob}%)')
        print('  flash_flood:        ', hz.get('flash_flood', {}).get('probability'), '%')
        print('  overall:            ', hz.get('overall', {}).get('probability'), '%')
