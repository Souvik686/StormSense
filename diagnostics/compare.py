import sys, os
sys.path.insert(0, os.path.abspath('.'))
import os
import asyncio
import json
import csv
from datetime import datetime, timezone, timedelta
import httpx
import numpy as np

from src.inference.nowcast_service import get_nowcast_service
from backend.openweather import get_current_weather

async def get_rainviewer_past():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.rainviewer.com/public/weather-maps.json")
            data = resp.json()
            if data.get("radar") and data["radar"].get("past"):
                latest = data["radar"]["past"][-1]
                return latest
    except Exception as e:
        print("RainViewer fetch failed:", e)
    return None

async def run_comparison():
    print('Loading NowcastService...')
    svc = get_nowcast_service()
    
    districts = {
        'Kolkata': (22.5726, 88.3639),
        'North 24 Parganas': (22.7533, 88.7533),
        'Howrah': (22.5958, 88.2636),
        'Durgapur': (23.5204, 87.3119),
        'Siliguri': (26.7271, 88.3953),
        'Digha (Coastal)': (21.6266, 87.5074),
        'Purulia (Western)': (23.332, 86.365)
    }
    
    if svc.live_pred is None:
        print('Refreshing live state (may take 2-3 minutes for GFS download)...')
        svc.refresh_live_state()
        
    rv_latest = await get_rainviewer_past()
    rv_time = datetime.fromtimestamp(rv_latest['time'], tz=timezone.utc).isoformat() if rv_latest else None
    
    results = []
    
    print('Querying points...')
    for name, (lat, lon) in districts.items():
        # Using NOW (lead=0) for precipitation comparison, and +2h for forecast
        ss_now = svc.get_point_inspection(lat, lon, lead_hours=0, mode='live')
        ss_2h = svc.get_point_inspection(lat, lon, lead_hours=2, mode='live')
        
        try:
            ow = await get_current_weather(lat, lon)
        except Exception:
            ow = None
            
        row = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'timestamp_ist': datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
            'latitude': lat,
            'longitude': lon,
            'location_name': name,
            
            'stormsense_input_time': ss_2h.get('issue_time_utc') if ss_2h else None,
            'stormsense_valid_time': ss_2h.get('forecast_valid_utc') if ss_2h else None,
            'stormsense_risk': ss_2h.get('predictions', {}).get('thunderstorm_prob_pct') if ss_2h else None,
            'stormsense_rain': ss_2h.get('predictions', {}).get('heavy_rain_mm') if ss_2h else None,
            
            'openweather_timestamp': datetime.fromtimestamp(ow['dt'], tz=timezone.utc).isoformat() if ow else None,
            'openweather_product_type': 'Observation/Current',
            'openweather_rain': ow.get('rain', {}).get('1h', 0) if ow else None,
            'openweather_temperature': ow.get('main', {}).get('temp') if ow else None,
            'openweather_humidity': ow.get('main', {}).get('humidity') if ow else None,
            'openweather_wind_speed': ow.get('wind', {}).get('speed') if ow else None,
            'openweather_wind_direction': ow.get('wind', {}).get('deg') if ow else None,
            'openweather_pressure': ow.get('main', {}).get('pressure') if ow else None,
            
            'rainviewer_timestamp': rv_time,
            'rainviewer_product_type': 'Radar Observation (dBZ)',
            'rainviewer_precipitation_status': 'Data available in raster' if rv_time else None,
            
            'microsoft_timestamp': None,
            'microsoft_product_type': 'Azure Maps Weather Radar Tile (AccuWeather)',
            'microsoft_precipitation_status': None,
            'microsoft_product_description': 'No authenticated programmatic API available',
            
            'time_difference_seconds': None,
            'spatial_difference_km': None,
            'source_status': 'Live',
            'comparison_status': 'Valid'
        }
        
        if row['stormsense_input_time'] and row['openweather_timestamp']:
            try:
                t1 = datetime.strptime(row['stormsense_input_time'].replace(' UTC', '+0000'), '%d %b %Y %H:%M%z')
            except ValueError:
                t1 = datetime.now(timezone.utc)
            t2 = datetime.fromisoformat(row['openweather_timestamp'].replace('Z', '+00:00'))
            row['time_difference_seconds'] = abs((t1 - t2).total_seconds())
            
        results.append(row)
        
    print('Writing CSV...')
    keys = results[0].keys()
    with open('diagnostics/results/weather_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print('Done comparison CSV.')

if __name__ == '__main__':
    asyncio.run(run_comparison())
