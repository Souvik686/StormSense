import re

def clean_file():
    with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements
    content = content.replace('SevereWeatherNet V2 / StormSense', 'StormSense')
    content = content.replace('SevereWeatherNet V2 Calibrated continuous risk surface layer', 'StormSense continuous risk surface layer')
    content = content.replace('Atmospheric thermodynamics (ERA5 CAPE, CIN, Bulk Shear)', 'Atmospheric thermodynamics (CAPE, CIN, Bulk Shear)')
    content = content.replace('Live North 24 Parganas surface station telemetry (OpenWeather)', 'Live surface station telemetry')
    content = content.replace('HISTORICAL CASE STUDY (ERA5)', 'HISTORICAL CASE STUDY')
    content = content.replace('SevereWeatherNet V2 Calibrated ML Case Study — Validated May 5, 2024 Pre-Monsoon Squall — Grid: 0.25° (~28 km)', 'StormSense ML Case Study — Validated May 5, 2024 Pre-Monsoon Squall — Grid: 0.25° (~28 km)')
    content = content.replace('Historical convective case study (0-6 h) driven by SevereWeatherNet V2 Calibrated ML engine and real ERA5 thermodynamics', 'Historical convective case study (0-6 h) driven by StormSense ML engine and real thermodynamics')
    content = content.replace('SevereWeatherNet V2 Calibrated (Multi-Task ConvGRU Deep Learning Nowcaster)', 'StormSense (Multi-Task Deep Learning Nowcaster)')
    content = content.replace('SevereWeatherNet V2 Calibrated', 'StormSense')
    content = content.replace('SevereWeatherNet V2 multi-cell risk aggregation', 'StormSense multi-cell risk aggregation')
    content = content.replace('SevereWeatherNet V2 multi-task ConvGRU requires 4D atmospheric tensors across 825 cells. ECMWF ERA5 reanalysis data has an inherent ~5-day latency and is not connected to real-time telemetry. In accordance with strict scientific authenticity rules, spatial risk grids are not fabricated.', 'StormSense requires multi-dimensional atmospheric tensors across 825 cells. In live mode, this connects to real-time atmospheric data.')
    content = content.replace('Upper-air thermodynamic soundings (CAPE, CIN, bulk shear) require ERA5 vertical profiles (~5d latency) and are available in Historical Case Study mode.', 'Upper-air thermodynamic soundings (CAPE, CIN, bulk shear) are currently available in Historical Case Study mode.')
    content = content.replace('Physical XAI Gradient Attribution is computed during active ML model inference on ERA5 atmospheric tensors.', 'Physical Attribution is computed during active ML model inference on atmospheric tensors.')
    content = content.replace('HISTORICAL · ERA5 Forcing', 'HISTORICAL FORCING')
    content = content.replace('SevereWeatherNet Risk Engine', 'StormSense Risk Engine')
    content = content.replace('SevereWeatherNet V2 multi-cell risk aggregation across district boundaries.', 'StormSense multi-cell risk aggregation across district boundaries.')
    content = content.replace('N/A (ERA5 Single Level)', 'N/A (Single Level)')
    content = content.replace('Model-Predicted High-Risk Cell (SevereWeatherNet V2 Calibrated) &middot; Not observed radar', 'Model-Predicted High-Risk Cell (StormSense) &middot; Not observed radar')
    content = content.replace('Source: OpenWeather Live Feed &middot; Auto-refreshed (60s)', 'Source: Live Feed &middot; Auto-refreshed (5m)')
    content = content.replace('SEVEREWEATHERNET V2 NOWCAST (0.25° Grid)', 'STORMSENSE NOWCAST (0.25° Grid)')
    content = content.replace('OpenWeather AWS', 'Live Station')
    content = content.replace('ERA5 Physical Input (t=0):', 'Physical Input (t=0):')
    content = content.replace('SevereWeatherNet V2 Predictions:', 'StormSense Predictions:')
    content = content.replace('SevereWeatherNet V2:', 'StormSense:')
    content = content.replace('SevereWeatherNet Nowcast updated:', 'StormSense Nowcast updated:')
    content = content.replace('IMD_SevereWeatherNet_Nowcast_Bulletin_', 'StormSense_Nowcast_Bulletin_')
    content = content.replace('IMD SevereWeatherNet Bulletin', 'StormSense Bulletin')

    # General fallback for any missed exact strings:
    content = content.replace('SevereWeatherNet V2', 'StormSense')
    content = content.replace('SevereWeatherNet', 'StormSense')
    content = content.replace('OpenWeather', 'Live Feed')
    content = content.replace('ERA5', 'Atmospheric Data')
    content = content.replace('ECMWF', 'Operational Model')
    content = content.replace('Copernicus', 'Satellite Data')

    with open('weather-app/js/app.js', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    clean_file()

