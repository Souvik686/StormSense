# DATA PIPELINE AUDIT

## Current StormSense Pipeline

### Model Architecture
- **Model**: SevereWeatherNetV2Plus (from src.models.v2_plus_model or similar, depending on checkpoint)
- **Checkpoint**: Uses 2_calibrated_best.pt by default, falling back to 2_best.pt or est.pt.
- **Decision Threshold**: Optimized on validation set, stored in checkpoint (defaults to 0.35 if not found).

### Variables & Inputs
- **Inputs**: (n_surface_vars=17, n_wind_vars=2, n_wind_levels=3, n_thermo_vars=3, n_thermo_levels=4, n_dem_vars=1). Accepts ERA5/GFS feature arrays.
- **Outputs**: Severe weather probability and predicted 3h rainfall.
- **Historical Source**: ERA5 Reanalysis data (cached) for the Cyclone Remal case study.
- **Live Source**: GFS (Global Forecast System) AWS S3 buckets (0.25-degree resolution) for real-time predictions.

### Spatial & Temporal Dimensions
- **Grid**: 33x25 grid covering West Bengal (approx 0.25-degree resolution).
- **Time/Lead Horizons**: NOW (t=0 analysis), +2h, +4h, +6h.
- **Timezones**: Internally UTC.

### External Providers
- **OpenWeather**: Configured via .env but typically used for fetching current point observation weather, not radar.
- **RainViewer**: Configured for frontend radar visualization (past radar frames and short-term radar extrapolation).

