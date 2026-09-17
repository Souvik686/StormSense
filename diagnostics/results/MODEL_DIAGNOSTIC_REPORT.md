# StormSense Model Diagnostic

## 1. Executive Summary
This report analyzes the structural and numerical differences between StormSense's ML-based predictions and external weather products (like MSN Weather/Azure Maps, OpenWeather, and RainViewer) to determine if discrepancies stem from model errors, spatial/temporal mismatches, or product differences. No retraining or model modification was performed. The discrepancies observed are primarily caused by product semantic differences (predictive severe weather risk vs. observational precipitation reflectivity), spatial resolution mismatches (0.25 deg grid vs 1km radar tiles), and different data providers.

## 2. Current StormSense Pipeline
- **Model**: `SevereWeatherNetV2Plus`, an advanced multimodal spatiotemporal model.
- **Checkpoint**: `v2_calibrated_best.pt` (a 9 MB artifact).
- **Inputs/Outputs**: Ingests surface and upper-air variables (e.g., U/V winds, CAPE, CIN, TCWV) and DEM on a 33x25 grid (0.25-degree resolution). Outputs Severe Weather Risk Probability and 3-hour precipitation accumulation.
- **Sources**: GFS (Global Forecast System) AWS S3 buckets for live mode, ERA5 reanalysis for historical mode.

## 3. MSN/Microsoft Weather Source Investigation
- **Product**: MSN Weather utilizes **Azure Maps Weather Services**.
- **Provider**: **AccuWeather** is the exclusive provider.
- **Data Source**: The radar map shown on MSN is a real-time observation product (`microsoft.weather.radar.main` tileset) representing radar reflectivity (dBZ) combined from ground stations and satellite derivations.
- **Comparability**: MSN shows *current or forecasted precipitation reflectivity*, whereas StormSense shows *probability of severe weather*. A rain cell on MSN does not strictly equate to severe risk on StormSense, leading to visual divergence.

## 4. OpenWeather Comparison
- **Usage**: Used for current point observations (temperature, humidity, precipitation).
- **Comparison**: OpenWeather provides point-level current weather conditions, while StormSense forecasts risk and precipitation accumulation (+2h, +4h, +6h) over a coarse 0.25-degree grid. Time synchronization is vital when comparing StormSense's target valid time with OpenWeather's current observation time.

## 5. RainViewer Comparison
- **Usage**: Provides the live radar layer in the StormSense frontend.
- **Product**: Observational radar reflectivity tiles based on past/current radar scans. It is strictly a visual raster product with high spatial resolution but is not a forecast model.

## 6. Existing Observation Comparison
From our automated point sampling across West Bengal, we compared StormSense's 2-hour forecast against current OpenWeather and RainViewer observations. 
We observed:
- OpenWeather mostly recorded 0 mm or low precipitation.
- StormSense predicted 0.0 to 3.2 mm/3h of accumulation.
- The values are physically consistent but represent different variables: OpenWeather is an instantaneous point reading, while StormSense is a 3-hour accumulation over a 0.25-degree area.

## 7. Timestamp Alignment
StormSense's GFS input corresponds to an initialization time (e.g., 00Z, 06Z, 12Z, 18Z), and predictions are valid for +2h, +4h, +6h. External observational data (OpenWeather, RainViewer) corresponds to current wall-clock time. Aligning these requires accounting for the StormSense target time vs. observation time, which leads to typical lag offsets of ~7.5 hours from GFS analysis time.

## 8. Spatial Alignment
- **StormSense**: 0.25-degree grid (~25-28 km resolution).
- **MSN/RainViewer**: High-resolution web mercator tiles (~1-2 km or better).
- **Result**: Fine-scale rain cells in radar products may not perfectly align with the coarse StormSense grid cells.

## 9. Precipitation Comparison
Because StormSense outputs 3-hour accumulation and OpenWeather/MSN outputs 1-hour or instantaneous rates, exact numerical equivalence is mathematically impossible. A cell dropping heavy rain for 15 minutes will appear intense on MSN/RainViewer but may average out to a moderate value on StormSense's 3-hour, 25-km block.

## 10. Severe-Weather Risk Comparison
Severe-weather risk is a probabilistic atmospheric instability metric, not directly comparable to precipitation occurrence on an external provider's radar map. External rain data cannot be used as ground truth for severe-weather risk without explicitly conflating the two. The model is tuned to CAPE/CIN/wind shear triggers, not merely reflectivity.

## 11. Detected Pipeline Problems
No egregious pipeline bugs (e.g., latitude/longitude inversions, UTC parsing errors) were detected. The interpolation and grid mapping work as designed. 

## 12. Model Problems
The model operates exactly as expected given its 0.25-degree input resolution and physics-based target variables. Visual discrepancies with MSN are an expected artifact of comparing radar reflectivity (MSN) against instability probability (StormSense).

## 13. Data Limitations
Azure Maps Weather API cannot be accessed legitimately without a subscription key, so programmatic comparisons against MSN are limited to public providers like OpenWeather and RainViewer.

## 14. MSN Visual Comparison Guidance
To manually compare MSN Weather with StormSense:
1. **Location**: Center both maps on Kolkata.
2. **Product Alignment**: On MSN, select "Precipitation" or "Radar". On StormSense, look at the "+2h Forecast" Heavy Rain layer, NOT the Severe Risk layer.
3. **Temporal Alignment**: Ensure MSN is showing the forecast for +2 hours from the StormSense GFS Initialization Time.
4. **Expected Differences**: MSN will show finer spatial detail (AccuWeather data) compared to StormSense's 0.25-degree blocks (GFS-based).

## 15. Final Recommendation

**DO NOT RETRAIN**.

1. Is the current StormSense model demonstrably wrong? **No.**
2. Are the observed differences mainly caused by different data sources/products? **Yes (AccuWeather vs GFS, Reflectivity vs Risk).**
3. Are there pipeline bugs? **No.**
4. Is retraining justified? **No.**
5. If retraining is NOT justified, what should be changed instead? **Educate stakeholders on the distinction between Severe Weather Risk and radar reflectivity, and clarify the 0.25-degree spatial resolution limitations.**

