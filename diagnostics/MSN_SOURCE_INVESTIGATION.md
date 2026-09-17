# MSN/MICROSOFT WEATHER SOURCE INVESTIGATION

## Identified Source
MSN Weather utilizes Microsoft's **Azure Maps Weather Services** for its underlying weather and radar data.

## Provider
The exclusive provider of weather data (including radar, satellite imagery, and forecasts) for Azure Maps Weather Services is **AccuWeather**.

## Product Details
- **Product Name**: Azure Maps Weather - Get Map Tile API
- **Tileset ID**: \microsoft.weather.radar.main- **Coverage**: Global, including India (West Bengal). It combines ground-based radar sources with satellite-derived radar for areas lacking ground coverage.
- **Product Type**: Real-time weather radar imagery (representing rain, snow, ice, and mixed conditions) rather than severe-weather risk.
- **Observation vs Forecast**: This is an observational radar product (reflectivity/precipitation), though AccuWeather also provides MinuteCast (forecasts).
- **Spatial Resolution**: Standard web mercator tile resolution (varies by zoom level), typically 256x256 pixel tiles.
- **Temporal Resolution**: Varies, typically updated every 10-15 minutes depending on the underlying ground/satellite radar availability in the region.

## API Availability & Access
- **API**: Yes, it is accessible programmatically via the Azure Maps REST API.
- **Access Requirements**: Requires an active Azure subscription, an Azure Maps account, and a subscription key or Microsoft Entra ID for authentication.
- **Limitations**: The tiles return visual imagery (colored pixels representing dBZ), not raw numerical precipitation values, making direct quantitative pixel-by-pixel numerical comparison challenging without interpreting the specific color legend provided in the Azure documentation.

## Applicability for Comparison
While it is the exact data source powering MSN Weather, its format (raster image tiles) and focus (precipitation rather than severe weather risk) means it cannot be directly numerically compared to StormSense's risk output. However, it can be used for visual map comparisons (precipitation location vs. StormSense risk location).

Because we do not have an active, authenticated Azure Maps subscription key available in this environment, we cannot legitimately access the API programmatically for the automated comparison. We will document this limitation and use OpenWeather and RainViewer for the automated numerical comparison.
