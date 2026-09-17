# MSN Manual Check

To manually verify the discrepancy between MSN Weather and StormSense, perform the following checks on the live MSN Weather website and the StormSense local application.

## 1. Setup & Alignment
1. **MSN Weather**: Open the MSN Weather map and select the "Precipitation" or "Radar" layer. Center the map on West Bengal (e.g., Kolkata).
2. **StormSense**: Open the StormSense frontend on `http://127.0.0.1:8000`. Make sure you are in **Live Mode** (the default).

## 2. Temporal Alignment Check
- **StormSense Time**: Note the exact "issue time" and "+2h" valid time shown on the dashboard.
- **MSN Time**: Adjust the MSN Weather timeline slider to closely match the StormSense +2h valid time. (Remember to align timezones if MSN displays in local time and StormSense displays UTC/IST).

## 3. The Comparison
- **Look at MSN**: Identify specific, isolated rain cells (green/yellow/red blobs). These represent expected *precipitation reflectivity*.
- **Look at StormSense (Heavy Rain Layer)**: Switch StormSense to the "Heavy Rain (mm/3h)" forecast. Does it show rain in roughly the same broad 0.25-degree grid cells?
- **Look at StormSense (Severe Risk Layer)**: Switch to "Severe Risk (%)". Note how the risk probability covers a broad, smooth area (often larger than individual rain cells). 

## 4. Key Takeaways
- If StormSense's **Heavy Rain** layer matches the broad location of MSN's rain cells, the model's precipitation physics are correct, but MSN simply has a higher spatial resolution.
- If StormSense's **Severe Risk** layer looks different from MSN's rain cells, *this is expected*. Severe weather risk (convective instability, CAPE, etc.) is not the same as precipitation. You can have high severe-weather potential (instability) without current radar reflectivity, or high radar reflectivity (heavy stratiform rain) with low severe-weather risk.

