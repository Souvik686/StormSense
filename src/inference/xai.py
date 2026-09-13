def _clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

def calculate_xai_factors(xai_inputs: dict):
    """
    StormSense rule-based atmospheric factor attribution.

    This is a physics-inspired attribution layer, NOT an ML
    model explanation. It estimates the relative influence of
    available atmospheric variables on hazardous weather.
    """

    rainfall_1h = xai_inputs.get("rainfall_1h_mm") or 0
    rainfall_3h = xai_inputs.get("rainfall_3h_mm") or 0
    rainfall_6h = xai_inputs.get("rainfall_6h_mm") or 0

    humidity = xai_inputs.get("humidity_percent") or 0
    dew_point = xai_inputs.get("dew_point_c")
    cape = xai_inputs.get("cape_jkg") or 0

    wind_speed = xai_inputs.get("wind_speed_kmh") or 0

    radar_dbz = xai_inputs.get("radar_dbz")

    # =====================================================
    # 1. RAINFALL FACTOR
    # =====================================================
    rainfall_score = (rainfall_1h * 5 + rainfall_3h * 2 + rainfall_6h)
    rainfall_score = _clamp(rainfall_score)

    # =====================================================
    # 2. MOISTURE FACTOR
    # =====================================================
    moisture_score = _clamp((humidity - 60) / 40 * 100)
    if dew_point is not None:
        if dew_point >= 26:
            moisture_score += 10
        elif dew_point >= 24:
            moisture_score += 5
    moisture_score = _clamp(moisture_score)

    # =====================================================
    # 3. WIND FACTOR
    # =====================================================
    wind_score = _clamp(wind_speed / 30 * 100)

    # =====================================================
    # 4. INSTABILITY / CAPE FACTOR
    # =====================================================
    if cape < 500:
        cape_score = cape / 500 * 20
    elif cape < 1000:
        cape_score = 20 + ((cape - 500) / 500 * 20)
    elif cape < 2000:
        cape_score = 40 + ((cape - 1000) / 1000 * 25)
    elif cape < 3000:
        cape_score = 65 + ((cape - 2000) / 1000 * 20)
    else:
        cape_score = 85 + ((cape - 3000) / 2000 * 15)
    cape_score = _clamp(cape_score)

    # =====================================================
    # 5. RADAR FACTOR
    # =====================================================
    radar_score = None
    if radar_dbz is not None:
        radar_score = _clamp((radar_dbz - 15) / 40 * 100)

    # =====================================================
    # AVAILABLE FACTORS
    # =====================================================
    raw_factors = {
        "rainfall": rainfall_score,
        "moisture": moisture_score,
        "wind": wind_score,
        "instability": cape_score,
    }

    if radar_score is not None:
        raw_factors["radar"] = radar_score

    # =====================================================
    # NORMALIZE CONTRIBUTIONS
    # =====================================================
    total_score = sum(raw_factors.values())
    if total_score > 0:
        contributions = {key: round((score / total_score) * 100, 1) for key, score in raw_factors.items()}
    else:
        contributions = {key: 0 for key in raw_factors}

    # Format like historical attribution for frontend compatibility
    # but strictly labelled as physics-inspired.
    factors_list = [
        {
            "name": "Convective Instability (CAPE)",
            "value": f"{round(cape, 1)} J/kg",
            "physical_role": "Physics-inspired atmospheric factor. Instability proxy.",
            "importance_rank": 1,
            "impact": f"{contributions.get('instability', 0)}%"
        },
        {
            "name": "Moisture / Humidity",
            "value": f"{round(humidity, 1)}%",
            "physical_role": "Physics-inspired atmospheric factor. Environmental moisture.",
            "importance_rank": 2,
            "impact": f"{contributions.get('moisture', 0)}%"
        },
        {
            "name": "Wind Influence",
            "value": f"{round(wind_speed, 1)} km/h",
            "physical_role": "Physics-inspired atmospheric factor. Wind proxy.",
            "importance_rank": 3,
            "impact": f"{contributions.get('wind', 0)}%"
        },
        {
            "name": "Recent Rainfall",
            "value": f"{round(rainfall_1h, 1)} mm/h",
            "physical_role": "Physics-inspired atmospheric factor. Precipitation proxy.",
            "importance_rank": 4,
            "impact": f"{contributions.get('rainfall', 0)}%"
        }
    ]

    # Sort by contribution descending, then assign importance ranks
    factors_list.sort(key=lambda x: float(x["impact"].strip("%")), reverse=True)
    for i, f in enumerate(factors_list):
        f["importance_rank"] = i + 1
        f["impact"] = "Dominant" if i == 0 else "High" if i == 1 else "Moderate" if i == 2 else "Secondary"

    return {
        "model_architecture": "Atmospheric Conditions",
        "parameters": "Rule-Based Physics Proxy",
        "attribution_method": "Physics-Inspired Factor Attribution",
        "factors": factors_list
    }

