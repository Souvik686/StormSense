def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def risk_level(probability):
    if probability >= 75:
        return "red"
    elif probability >= 50:
        return "orange"
    elif probability >= 25:
        return "yellow"
    else:
        return "green"


def calculate_thunderstorm_risk(current, forecast):
    score = 0

    weather = (current.get("weather") or "").lower()

    # Existing thunderstorm condition
    if "thunderstorm" in weather:
        score += 70
    elif "rain" in weather:
        score += 20
    elif "cloud" in weather:
        score += 10

    # Humidity
    humidity = current.get("humidity", 0) or 0

    if humidity >= 85:
        score += 15
    elif humidity >= 75:
        score += 10
    elif humidity >= 65:
        score += 5

    # Wind
    wind = current.get("wind_speed", 0) or 0

    if wind >= 40:
        score += 15
    elif wind >= 25:
        score += 10
    elif wind >= 15:
        score += 5

    # Forecast thunderstorm signals
    for item in forecast:
        forecast_weather = (item.get("weather") or "").lower()

        if "thunderstorm" in forecast_weather:
            score += 20
            break

    probability = clamp(score)

    return {
        "probability": round(probability),
        "level": risk_level(probability)
    }


def calculate_heavy_rain_risk(current, forecast):
    score = 0

    current_rain = current.get("rainfall_1h", 0) or 0

    # Current rainfall
    if current_rain >= 20:
        score += 60
    elif current_rain >= 10:
        score += 45
    elif current_rain >= 5:
        score += 30
    elif current_rain > 0:
        score += 15

    # Forecast rainfall
    max_forecast_rain = 0

    for item in forecast:
        rain = item.get("rainfall_3h", 0) or 0
        max_forecast_rain = max(max_forecast_rain, rain)

    if max_forecast_rain >= 30:
        score += 40
    elif max_forecast_rain >= 20:
        score += 30
    elif max_forecast_rain >= 10:
        score += 20
    elif max_forecast_rain > 0:
        score += 10

    # Rain persistence
    rainy_periods = sum(
        1
        for item in forecast
        if (item.get("rainfall_3h", 0) or 0) > 0
    )

    if rainy_periods >= 3:
        score += 15
    elif rainy_periods >= 2:
        score += 10

    probability = clamp(score)

    return {
        "probability": round(probability),
        "level": risk_level(probability)
    }


def calculate_flash_flood_risk(current, forecast):
    score = 0

    current_rain = current.get("rainfall_1h", 0) or 0

    # Intense recent rainfall
    if current_rain >= 30:
        score += 50
    elif current_rain >= 20:
        score += 40
    elif current_rain >= 10:
        score += 25
    elif current_rain > 0:
        score += 10

    # Forecast accumulation
    total_forecast_rain = sum(
        item.get("rainfall_3h", 0) or 0
        for item in forecast
    )

    if total_forecast_rain >= 60:
        score += 35
    elif total_forecast_rain >= 40:
        score += 25
    elif total_forecast_rain >= 20:
        score += 15
    elif total_forecast_rain > 0:
        score += 5

    # Persistent rainfall
    rainy_periods = sum(
        1
        for item in forecast
        if (item.get("rainfall_3h", 0) or 0) > 0
    )

    if rainy_periods >= 3:
        score += 15
    elif rainy_periods >= 2:
        score += 10

    probability = clamp(score)

    return {
        "probability": round(probability),
        "level": risk_level(probability)
    }


def calculate_hazards(current=None, forecast=None, lead_hours=2):
    """Calculate multi-hazard probabilities using SevereWeatherNet V2 ML nowcasting."""
    try:
        from src.inference.nowcast_service import get_nowcast_service
        svc = get_nowcast_service()
        summary = svc.get_summary(lead_hours=int(lead_hours) if lead_hours in [2, 3, 4, 5, 6] else 2)
        hz = summary["hazards"]
        return {
            "thunderstorm": {
                "probability": hz["thunderstorm"]["probability"],
                "level": hz["thunderstorm"]["level"]
            },
            "heavy_rainfall": {
                "probability": hz["heavy_rainfall"]["probability"],
                "level": hz["heavy_rainfall"]["level"]
            },
            "flash_flood": {
                "probability": hz["flash_flood"]["probability"],
                "level": hz["flash_flood"]["level"]
            },
            "overall": {
                "probability": hz["overall"]["probability"],
                "level": hz["overall"]["level"]
            }
        }
    except Exception as e:
        # Fallback to rule-based heuristic if ML service is unavailable
        pass

    thunderstorm = calculate_thunderstorm_risk(
        current or {},
        forecast or []
    )

    heavy_rainfall = calculate_heavy_rain_risk(
        current or {},
        forecast or []
    )

    flash_flood = calculate_flash_flood_risk(
        current or {},
        forecast or []
    )

    probabilities = [
        thunderstorm["probability"],
        heavy_rainfall["probability"],
        flash_flood["probability"]
    ]
    highest_risk = max(probabilities)
    combined_risk = sum(probabilities) / 3
    overall_probability = round((highest_risk * 0.7) + (combined_risk * 0.3))
    overall_probability = clamp(overall_probability)

    return {
        "thunderstorm": thunderstorm,
        "heavy_rainfall": heavy_rainfall,
        "flash_flood": flash_flood,
        "overall": {
            "probability": overall_probability,
            "level": risk_level(overall_probability)
        }
    }

def calculate_horizon_hazards(current, forecast, target_hours):
    """
    Calculate hazard probabilities for requested horizon using SevereWeatherNet V2 ML.
    """
    try:
        # Map target hours to closest supported ML horizon: 2, 3, 4, 5, 6
        target_f = float(target_hours)
        supported_leads = [2, 3, 4, 5, 6]
        closest_lead = min(supported_leads, key=lambda x: abs(x - target_f))
        
        from src.inference.nowcast_service import get_nowcast_service
        svc = get_nowcast_service()
        summary = svc.get_summary(lead_hours=closest_lead)
        hz = summary["hazards"]
        
        return {
            "target_hours": target_hours,
            "actual_hours": float(closest_lead),
            "hazards": {
                "thunderstorm": {
                    "probability": hz["thunderstorm"]["probability"],
                    "level": hz["thunderstorm"]["level"]
                },
                "heavy_rainfall": {
                    "probability": hz["heavy_rainfall"]["probability"],
                    "level": hz["heavy_rainfall"]["level"]
                },
                "flash_flood": {
                    "probability": hz["flash_flood"]["probability"],
                    "level": hz["flash_flood"]["level"]
                },
                "overall": {
                    "probability": hz["overall"]["probability"],
                    "level": hz["overall"]["level"]
                }
            }
        }
    except Exception:
        pass
    No weather values are interpolated or invented.
    """

    if not forecast:
        return calculate_hazards(
            current,
            []
        )

    selected_point = None
    smallest_difference = float("inf")

    for item in forecast:
        hours_from_now = item.get(
            "hours_from_now"
        )

        if hours_from_now is None:
            continue

        difference = abs(
            float(hours_from_now) -
            float(target_hours)
        )

        if difference < smallest_difference:
            smallest_difference = difference
            selected_point = item

    if selected_point is None:
        return calculate_hazards(
            current,
            []
        )

    # Convert the selected forecast point into
    # the same structure expected by the
    # existing hazard calculation functions.

    horizon_current = {
        "temperature": selected_point.get(
            "temperature"
        ),
        "humidity": selected_point.get(
            "humidity",
            0
        ),
        "pressure": selected_point.get(
            "pressure"
        ),
        "wind_speed": selected_point.get(
            "wind_speed",
            0
        ),
        "rainfall_1h": 0,
        "weather": selected_point.get(
            "weather"
        )
    }

    # Use forecast points from the selected
    # horizon onward for accumulation/persistence.

    remaining_forecast = [
        item
        for item in forecast
        if item.get("hours_from_now") is not None
        and float(item.get("hours_from_now")) >=
            float(target_hours)
    ]

    for item in remaining_forecast:
     if "rainfall_mm" in item:
        item["rainfall_3h"] = (
            item.get("rainfall_mm") or 0
        )

    hazards = calculate_hazards(
        horizon_current,
        remaining_forecast
    )

    return {
        "target_hours": target_hours,
        "actual_hours": round(
            float(
                selected_point.get(
                    "hours_from_now",
                    target_hours
                )
            ),
            1
        ),
        "hazards": hazards
    }