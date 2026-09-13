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

    # Recent rainfall receives more weight than older rainfall.
    rainfall_score = (
        rainfall_1h * 5
        + rainfall_3h * 2
        + rainfall_6h
    )

    rainfall_score = _clamp(rainfall_score)

    # =====================================================
    # 2. MOISTURE FACTOR
    # =====================================================

    # High humidity means a moisture-rich environment.
    moisture_score = _clamp(
        (humidity - 60) / 40 * 100
    )

    # Dew-point reinforcement.
    if dew_point is not None:
        if dew_point >= 26:
            moisture_score += 10
        elif dew_point >= 24:
            moisture_score += 5

    moisture_score = _clamp(moisture_score)

    # =====================================================
    # 3. WIND FACTOR
    # =====================================================

    # This is a wind-influence proxy.
    # It is NOT true wind convergence because only one
    # spatial wind observation is currently available.
    wind_score = _clamp(
        wind_speed / 30 * 100
    )

    # =====================================================
    # 4. INSTABILITY / CAPE FACTOR
    # =====================================================

    # CAPE is a proxy for convective instability.
    if cape < 500:
        cape_score = cape / 500 * 20

    elif cape < 1000:
        cape_score = 20 + (
            (cape - 500) / 500 * 20
        )

    elif cape < 2000:
        cape_score = 40 + (
            (cape - 1000) / 1000 * 25
        )

    elif cape < 3000:
        cape_score = 65 + (
            (cape - 2000) / 1000 * 20
        )

    else:
        cape_score = 85 + (
            (cape - 3000) / 2000 * 15
        )

    cape_score = _clamp(cape_score)

    # =====================================================
    # 5. RADAR FACTOR
    # =====================================================

    radar_score = None

    if radar_dbz is not None:
        radar_score = _clamp(
            (radar_dbz - 15) / 40 * 100
        )

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
        contributions = {
            key: round(
                (score / total_score) * 100,
                1
            )
            for key, score in raw_factors.items()
        }
    else:
        contributions = {
            key: 0
            for key in raw_factors
        }

    # =====================================================
    # RESULT
    # =====================================================

    return {
        "method": "rule_based_attribution",

        "factors": {
            "rainfall": {
                "score": round(rainfall_score, 1),
                "contribution": contributions["rainfall"],
                "source": "Open-Meteo precipitation"
            },

            "moisture": {
                "score": round(moisture_score, 1),
                "contribution": contributions["moisture"],
                "source": "Open-Meteo humidity + dew point"
            },

            "wind": {
                "score": round(wind_score, 1),
                "contribution": contributions["wind"],
                "source": "Open-Meteo wind",
                "type": "wind_influence_proxy"
            },

            "instability": {
                "score": round(cape_score, 1),
                "contribution": contributions["instability"],
                "source": "Open-Meteo CAPE",
                "type": "convective_instability_proxy"
            }
        },

        "radar": {
            "score": (
                round(radar_score, 1)
                if radar_score is not None
                else None
            ),
            "contribution": (
                contributions.get("radar")
                if radar_score is not None
                else None
            ),
            "source": (
                "Radar data"
                if radar_score is not None
                else None
            )
        }
    }