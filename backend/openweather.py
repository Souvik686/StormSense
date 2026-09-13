import httpx

try:
    from .config import OPENWEATHER_API_KEY
except (ImportError, ValueError):
    import os
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv()
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


async def get_current_weather(lat: float, lon: float):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            CURRENT_WEATHER_URL,
            params=params
        )
        response.raise_for_status()

        return response.json()


async def get_weather_forecast(lat: float, lon: float):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            FORECAST_URL,
            params=params
        )
        response.raise_for_status()

        return response.json()