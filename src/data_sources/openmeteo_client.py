"""Open-Meteo API client: air-quality (the AQI target + pollutant features) and
weather (feature) data, both live (used by hourly ingestion) and historical
(used by the backfill script). No API key required.
"""

import requests

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_QUALITY_HOURLY_VARS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
WEATHER_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation"
)

REQUEST_TIMEOUT_SECONDS = 30


def fetch_air_quality_hourly(latitude: float, longitude: float, past_days: int = 1, forecast_days: int = 1) -> dict:
    """Live/recent hourly air quality (AQI + pollutants) for the hourly ingestion pipeline."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": AIR_QUALITY_HOURLY_VARS,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def fetch_weather_hourly(latitude: float, longitude: float, forecast_days: int = 1) -> dict:
    """Live/recent hourly weather for the hourly ingestion pipeline."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": WEATHER_HOURLY_VARS,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = requests.get(WEATHER_FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def fetch_air_quality_historical(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """Historical hourly air quality for a date range (YYYY-MM-DD), used by the backfill script."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": AIR_QUALITY_HOURLY_VARS,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def fetch_weather_historical(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """Historical hourly weather for a date range (YYYY-MM-DD), used by the backfill script."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": WEATHER_HOURLY_VARS,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()
