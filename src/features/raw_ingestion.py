"""Module 1: hourly raw ingestion.

Fetches the latest available hour of air-quality + weather data for a city
from Open-Meteo (plus a display-only AQICN reading), and returns one row
matching the `aqi_hourly_raw` feature group schema (see PROJECT_PLAN.md section 3).
"""

from datetime import datetime, timezone

from src.config import CityConfig
from src.data_sources import aqicn_client, openmeteo_client


def select_latest_hour_index(timestamps: list[str], now_utc: datetime) -> int:
    """Returns the index of the latest hourly timestamp that is <= now_utc.

    Open-Meteo's air-quality model has a small reporting lag, so "now" itself
    isn't always present yet — we take the most recent hour that *is* available.
    """
    now_str = now_utc.strftime("%Y-%m-%dT%H:00")
    candidates = [i for i, t in enumerate(timestamps) if t <= now_str]
    if not candidates:
        raise ValueError(f"No hourly data available at or before {now_str}")
    return candidates[-1]


def fetch_raw_observation(city: CityConfig, aqicn_token: str) -> dict:
    """Fetches one hourly row of raw features for `city`."""
    now_utc = datetime.now(timezone.utc)

    aq = openmeteo_client.fetch_air_quality_hourly(city.latitude, city.longitude, past_days=1, forecast_days=1)
    wx = openmeteo_client.fetch_weather_hourly(city.latitude, city.longitude, forecast_days=1)

    aq_idx = select_latest_hour_index(aq["hourly"]["time"], now_utc)
    wx_idx = select_latest_hour_index(wx["hourly"]["time"], now_utc)

    try:
        aqicn_reading = aqicn_client.fetch_live_reading(city.aqicn_station_uid, aqicn_token)
    except Exception:
        # Display-only data source — never fail ingestion because AQICN hiccups.
        aqicn_reading = None

    return {
        "city": city.slug,
        "ts": aq["hourly"]["time"][aq_idx],
        "aqi": aq["hourly"]["us_aqi"][aq_idx],
        "pm2_5": aq["hourly"]["pm2_5"][aq_idx],
        "pm10": aq["hourly"]["pm10"][aq_idx],
        "carbon_monoxide": aq["hourly"]["carbon_monoxide"][aq_idx],
        "nitrogen_dioxide": aq["hourly"]["nitrogen_dioxide"][aq_idx],
        "sulphur_dioxide": aq["hourly"]["sulphur_dioxide"][aq_idx],
        "ozone": aq["hourly"]["ozone"][aq_idx],
        "temperature_2m": wx["hourly"]["temperature_2m"][wx_idx],
        "relative_humidity_2m": wx["hourly"]["relative_humidity_2m"][wx_idx],
        "wind_speed_10m": wx["hourly"]["wind_speed_10m"][wx_idx],
        "wind_direction_10m": wx["hourly"]["wind_direction_10m"][wx_idx],
        "surface_pressure": wx["hourly"]["surface_pressure"][wx_idx],
        "precipitation": wx["hourly"]["precipitation"][wx_idx],
        "aqicn_live_aqi": aqicn_reading["aqi"] if aqicn_reading else None,
    }
