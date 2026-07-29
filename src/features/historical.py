"""Module 3: historical backfill.

Fetches years of past hourly weather + air-quality data from Open-Meteo and
shapes it into exactly the same format the hourly ingestion pipeline (M1)
produces, so that the same feature engineering code (M2) can be applied to it.

Two API behaviours worth knowing, both established by probing the live API:

1. Air-quality history for this location begins **2022-08-05**. Requests for
   earlier dates do NOT error — they return HTTP 200 with rows whose values are
   all `null`. Those rows must be dropped, or the feature store fills up with
   null-AQI garbage that silently corrupts training.
2. Multi-year ranges are served in a single request (a 3.5-year range returned
   ~31k complete rows), so chunking is for progress reporting and retry
   granularity, not because the API demands it.
"""

import datetime as dt

import numpy as np
import pandas as pd

from src.config import CityConfig
from src.data_sources import openmeteo_client

# Earliest date Open-Meteo has air-quality data for. Verified by binary search
# against the live API: 2022-08-04 returns all-null, 2022-08-05 returns 24/24
# non-null hours for every pollutant.
EARLIEST_AIR_QUALITY_DATE = dt.date(2022, 8, 5)

HOURLY_SCHEMA_COLUMNS = [
    "city",
    "ts",
    "aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "aqicn_live_aqi",
]


def date_chunks(start: dt.date, end: dt.date, chunk_days: int = 365):
    """Splits [start, end] into inclusive (chunk_start, chunk_end) pairs."""
    if start > end:
        return []
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return chunks


def clamp_start_date(start: dt.date) -> dt.date:
    """Open-Meteo silently returns null rows before EARLIEST_AIR_QUALITY_DATE."""
    return max(start, EARLIEST_AIR_QUALITY_DATE)


def shape_hourly_payloads(air_quality: dict, weather: dict, city_slug: str) -> pd.DataFrame:
    """Merges one air-quality + one weather API payload into the hourly schema.

    Kept separate from the fetching so it can be unit tested without network access.
    """
    aq_df = pd.DataFrame(air_quality["hourly"]).rename(columns={"time": "ts", "us_aqi": "aqi"})
    wx_df = pd.DataFrame(weather["hourly"]).rename(columns={"time": "ts"})

    merged = aq_df.merge(wx_df, on="ts", how="inner")
    merged["city"] = city_slug
    # AQICN has no historical API — this column only ever has live values.
    merged["aqicn_live_aqi"] = np.nan

    # Drop hours with no AQI reading: out-of-range dates come back as all-null
    # rows rather than an error, and a null target is useless for training.
    merged = merged[merged["aqi"].notna()].reset_index(drop=True)

    return merged[HOURLY_SCHEMA_COLUMNS]


def fetch_historical_hourly(
    city: CityConfig,
    start_date: dt.date,
    end_date: dt.date,
    chunk_days: int = 365,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetches historical hourly observations for `city` between two dates."""
    start_date = clamp_start_date(start_date)
    if start_date > end_date:
        return pd.DataFrame(columns=HOURLY_SCHEMA_COLUMNS)

    frames = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_days):
        air_quality = openmeteo_client.fetch_air_quality_historical(
            city.latitude, city.longitude, chunk_start.isoformat(), chunk_end.isoformat()
        )
        weather = openmeteo_client.fetch_weather_historical(
            city.latitude, city.longitude, chunk_start.isoformat(), chunk_end.isoformat()
        )
        chunk = shape_hourly_payloads(air_quality, weather, city.slug)
        frames.append(chunk)
        if verbose:
            print(f"  {chunk_start} .. {chunk_end}: {len(chunk):>6,} usable hourly rows")

    if not frames:
        return pd.DataFrame(columns=HOURLY_SCHEMA_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["ts"], utc=True)
    combined = combined.sort_values("ts").drop_duplicates(subset=["city", "ts"], keep="last")
    return combined.reset_index(drop=True)
