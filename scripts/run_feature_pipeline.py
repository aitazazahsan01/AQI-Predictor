"""Module 1 entrypoint — run hourly (by GitHub Actions, see .github/workflows/feature_pipeline.yml).

Usage: python scripts/run_feature_pipeline.py
Requires HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, AQICN_TOKEN in the environment (or a local .env file).
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CITIES  # noqa: E402
from src.features.raw_ingestion import fetch_raw_observation  # noqa: E402
from src.hopsworks_utils.connection import get_feature_store  # noqa: E402
from src.hopsworks_utils.feature_groups import get_or_create_hourly_raw_fg  # noqa: E402

# Explicitly typed so a single all-null column (e.g. aqicn_live_aqi when that
# station is stale/offline) doesn't get inferred as an untyped "null" dtype,
# which Hopsworks' schema inference rejects.
NUMERIC_COLUMNS = [
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


def main():
    aqicn_token = os.environ["AQICN_TOKEN"]
    fs = get_feature_store()
    fg = get_or_create_hourly_raw_fg(fs)

    for city in CITIES.values():
        row = fetch_raw_observation(city, aqicn_token)
        df = pd.DataFrame([row])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        for col in NUMERIC_COLUMNS:
            df[col] = df[col].astype("float64")
        fg.insert(df, wait=True)
        print(f"[{city.display_name}] inserted row for {row['ts']} -> AQI={row['aqi']}")


if __name__ == "__main__":
    main()
