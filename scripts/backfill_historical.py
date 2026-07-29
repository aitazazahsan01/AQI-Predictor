"""Module 3 entrypoint — one-off historical backfill.

Loads years of past weather + air-quality data into the feature store so there
is enough history to train on immediately, instead of waiting months for the
hourly pipeline to accumulate it.

Writes BOTH feature groups:
  * aqi_hourly_raw     — so daily features stay fully reproducible from raw data
                         and intraday patterns are available for EDA
  * aqi_daily_features — the engineered rows the training pipeline reads

Usage:
    python scripts/backfill_historical.py --dry-run
    python scripts/backfill_historical.py
    python scripts/backfill_historical.py --start-date 2024-01-01 --city islamabad
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CITIES  # noqa: E402
from src.features.feature_engineering import compute_daily_features  # noqa: E402
from src.features.historical import (  # noqa: E402
    EARLIEST_AIR_QUALITY_DATE,
    clamp_start_date,
    fetch_historical_hourly,
)
from src.hopsworks_utils.connection import get_feature_store  # noqa: E402
from src.hopsworks_utils.feature_groups import (  # noqa: E402
    get_or_create_daily_features_fg,
    get_or_create_hourly_raw_fg,
)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill historical AQI features into the feature store.")
    parser.add_argument(
        "--start-date",
        default=EARLIEST_AIR_QUALITY_DATE.isoformat(),
        help=f"First day to backfill (YYYY-MM-DD). Defaults to {EARLIEST_AIR_QUALITY_DATE} (earliest available).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last day to backfill (YYYY-MM-DD). Defaults to yesterday — today is excluded because "
        "Open-Meteo fills the remaining hours of the current day with forecast values, which must "
        "not enter training data as if they were observations.",
    )
    parser.add_argument("--city", default=None, help="City slug to backfill. Defaults to all configured cities.")
    parser.add_argument("--chunk-days", type=int, default=365, help="Days per API request (default 365).")
    parser.add_argument("--skip-hourly", action="store_true", help="Only write daily features, not raw hourly rows.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and summarise, but write nothing.")
    return parser.parse_args()


def main():
    args = parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    # Yesterday, not today: the current day's later hours come back as forecast
    # values, and a forecast must never be stored as if it were an observation.
    end_date = dt.date.fromisoformat(args.end_date) if args.end_date else dt.date.today() - dt.timedelta(days=1)

    clamped = clamp_start_date(start_date)
    if clamped != start_date:
        print(f"NOTE: start date moved {start_date} -> {clamped} (no air-quality data exists before then)\n")
    start_date = clamped

    if args.city:
        if args.city not in CITIES:
            raise SystemExit(f"Unknown city '{args.city}'. Configured: {list(CITIES)}")
        cities = [CITIES[args.city]]
    else:
        cities = list(CITIES.values())

    fs = None if args.dry_run else get_feature_store()

    for city in cities:
        print(f"=== {city.display_name}: {start_date} .. {end_date} ===")
        hourly = fetch_historical_hourly(city, start_date, end_date, chunk_days=args.chunk_days)

        if hourly.empty:
            print("  No usable data returned — skipping.\n")
            continue

        for col in NUMERIC_COLUMNS:
            hourly[col] = hourly[col].astype("float64")

        daily = compute_daily_features(hourly)
        daily["date"] = pd.to_datetime(daily["date"], utc=True)

        print(
            f"  fetched {len(hourly):,} hourly rows -> {len(daily):,} daily rows "
            f"({daily['date'].min().date()} .. {daily['date'].max().date()})"
        )
        print(
            f"  AQI mean={daily['aqi_mean'].mean():.1f}  "
            f"min={daily['aqi_mean'].min():.1f}  max={daily['aqi_mean'].max():.1f}  "
            f"complete 24h days={int((daily['hours_observed'] == 24).sum()):,}/{len(daily):,}"
        )

        if args.dry_run:
            print("  [dry run] nothing written\n")
            continue

        if not args.skip_hourly:
            print(f"  writing {len(hourly):,} rows to aqi_hourly_raw ...")
            get_or_create_hourly_raw_fg(fs).insert(hourly, wait=True)

        print(f"  writing {len(daily):,} rows to aqi_daily_features ...")
        get_or_create_daily_features_fg(fs).insert(daily, wait=True)
        print("  done\n")


if __name__ == "__main__":
    main()
