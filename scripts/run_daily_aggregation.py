"""Module 2 entrypoint — run daily (by GitHub Actions, shortly after midnight UTC).

Reads hourly raw observations from the feature store, engineers one daily row
per city, and writes it to the `aqi_daily_features` feature group.

Usage:
    python scripts/run_daily_aggregation.py                 # yesterday (UTC)
    python scripts/run_daily_aggregation.py --date 2026-07-27
    python scripts/run_daily_aggregation.py --all           # every computable day

Requires HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME in the environment.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import (  # noqa: E402
    MIN_HOURS_FOR_RELIABLE_DAY,
    compute_daily_features,
)
from src.hopsworks_utils.connection import get_feature_store  # noqa: E402
from src.hopsworks_utils.feature_groups import (  # noqa: E402
    get_or_create_daily_features_fg,
    get_or_create_hourly_raw_fg,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate hourly raw AQI data into daily features.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", help="UTC day to compute, as YYYY-MM-DD. Defaults to yesterday.")
    group.add_argument("--all", action="store_true", help="Write every day that can be computed.")
    return parser.parse_args()


def main():
    args = parse_args()

    fs = get_feature_store()
    hourly_fg = get_or_create_hourly_raw_fg(fs)

    hourly = hourly_fg.read()
    if hourly.empty:
        print("No hourly data in the feature store yet — nothing to aggregate.")
        return

    daily = compute_daily_features(hourly)
    if daily.empty:
        print("Hourly data produced no daily rows — nothing to aggregate.")
        return

    daily["date"] = pd.to_datetime(daily["date"], utc=True)

    if args.all:
        to_write = daily
    else:
        if args.date:
            target = pd.Timestamp(args.date, tz="UTC")
        else:
            target = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1), tz="UTC")
        to_write = daily[daily["date"] == target]
        if to_write.empty:
            available = daily["date"].dt.strftime("%Y-%m-%d").tolist()
            print(f"No hourly data for {target.date()}. Days available: {available}")
            return

    daily_fg = get_or_create_daily_features_fg(fs)
    daily_fg.insert(to_write, wait=True)

    for _, row in to_write.iterrows():
        reliability = "" if row["hours_observed"] >= MIN_HOURS_FOR_RELIABLE_DAY else "  [PARTIAL DAY]"
        print(
            f"[{row['city']}] {row['date'].date()}  "
            f"aqi_mean={row['aqi_mean']:.1f}  hours={int(row['hours_observed'])}{reliability}"
        )
    print(f"Wrote {len(to_write)} daily row(s) to aqi_daily_features.")


if __name__ == "__main__":
    main()
