"""Module 4 entrypoint — run daily (by GitHub Actions, after the aggregation job).

Pulls engineered features from the feature store, trains every candidate model
for each forecast horizon, and registers the best per horizon.

Usage:
    python scripts/run_training_pipeline.py
    python scripts/run_training_pipeline.py --test-days 120 --no-register
    python scripts/run_training_pipeline.py --offline   # fetch data straight from
                                                        # Open-Meteo, skip Hopsworks

`--offline` exists so the pipeline can be exercised end-to-end on real data from
networks that block the Hopsworks data ports (see README troubleshooting).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_prep import HORIZONS, build_targets, get_feature_columns  # noqa: E402
from src.training.train import results_to_frame, select_best, train_all_horizons  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train and register AQI forecasting models.")
    parser.add_argument("--city", default=None, help="Restrict to one city slug.")
    parser.add_argument(
        "--test-days", type=int, default=90, help="Size of the chronological hold-out window."
    )
    parser.add_argument(
        "--min-hours", type=int, default=18, help="Drop days assembled from fewer hours than this."
    )
    parser.add_argument(
        "--no-register", action="store_true", help="Train and report, but don't touch the registry."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fetch features directly from Open-Meteo instead of the feature store.",
    )
    return parser.parse_args()


def load_offline_frame(city_slug: str | None) -> pd.DataFrame:
    """Rebuilds the feature frame straight from the source APIs, using the same
    feature-engineering code the stored features were built with."""
    import datetime as dt

    from src.config import CITIES, DEFAULT_CITY
    from src.features.feature_engineering import compute_daily_features
    from src.features.historical import EARLIEST_AIR_QUALITY_DATE, fetch_historical_hourly

    city = CITIES[city_slug or DEFAULT_CITY]
    end = dt.date.today() - dt.timedelta(days=1)
    print(f"Offline mode: fetching {city.display_name} from {EARLIEST_AIR_QUALITY_DATE} to {end}\n")

    hourly = fetch_historical_hourly(city, EARLIEST_AIR_QUALITY_DATE, end)
    daily = compute_daily_features(hourly)
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    return build_targets(daily)


def main():
    args = parse_args()

    if args.offline:
        df = load_offline_frame(args.city)
    else:
        from src.training.data_prep import load_training_frame

        df = load_training_frame(args.city)

    if df.empty:
        print("No feature rows available — run the backfill or the daily aggregation first.")
        return

    span = f"{df['date'].min().date()} .. {df['date'].max().date()}"
    print(f"Loaded {len(df):,} daily rows ({span})\n")

    all_results = train_all_horizons(
        df,
        horizons=HORIZONS,
        test_days=args.test_days,
        min_hours_observed=args.min_hours,
    )

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    summary = pd.concat([results_to_frame(r) for r in all_results.values()], ignore_index=True)
    print(summary.to_string(index=False))

    print("\nWinners:")
    feature_columns = get_feature_columns(df)
    training_window = (str(df["date"].min().date()), str(df["date"].max().date()))
    winners = {h: select_best(r) for h, r in all_results.items()}
    for horizon, best in winners.items():
        baseline = next(
            (r for r in all_results[horizon] if r.model_name == "persistence_baseline" and r.ok),
            None,
        )
        if baseline is not None and baseline.rmse > 0:
            lift = (baseline.rmse - best.rmse) / baseline.rmse * 100
            verdict = f"({lift:+.1f}% vs persistence baseline)"
        else:
            verdict = "(no baseline to compare against)"
        print(f"  h{horizon}: {best.model_name:<20} RMSE {best.rmse:6.2f}  {verdict}")

    if args.no_register or args.offline:
        reason = "--no-register" if args.no_register else "--offline"
        print(f"\nSkipping model registry ({reason}).")
        return

    print("\nRegistering best model per horizon...")
    from src.training.register import register_best_model

    for best in winners.values():
        register_best_model(best, feature_columns, training_window)


if __name__ == "__main__":
    main()
