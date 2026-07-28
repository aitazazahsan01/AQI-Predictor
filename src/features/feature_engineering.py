"""Module 2: daily feature engineering.

Pure functions (no network, no feature-store calls) that turn hourly raw
observations into one engineered row per (city, day).

These are deliberately pure so that BOTH the daily aggregation job (M2) and the
historical backfill (M3) call the exact same code — that way features computed
from live data and features computed from backfilled history are guaranteed to
be identical in definition, which is a classic source of train/serve skew.

All features here describe data that is *known as of* `date`, so none of them
leak future information. Targets (tomorrow's AQI, etc.) are derived separately
at training time — see PROJECT_PLAN.md section 3.
"""

import numpy as np
import pandas as pd

# hourly column -> daily column, aggregated by mean
MEAN_COLUMNS = {
    "pm2_5": "pm2_5_mean",
    "pm10": "pm10_mean",
    "carbon_monoxide": "co_mean",
    "nitrogen_dioxide": "no2_mean",
    "sulphur_dioxide": "so2_mean",
    "ozone": "o3_mean",
    "temperature_2m": "temp_mean",
    "relative_humidity_2m": "humidity_mean",
    "wind_speed_10m": "wind_speed_mean",
    "surface_pressure": "pressure_mean",
}

LAG_DAYS = (1, 2, 3, 7)

# A day assembled from only a handful of hours isn't a trustworthy daily mean.
# Days below this are still stored (with hours_observed recorded) so nothing is
# silently dropped, but training can filter on it.
MIN_HOURS_FOR_RELIABLE_DAY = 18

DAILY_FEATURE_COLUMNS = [
    "city",
    "date",
    "aqi_mean",
    "aqi_max",
    "aqi_min",
    *MEAN_COLUMNS.values(),
    "precipitation_sum",
    "hours_observed",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    *[f"aqi_lag{lag}" for lag in LAG_DAYS],
    "aqi_change_rate",
    "aqi_roll3_mean",
    "aqi_roll7_mean",
    "aqi_roll3_std",
]


def aggregate_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapses hourly observations into one row per (city, UTC day).

    Duplicate (city, ts) rows are de-duplicated first — the hourly pipeline can
    legitimately insert the same hour twice (e.g. a re-run within the same hour,
    or a GitHub Actions retry).
    """
    if hourly.empty:
        return pd.DataFrame(columns=["city", "date", "aqi_mean", "aqi_max", "aqi_min", "hours_observed"])

    df = hourly.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["city", "ts"], keep="last")
    df["date"] = df["ts"].dt.floor("D")

    grouped = df.groupby(["city", "date"], sort=True)

    daily = pd.DataFrame(
        {
            "aqi_mean": grouped["aqi"].mean(),
            "aqi_max": grouped["aqi"].max(),
            "aqi_min": grouped["aqi"].min(),
            **{daily_col: grouped[hourly_col].mean() for hourly_col, daily_col in MEAN_COLUMNS.items()},
            "precipitation_sum": grouped["precipitation"].sum(),
            "hours_observed": grouped["aqi"].count(),
        }
    ).reset_index()

    return daily


def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. Pollution has strong weekly and seasonal cycles, so
    telling the model *when* a reading happened is genuinely predictive."""
    if daily.empty:
        return daily

    daily = daily.copy()
    dates = pd.to_datetime(daily["date"], utc=True)
    daily["day_of_week"] = dates.dt.dayofweek.astype("int64")  # Monday=0
    daily["day_of_month"] = dates.dt.day.astype("int64")
    daily["month"] = dates.dt.month.astype("int64")
    daily["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int64")
    return daily


def add_derived_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Lags, rolling statistics and AQI change rate.

    These give the model *trend* information — "72 and rising for three days"
    is a very different situation from "72 and falling", but a single day's
    snapshot can't express that.

    Note: lags/rolling windows are positional (previous *rows*), not strict
    calendar offsets. After a contiguous backfill that's the same thing; if the
    history has gaps, a "lag1" is the previous *available* day.
    """
    if daily.empty:
        return daily

    daily = daily.sort_values(["city", "date"]).reset_index(drop=True)

    for lag in LAG_DAYS:
        daily[f"aqi_lag{lag}"] = daily.groupby("city")["aqi_mean"].shift(lag)

    # Guarded against divide-by-zero: AQI of exactly 0 is possible in clean air.
    prev = daily["aqi_lag1"]
    daily["aqi_change_rate"] = np.where(
        prev.notna() & (prev != 0),
        (daily["aqi_mean"] - prev) / prev.replace(0, np.nan),
        np.nan,
    )

    # min_periods=1 so early history still produces usable values instead of
    # throwing away the first week of an already-small dataset.
    grouped_aqi = daily.groupby("city")["aqi_mean"]
    daily["aqi_roll3_mean"] = grouped_aqi.transform(lambda s: s.rolling(3, min_periods=1).mean())
    daily["aqi_roll7_mean"] = grouped_aqi.transform(lambda s: s.rolling(7, min_periods=1).mean())
    daily["aqi_roll3_std"] = grouped_aqi.transform(lambda s: s.rolling(3, min_periods=2).std())

    return daily


def compute_daily_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Full hourly -> engineered-daily pipeline.

    Pass in as much contiguous hourly history as is available: lag7/roll7 need
    the preceding 7 days to be present to produce non-null values for the most
    recent day.
    """
    daily = aggregate_hourly_to_daily(hourly)
    daily = add_time_features(daily)
    daily = add_derived_features(daily)
    return daily
