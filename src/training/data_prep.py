"""Module 4, part 1: turning stored daily features into a supervised learning problem.

The feature store holds one row per (city, day) describing what was known *on*
that day. To forecast, we need to pair each of those rows with what actually
happened 1, 2 and 3 days later — that pairing happens here, not in the feature
store, because the newest rows legitimately have no "tomorrow" yet and must
still be usable for live prediction.
"""

import pandas as pd

HORIZONS = (1, 2, 3)

KEY_COLUMNS = ["city", "date"]

# Excluded from the feature matrix on purpose. Note `aqi_mean` is NOT excluded:
# today's AQI is known at prediction time and is the single strongest predictor
# of tomorrow's, so it is a legitimate input. What must never be an input is a
# *target* column, which holds genuinely future information.
#   hours_observed -> describes our collection pipeline, not the air itself.
#                     Letting a model learn from it risks fitting artefacts of
#                     our own outages rather than anything about air quality.
NON_FEATURE_COLUMNS = {"city", "date", "hours_observed"}


def target_column(horizon: int) -> str:
    return f"target_h{horizon}"


def build_targets(daily: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Attaches target_h1/h2/h3 = the AQI 1/2/3 days after each row.

    Uses a negative shift within each city, so row for day D gets day D+h's
    aqi_mean. The final `h` rows of each city necessarily get NaN — that future
    hasn't happened yet — and are dropped per-horizon at training time.
    """
    if daily.empty:
        return daily

    df = daily.sort_values(KEY_COLUMNS).reset_index(drop=True)
    for horizon in horizons:
        df[target_column(horizon)] = df.groupby("city")["aqi_mean"].shift(-horizon)
    return df


def get_feature_columns(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> list[str]:
    """Every numeric column that is a legitimate model input.

    Deliberately excludes keys, the collection-quality flag, and all target
    columns — leaking a target into the feature matrix would produce a model
    that looks perfect and is worthless.
    """
    targets = {target_column(h) for h in horizons}
    excluded = NON_FEATURE_COLUMNS | targets
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def prepare_horizon_dataset(
    df: pd.DataFrame, horizon: int, min_hours_observed: int | None = None
) -> pd.DataFrame:
    """Rows usable for training one horizon: known target, and (optionally) only
    days assembled from enough hours to trust their daily mean."""
    target = target_column(horizon)
    usable = df[df[target].notna()]
    if min_hours_observed is not None and "hours_observed" in usable.columns:
        usable = usable[usable["hours_observed"] >= min_hours_observed]
    return usable.reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits chronologically: the most recent `test_days` days are the test set.

    Never shuffles. A random split would let the model train on data from *after*
    the days it is tested on, which inflates scores enormously on time series and
    tells you nothing about real forecasting skill.
    """
    if df.empty:
        return df, df

    ordered = df.sort_values("date").reset_index(drop=True)
    cutoff = ordered["date"].max() - pd.Timedelta(days=test_days)
    train = ordered[ordered["date"] <= cutoff].reset_index(drop=True)
    test = ordered[ordered["date"] > cutoff].reset_index(drop=True)
    return train, test


def load_training_frame(city: str | None = None) -> pd.DataFrame:
    """Reads engineered daily features from the Hopsworks feature store and
    attaches forecast targets. Imported lazily so the pure functions above stay
    testable without credentials or a network."""
    from src.hopsworks_utils.connection import get_feature_store
    from src.hopsworks_utils.feature_groups import get_or_create_daily_features_fg

    fs = get_feature_store()
    fg = get_or_create_daily_features_fg(fs)
    daily = fg.read()

    if city is not None:
        daily = daily[daily["city"] == city]

    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    return build_targets(daily)
