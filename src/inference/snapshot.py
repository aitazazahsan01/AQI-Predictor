"""A self-contained snapshot of everything a front end needs to render.

The Next.js site is a static build with no server and no credentials, so it
cannot reach Hopsworks, load a model or run SHAP. Instead the pipeline runs
this module and publishes the result as JSON; the site just draws it.

That keeps the trust boundary where it already was - secrets stay inside
GitHub Actions - and means the published numbers were produced by exactly the
same code path as the Streamlit dashboard, rather than a parallel
reimplementation that could drift.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from src.alerts.aqi_scale import AQI_CATEGORIES, categorize, summarize_alert
from src.config import CITIES

TREND_DAYS = 45

# How many rows SHAP compares the current day against, and how many drivers to
# publish per horizon. Eight is what the Streamlit chart shows.
SHAP_BACKGROUND_ROWS = 200
SHAP_TOP_N = 8

# Columns worth surfacing as the conditions behind today's reading, in display
# order. Anything absent from the frame is skipped rather than published null.
CONDITION_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("pm2_5_mean", "PM2.5", "ug/m3"),
    ("pm10_mean", "PM10", "ug/m3"),
    ("o3_mean", "Ozone", "ug/m3"),
    ("no2_mean", "Nitrogen dioxide", "ug/m3"),
    ("temp_mean", "Temperature", "C"),
    ("humidity_mean", "Humidity", "%"),
    ("wind_speed_mean", "Wind speed", "km/h"),
    ("precipitation_sum", "Rainfall", "mm"),
)


def load_daily_features(city_slug: str, trend_days: int = TREND_DAYS) -> tuple[pd.DataFrame, str]:
    """Engineered daily features, preferring the feature store.

    Falls back to rebuilding from Open-Meteo when the stored features are
    absent, too short or stale - showing a month-old reading as the latest is
    worse than refetching. Both paths run the same feature-engineering code, so
    either way what is published matches what the models were trained on.

    Returns the frame plus a short description of where it came from, so the
    fallback is visible to the reader rather than silent.
    """
    from src.features.availability import describe_usability

    try:
        from src.training.data_prep import load_training_frame

        stored = load_training_frame(city_slug)
        usable, reason = describe_usability(stored)
        if usable:
            return stored, f"Hopsworks feature store ({reason})"
        fallback_reason = f"feature store unusable: {reason}"
    except Exception as exc:
        fallback_reason = f"feature store unreachable ({type(exc).__name__})"

    from src.features.feature_engineering import compute_daily_features
    from src.features.historical import fetch_historical_hourly

    city = CITIES[city_slug]
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=trend_days + 10)
    hourly = fetch_historical_hourly(city, start, end)
    daily = compute_daily_features(hourly)
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    return daily, f"live Open-Meteo fetch - {fallback_reason}"


def _clean(value: Any) -> float | None:
    """JSON has no NaN. Anything unrepresentable becomes null."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else round(number, 2)


def _iso_date(value: Any) -> str:
    return pd.to_datetime(value).date().isoformat()


def build_latest(features: pd.DataFrame) -> dict:
    """The most recent observed day, with its conditions and AQI band."""
    row = features.iloc[-1]
    category = categorize(row.get("aqi_mean"))

    return {
        "date": _iso_date(row["date"]),
        "aqi": _clean(row.get("aqi_mean")),
        "aqi_max": _clean(row.get("aqi_max")),
        "aqi_min": _clean(row.get("aqi_min")),
        "category": category.name if category else None,
        "color": category.color if category else None,
        "advice": category.health_advice if category else None,
        "conditions": [
            {"label": label, "value": _clean(row.get(column)), "unit": unit}
            for column, label, unit in CONDITION_FIELDS
            if column in features.columns and _clean(row.get(column)) is not None
        ],
    }


def build_history(features: pd.DataFrame, trend_days: int = TREND_DAYS) -> list[dict]:
    tail = features.tail(trend_days)
    return [
        {"date": _iso_date(row["date"]), "aqi": _clean(row.get("aqi_mean"))}
        for _, row in tail.iterrows()
    ]


def build_models_summary(models: dict) -> list[dict]:
    return [
        {
            "horizon": horizon,
            "model_type": loaded.model_type,
            "source": loaded.source,
            "metrics": {k: _clean(v) for k, v in (loaded.metrics or {}).items()},
            "n_features": len(loaded.feature_columns),
        }
        for horizon, loaded in sorted(models.items())
    ]


def build_drivers(features: pd.DataFrame, models: dict) -> list[dict]:
    """SHAP contributions per horizon, precomputed so the site needs no runtime.

    A model family SHAP cannot explain is reported as such rather than
    omitted, so the front end can say why a panel is empty.
    """
    from src.explainability.shap_utils import (
        FEATURE_LABELS,
        explain_prediction,
        top_contributions,
    )

    background = features.tail(SHAP_BACKGROUND_ROWS)
    latest = features.tail(1)
    drivers: list[dict] = []

    for horizon, loaded in sorted(models.items()):
        try:
            contributions = explain_prediction(
                loaded.model, loaded.model_type, loaded.feature_columns, latest, background
            )
            drivers.append(
                {
                    "horizon": horizon,
                    "model_type": loaded.model_type,
                    "unavailable": None,
                    "features": [
                        {
                            "feature": c.feature,
                            "label": FEATURE_LABELS.get(c.feature, c.feature),
                            "value": _clean(c.value),
                            "contribution": _clean(c.contribution),
                        }
                        for c in top_contributions(contributions, SHAP_TOP_N)
                    ],
                }
            )
        except Exception as exc:
            drivers.append(
                {
                    "horizon": horizon,
                    "model_type": loaded.model_type,
                    "unavailable": f"{type(exc).__name__}: {exc}",
                    "features": [],
                }
            )

    return drivers


def build_scale() -> list[dict]:
    return [
        {
            "name": c.name,
            "lower": c.lower,
            "upper": None if c.upper == float("inf") else c.upper,
            "color": c.color,
            "alert_level": c.alert_level,
            "advice": c.health_advice,
        }
        for c in AQI_CATEGORIES
    ]


def build_snapshot(
    city_slug: str,
    features: pd.DataFrame,
    forecasts: list,
    models: dict,
    feature_source: str,
    station: dict | None = None,
    include_drivers: bool = True,
) -> dict:
    """Assembles the published payload. Pure: every input is already loaded.

    `include_drivers` exists because SHAP dominates the runtime; turning it off
    gives a fast snapshot with every panel but the explanations.
    """
    city = CITIES[city_slug]
    alert = summarize_alert({f.date.strftime("%a %d %b"): f.aqi for f in forecasts})

    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "city": {
            "slug": city_slug,
            "name": city.display_name,
            "latitude": city.latitude,
            "longitude": city.longitude,
        },
        "feature_source": feature_source,
        "observed_days": int(len(features)),
        "latest": build_latest(features),
        "station": station,
        "forecast": [f.as_dict() for f in forecasts],
        "alert": alert,
        "history": build_history(features),
        "drivers": build_drivers(features, models) if include_drivers else [],
        "models": build_models_summary(models),
        "scale": build_scale(),
    }
