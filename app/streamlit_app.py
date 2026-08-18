"""Module 6: the AQI forecast dashboard.

Loads models and features that the pipelines already produced, and shows the
3-day forecast, recent trends, why the model said what it said, and a warning
when unhealthy air is expected.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.aqi_scale import AQI_CATEGORIES, categorize, summarize_alert  # noqa: E402
from src.config import CITIES, DEFAULT_CITY  # noqa: E402
from src.data_sources import aqicn_client  # noqa: E402
from src.inference.predict import build_forecast, load_models  # noqa: E402

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

TREND_DAYS = 45
CACHE_SECONDS = 30 * 60


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def load_features(city_slug: str) -> tuple[pd.DataFrame, str]:
    """Engineered daily features, preferring the feature store.

    Falls back to rebuilding from Open-Meteo when the stored features are absent,
    too short or stale - showing a month-old reading as "latest" is worse than
    refetching. Both paths run the same feature-engineering code, so either way
    what's on screen matches what the models were trained on.

    Returns the frame plus a short description of where it came from, so the
    fallback is visible to the user rather than silent.
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
    start = end - dt.timedelta(days=TREND_DAYS + 10)
    hourly = fetch_historical_hourly(city, start, end)
    daily = compute_daily_features(hourly)
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    return daily, f"live Open-Meteo fetch — {fallback_reason}"


@st.cache_resource(show_spinner=False)
def load_forecast_models():
    return load_models()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def load_station_reading(city_slug: str):
    """Live reading from the official monitoring station, or None if stale.

    Display only - never training data. The Islamabad station has gone quiet
    for months at a time, so anything older than a few hours is treated as
    unavailable rather than shown as current.
    """
    import os

    token = os.environ.get("AQICN_TOKEN", "")
    if not token:
        # st.secrets raises rather than returning empty when no secrets file
        # exists, which is the normal case for a local run driven by .env.
        try:
            token = st.secrets.get("AQICN_TOKEN", "")
        except Exception:
            token = ""
    if not token:
        return None
    try:
        return aqicn_client.fetch_live_reading(CITIES[city_slug].aqicn_station_uid, token)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_alert(forecasts):
    alert = summarize_alert({f.date.strftime("%a %d %b"): f.aqi for f in forecasts})
    if alert is None:
        st.success("**No air quality alerts.** Nothing in the next 3 days is forecast to reach unhealthy levels.")
        return

    days = ", ".join(alert["days"])
    headline = f"**{alert['category']} air expected — {days}**"
    body = f"{headline}\n\nPeak forecast **AQI {alert['worst_aqi']:.0f}** on {alert['worst_day']}. {alert['advice']}"
    if alert["level"] == "critical":
        st.error(body, icon="🚨")
    else:
        st.warning(body, icon="⚠️")


def render_today(features: pd.DataFrame, station):
    latest = features.iloc[-1]
    observed = float(latest["aqi_mean"])
    category = categorize(observed)
    as_of = pd.to_datetime(latest["date"]).date()

    left, right = st.columns([2, 1])
    with left:
        st.metric(
            label=f"Latest daily average — {as_of:%d %b %Y}",
            value=f"{observed:.0f}",
            help="Mean AQI across the most recent complete day of observations.",
        )
        st.markdown(
            f"<span style='background:{category.color};padding:4px 12px;border-radius:12px;"
            f"color:#111;font-weight:600'>{category.name}</span>",
            unsafe_allow_html=True,
        )
        st.caption(category.health_advice)

    with right:
        if station:
            st.metric("Live station reading", f"{station['aqi']:.0f}")
            st.caption(f"{station['station_name']} · dominant pollutant: {station.get('dominant_pollutant', 'n/a')}")
        else:
            st.metric("Live station reading", "—")
            st.caption("Official station reading unavailable or stale.")


def render_forecast(forecasts):
    st.subheader("Next 3 days")
    columns = st.columns(len(forecasts))
    for column, forecast in zip(columns, forecasts):
        with column:
            st.markdown(f"**{forecast.date:%A %d %b}**")
            st.markdown(
                f"<div style='background:{forecast.color};border-radius:10px;padding:18px;"
                f"text-align:center;color:#111'>"
                f"<div style='font-size:40px;font-weight:700;line-height:1'>{forecast.aqi:.0f}</div>"
                f"<div style='font-size:13px;margin-top:6px'>{forecast.category}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"+{forecast.horizon}d · {forecast.model_type}")


def render_trend(features: pd.DataFrame, forecasts):
    st.subheader(f"Recent trend and forecast")

    history = features.tail(TREND_DAYS)[["date", "aqi_mean"]].copy()
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)
    history = history.rename(columns={"aqi_mean": "Observed"}).set_index("date")

    predicted = pd.DataFrame(
        {"Forecast": [f.aqi for f in forecasts]},
        index=pd.to_datetime([f.date for f in forecasts]),
    )

    st.line_chart(pd.concat([history, predicted], axis=1), height=320)
    st.caption(
        "Observed daily averages with the 3-day forecast appended. "
        "Forecast accuracy degrades with distance — day 3 is substantially less certain than day 1."
    )


def render_explanation(features: pd.DataFrame, models, forecasts):
    st.subheader("Why this forecast?")

    horizon = st.radio(
        "Forecast day", [f.horizon for f in forecasts], horizontal=True,
        format_func=lambda h: f"Day {h}",
    )
    loaded = models[horizon]

    try:
        from src.explainability.shap_utils import (
            FEATURE_LABELS,
            explain_prediction,
            top_contributions,
        )

        background = features.tail(200)
        contributions = explain_prediction(
            loaded.model, loaded.model_type, loaded.feature_columns,
            features.tail(1), background,
        )
        top = top_contributions(contributions)

        chart = pd.DataFrame(
            {"contribution": [c.contribution for c in top]},
            index=[FEATURE_LABELS.get(c.feature, c.feature) for c in top],
        )
        st.bar_chart(chart, horizontal=True, height=320)
        st.caption(
            "How much each input pushed this forecast up (positive) or down (negative), "
            "in AQI points, relative to a typical recent day."
        )
    except Exception as exc:
        st.info(f"Explanation unavailable for this model type ({type(exc).__name__}: {exc}).")


def render_scale():
    with st.expander("What do these numbers mean?"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Range": f"{c.lower}–{'∞' if c.upper == float('inf') else c.upper}",
                        "Category": c.name,
                        "Advice": c.health_advice,
                    }
                    for c in AQI_CATEGORIES
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def main():
    st.title("🌫️ Pearls AQI Predictor")

    city_slug = st.sidebar.selectbox(
        "City", list(CITIES), index=list(CITIES).index(DEFAULT_CITY),
        format_func=lambda slug: CITIES[slug].display_name,
    )
    st.sidebar.caption("Forecasts are produced daily from an automated feature and training pipeline.")

    with st.spinner("Loading features and models..."):
        features, feature_source = load_features(city_slug)
        models = load_forecast_models()
        station = load_station_reading(city_slug)

    st.sidebar.caption(f"**Feature source:** {feature_source}")

    if features.empty:
        st.error("No feature data available. Run the backfill or the hourly pipeline first.")
        return
    if not models:
        # Features loaded but no model did, so show what we can rather than a
        # bare error page: the observed history is still worth looking at.
        st.error("No trained models available, so no forecast can be produced yet.")
        st.markdown(
            """
The dashboard reads models from the **Hopsworks Model Registry**, which stays
empty until a training run registers one.

- **Deployed app:** run the *Daily Aggregation and Training* workflow in GitHub
  Actions (Actions -> Daily Aggregation and Training -> Run workflow). It trains
  on the feature store and registers the winning model per horizon.
- **Locally:** `python scripts/run_training_pipeline.py --save-local models`,
  which writes the bundle the dashboard falls back to when the registry is
  unreachable.
"""
        )
        st.divider()
        render_today(features, station)
        st.divider()
        render_trend(features, forecasts=[])
        render_scale()
        return

    forecasts = build_forecast(features.tail(1), models)

    render_alert(forecasts)
    st.divider()
    render_today(features, station)
    st.divider()
    render_forecast(forecasts)
    st.divider()
    render_trend(features, forecasts)
    st.divider()
    render_explanation(features, models, forecasts)
    render_scale()

    with st.sidebar.expander("Model details"):
        for horizon in sorted(models):
            loaded = models[horizon]
            rmse = loaded.metrics.get("rmse")
            st.write(f"**Day {horizon}** — {loaded.model_type}")
            if rmse is not None:
                st.caption(f"hold-out RMSE {rmse:.2f} · source: {loaded.source}")


if __name__ == "__main__":
    main()
