"""Module 7: explaining individual forecasts with SHAP.

A model that only emits a number is a black box - nobody can question it or
learn from it. SHAP attributes a prediction across its input features, turning
"tomorrow will be 150" into "tomorrow will be 150, mostly because today was
already high and the wind has dropped".

The explainer is chosen to match the winning model, because the fast exact
methods only apply to particular model families:
  * tree ensembles (Random Forest, XGBoost) -> TreeExplainer, exact and fast
  * linear models (Ridge)                   -> LinearExplainer, exact
  * anything else (LSTM, SARIMAX)           -> KernelExplainer, approximate
    and slow, so it is sampled down hard
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# KernelExplainer cost grows with both background size and features; these keep
# a dashboard explanation to roughly a second rather than a minute.
KERNEL_BACKGROUND_SAMPLES = 50
KERNEL_EVAL_SAMPLES = 100

TREE_MODEL_TYPES = {"random_forest", "xgboost"}
LINEAR_MODEL_TYPES = {"ridge"}


@dataclass
class FeatureContribution:
    feature: str
    value: float
    contribution: float  # signed: positive pushes the forecast up

    @property
    def direction(self) -> str:
        return "increases" if self.contribution >= 0 else "decreases"


def _unwrap_estimator(forecaster):
    """Digs the fitted sklearn estimator out of our Forecaster wrapper.

    SklearnForecaster hides the model behind a Pipeline (imputer -> optional
    scaler -> model). SHAP needs the estimator itself, plus data transformed the
    same way the estimator saw it during training.
    """
    pipeline = getattr(forecaster, "_pipeline", None)
    if pipeline is None:
        return None, None
    return pipeline.named_steps.get("model"), pipeline


def _transform_like_training(pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Applies every pipeline step except the final estimator."""
    data = frame[feature_columns]
    for name, step in pipeline.steps[:-1]:
        data = step.transform(data)
    return np.asarray(data, dtype="float64")


def explain_prediction(
    forecaster,
    model_type: str,
    feature_columns: list[str],
    row: pd.DataFrame,
    background: pd.DataFrame,
) -> list[FeatureContribution]:
    """Per-feature contributions to a single forecast, largest magnitude first.

    `background` is the reference distribution SHAP compares against - it should
    be a sample of training rows, since a contribution is always "relative to
    what this feature usually looks like".
    """
    import shap

    estimator, pipeline = _unwrap_estimator(forecaster)
    if estimator is None:
        raise TypeError(f"{model_type} does not expose a SHAP-compatible estimator")

    row_values = _transform_like_training(pipeline, row, feature_columns)
    background_values = _transform_like_training(pipeline, background, feature_columns)

    if model_type in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(row_values)
    elif model_type in LINEAR_MODEL_TYPES:
        explainer = shap.LinearExplainer(estimator, background_values)
        shap_values = explainer.shap_values(row_values)
    else:
        sampled = shap.sample(background_values, min(KERNEL_BACKGROUND_SAMPLES, len(background_values)))
        explainer = shap.KernelExplainer(estimator.predict, sampled)
        shap_values = explainer.shap_values(row_values, nsamples=KERNEL_EVAL_SAMPLES)

    contributions = np.asarray(shap_values).reshape(-1)
    raw_values = row[feature_columns].iloc[0]

    result = [
        FeatureContribution(
            feature=name,
            value=float(raw_values[name]),
            contribution=float(contributions[i]),
        )
        for i, name in enumerate(feature_columns)
    ]
    result.sort(key=lambda c: abs(c.contribution), reverse=True)
    return result


def top_contributions(
    contributions: list[FeatureContribution], limit: int = 8
) -> list[FeatureContribution]:
    return contributions[:limit]


def describe_contribution(contribution: FeatureContribution) -> str:
    """One human-readable line, for the dashboard and the report."""
    return (
        f"{FEATURE_LABELS.get(contribution.feature, contribution.feature)} "
        f"({contribution.value:.1f}) {contribution.direction} the forecast "
        f"by {abs(contribution.contribution):.1f} AQI points"
    )


# Raw column names are fine in code but poor on a dashboard.
FEATURE_LABELS = {
    "aqi_mean": "Today's average AQI",
    "aqi_max": "Today's peak AQI",
    "aqi_min": "Today's lowest AQI",
    "pm2_5_mean": "PM2.5",
    "pm10_mean": "PM10",
    "co_mean": "Carbon monoxide",
    "no2_mean": "Nitrogen dioxide",
    "so2_mean": "Sulphur dioxide",
    "o3_mean": "Ozone",
    "temp_mean": "Temperature",
    "humidity_mean": "Humidity",
    "wind_speed_mean": "Wind speed",
    "pressure_mean": "Air pressure",
    "precipitation_sum": "Rainfall",
    "day_of_week": "Day of week",
    "day_of_month": "Day of month",
    "month": "Month",
    "is_weekend": "Weekend",
    "aqi_lag1": "AQI yesterday",
    "aqi_lag2": "AQI 2 days ago",
    "aqi_lag3": "AQI 3 days ago",
    "aqi_lag7": "AQI a week ago",
    "aqi_change_rate": "Day-on-day change rate",
    "aqi_roll3_mean": "3-day average AQI",
    "aqi_roll7_mean": "7-day average AQI",
    "aqi_roll3_std": "3-day volatility",
}
