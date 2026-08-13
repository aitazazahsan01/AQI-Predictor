"""Turning stored models plus the newest feature row into a 3-day forecast.

The dashboard never computes features itself - it only ever consumes rows the
feature pipeline already produced. That guarantees the numbers on screen were
derived exactly the same way as the numbers the models were trained on.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.alerts.aqi_scale import categorize
from src.training.data_prep import HORIZONS

DEFAULT_LOCAL_MODEL_DIR = Path("models")


@dataclass
class LoadedModel:
    horizon: int
    model: object
    feature_columns: list[str]
    model_type: str
    metrics: dict
    source: str  # "registry" or "local"


@dataclass
class ForecastDay:
    horizon: int
    date: dt.date
    aqi: float
    category: str
    color: str
    model_type: str

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "date": self.date.isoformat(),
            "aqi": round(self.aqi, 1),
            "category": self.category,
            "color": self.color,
            "model_type": self.model_type,
        }


def load_local_models(root: Path = DEFAULT_LOCAL_MODEL_DIR) -> dict[int, LoadedModel]:
    """Loads models previously written by `run_training_pipeline.py --save-local`."""
    import joblib

    models: dict[int, LoadedModel] = {}
    for horizon in HORIZONS:
        directory = Path(root) / f"h{horizon}"
        metadata_path = directory / "metadata.json"
        model_path = directory / "model.joblib"
        if not (metadata_path.exists() and model_path.exists()):
            continue

        metadata = json.loads(metadata_path.read_text())
        models[horizon] = LoadedModel(
            horizon=horizon,
            model=joblib.load(model_path),
            feature_columns=metadata["feature_columns"],
            model_type=metadata.get("model_type", "unknown"),
            metrics=metadata.get("metrics", {}),
            source="local",
        )
    return models


def load_registry_models(model_registry=None) -> dict[int, LoadedModel]:
    """Loads the current best model per horizon from the Hopsworks Model Registry."""
    import joblib

    from src.training.register import model_name

    if model_registry is None:
        from src.hopsworks_utils.connection import get_model_registry

        model_registry = get_model_registry()

    models: dict[int, LoadedModel] = {}
    for horizon in HORIZONS:
        registered = model_registry.get_model(model_name(horizon))
        directory = Path(registered.download())
        metadata = json.loads((directory / "metadata.json").read_text())
        models[horizon] = LoadedModel(
            horizon=horizon,
            model=joblib.load(directory / "model.joblib"),
            feature_columns=metadata["feature_columns"],
            model_type=metadata.get("model_type", "unknown"),
            metrics=metadata.get("metrics", {}),
            source="registry",
        )
    return models


def load_models(local_dir: Path = DEFAULT_LOCAL_MODEL_DIR) -> dict[int, LoadedModel]:
    """Registry first, local bundle as fallback.

    Production reads the registry; the fallback keeps the dashboard usable when
    the Hopsworks data ports are unreachable (see README troubleshooting).
    """
    try:
        models = load_registry_models()
        if models:
            return models
    except Exception as exc:
        print(f"Model registry unavailable ({type(exc).__name__}), falling back to local models.")
    return load_local_models(local_dir)


def build_forecast(latest_row: pd.DataFrame, models: dict[int, LoadedModel]) -> list[ForecastDay]:
    """Runs each horizon's model on the most recent feature row.

    `feature_columns` comes from the model's own metadata rather than being
    recomputed here, so a model trained on a different feature set can never be
    silently fed the wrong columns in the wrong order.
    """
    if latest_row.empty:
        return []

    as_of = pd.to_datetime(latest_row.iloc[-1]["date"]).date()
    forecasts: list[ForecastDay] = []

    for horizon in sorted(models):
        loaded = models[horizon]
        missing = [c for c in loaded.feature_columns if c not in latest_row.columns]
        if missing:
            raise ValueError(
                f"h{horizon} model needs features missing from the feature row: {missing}"
            )

        value = float(loaded.model.predict(latest_row, loaded.feature_columns)[-1])
        category = categorize(value)
        forecasts.append(
            ForecastDay(
                horizon=horizon,
                date=as_of + dt.timedelta(days=horizon),
                aqi=value,
                category=category.name if category else "Unknown",
                color=category.color if category else "#888888",
                model_type=loaded.model_type,
            )
        )

    return forecasts
