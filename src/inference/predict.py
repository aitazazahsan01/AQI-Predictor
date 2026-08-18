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


def _load_artifact(directory: Path, metadata: dict):
    """Rebuilds a fitted forecaster from a saved bundle.

    Branches on `family` because the LSTM cannot be pickled: it is saved as a
    Keras file plus its preprocessing, and has to be reassembled around them.
    """
    import joblib

    if metadata.get("family") == "tensorflow":
        from tensorflow import keras  # only installed where training happens

        from src.training.models import LstmForecaster

        preprocessing = joblib.load(directory / "preprocessing.joblib")
        forecaster = LstmForecaster(sequence_length=preprocessing["sequence_length"])
        forecaster._model = keras.models.load_model(directory / "model.keras")
        forecaster._imputer = preprocessing["imputer"]
        forecaster._scaler = preprocessing["scaler"]
        forecaster._history_tail = preprocessing["history_tail"]
        return forecaster

    return joblib.load(directory / "model.joblib")


def _build_loaded(horizon: int, directory: Path, source: str) -> LoadedModel:
    metadata = json.loads((directory / "metadata.json").read_text())
    return LoadedModel(
        horizon=horizon,
        model=_load_artifact(directory, metadata),
        feature_columns=metadata["feature_columns"],
        model_type=metadata.get("model_type", "unknown"),
        metrics=metadata.get("metrics", {}),
        source=source,
    )


def load_local_models(root: Path = DEFAULT_LOCAL_MODEL_DIR) -> dict[int, LoadedModel]:
    """Loads models previously written by `run_training_pipeline.py --save-local`."""
    models: dict[int, LoadedModel] = {}
    for horizon in HORIZONS:
        directory = Path(root) / f"h{horizon}"
        if not (directory / "metadata.json").exists():
            continue
        try:
            models[horizon] = _build_loaded(horizon, directory, source="local")
        except Exception as exc:
            print(f"Skipping local h{horizon}: {type(exc).__name__}: {exc}")
    return models


def load_registry_models(model_registry=None) -> dict[int, LoadedModel]:
    """Loads the current best model per horizon from the Hopsworks Model Registry.

    Each horizon is loaded independently so that one unusable entry - a model
    whose framework isn't installed here, or one that was never registered -
    costs only that horizon rather than the whole forecast.
    """
    if model_registry is None:
        from src.hopsworks_utils.connection import get_model_registry

        model_registry = get_model_registry()

    from src.training.register import model_name

    models: dict[int, LoadedModel] = {}
    for horizon in HORIZONS:
        try:
            registered = model_registry.get_model(model_name(horizon))
            if registered is None:
                raise LookupError(f"{model_name(horizon)} is not in the registry")
            directory = Path(registered.download())
            models[horizon] = _build_loaded(horizon, directory, source="registry")
        except Exception as exc:
            print(f"Skipping registry h{horizon}: {type(exc).__name__}: {exc}")
    return models


def load_models(local_dir: Path = DEFAULT_LOCAL_MODEL_DIR) -> dict[int, LoadedModel]:
    """Registry first, local bundle filling any gaps.

    Production reads the registry; the local fallback keeps the dashboard usable
    when the Hopsworks data ports are unreachable (see README troubleshooting),
    and covers individual horizons the registry could not supply.
    """
    try:
        models = load_registry_models()
    except Exception as exc:
        print(f"Model registry unavailable ({type(exc).__name__}: {exc}), using local models.")
        models = {}

    if len(models) < len(HORIZONS):
        for horizon, loaded in load_local_models(local_dir).items():
            models.setdefault(horizon, loaded)

    return models


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
