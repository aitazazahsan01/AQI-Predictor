"""Module 4, part 4: saving winning models to the Hopsworks Model Registry.

Each registered model carries the metadata needed to use it correctly later.
`feature_columns` in particular is authoritative: the dashboard builds its
inference vector in exactly that order, so a model trained on a different
feature set can never be silently fed the wrong columns.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from src.training.train import Evaluation

MODEL_NAME_TEMPLATE = "aqi_forecast_h{horizon}"

# Hopsworks caps entity descriptions at 256 characters (learned the hard way
# when creating the daily feature group).
MAX_DESCRIPTION_CHARS = 256


def model_name(horizon: int) -> str:
    return MODEL_NAME_TEMPLATE.format(horizon=horizon)


def build_metadata(
    evaluation: Evaluation,
    feature_columns: list[str],
    training_window: tuple[str, str],
) -> dict:
    return {
        "model_type": evaluation.model_name,
        "family": evaluation.family,
        "horizon_days": evaluation.horizon,
        "feature_columns": feature_columns,
        "metrics": {
            "rmse": round(evaluation.rmse, 4),
            "mae": round(evaluation.mae, 4),
            "r2": round(evaluation.r2, 4),
        },
        "training_window": {"start": training_window[0], "end": training_window[1]},
        "n_train": evaluation.n_train,
        "n_test": evaluation.n_test,
    }


def build_description(evaluation: Evaluation) -> str:
    description = (
        f"{evaluation.model_name} forecasting AQI {evaluation.horizon} day(s) ahead. "
        f"RMSE {evaluation.rmse:.2f}, MAE {evaluation.mae:.2f}, R2 {evaluation.r2:.3f} "
        f"on a {evaluation.n_test}-day held-out window."
    )
    return description[:MAX_DESCRIPTION_CHARS]


def save_model_artifact(evaluation: Evaluation, directory: Path) -> None:
    """Persists the fitted model. Keras models need their own format; everything
    else pickles cleanly via joblib."""
    directory.mkdir(parents=True, exist_ok=True)
    model = evaluation.model

    if evaluation.family == "tensorflow":
        model._model.save(directory / "model.keras")
        import joblib

        joblib.dump(
            {"imputer": model._imputer, "scaler": model._scaler, "sequence_length": model._sequence_length,
             "history_tail": model._history_tail},
            directory / "preprocessing.joblib",
        )
    else:
        import joblib

        joblib.dump(model, directory / "model.joblib")


def save_local_bundle(
    evaluation: Evaluation,
    feature_columns: list[str],
    training_window: tuple[str, str],
    root: Path,
) -> Path:
    """Writes a model plus its metadata to disk, mirroring the registry layout.

    The dashboard prefers the Model Registry, but falls back to this so it can
    still run on networks that can't reach the Hopsworks data ports.
    """
    directory = root / f"h{evaluation.horizon}"
    if directory.exists():
        shutil.rmtree(directory)

    save_model_artifact(evaluation, directory)
    metadata = build_metadata(evaluation, feature_columns, training_window)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return directory


def register_best_model(
    evaluation: Evaluation,
    feature_columns: list[str],
    training_window: tuple[str, str],
    model_registry=None,
) -> None:
    """Uploads the winning model for one horizon to the Model Registry."""
    if model_registry is None:
        from src.hopsworks_utils.connection import get_model_registry

        model_registry = get_model_registry()

    metadata = build_metadata(evaluation, feature_columns, training_window)
    staging = Path(tempfile.mkdtemp(prefix=f"aqi_h{evaluation.horizon}_"))

    try:
        save_model_artifact(evaluation, staging)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))

        registered = model_registry.python.create_model(
            name=model_name(evaluation.horizon),
            metrics=metadata["metrics"],
            description=build_description(evaluation),
            input_example=None,
        )
        registered.save(str(staging))
        print(
            f"  registered {model_name(evaluation.horizon)} "
            f"({evaluation.model_name}, RMSE {evaluation.rmse:.2f})"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
