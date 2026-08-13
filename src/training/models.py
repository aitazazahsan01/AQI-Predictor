"""Module 4, part 2: the candidate forecasting models.

The brief asks for a range "from statistical modelling to deep learning", so
this covers five families plus a naive baseline, all behind one interface so
train.py can score them identically.

Optional heavy dependencies (xgboost, statsmodels, tensorflow) are imported
lazily. If one isn't installed the candidate is skipped with a clear message
rather than crashing the whole run — a training pipeline that dies because an
optional extra is missing is worse than one that trains four models out of six.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# The underlying observed series that targets are derived from. Sequence models
# forecast this directly rather than the pre-shifted target column.
BASE_SERIES_COLUMN = "aqi_mean"


def _horizon_from_target(target_column: str) -> int:
    """'target_h3' -> 3."""
    return int(target_column.rsplit("_h", 1)[-1])


class Forecaster(ABC):
    """Common interface so every candidate can be trained and scored the same way."""

    name: str = "unnamed"
    family: str = "unknown"

    @abstractmethod
    def fit(self, train: pd.DataFrame, feature_columns: list[str], target_column: str) -> None: ...

    @abstractmethod
    def predict(self, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray: ...


class PersistenceForecaster(Forecaster):
    """Naive baseline: "tomorrow will be the same as today".

    This exists to keep everyone honest. Air quality is strongly autocorrelated,
    so a model can post an impressive-looking R² while being *worse* than simply
    repeating today's value. Any model that cannot beat this is not adding value,
    and reporting it is the difference between an honest evaluation and a
    flattering one.
    """

    name = "persistence_baseline"
    family = "baseline"

    def fit(self, train, feature_columns, target_column):
        return None

    def predict(self, frame, feature_columns):
        return frame["aqi_mean"].to_numpy(dtype="float64")


class SklearnForecaster(Forecaster):
    """Wraps any sklearn-compatible regressor, optionally behind a scaler."""

    family = "sklearn"

    def __init__(self, name: str, estimator, needs_scaling: bool = False):
        self.name = name
        self._estimator = estimator
        self._needs_scaling = needs_scaling
        self._pipeline = None

    def fit(self, train, feature_columns, target_column):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps = [("impute", SimpleImputer(strategy="median"))]
        if self._needs_scaling:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", self._estimator))

        self._pipeline = Pipeline(steps)
        self._pipeline.fit(train[feature_columns], train[target_column])

    def predict(self, frame, feature_columns):
        return self._pipeline.predict(frame[feature_columns])

    @property
    def fitted_estimator(self):
        return self._pipeline.named_steps["model"] if self._pipeline else None


class SarimaxForecaster(Forecaster):
    """Classical statistical time-series model (the 'statistical' end of the range).

    Models the AQI series itself with weekly seasonality. Unlike the tree/linear
    models it never sees the engineered lag features - it derives its own
    temporal structure from the series.

    Deliberately univariate: SARIMAX can take exogenous weather regressors, but
    forecasting h steps ahead would then require h days of *future* weather,
    which we don't have at prediction time.
    """

    name = "sarimax"
    family = "statsmodels"

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)):
        self._order = order
        self._seasonal_order = seasonal_order
        self._result = None
        self._horizon = 1

    def fit(self, train, feature_columns, target_column):
        """Fits on the AQI series itself, not on the shifted target.

        Modelling the shifted target and walking forward over it would leak: the
        observation fed back at step i for horizon 3 is the AQI three days after
        that row, so the model would only ever be predicting one step past what
        it already knew - and would score the same at h1 and h3, which is the
        giveaway that something is wrong.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._horizon = _horizon_from_target(target_column)

        model = SARIMAX(
            train[BASE_SERIES_COLUMN].astype("float64").to_numpy(),
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)

    def predict(self, frame, feature_columns):
        """Walk-forward, forecasting a genuine `horizon` steps at each point.

        For each test day D we first fold in D's own observed AQI (which is known
        at D), then forecast `horizon` steps and keep the last one - that is the
        AQI at D + horizon, which is exactly what the target column holds. Only
        data available at D is ever used, matching what the lag features give the
        tabular models.
        """
        if BASE_SERIES_COLUMN not in frame.columns:
            forecast = self._result.forecast(steps=self._horizon)
            return np.repeat(float(np.asarray(forecast)[-1]), len(frame))

        observed = frame[BASE_SERIES_COLUMN].astype("float64").to_numpy()
        result = self._result
        predictions = []

        for i in range(len(frame)):
            # refit=False keeps parameters fixed and only advances the state,
            # which is what makes 90 walk-forward steps affordable.
            result = result.append(observed[i : i + 1], refit=False)
            forecast = np.asarray(result.forecast(steps=self._horizon), dtype="float64")
            predictions.append(float(forecast[-1]))

        return np.asarray(predictions, dtype="float64")


class LstmForecaster(Forecaster):
    """Deep learning end of the range: an LSTM over a rolling window of recent days.

    Where the tabular models see one row at a time, this sees the last
    `sequence_length` days as an ordered sequence, so it can in principle learn
    temporal shapes (a multi-day build-up, a sharp clearing) that a single row
    can't express.
    """

    name = "lstm"
    family = "tensorflow"

    def __init__(self, sequence_length: int = 14, epochs: int = 60, batch_size: int = 32):
        self._sequence_length = sequence_length
        self._epochs = epochs
        self._batch_size = batch_size
        self._model = None
        self._scaler = None
        self._feature_columns: list[str] = []
        self._history_tail: pd.DataFrame | None = None

    def fit(self, train, feature_columns, target_column):
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        tf.keras.utils.set_random_seed(RANDOM_SEED)

        self._feature_columns = feature_columns
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

        values = self._imputer.fit_transform(train[feature_columns])
        values = self._scaler.fit_transform(values)
        targets = train[target_column].to_numpy(dtype="float64")

        x, y = self._to_sequences(values, targets)
        if len(x) == 0:
            raise ValueError(
                f"Not enough rows ({len(train)}) to build sequences of length {self._sequence_length}"
            )

        self._model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._sequence_length, len(feature_columns))),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        self._model.compile(optimizer="adam", loss="mse")
        self._model.fit(
            x,
            y,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=0,
            validation_split=0.15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True
                )
            ],
        )

        # Keep the tail of training so the first test rows have the history they
        # need to form a full sequence.
        self._history_tail = train[feature_columns].tail(self._sequence_length - 1).copy()

    def predict(self, frame, feature_columns):
        combined = pd.concat([self._history_tail, frame[feature_columns]], ignore_index=True)
        values = self._imputer.transform(combined)
        values = self._scaler.transform(values)

        windows = []
        offset = len(self._history_tail)
        for i in range(len(frame)):
            end = offset + i + 1
            windows.append(values[end - self._sequence_length : end])

        predictions = self._model.predict(np.asarray(windows), verbose=0)
        return predictions.reshape(-1)

    def _to_sequences(self, values: np.ndarray, targets: np.ndarray):
        x, y = [], []
        for end in range(self._sequence_length, len(values) + 1):
            x.append(values[end - self._sequence_length : end])
            y.append(targets[end - 1])
        return np.asarray(x), np.asarray(y)


def build_candidates() -> list[Forecaster]:
    """Every candidate that can actually run in this environment.

    Optional dependencies are probed here rather than at import time so a
    missing extra costs one candidate, not the whole training run.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    candidates: list[Forecaster] = [
        PersistenceForecaster(),
        SklearnForecaster("ridge", Ridge(alpha=1.0), needs_scaling=True),
        SklearnForecaster(
            "random_forest",
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
            ),
        ),
    ]

    try:
        from xgboost import XGBRegressor

        candidates.append(
            SklearnForecaster(
                "xgboost",
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            )
        )
    except ImportError:
        print("  (skipping xgboost - not installed)")

    try:
        import statsmodels  # noqa: F401

        candidates.append(SarimaxForecaster())
    except ImportError:
        print("  (skipping sarimax - statsmodels not installed)")

    try:
        import tensorflow  # noqa: F401

        candidates.append(LstmForecaster())
    except ImportError:
        print("  (skipping lstm - tensorflow not installed)")

    return candidates
