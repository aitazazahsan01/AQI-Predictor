"""Module 4, part 3: train every candidate for every horizon and pick winners.

Scoring is deliberately blunt and comparable: identical train/test split,
identical metrics, one table per horizon. The best model for day 1 is often not
the best for day 3, so selection happens per horizon rather than once overall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.training.data_prep import (
    HORIZONS,
    get_feature_columns,
    prepare_horizon_dataset,
    target_column,
    time_based_split,
)
from src.training.models import Forecaster, build_candidates


@dataclass
class Evaluation:
    model_name: str
    family: str
    horizon: int
    rmse: float
    mae: float
    r2: float
    train_seconds: float
    n_train: int
    n_test: int
    failed: str | None = None
    model: Forecaster | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.failed is None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, computed directly so the metric doesn't
    depend on which sklearn version is installed."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def evaluate_candidate(
    candidate: Forecaster,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
) -> Evaluation:
    target = target_column(horizon)
    started = time.perf_counter()

    try:
        candidate.fit(train, feature_columns, target)
        predictions = np.asarray(candidate.predict(test, feature_columns), dtype="float64")
        actuals = test[target].to_numpy(dtype="float64")

        if predictions.shape != actuals.shape:
            raise ValueError(f"predicted {predictions.shape}, expected {actuals.shape}")
        if not np.isfinite(predictions).all():
            raise ValueError("model produced non-finite predictions")

        return Evaluation(
            model_name=candidate.name,
            family=candidate.family,
            horizon=horizon,
            rmse=rmse(actuals, predictions),
            mae=mae(actuals, predictions),
            r2=r2(actuals, predictions),
            train_seconds=time.perf_counter() - started,
            n_train=len(train),
            n_test=len(test),
            model=candidate,
        )
    except Exception as exc:  # one broken candidate must not sink the run
        return Evaluation(
            model_name=candidate.name,
            family=candidate.family,
            horizon=horizon,
            rmse=float("inf"),
            mae=float("inf"),
            r2=float("-inf"),
            train_seconds=time.perf_counter() - started,
            n_train=len(train),
            n_test=len(test),
            failed=f"{type(exc).__name__}: {exc}",
        )


def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
    test_days: int = 90,
    min_hours_observed: int | None = None,
) -> list[Evaluation]:
    usable = prepare_horizon_dataset(df, horizon, min_hours_observed=min_hours_observed)
    train, test = time_based_split(usable, test_days=test_days)

    if train.empty or test.empty:
        raise ValueError(
            f"horizon {horizon}: not enough data to split "
            f"({len(usable)} usable rows, {test_days}-day test window)"
        )

    results = []
    for candidate in build_candidates():
        result = evaluate_candidate(candidate, train, test, feature_columns, horizon)
        status = "FAILED" if not result.ok else f"RMSE={result.rmse:6.2f}  MAE={result.mae:6.2f}  R2={result.r2:6.3f}"
        print(f"    {result.model_name:<22} {status}")
        if not result.ok:
            print(f"      -> {result.failed}")
        results.append(result)

    return results


def select_best(results: list[Evaluation]) -> Evaluation:
    """Lowest RMSE wins. RMSE (over MAE) because large misses matter more here —
    being 60 points wrong on a hazardous day is far worse than being 6 points
    wrong on ten ordinary days, and RMSE penalises that accordingly."""
    successful = [r for r in results if r.ok]
    if not successful:
        raise RuntimeError("every candidate failed to train")
    return min(successful, key=lambda r: r.rmse)


def results_to_frame(results: list[Evaluation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": r.horizon,
                "model": r.model_name,
                "family": r.family,
                "rmse": r.rmse,
                "mae": r.mae,
                "r2": r.r2,
                "train_s": round(r.train_seconds, 1),
                "status": "ok" if r.ok else "failed",
            }
            for r in results
        ]
    )


def train_all_horizons(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    test_days: int = 90,
    min_hours_observed: int | None = None,
) -> dict[int, list[Evaluation]]:
    feature_columns = get_feature_columns(df, horizons)
    print(f"Using {len(feature_columns)} features: {', '.join(feature_columns)}\n")

    all_results = {}
    for horizon in horizons:
        print(f"  --- horizon: {horizon} day(s) ahead ---")
        results = train_horizon(
            df,
            horizon,
            feature_columns,
            test_days=test_days,
            min_hours_observed=min_hours_observed,
        )
        best = select_best(results)
        print(f"    => best: {best.model_name} (RMSE {best.rmse:.2f})\n")
        all_results[horizon] = results

    return all_results
