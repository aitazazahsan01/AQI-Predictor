"""Scoring published forecasts against what actually happened.

Every number in the report so far comes from a chronological hold-out: the
model is fitted on old days and scored on recent ones. That is the right way to
*select* a model, but it is not the same as measuring one in operation. A
hold-out score is computed once, by the same process that chose the model, on
data that was sitting on disk the whole time.

This measures the other thing. Each nightly run committed a snapshot recording
what the system predicted for the next three days; later runs recorded what the
air actually did. So the git history of `web/public/data/forecast.json` is
already a forecast ledger with no extra infrastructure - it just has to be read
back and joined to itself.

Two properties make this an honest measurement rather than a flattering one:

- A forecast is scored against an observation that did not exist when it was
  made. There is no way to leak.
- The persistence baseline is reconstructed from the same ledger: the last AQI
  observed at the moment each forecast was issued. That is genuinely what a
  person could have said for free, at the time, without a model.

The functions here are pure - reading git is the caller's job (see
`scripts/run_backtest.py`) - so the scoring is testable without a repository.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from src.alerts.aqi_scale import categorize

# A prediction is only comparable to an observation once that day is over and
# has been aggregated. Anything still unresolved is reported, never scored.
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Prediction:
    """One forecast, as it was published."""

    issued_for: str      # the last observed day at the time of issue
    horizon: int
    target_date: str
    predicted: float
    model_type: str
    baseline: float | None   # last observed AQI when this was issued


@dataclass(frozen=True)
class Matched:
    prediction: Prediction
    actual: float

    @property
    def error(self) -> float:
        """Signed: positive means the forecast was too high."""
        return self.prediction.predicted - self.actual

    @property
    def baseline_error(self) -> float | None:
        if self.prediction.baseline is None:
            return None
        return self.prediction.baseline - self.actual

    @property
    def category_hit(self) -> bool:
        predicted = categorize(self.prediction.predicted)
        actual = categorize(self.actual)
        return predicted is not None and actual is not None and predicted.name == actual.name


@dataclass(frozen=True)
class HorizonScore:
    horizon: int
    n: int
    rmse: float
    mae: float
    bias: float
    category_accuracy: float
    baseline_rmse: float | None
    baseline_mae: float | None
    beat_baseline_rate: float | None

    @property
    def skill(self) -> float | None:
        """Percent reduction in RMSE against persistence. Negative means worse."""
        if self.baseline_rmse is None or self.baseline_rmse == 0:
            return None
        return (1 - self.rmse / self.baseline_rmse) * 100


def extract_predictions(snapshot: dict) -> list[Prediction]:
    """The forecasts one published snapshot contains.

    `baseline` is the snapshot's own latest observed AQI - what persistence
    would have predicted for every horizon at that moment.
    """
    latest = snapshot.get("latest") or {}
    issued_for = latest.get("date")
    baseline = latest.get("aqi")

    predictions = []
    for entry in snapshot.get("forecast") or []:
        if entry.get("aqi") is None or not entry.get("date"):
            continue
        predictions.append(
            Prediction(
                issued_for=issued_for or "",
                horizon=int(entry["horizon"]),
                target_date=entry["date"],
                predicted=float(entry["aqi"]),
                model_type=entry.get("model_type", "unknown"),
                baseline=None if baseline is None else float(baseline),
            )
        )
    return predictions


def extract_observations(snapshot: dict) -> dict[str, float]:
    """Every observed daily AQI a snapshot carries, from its history and latest row."""
    observations: dict[str, float] = {}

    for point in snapshot.get("history") or []:
        if point.get("aqi") is not None and point.get("date"):
            observations[point["date"]] = float(point["aqi"])

    latest = snapshot.get("latest") or {}
    if latest.get("aqi") is not None and latest.get("date"):
        observations[latest["date"]] = float(latest["aqi"])

    return observations


def build_ledger(snapshots: list[dict]) -> tuple[list[Prediction], dict[str, float]]:
    """Folds a chronological list of snapshots into predictions plus observations.

    Snapshots overlap heavily - each carries 45 days of history - so
    observations are merged, and a later snapshot wins where they disagree
    (features can be revised as late hours arrive).

    Predictions are deduplicated on (target_date, horizon), keeping the
    earliest. Re-running a workflow must not let the same day be forecast twice
    and counted twice.
    """
    predictions: dict[tuple[str, int], Prediction] = {}
    observations: dict[str, float] = {}

    for snapshot in snapshots:
        observations.update(extract_observations(snapshot))
        for prediction in extract_predictions(snapshot):
            key = (prediction.target_date, prediction.horizon)
            predictions.setdefault(key, prediction)

    ordered = sorted(predictions.values(), key=lambda p: (p.target_date, p.horizon))
    return ordered, observations


def match(predictions: list[Prediction], observations: dict[str, float]) -> list[Matched]:
    """Pairs each prediction with the day it predicted, where that day is known."""
    return [
        Matched(prediction=p, actual=observations[p.target_date])
        for p in predictions
        if p.target_date in observations
    ]


def unresolved(predictions: list[Prediction], observations: dict[str, float]) -> list[Prediction]:
    """Forecasts whose target day has not been observed yet."""
    return [p for p in predictions if p.target_date not in observations]


def coverage_gaps(observations: dict[str, float]) -> list[str]:
    """Calendar days with no observation between the first and last that has one.

    A forecast can only be scored against a day the feature store actually
    recorded, so a gap silently shrinks the sample. Reporting the gaps keeps
    that visible instead of letting a small `n` look like a short history.
    """
    if not observations:
        return []

    known = sorted(dt.date.fromisoformat(d) for d in observations)
    span = (known[-1] - known[0]).days + 1
    every_day = {known[0] + dt.timedelta(days=i) for i in range(span)}
    return [d.isoformat() for d in sorted(every_day - set(known))]


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def _mae(errors: list[float]) -> float:
    return sum(abs(e) for e in errors) / len(errors)


def score_by_horizon(matched: list[Matched]) -> list[HorizonScore]:
    """Per-horizon accuracy, alongside the persistence baseline over the same days.

    The baseline is scored only on the rows where it exists, and those are the
    same rows the model is scored on, so the comparison is like for like.
    """
    scores: list[HorizonScore] = []

    for horizon in sorted({m.prediction.horizon for m in matched}):
        rows = [m for m in matched if m.prediction.horizon == horizon]
        errors = [m.error for m in rows]

        with_baseline = [m for m in rows if m.baseline_error is not None]
        baseline_errors = [m.baseline_error for m in with_baseline]

        beat = None
        if with_baseline:
            wins = sum(1 for m in with_baseline if abs(m.error) < abs(m.baseline_error))
            beat = wins / len(with_baseline) * 100

        scores.append(
            HorizonScore(
                horizon=horizon,
                n=len(rows),
                rmse=_rmse(errors),
                mae=_mae(errors),
                bias=sum(errors) / len(errors),
                category_accuracy=sum(1 for m in rows if m.category_hit) / len(rows) * 100,
                baseline_rmse=_rmse(baseline_errors) if baseline_errors else None,
                baseline_mae=_mae(baseline_errors) if baseline_errors else None,
                beat_baseline_rate=beat,
            )
        )

    return scores
