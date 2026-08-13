import datetime as dt

import pandas as pd
import pytest

from src.inference.predict import ForecastDay, LoadedModel, build_forecast


class FakeModel:
    """Returns a fixed value, and records what it was asked to predict on."""

    def __init__(self, value):
        self.value = value
        self.seen_columns = None

    def predict(self, frame, feature_columns):
        self.seen_columns = list(feature_columns)
        return [self.value] * len(frame)


def make_loaded(horizon, value, feature_columns=("aqi_mean", "temp_mean"), model_type="ridge"):
    return LoadedModel(
        horizon=horizon,
        model=FakeModel(value),
        feature_columns=list(feature_columns),
        model_type=model_type,
        metrics={"rmse": 9.0},
        source="local",
    )


def make_latest_row(date="2026-08-13", aqi_mean=120.0):
    return pd.DataFrame(
        [{"city": "islamabad", "date": pd.Timestamp(date, tz="UTC"), "aqi_mean": aqi_mean, "temp_mean": 30.0}]
    )


class TestBuildForecast:
    def test_returns_one_entry_per_horizon(self):
        models = {1: make_loaded(1, 100.0), 2: make_loaded(2, 110.0), 3: make_loaded(3, 120.0)}
        forecasts = build_forecast(make_latest_row(), models)
        assert [f.horizon for f in forecasts] == [1, 2, 3]

    def test_dates_are_offset_from_the_latest_observation(self):
        models = {1: make_loaded(1, 100.0), 3: make_loaded(3, 120.0)}
        forecasts = build_forecast(make_latest_row("2026-08-13"), models)
        assert forecasts[0].date == dt.date(2026, 8, 14)
        assert forecasts[1].date == dt.date(2026, 8, 16)

    def test_categories_come_from_the_shared_aqi_scale(self):
        models = {1: make_loaded(1, 45.0), 2: make_loaded(2, 175.0)}
        forecasts = build_forecast(make_latest_row(), models)
        assert forecasts[0].category == "Good"
        assert forecasts[1].category == "Unhealthy"

    def test_each_model_is_fed_its_own_recorded_feature_columns(self):
        # The metadata is authoritative - a model trained on a different feature
        # set must never be silently handed the wrong columns.
        models = {
            1: make_loaded(1, 100.0, feature_columns=["aqi_mean"]),
            2: make_loaded(2, 110.0, feature_columns=["aqi_mean", "temp_mean"]),
        }
        build_forecast(make_latest_row(), models)
        assert models[1].model.seen_columns == ["aqi_mean"]
        assert models[2].model.seen_columns == ["aqi_mean", "temp_mean"]

    def test_missing_feature_raises_rather_than_predicting_nonsense(self):
        models = {1: make_loaded(1, 100.0, feature_columns=["aqi_mean", "not_a_real_column"])}
        with pytest.raises(ValueError, match="not_a_real_column"):
            build_forecast(make_latest_row(), models)

    def test_empty_feature_row_yields_no_forecast(self):
        assert build_forecast(pd.DataFrame(), {1: make_loaded(1, 100.0)}) == []

    def test_model_type_is_carried_through_for_display(self):
        models = {1: make_loaded(1, 100.0, model_type="random_forest")}
        assert build_forecast(make_latest_row(), models)[0].model_type == "random_forest"


class TestForecastDay:
    def test_serialises_with_rounded_aqi(self):
        day = ForecastDay(1, dt.date(2026, 8, 14), 150.2647, "Unhealthy", "#FF0000", "ridge")
        assert day.as_dict() == {
            "horizon": 1,
            "date": "2026-08-14",
            "aqi": 150.3,
            "category": "Unhealthy",
            "color": "#FF0000",
            "model_type": "ridge",
        }
