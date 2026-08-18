import datetime as dt

import pandas as pd
import pytest

from src.inference.predict import (
    ForecastDay,
    LoadedModel,
    build_forecast,
    load_local_models,
    load_models,
    load_registry_models,
)


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


class FakeRegistered:
    """Stands in for a Hopsworks model entry, whose download() yields a directory."""

    def __init__(self, directory):
        self._directory = directory

    def download(self):
        return str(self._directory)


class FakeRegistry:
    def __init__(self, entries):
        self._entries = entries

    def get_model(self, name):
        if name not in self._entries:
            raise RuntimeError(f"no such model: {name}")
        return self._entries[name]


def write_bundle(root, horizon, model_type="ridge", family="sklearn"):
    """Writes a bundle in the same layout `save_local_bundle` produces."""
    import json

    import joblib

    directory = root / f"h{horizon}"
    directory.mkdir(parents=True)
    joblib.dump(FakeModel(100.0 + horizon), directory / "model.joblib")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "family": family,
                "feature_columns": ["aqi_mean", "temp_mean"],
                "metrics": {"rmse": 9.0},
            }
        )
    )
    return directory


class TestLoadLocalModels:
    def test_loads_every_bundle_present(self, tmp_path):
        for horizon in (1, 2, 3):
            write_bundle(tmp_path, horizon)

        models = load_local_models(tmp_path)

        assert sorted(models) == [1, 2, 3]
        assert all(m.source == "local" for m in models.values())

    def test_absent_horizon_is_skipped_without_error(self, tmp_path):
        write_bundle(tmp_path, 1)

        assert sorted(load_local_models(tmp_path)) == [1]

    def test_unreadable_bundle_costs_only_its_own_horizon(self, tmp_path):
        write_bundle(tmp_path, 1)
        broken = write_bundle(tmp_path, 2)
        (broken / "model.joblib").write_text("not a pickle")
        write_bundle(tmp_path, 3)

        assert sorted(load_local_models(tmp_path)) == [1, 3]


class TestLoadRegistryModels:
    def test_loads_each_horizon_from_its_downloaded_directory(self, tmp_path):
        entries = {
            f"aqi_forecast_h{h}": FakeRegistered(write_bundle(tmp_path, h)) for h in (1, 2, 3)
        }

        models = load_registry_models(FakeRegistry(entries))

        assert sorted(models) == [1, 2, 3]
        assert all(m.source == "registry" for m in models.values())

    def test_one_missing_horizon_does_not_lose_the_others(self, tmp_path):
        entries = {
            f"aqi_forecast_h{h}": FakeRegistered(write_bundle(tmp_path, h)) for h in (1, 3)
        }

        assert sorted(load_registry_models(FakeRegistry(entries))) == [1, 3]

    def test_empty_registry_yields_no_models_rather_than_raising(self, tmp_path):
        assert load_registry_models(FakeRegistry({})) == {}


class TestLoadModels:
    def test_local_bundles_fill_horizons_the_registry_could_not_supply(self, tmp_path, monkeypatch):
        registry_dir = tmp_path / "registry"
        local_dir = tmp_path / "local"
        entries = {"aqi_forecast_h1": FakeRegistered(write_bundle(registry_dir, 1))}
        for horizon in (1, 2, 3):
            write_bundle(local_dir, horizon)

        monkeypatch.setattr(
            "src.inference.predict.load_registry_models",
            lambda: load_registry_models(FakeRegistry(entries)),
        )

        models = load_models(local_dir)

        assert sorted(models) == [1, 2, 3]
        # The registry wins where it has an answer; local only fills the gaps.
        assert models[1].source == "registry"
        assert models[2].source == "local"

    def test_unreachable_registry_falls_back_entirely_to_local(self, tmp_path, monkeypatch):
        for horizon in (1, 2, 3):
            write_bundle(tmp_path, horizon)

        def unreachable():
            raise ConnectionError("hopsworks unreachable")

        monkeypatch.setattr("src.inference.predict.load_registry_models", unreachable)

        models = load_models(tmp_path)

        assert sorted(models) == [1, 2, 3]
        assert all(m.source == "local" for m in models.values())
