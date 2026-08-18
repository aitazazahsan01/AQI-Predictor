import datetime as dt
import json
import math

import pandas as pd
import pytest

from src.inference.predict import ForecastDay
from src.inference.snapshot import (
    build_history,
    build_latest,
    build_models_summary,
    build_scale,
    build_snapshot,
)


class FakeLoaded:
    def __init__(self, model_type="ridge", source="registry", metrics=None, n_features=26):
        self.model_type = model_type
        self.source = source
        self.metrics = metrics if metrics is not None else {"rmse": 9.0312, "mae": 6.7955}
        self.feature_columns = [f"f{i}" for i in range(n_features)]


def make_features(rows=5, **overrides):
    base = {
        "city": "islamabad",
        "aqi_mean": 120.0,
        "aqi_max": 150.0,
        "aqi_min": 90.0,
        "pm2_5_mean": 36.5,
        "temp_mean": 31.2,
    }
    base.update(overrides)
    return pd.DataFrame(
        [
            {**base, "date": pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(days=i)}
            for i in range(rows)
        ]
    )


def make_forecasts():
    return [
        ForecastDay(
            horizon=h,
            date=dt.date(2026, 8, 18) + dt.timedelta(days=h - 1),
            aqi=110.0 + h,
            category="Unhealthy for Sensitive Groups",
            color="#FF7E00",
            model_type="ridge",
        )
        for h in (1, 2, 3)
    ]


class TestBuildLatest:
    def test_uses_the_last_row_not_the_first(self):
        features = make_features(rows=3)
        features.loc[2, "aqi_mean"] = 200.0

        assert build_latest(features)["aqi"] == 200.0

    def test_categorises_from_the_shared_scale(self):
        latest = build_latest(make_features(aqi_mean=45.0))

        assert latest["category"] == "Good"
        assert latest["color"] == "#00E400"

    def test_conditions_skip_columns_the_frame_does_not_have(self):
        latest = build_latest(make_features())
        labels = [c["label"] for c in latest["conditions"]]

        assert "PM2.5" in labels
        # so2_mean is absent from the fixture, so it must not appear as a null.
        assert "Sulphur dioxide" not in labels

    def test_nan_becomes_null_rather_than_a_nan_literal(self):
        features = make_features()
        features.loc[features.index[-1], "aqi_max"] = math.nan

        assert build_latest(features)["aqi_max"] is None

    def test_date_is_serialised_as_a_plain_iso_day(self):
        assert build_latest(make_features(rows=1))["date"] == "2026-08-01"


class TestBuildHistory:
    def test_returns_at_most_the_trend_window(self):
        assert len(build_history(make_features(rows=90), trend_days=45)) == 45

    def test_keeps_chronological_order(self):
        history = build_history(make_features(rows=4))

        assert [point["date"] for point in history] == sorted(point["date"] for point in history)


class TestBuildScale:
    def test_covers_every_category(self):
        assert len(build_scale()) == 6

    def test_open_ended_top_band_serialises_as_null_not_infinity(self):
        top = build_scale()[-1]

        assert top["name"] == "Hazardous"
        assert top["upper"] is None


class TestBuildModelsSummary:
    def test_orders_by_horizon(self):
        summary = build_models_summary({3: FakeLoaded(), 1: FakeLoaded(), 2: FakeLoaded()})

        assert [entry["horizon"] for entry in summary] == [1, 2, 3]

    def test_carries_the_source_so_the_page_can_show_where_a_model_came_from(self):
        summary = build_models_summary({1: FakeLoaded(source="local")})

        assert summary[0]["source"] == "local"


class TestBuildSnapshot:
    def build(self, **kwargs):
        defaults = dict(
            city_slug="islamabad",
            features=make_features(rows=10),
            forecasts=make_forecasts(),
            models={h: FakeLoaded() for h in (1, 2, 3)},
            feature_source="test fixture",
            include_drivers=False,
        )
        defaults.update(kwargs)
        return build_snapshot(**defaults)

    def test_is_json_serialisable(self):
        # The whole point of the module: if this raises, the website breaks.
        json.dumps(self.build())

    def test_declares_its_schema_version(self):
        assert self.build()["schema_version"] == 1

    def test_quiet_air_produces_no_alert(self):
        snapshot = self.build(features=make_features(rows=10, aqi_mean=40.0))

        assert snapshot["alert"] is None

    def test_unhealthy_forecast_raises_an_alert(self):
        forecasts = make_forecasts()
        forecasts[1].aqi = 210.0

        alert = self.build(forecasts=forecasts)["alert"]

        assert alert is not None
        assert alert["level"] == "critical"
        assert alert["worst_aqi"] == 210.0

    def test_drivers_are_omitted_when_not_requested(self):
        assert self.build(include_drivers=False)["drivers"] == []

    def test_records_where_the_features_came_from(self):
        snapshot = self.build(feature_source="Hopsworks feature store (1,474 rows)")

        assert snapshot["feature_source"] == "Hopsworks feature store (1,474 rows)"

    def test_counts_every_observed_day_not_just_the_charted_window(self):
        assert self.build(features=make_features(rows=90))["observed_days"] == 90


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_forecast_entries_survive_serialisation(horizon):
    snapshot = build_snapshot(
        "islamabad",
        features=make_features(rows=5),
        forecasts=make_forecasts(),
        models={h: FakeLoaded() for h in (1, 2, 3)},
        feature_source="test fixture",
        include_drivers=False,
    )
    entry = next(f for f in snapshot["forecast"] if f["horizon"] == horizon)

    assert entry["aqi"] == pytest.approx(110.0 + horizon)
    assert entry["date"] == (dt.date(2026, 8, 18) + dt.timedelta(days=horizon - 1)).isoformat()
