import numpy as np
import pandas as pd
import pytest

from src.training.data_prep import (
    build_targets,
    get_feature_columns,
    prepare_horizon_dataset,
    target_column,
    time_based_split,
)
from src.training.models import BASE_SERIES_COLUMN, PersistenceForecaster, _horizon_from_target
from src.training.train import Evaluation, mae, r2, rmse, select_best


def make_daily(aqi_values, city="islamabad", start="2026-01-01"):
    n = len(aqi_values)
    return pd.DataFrame(
        {
            "city": [city] * n,
            "date": pd.date_range(start, periods=n, tz="UTC"),
            "aqi_mean": [float(v) for v in aqi_values],
            "pm2_5_mean": np.linspace(10, 40, n),
            "temp_mean": np.linspace(15, 35, n),
            "hours_observed": [24] * n,
        }
    )


class TestBuildTargets:
    def test_target_is_the_future_value(self):
        out = build_targets(make_daily([10, 20, 30, 40]))
        assert out.iloc[0]["target_h1"] == 20.0
        assert out.iloc[0]["target_h2"] == 30.0
        assert out.iloc[0]["target_h3"] == 40.0

    def test_last_rows_have_no_target(self):
        # The newest day genuinely has no "tomorrow" yet.
        out = build_targets(make_daily([10, 20, 30, 40]))
        assert pd.isna(out.iloc[3]["target_h1"])
        assert pd.isna(out.iloc[2]["target_h2"])

    def test_targets_do_not_bleed_across_cities(self):
        isb = make_daily([10, 20, 30])
        lhr = make_daily([500, 600, 700], city="lahore")
        out = build_targets(pd.concat([isb, lhr], ignore_index=True))
        last_isb = out[out["city"] == "islamabad"].iloc[-1]
        assert pd.isna(last_isb["target_h1"]), "Islamabad's last day must not borrow Lahore's value"

    def test_empty_input_is_handled(self):
        assert build_targets(pd.DataFrame()).empty


class TestFeatureColumns:
    def test_target_columns_are_never_features(self):
        # A target leaking into the feature matrix produces a perfect, useless model.
        df = build_targets(make_daily([10, 20, 30, 40, 50]))
        features = get_feature_columns(df)
        for horizon in (1, 2, 3):
            assert target_column(horizon) not in features

    def test_keys_and_quality_flag_are_excluded(self):
        df = build_targets(make_daily([10, 20, 30, 40, 50]))
        features = get_feature_columns(df)
        assert "city" not in features
        assert "date" not in features
        assert "hours_observed" not in features

    def test_real_features_are_included(self):
        df = build_targets(make_daily([10, 20, 30, 40, 50]))
        features = get_feature_columns(df)
        assert "aqi_mean" in features  # today's AQI is known at prediction time
        assert "pm2_5_mean" in features
        assert "temp_mean" in features


class TestHorizonDataset:
    def test_rows_without_a_known_target_are_dropped(self):
        df = build_targets(make_daily([10, 20, 30, 40]))
        assert len(prepare_horizon_dataset(df, 1)) == 3
        assert len(prepare_horizon_dataset(df, 3)) == 1

    def test_partial_days_can_be_filtered_out(self):
        df = build_targets(make_daily([10, 20, 30, 40, 50]))
        df.loc[1, "hours_observed"] = 4
        kept = prepare_horizon_dataset(df, 1, min_hours_observed=18)
        assert 4 not in kept["hours_observed"].tolist()


class TestTimeBasedSplit:
    def test_test_set_is_strictly_later_than_train_set(self):
        df = make_daily(list(range(100)))
        train, test = time_based_split(df, test_days=20)
        assert train["date"].max() < test["date"].min()

    def test_split_sizes_are_sane(self):
        df = make_daily(list(range(100)))
        train, test = time_based_split(df, test_days=20)
        assert len(test) == 20
        assert len(train) + len(test) == 100

    def test_split_is_chronological_even_if_input_is_shuffled(self):
        df = make_daily(list(range(50))).sample(frac=1, random_state=0)
        train, test = time_based_split(df, test_days=10)
        assert train["date"].max() < test["date"].min()
        assert test["date"].is_monotonic_increasing


class TestMetrics:
    def test_perfect_prediction_scores_perfectly(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0
        assert r2(y, y) == 1.0

    def test_rmse_punishes_large_errors_more_than_mae(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        one_big_miss = np.array([0.0, 0.0, 0.0, 20.0])
        spread_out = np.array([5.0, 5.0, 5.0, 5.0])
        assert mae(y_true, one_big_miss) == mae(y_true, spread_out)
        assert rmse(y_true, one_big_miss) > rmse(y_true, spread_out)

    def test_r2_is_zero_for_predicting_the_mean(self):
        y_true = np.array([10.0, 20.0, 30.0])
        assert r2(y_true, np.full(3, 20.0)) == pytest.approx(0.0)


class TestPersistenceBaseline:
    def test_it_predicts_todays_value(self):
        df = build_targets(make_daily([10, 20, 30, 40]))
        model = PersistenceForecaster()
        model.fit(df, [], "target_h1")
        np.testing.assert_array_equal(model.predict(df, []), df["aqi_mean"].to_numpy())


class TestSelectBest:
    def _evaluation(self, name, rmse_value, failed=None):
        return Evaluation(
            model_name=name,
            family="test",
            horizon=1,
            rmse=rmse_value,
            mae=0.0,
            r2=0.0,
            train_seconds=0.0,
            n_train=10,
            n_test=5,
            failed=failed,
        )

    def test_lowest_rmse_wins(self):
        results = [self._evaluation("a", 10.0), self._evaluation("b", 5.0), self._evaluation("c", 8.0)]
        assert select_best(results).model_name == "b"

    def test_failed_candidates_are_never_selected(self):
        results = [
            self._evaluation("broken", 0.0, failed="ValueError: boom"),
            self._evaluation("working", 9.0),
        ]
        assert select_best(results).model_name == "working"

    def test_all_failed_raises(self):
        with pytest.raises(RuntimeError):
            select_best([self._evaluation("x", 1.0, failed="nope")])


class TestSequenceModelHorizon:
    """Guards the leak that made SARIMAX score identically at h1 and h3.

    Sequence models must forecast `horizon` steps from the *base* AQI series.
    Walking forward over the pre-shifted target column instead hands the model
    the actual value from `horizon` days later, so it only ever predicts one
    step past what it knows - which looks like a great score and is nonsense.
    """

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_horizon_is_recovered_from_target_name(self, horizon):
        assert _horizon_from_target(target_column(horizon)) == horizon

    def test_base_series_is_the_unshifted_aqi_column(self):
        # If this ever points at a target column, the leak is back.
        assert BASE_SERIES_COLUMN == "aqi_mean"
        assert not BASE_SERIES_COLUMN.startswith("target_")

    def test_base_series_column_survives_target_building(self):
        df = build_targets(make_daily([10, 20, 30, 40, 50]))
        assert BASE_SERIES_COLUMN in df.columns
