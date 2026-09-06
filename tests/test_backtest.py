import pytest

from src.evaluation.backtest import (
    Matched,
    Prediction,
    build_ledger,
    coverage_gaps,
    extract_observations,
    extract_predictions,
    match,
    score_by_horizon,
    unresolved,
)


def make_snapshot(latest_date="2026-09-05", latest_aqi=150.0, forecasts=None, history=None):
    if forecasts is None:
        forecasts = [(1, "2026-09-06", 140.0), (2, "2026-09-07", 130.0), (3, "2026-09-08", 120.0)]
    return {
        "latest": {"date": latest_date, "aqi": latest_aqi},
        "forecast": [
            {
                "horizon": h,
                "date": d,
                "aqi": v,
                "category": "Unhealthy for Sensitive Groups",
                "color": "#FF7E00",
                "model_type": "ridge",
            }
            for h, d, v in forecasts
        ],
        "history": history if history is not None else [],
    }


def make_prediction(horizon=1, target="2026-09-06", predicted=140.0, baseline=150.0):
    return Prediction(
        issued_for="2026-09-05",
        horizon=horizon,
        target_date=target,
        predicted=predicted,
        model_type="ridge",
        baseline=baseline,
    )


class TestExtractPredictions:
    def test_reads_every_horizon(self):
        assert len(extract_predictions(make_snapshot())) == 3

    def test_carries_the_latest_observation_as_the_persistence_baseline(self):
        predictions = extract_predictions(make_snapshot(latest_aqi=150.0))

        assert all(p.baseline == 150.0 for p in predictions)

    def test_skips_entries_with_no_value(self):
        snapshot = make_snapshot()
        snapshot["forecast"][1]["aqi"] = None

        assert len(extract_predictions(snapshot)) == 2

    def test_a_snapshot_with_no_forecast_yields_nothing(self):
        assert extract_predictions({"latest": {"date": "2026-09-05", "aqi": 100.0}}) == []


class TestExtractObservations:
    def test_merges_history_and_the_latest_row(self):
        snapshot = make_snapshot(
            latest_date="2026-09-05",
            latest_aqi=150.0,
            history=[{"date": "2026-09-03", "aqi": 120.0}, {"date": "2026-09-04", "aqi": 130.0}],
        )

        observations = extract_observations(snapshot)

        assert observations == {"2026-09-03": 120.0, "2026-09-04": 130.0, "2026-09-05": 150.0}

    def test_null_readings_are_not_recorded_as_observations(self):
        snapshot = make_snapshot(history=[{"date": "2026-09-03", "aqi": None}])

        assert "2026-09-03" not in extract_observations(snapshot)


class TestBuildLedger:
    def test_a_forecast_is_counted_once_even_if_a_run_repeats(self):
        # Re-running a workflow must not let the same day be scored twice.
        snapshot = make_snapshot()

        predictions, _ = build_ledger([snapshot, snapshot, snapshot])

        assert len(predictions) == 3

    def test_the_earliest_issue_of_a_forecast_wins(self):
        first = make_snapshot(forecasts=[(1, "2026-09-06", 140.0)])
        later = make_snapshot(forecasts=[(1, "2026-09-06", 999.0)])

        predictions, _ = build_ledger([first, later])

        assert predictions[0].predicted == 140.0

    def test_observations_accumulate_across_snapshots(self):
        a = make_snapshot(latest_date="2026-09-04", latest_aqi=100.0, history=[])
        b = make_snapshot(latest_date="2026-09-05", latest_aqi=150.0, history=[])

        _, observations = build_ledger([a, b])

        assert observations == {"2026-09-04": 100.0, "2026-09-05": 150.0}

    def test_a_later_snapshot_wins_when_an_observation_is_revised(self):
        a = make_snapshot(latest_date="2026-09-05", latest_aqi=100.0)
        b = make_snapshot(latest_date="2026-09-05", latest_aqi=155.0)

        _, observations = build_ledger([a, b])

        assert observations["2026-09-05"] == 155.0


class TestMatch:
    def test_pairs_a_forecast_with_the_day_it_predicted(self):
        matched = match([make_prediction()], {"2026-09-06": 135.0})

        assert len(matched) == 1
        assert matched[0].actual == 135.0

    def test_a_day_that_has_not_happened_yet_is_not_scored(self):
        predictions = [make_prediction(target="2026-12-01")]

        assert match(predictions, {"2026-09-06": 135.0}) == []
        assert len(unresolved(predictions, {"2026-09-06": 135.0})) == 1


class TestMatchedArithmetic:
    def test_error_is_positive_when_the_forecast_was_too_high(self):
        m = Matched(prediction=make_prediction(predicted=150.0), actual=120.0)

        assert m.error == pytest.approx(30.0)

    def test_baseline_error_uses_the_value_known_at_issue_time(self):
        m = Matched(prediction=make_prediction(baseline=100.0), actual=120.0)

        assert m.baseline_error == pytest.approx(-20.0)

    def test_baseline_error_is_absent_when_no_baseline_was_recorded(self):
        m = Matched(prediction=make_prediction(baseline=None), actual=120.0)

        assert m.baseline_error is None

    def test_category_hit_compares_bands_not_values(self):
        # Both land in "Unhealthy for Sensitive Groups" (101-150).
        assert Matched(prediction=make_prediction(predicted=110.0), actual=145.0).category_hit

    def test_category_miss_when_the_band_differs(self):
        assert not Matched(prediction=make_prediction(predicted=95.0), actual=160.0).category_hit


class TestScoreByHorizon:
    def build(self, rows):
        """rows: (horizon, predicted, actual, baseline)"""
        return [
            Matched(
                prediction=make_prediction(
                    horizon=h, target=f"2026-09-{10 + i:02d}", predicted=p, baseline=b
                ),
                actual=a,
            )
            for i, (h, p, a, b) in enumerate(rows)
        ]

    def test_one_score_per_horizon(self):
        scores = self.build([(1, 100.0, 100.0, 100.0), (2, 100.0, 100.0, 100.0)])

        assert [s.horizon for s in score_by_horizon(scores)] == [1, 2]

    def test_a_perfect_forecast_scores_zero_error(self):
        score = score_by_horizon(self.build([(1, 120.0, 120.0, 90.0)]))[0]

        assert score.rmse == pytest.approx(0.0)
        assert score.mae == pytest.approx(0.0)
        assert score.bias == pytest.approx(0.0)

    def test_bias_is_signed_so_a_consistent_overshoot_is_visible(self):
        score = score_by_horizon(self.build([(1, 110.0, 100.0, 100.0), (1, 120.0, 100.0, 100.0)]))[0]

        assert score.bias == pytest.approx(15.0)

    def test_mae_ignores_sign_where_bias_does_not(self):
        score = score_by_horizon(self.build([(1, 110.0, 100.0, 100.0), (1, 90.0, 100.0, 100.0)]))[0]

        assert score.bias == pytest.approx(0.0)
        assert score.mae == pytest.approx(10.0)

    def test_skill_is_positive_when_the_model_beats_persistence(self):
        # Model is 5 off, persistence is 20 off.
        score = score_by_horizon(self.build([(1, 105.0, 100.0, 120.0)]))[0]

        assert score.skill > 0

    def test_skill_is_negative_when_persistence_wins(self):
        score = score_by_horizon(self.build([(1, 130.0, 100.0, 105.0)]))[0]

        assert score.skill < 0

    def test_beat_baseline_rate_counts_days_not_magnitude(self):
        rows = self.build(
            [
                (1, 101.0, 100.0, 150.0),   # model wins
                (1, 102.0, 100.0, 150.0),   # model wins
                (1, 190.0, 100.0, 101.0),   # persistence wins, by a lot
            ]
        )

        assert score_by_horizon(rows)[0].beat_baseline_rate == pytest.approx(200 / 3)

    def test_skill_is_absent_when_no_baseline_was_recorded(self):
        score = score_by_horizon(self.build([(1, 110.0, 100.0, None)]))[0]

        assert score.baseline_rmse is None
        assert score.skill is None

    def test_category_accuracy_is_a_percentage(self):
        rows = self.build([(1, 110.0, 145.0, 100.0), (1, 40.0, 160.0, 100.0)])

        assert score_by_horizon(rows)[0].category_accuracy == pytest.approx(50.0)


class TestCoverageGaps:
    def test_a_contiguous_run_has_no_gaps(self):
        observations = {"2026-09-01": 1.0, "2026-09-02": 2.0, "2026-09-03": 3.0}

        assert coverage_gaps(observations) == []

    def test_a_missing_middle_day_is_reported(self):
        observations = {"2026-09-01": 1.0, "2026-09-03": 3.0}

        assert coverage_gaps(observations) == ["2026-09-02"]

    def test_several_gaps_come_back_in_order(self):
        observations = {"2026-09-01": 1.0, "2026-09-03": 3.0, "2026-09-06": 6.0}

        assert coverage_gaps(observations) == ["2026-09-02", "2026-09-04", "2026-09-05"]

    def test_gaps_outside_the_observed_range_are_not_invented(self):
        # Nothing before the first or after the last observation counts.
        observations = {"2026-09-05": 5.0}

        assert coverage_gaps(observations) == []

    def test_no_observations_yields_no_gaps(self):
        assert coverage_gaps({}) == []
