import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    add_derived_features,
    add_time_features,
    aggregate_hourly_to_daily,
    compute_daily_features,
)


def make_hourly(city="islamabad", start="2026-07-01", days=1, hours_per_day=24, aqi_by_day=None):
    """Builds a synthetic hourly frame shaped like the aqi_hourly_raw feature group."""
    rows = []
    start_ts = pd.Timestamp(start, tz="UTC")
    for day in range(days):
        aqi = aqi_by_day[day] if aqi_by_day else 100.0
        for hour in range(hours_per_day):
            rows.append(
                {
                    "city": city,
                    "ts": start_ts + pd.Timedelta(days=day, hours=hour),
                    "aqi": aqi,
                    "pm2_5": 20.0,
                    "pm10": 30.0,
                    "carbon_monoxide": 300.0,
                    "nitrogen_dioxide": 10.0,
                    "sulphur_dioxide": 5.0,
                    "ozone": 60.0,
                    "temperature_2m": 30.0,
                    "relative_humidity_2m": 50.0,
                    "wind_speed_10m": 10.0,
                    "wind_direction_10m": 180.0,
                    "surface_pressure": 950.0,
                    "precipitation": 1.0,
                    "aqicn_live_aqi": np.nan,
                }
            )
    return pd.DataFrame(rows)


class TestAggregation:
    def test_collapses_24_hours_into_one_day(self):
        daily = aggregate_hourly_to_daily(make_hourly(days=1))
        assert len(daily) == 1
        assert daily.iloc[0]["hours_observed"] == 24

    def test_aqi_min_mean_max_computed_over_the_day(self):
        hourly = make_hourly(days=1, hours_per_day=3)
        hourly["aqi"] = [50.0, 100.0, 150.0]
        daily = aggregate_hourly_to_daily(hourly)
        row = daily.iloc[0]
        assert row["aqi_min"] == 50.0
        assert row["aqi_mean"] == 100.0
        assert row["aqi_max"] == 150.0

    def test_precipitation_is_summed_not_averaged(self):
        # Rain is a daily total, unlike temperature which is an average.
        daily = aggregate_hourly_to_daily(make_hourly(days=1, hours_per_day=5))
        assert daily.iloc[0]["precipitation_sum"] == 5.0

    def test_duplicate_hours_are_deduplicated(self):
        # The hourly pipeline can insert the same hour twice (re-run / CI retry).
        hourly = make_hourly(days=1, hours_per_day=2)
        doubled = pd.concat([hourly, hourly], ignore_index=True)
        daily = aggregate_hourly_to_daily(doubled)
        assert daily.iloc[0]["hours_observed"] == 2

    def test_partial_day_is_kept_but_flagged_by_hours_observed(self):
        daily = aggregate_hourly_to_daily(make_hourly(days=1, hours_per_day=4))
        assert len(daily) == 1
        assert daily.iloc[0]["hours_observed"] == 4

    def test_separate_cities_do_not_merge(self):
        a = make_hourly(city="islamabad", days=1, hours_per_day=2)
        b = make_hourly(city="lahore", days=1, hours_per_day=2)
        daily = aggregate_hourly_to_daily(pd.concat([a, b], ignore_index=True))
        assert len(daily) == 2
        assert set(daily["city"]) == {"islamabad", "lahore"}

    def test_empty_input_returns_empty_frame_not_error(self):
        assert aggregate_hourly_to_daily(pd.DataFrame()).empty


class TestTimeFeatures:
    def test_calendar_fields_match_the_date(self):
        # 2026-07-01 is a Wednesday.
        daily = add_time_features(pd.DataFrame({"date": [pd.Timestamp("2026-07-01", tz="UTC")]}))
        row = daily.iloc[0]
        assert row["day_of_week"] == 2
        assert row["day_of_month"] == 1
        assert row["month"] == 7
        assert row["is_weekend"] == 0

    @pytest.mark.parametrize(
        "date,expected",
        [("2026-07-03", 0), ("2026-07-04", 1), ("2026-07-05", 1), ("2026-07-06", 0)],
    )
    def test_weekend_flag(self, date, expected):
        # Fri=0, Sat=1, Sun=1, Mon=0
        daily = add_time_features(pd.DataFrame({"date": [pd.Timestamp(date, tz="UTC")]}))
        assert daily.iloc[0]["is_weekend"] == expected


class TestDerivedFeatures:
    def _daily(self, aqi_values):
        return pd.DataFrame(
            {
                "city": ["islamabad"] * len(aqi_values),
                "date": pd.date_range("2026-07-01", periods=len(aqi_values), tz="UTC"),
                "aqi_mean": [float(v) for v in aqi_values],
            }
        )

    def test_lags_pull_previous_days_values(self):
        out = add_derived_features(self._daily([10, 20, 30, 40, 50, 60, 70, 80]))
        last = out.iloc[7]
        assert last["aqi_lag1"] == 70.0
        assert last["aqi_lag2"] == 60.0
        assert last["aqi_lag3"] == 50.0
        assert last["aqi_lag7"] == 10.0

    def test_earliest_row_has_no_lag(self):
        out = add_derived_features(self._daily([10, 20]))
        assert pd.isna(out.iloc[0]["aqi_lag1"])

    def test_change_rate_is_relative_not_absolute(self):
        out = add_derived_features(self._daily([100, 150]))
        assert out.iloc[1]["aqi_change_rate"] == pytest.approx(0.5)

    def test_change_rate_handles_decline(self):
        out = add_derived_features(self._daily([100, 75]))
        assert out.iloc[1]["aqi_change_rate"] == pytest.approx(-0.25)

    def test_change_rate_survives_zero_previous_day(self):
        # AQI of exactly 0 is possible; this must not raise or produce inf.
        out = add_derived_features(self._daily([0, 50]))
        assert pd.isna(out.iloc[1]["aqi_change_rate"])

    def test_rolling_mean_uses_current_and_previous_days(self):
        out = add_derived_features(self._daily([10, 20, 30, 40]))
        assert out.iloc[3]["aqi_roll3_mean"] == pytest.approx(30.0)  # (20+30+40)/3

    def test_rolling_std_needs_two_points(self):
        out = add_derived_features(self._daily([10, 20]))
        assert pd.isna(out.iloc[0]["aqi_roll3_std"])
        assert not pd.isna(out.iloc[1]["aqi_roll3_std"])

    def test_lags_do_not_bleed_across_cities(self):
        isb = self._daily([10, 20, 30])
        lhr = self._daily([500, 600, 700])
        lhr["city"] = "lahore"
        out = add_derived_features(pd.concat([isb, lhr], ignore_index=True))
        first_lahore = out[out["city"] == "lahore"].iloc[0]
        assert pd.isna(first_lahore["aqi_lag1"])

    def test_unsorted_input_is_sorted_before_lagging(self):
        daily = self._daily([10, 20, 30]).iloc[::-1].reset_index(drop=True)
        out = add_derived_features(daily)
        assert out["aqi_mean"].tolist() == [10.0, 20.0, 30.0]
        assert out.iloc[2]["aqi_lag1"] == 20.0


class TestFullPipeline:
    def test_end_to_end_produces_expected_columns_and_trend_features(self):
        hourly = make_hourly(days=8, aqi_by_day=[10, 20, 30, 40, 50, 60, 70, 80])
        daily = compute_daily_features(hourly)

        assert len(daily) == 8
        for column in ["aqi_mean", "day_of_week", "aqi_lag7", "aqi_change_rate", "aqi_roll7_mean"]:
            assert column in daily.columns

        last = daily.iloc[7]
        assert last["aqi_mean"] == 80.0
        assert last["aqi_lag7"] == 10.0
        assert last["aqi_change_rate"] == pytest.approx(80 / 70 - 1)

    def test_single_day_of_data_yields_row_with_null_lags(self):
        # This is the real state right after M1 first runs, before backfill.
        daily = compute_daily_features(make_hourly(days=1))
        assert len(daily) == 1
        assert pd.isna(daily.iloc[0]["aqi_lag1"])
        assert not pd.isna(daily.iloc[0]["aqi_mean"])
