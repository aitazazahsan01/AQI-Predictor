import datetime as dt

import pandas as pd
import pytest

from src.features.historical import (
    EARLIEST_AIR_QUALITY_DATE,
    HOURLY_SCHEMA_COLUMNS,
    clamp_start_date,
    date_chunks,
    shape_hourly_payloads,
)


class TestDateChunks:
    def test_single_chunk_when_range_fits(self):
        chunks = date_chunks(dt.date(2024, 1, 1), dt.date(2024, 1, 10), chunk_days=365)
        assert chunks == [(dt.date(2024, 1, 1), dt.date(2024, 1, 10))]

    def test_splits_long_range(self):
        chunks = date_chunks(dt.date(2024, 1, 1), dt.date(2024, 1, 10), chunk_days=4)
        assert chunks == [
            (dt.date(2024, 1, 1), dt.date(2024, 1, 4)),
            (dt.date(2024, 1, 5), dt.date(2024, 1, 8)),
            (dt.date(2024, 1, 9), dt.date(2024, 1, 10)),
        ]

    def test_chunks_are_contiguous_and_non_overlapping(self):
        chunks = date_chunks(dt.date(2022, 8, 5), dt.date(2026, 7, 28), chunk_days=365)
        for earlier, later in zip(chunks, chunks[1:]):
            assert later[0] == earlier[1] + dt.timedelta(days=1)

    def test_chunks_cover_the_whole_range_exactly(self):
        start, end = dt.date(2023, 3, 1), dt.date(2024, 9, 15)
        chunks = date_chunks(start, end, chunk_days=100)
        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        total_days = sum((c_end - c_start).days + 1 for c_start, c_end in chunks)
        assert total_days == (end - start).days + 1

    def test_single_day_range(self):
        assert date_chunks(dt.date(2024, 5, 1), dt.date(2024, 5, 1)) == [
            (dt.date(2024, 5, 1), dt.date(2024, 5, 1))
        ]

    def test_inverted_range_returns_nothing(self):
        assert date_chunks(dt.date(2024, 5, 10), dt.date(2024, 5, 1)) == []


class TestClampStartDate:
    def test_too_early_date_is_clamped(self):
        assert clamp_start_date(dt.date(2020, 1, 1)) == EARLIEST_AIR_QUALITY_DATE

    def test_valid_date_is_left_alone(self):
        assert clamp_start_date(dt.date(2024, 6, 1)) == dt.date(2024, 6, 1)

    def test_boundary_date_is_left_alone(self):
        assert clamp_start_date(EARLIEST_AIR_QUALITY_DATE) == EARLIEST_AIR_QUALITY_DATE


def make_payloads(times, aqi_values):
    """Builds fake API payloads shaped like real Open-Meteo responses."""
    n = len(times)
    air_quality = {
        "hourly": {
            "time": times,
            "us_aqi": aqi_values,
            "pm2_5": [20.0] * n,
            "pm10": [30.0] * n,
            "carbon_monoxide": [300.0] * n,
            "nitrogen_dioxide": [10.0] * n,
            "sulphur_dioxide": [5.0] * n,
            "ozone": [60.0] * n,
        }
    }
    weather = {
        "hourly": {
            "time": times,
            "temperature_2m": [30.0] * n,
            "relative_humidity_2m": [50.0] * n,
            "wind_speed_10m": [10.0] * n,
            "wind_direction_10m": [180.0] * n,
            "surface_pressure": [950.0] * n,
            "precipitation": [0.0] * n,
        }
    }
    return air_quality, weather


class TestShapeHourlyPayloads:
    def test_output_matches_the_hourly_feature_group_schema(self):
        aq, wx = make_payloads(["2024-01-01T00:00", "2024-01-01T01:00"], [80.0, 85.0])
        out = shape_hourly_payloads(aq, wx, "islamabad")
        assert list(out.columns) == HOURLY_SCHEMA_COLUMNS

    def test_air_quality_and_weather_are_joined_on_timestamp(self):
        aq, wx = make_payloads(["2024-01-01T00:00", "2024-01-01T01:00"], [80.0, 85.0])
        out = shape_hourly_payloads(aq, wx, "islamabad")
        assert len(out) == 2
        assert out.iloc[0]["aqi"] == 80.0
        assert out.iloc[0]["temperature_2m"] == 30.0

    def test_null_aqi_rows_are_dropped(self):
        # Open-Meteo returns HTTP 200 with all-null rows for out-of-range dates.
        # Keeping them would poison the feature store with null-target rows.
        aq, wx = make_payloads(
            ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"], [None, 85.0, None]
        )
        out = shape_hourly_payloads(aq, wx, "islamabad")
        assert len(out) == 1
        assert out.iloc[0]["aqi"] == 85.0

    def test_entirely_null_response_yields_empty_frame_not_error(self):
        aq, wx = make_payloads(["2022-01-01T00:00", "2022-01-01T01:00"], [None, None])
        out = shape_hourly_payloads(aq, wx, "islamabad")
        assert out.empty
        assert list(out.columns) == HOURLY_SCHEMA_COLUMNS

    def test_city_slug_is_stamped_on_every_row(self):
        aq, wx = make_payloads(["2024-01-01T00:00", "2024-01-01T01:00"], [80.0, 85.0])
        out = shape_hourly_payloads(aq, wx, "lahore")
        assert (out["city"] == "lahore").all()

    def test_aqicn_column_present_but_null_for_historical_data(self):
        # AQICN has no historical API, so backfilled rows can never have it.
        aq, wx = make_payloads(["2024-01-01T00:00"], [80.0])
        out = shape_hourly_payloads(aq, wx, "islamabad")
        assert "aqicn_live_aqi" in out.columns
        assert pd.isna(out.iloc[0]["aqicn_live_aqi"])

    def test_mismatched_hours_between_the_two_apis_are_dropped(self):
        # Inner join: an hour missing from either source can't produce a full row.
        aq, _ = make_payloads(["2024-01-01T00:00", "2024-01-01T01:00"], [80.0, 85.0])
        _, wx = make_payloads(["2024-01-01T01:00"], [85.0])
        out = shape_hourly_payloads(aq, wx, "islamabad")
        assert len(out) == 1
        assert out.iloc[0]["ts"] == "2024-01-01T01:00"


class TestBackfillFeedsFeatureEngineering:
    def test_shaped_output_can_be_consumed_by_compute_daily_features(self):
        """The whole point of M3: its output must be valid input to M2's code."""
        from src.features.feature_engineering import compute_daily_features

        times = [f"2024-01-{day:02d}T{hour:02d}:00" for day in range(1, 4) for hour in range(24)]
        aqi_values = [100.0] * len(times)
        aq, wx = make_payloads(times, aqi_values)

        hourly = shape_hourly_payloads(aq, wx, "islamabad")
        daily = compute_daily_features(hourly)

        assert len(daily) == 3
        assert (daily["hours_observed"] == 24).all()
        assert daily.iloc[1]["aqi_lag1"] == 100.0
        assert pytest.approx(daily.iloc[1]["aqi_change_rate"]) == 0.0
