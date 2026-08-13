import datetime as dt

import pandas as pd
import pytest

from src.features.availability import describe_usability, is_usable

TODAY = dt.date(2026, 8, 14)


def frame(n_rows, latest="2026-08-13"):
    end = pd.Timestamp(latest, tz="UTC")
    return pd.DataFrame(
        {
            "date": pd.date_range(end=end, periods=n_rows, freq="D"),
            "aqi_mean": [100.0] * n_rows,
        }
    )


class TestUsability:
    def test_fresh_and_long_enough_is_usable(self):
        usable, reason = describe_usability(frame(30), today=TODAY)
        assert usable
        assert "30 rows" in reason

    def test_empty_frame_is_rejected(self):
        usable, reason = describe_usability(pd.DataFrame(), today=TODAY)
        assert not usable
        assert "empty" in reason

    def test_too_few_rows_is_rejected(self):
        # The single leftover row from an early pipeline test must not be
        # presented as though it were a live dataset.
        usable, reason = describe_usability(frame(1), today=TODAY)
        assert not usable
        assert "only 1 row" in reason

    def test_stale_data_is_rejected_even_when_plentiful(self):
        usable, reason = describe_usability(frame(500, latest="2026-06-01"), today=TODAY)
        assert not usable
        assert "days old" in reason

    def test_yesterdays_data_is_still_fresh(self):
        # The pipelines aggregate yesterday, so yesterday is the normal case.
        assert is_usable(frame(30, latest="2026-08-13"), today=TODAY)

    @pytest.mark.parametrize("days_old,expected", [(0, True), (3, True), (4, False)])
    def test_staleness_boundary(self, days_old, expected):
        latest = (TODAY - dt.timedelta(days=days_old)).isoformat()
        assert is_usable(frame(30, latest=latest), today=TODAY) is expected

    def test_thresholds_are_configurable(self):
        # The EDA holds stored data to a higher bar than the dashboard.
        assert not is_usable(frame(30), min_rows=60, today=TODAY)
