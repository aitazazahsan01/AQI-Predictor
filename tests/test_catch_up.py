import datetime as dt

import pandas as pd

from src.features.availability import CATCH_UP_DAYS, select_recent_days


def make_frame(dates):
    return pd.DataFrame(
        [{"city": "islamabad", "date": pd.Timestamp(d, tz="UTC"), "aqi_mean": 100.0} for d in dates]
    )


def days(frame):
    return sorted(pd.to_datetime(frame["date"], utc=True).dt.date.astype(str))


TODAY = dt.date(2026, 9, 7)


class TestSelectRecentDays:
    def test_window_ends_yesterday_because_today_is_still_in_progress(self):
        frame = make_frame(["2026-09-05", "2026-09-06", "2026-09-07"])

        assert days(select_recent_days(frame, days=7, today=TODAY)) == ["2026-09-05", "2026-09-06"]

    def test_window_covers_exactly_the_requested_span(self):
        frame = make_frame([f"2026-09-{d:02d}" for d in range(1, 8)])

        # days=3 ending 2026-09-06 -> 04, 05, 06
        assert days(select_recent_days(frame, days=3, today=TODAY)) == [
            "2026-09-04",
            "2026-09-05",
            "2026-09-06",
        ]

    def test_days_older_than_the_window_are_excluded(self):
        frame = make_frame(["2026-08-01", "2026-09-06"])

        assert days(select_recent_days(frame, days=7, today=TODAY)) == ["2026-09-06"]

    def test_a_gap_inside_the_window_is_simply_absent_not_an_error(self):
        # This is the repair case: 09-03 was never written, the others were.
        frame = make_frame(["2026-09-02", "2026-09-04", "2026-09-06"])

        assert days(select_recent_days(frame, days=7, today=TODAY)) == [
            "2026-09-02",
            "2026-09-04",
            "2026-09-06",
        ]

    def test_returns_empty_when_nothing_falls_in_the_window(self):
        frame = make_frame(["2026-01-01"])

        assert select_recent_days(frame, days=7, today=TODAY).empty

    def test_an_empty_frame_passes_through(self):
        frame = pd.DataFrame(columns=["city", "date", "aqi_mean"])

        assert select_recent_days(frame, days=7, today=TODAY).empty

    def test_default_window_is_a_week(self):
        assert CATCH_UP_DAYS == 7

    def test_default_window_spans_seven_days_back_from_yesterday(self):
        frame = make_frame([f"2026-09-{d:02d}" for d in range(1, 8)])

        selected = days(select_recent_days(frame, today=TODAY))

        # Yesterday is 09-06; seven days back inclusive reaches 08-31, so
        # everything from 09-01 onward except today survives.
        assert selected == ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
                            "2026-09-05", "2026-09-06"]
