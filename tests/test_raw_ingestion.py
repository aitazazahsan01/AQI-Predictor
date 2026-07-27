from datetime import datetime, timezone

import pytest

from src.features.raw_ingestion import select_latest_hour_index

TIMESTAMPS = [
    "2026-07-27T21:00",
    "2026-07-27T22:00",
    "2026-07-27T23:00",
    "2026-07-28T00:00",
]


def test_picks_exact_hour_match():
    now = datetime(2026, 7, 27, 22, 30, tzinfo=timezone.utc)
    assert select_latest_hour_index(TIMESTAMPS, now) == 1


def test_falls_back_to_latest_available_hour_when_now_is_ahead():
    now = datetime(2026, 7, 28, 5, 0, tzinfo=timezone.utc)
    assert select_latest_hour_index(TIMESTAMPS, now) == 3


def test_raises_when_no_timestamp_is_available_yet():
    now = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        select_latest_hour_index(TIMESTAMPS, now)
