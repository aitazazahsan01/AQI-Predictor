"""Deciding whether stored features are good enough to use.

The feature store can legitimately hold a small amount of stale data - a couple
of rows from an early pipeline test, or a gap after the scheduled jobs have been
down. Reading it blindly then shows a month-old AQI as though it were current,
which is worse than having no feature store at all.

Consumers use this to decide between stored features and refetching from the
source API.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

# Enough history to populate lag7/roll7 and draw a meaningful trend line.
MIN_USABLE_ROWS = 14

# Beyond this, the newest stored row is too old to present as "latest".
MAX_STALENESS_DAYS = 3

# How far back the daily aggregation re-writes on each run. A scheduled job that
# only ever writes yesterday loses a day permanently every time its cron is
# skipped, and GitHub Actions skips scheduled runs regularly. Rewriting a
# trailing window instead means a missed run heals itself on the next one.
CATCH_UP_DAYS = 7


def describe_usability(
    frame: pd.DataFrame,
    min_rows: int = MIN_USABLE_ROWS,
    max_staleness_days: int = MAX_STALENESS_DAYS,
    today: dt.date | None = None,
) -> tuple[bool, str]:
    """Returns (usable, human-readable reason).

    The reason is surfaced to the user rather than swallowed, so a fallback is
    visible instead of looking like the data simply appeared from nowhere.
    """
    if frame is None or frame.empty:
        return False, "the feature store is empty"

    if len(frame) < min_rows:
        return False, f"only {len(frame)} row(s) stored, need at least {min_rows}"

    latest = pd.to_datetime(frame["date"]).max()
    latest_date = latest.date() if hasattr(latest, "date") else latest
    reference = today or dt.date.today()
    staleness = (reference - latest_date).days

    if staleness > max_staleness_days:
        return False, f"newest stored row is {staleness} days old ({latest_date})"

    return True, f"{len(frame):,} rows, current to {latest_date}"


def is_usable(frame: pd.DataFrame, **kwargs) -> bool:
    return describe_usability(frame, **kwargs)[0]


def select_recent_days(
    frame: pd.DataFrame,
    days: int = CATCH_UP_DAYS,
    today: dt.date | None = None,
) -> pd.DataFrame:
    """Rows from the last `days` complete days, newest window first.

    Used by the daily aggregation to rewrite a trailing window rather than a
    single day. Re-inserting a day that already exists is an upsert on
    (city, date), so overlap is harmless - and it is what lets a skipped
    scheduled run be repaired by the next one instead of leaving a permanent
    hole in the training data.

    The window ends yesterday: today is still in progress, and Open-Meteo fills
    the remaining hours of the current day with forecast values.
    """
    if frame is None or frame.empty:
        return frame

    reference = today or dt.date.today()
    end = reference - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days - 1)

    dates = pd.to_datetime(frame["date"], utc=True).dt.date
    return frame[(dates >= start) & (dates <= end)]
