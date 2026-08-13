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
