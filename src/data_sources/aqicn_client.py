"""AQICN client — used only for the dashboard's "live official station reading"
display widget, never for training data (see PROJECT_PLAN.md section 0 for why).

Requires a free token from https://aqicn.org/data-platform/token/.
"""

import time

import requests

FEED_URL_TEMPLATE = "https://api.waqi.info/feed/@{uid}/?token={token}"
REQUEST_TIMEOUT_SECONDS = 15

# The Islamabad US Embassy station has been observed going stale for months at a
# time (see PROJECT_PLAN.md). Treat anything older than this as "unavailable"
# rather than showing a misleading old reading.
STALE_THRESHOLD_SECONDS = 6 * 3600


def fetch_live_reading(station_uid: int, token: str) -> dict | None:
    """Returns the station's live reading, or None if unavailable/stale."""
    url = FEED_URL_TEMPLATE.format(uid=station_uid, token=token)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "ok":
        return None

    data = payload["data"]
    reported_at = data["time"]["v"]  # unix epoch seconds
    age_seconds = time.time() - reported_at
    if age_seconds > STALE_THRESHOLD_SECONDS:
        return None

    return {
        "aqi": data.get("aqi"),
        "dominant_pollutant": data.get("dominentpol"),
        "reported_at": reported_at,
        "station_name": data.get("city", {}).get("name"),
    }
