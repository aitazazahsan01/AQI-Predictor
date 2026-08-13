"""Module 8: the AQI scale and hazardous-air alerting.

This is the single source of truth for turning an AQI number into a category,
a colour and an alert level. The dashboard, any future notifier, and the report
all read from here, so they can never disagree about where "unhealthy" starts.

Breakpoints follow the US EPA scale, which is what Open-Meteo's `us_aqi` field
reports (https://www.airnow.gov/aqi/aqi-basics/).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Alert levels, ordered by increasing seriousness.
ALERT_NONE = "none"
ALERT_WARNING = "warning"
ALERT_CRITICAL = "critical"

# An AQI at or above this is "Unhealthy" for everyone, not just sensitive groups.
WARNING_THRESHOLD = 151
# At or above this is "Very Unhealthy" - health warnings of emergency conditions.
CRITICAL_THRESHOLD = 201


@dataclass(frozen=True)
class AqiCategory:
    name: str
    lower: int
    upper: int          # inclusive; math.inf for the open-ended top band
    color: str          # hex, matching the official AQI colours
    alert_level: str
    health_advice: str


# Ordered low to high. `upper` is inclusive.
AQI_CATEGORIES: tuple[AqiCategory, ...] = (
    AqiCategory(
        name="Good",
        lower=0,
        upper=50,
        color="#00E400",
        alert_level=ALERT_NONE,
        health_advice="Air quality is satisfactory and poses little or no risk.",
    ),
    AqiCategory(
        name="Moderate",
        lower=51,
        upper=100,
        color="#FFFF00",
        alert_level=ALERT_NONE,
        health_advice="Acceptable, though unusually sensitive people should consider limiting prolonged outdoor exertion.",
    ),
    AqiCategory(
        name="Unhealthy for Sensitive Groups",
        lower=101,
        upper=150,
        color="#FF7E00",
        alert_level=ALERT_NONE,
        health_advice="People with heart or lung disease, older adults and children should limit prolonged outdoor exertion.",
    ),
    AqiCategory(
        name="Unhealthy",
        lower=151,
        upper=200,
        color="#FF0000",
        alert_level=ALERT_WARNING,
        health_advice="Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
    ),
    AqiCategory(
        name="Very Unhealthy",
        lower=201,
        upper=300,
        color="#8F3F97",
        alert_level=ALERT_CRITICAL,
        health_advice="Health alert: everyone may experience more serious health effects. Avoid outdoor exertion.",
    ),
    AqiCategory(
        name="Hazardous",
        lower=301,
        upper=math.inf,
        color="#7E0023",
        alert_level=ALERT_CRITICAL,
        health_advice="Health warning of emergency conditions. Everyone should avoid all outdoor exertion.",
    ),
)


def categorize(aqi: float | None) -> AqiCategory | None:
    """Maps an AQI value to its category. Returns None for missing values.

    Values are rounded before banding so that e.g. 150.4 reads as "Unhealthy for
    Sensitive Groups" rather than being pushed into the next band. Negative
    values (never physically valid, but possible from a model extrapolating)
    clamp to the lowest band rather than returning None - a forecast of -5
    still means "good air", not "unknown".
    """
    if aqi is None:
        return None
    try:
        value = float(aqi)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None

    rounded = round(value)
    if rounded < 0:
        return AQI_CATEGORIES[0]

    for category in AQI_CATEGORIES:
        if rounded <= category.upper:
            return category
    return AQI_CATEGORIES[-1]


def alert_level(aqi: float | None) -> str:
    """Convenience wrapper: just the alert level, `none` when unknown."""
    category = categorize(aqi)
    return category.alert_level if category else ALERT_NONE


def is_hazardous(aqi: float | None) -> bool:
    """True when the value warrants a visible warning to the user."""
    return alert_level(aqi) in (ALERT_WARNING, ALERT_CRITICAL)


def worst_alert(values) -> str:
    """The most serious alert level across a set of forecasts.

    A three-day forecast should raise a warning if *any* day is dangerous, not
    just the first one - the whole point of forecasting is advance notice.
    """
    levels = [alert_level(v) for v in values]
    if ALERT_CRITICAL in levels:
        return ALERT_CRITICAL
    if ALERT_WARNING in levels:
        return ALERT_WARNING
    return ALERT_NONE


def summarize_alert(forecasts: dict[str, float | None]) -> dict | None:
    """Builds a user-facing alert from a {label: aqi} mapping, or None if the
    air is fine on every day.

    Returns the worst level, which days trigger it, and the advice for the worst
    value - so the dashboard can render "Unhealthy air expected Tue, Wed".
    """
    flagged = {label: value for label, value in forecasts.items() if is_hazardous(value)}
    if not flagged:
        return None

    worst_label = max(flagged, key=lambda label: flagged[label])
    worst_value = flagged[worst_label]
    category = categorize(worst_value)

    return {
        "level": worst_alert(forecasts.values()),
        "days": list(flagged.keys()),
        "worst_day": worst_label,
        "worst_aqi": worst_value,
        "category": category.name,
        "color": category.color,
        "advice": category.health_advice,
    }
