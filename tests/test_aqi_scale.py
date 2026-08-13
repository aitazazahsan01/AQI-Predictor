import math

import pytest

from src.alerts.aqi_scale import (
    ALERT_CRITICAL,
    ALERT_NONE,
    ALERT_WARNING,
    AQI_CATEGORIES,
    alert_level,
    categorize,
    is_hazardous,
    summarize_alert,
    worst_alert,
)


class TestCategoryBoundaries:
    @pytest.mark.parametrize(
        "aqi,expected",
        [
            (0, "Good"),
            (50, "Good"),
            (51, "Moderate"),
            (100, "Moderate"),
            (101, "Unhealthy for Sensitive Groups"),
            (150, "Unhealthy for Sensitive Groups"),
            (151, "Unhealthy"),
            (200, "Unhealthy"),
            (201, "Very Unhealthy"),
            (300, "Very Unhealthy"),
            (301, "Hazardous"),
            (500, "Hazardous"),
        ],
    )
    def test_epa_breakpoints(self, aqi, expected):
        assert categorize(aqi).name == expected

    def test_no_gaps_between_categories(self):
        # Every integer must land in exactly one band; an off-by-one in the
        # breakpoints would silently mislabel air quality at a boundary.
        for lower, upper in zip(AQI_CATEGORIES, AQI_CATEGORIES[1:]):
            assert lower.upper + 1 == upper.lower

    def test_top_band_is_open_ended(self):
        assert AQI_CATEGORIES[-1].upper == math.inf
        assert categorize(9999).name == "Hazardous"


class TestMissingAndOddValues:
    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "not a number"])
    def test_unusable_values_return_none(self, value):
        assert categorize(value) is None

    def test_missing_value_does_not_raise_an_alert(self):
        # Absent data must never be reported as dangerous air.
        assert alert_level(None) == ALERT_NONE
        assert is_hazardous(None) is False

    def test_negative_forecast_clamps_to_good(self):
        # Regression models can extrapolate below zero; that means clean air,
        # not unknown air.
        assert categorize(-5).name == "Good"

    def test_values_are_rounded_not_truncated(self):
        assert categorize(150.4).name == "Unhealthy for Sensitive Groups"
        assert categorize(150.6).name == "Unhealthy"


class TestAlertLevels:
    def test_moderate_air_raises_nothing(self):
        assert alert_level(75) == ALERT_NONE

    def test_sensitive_group_band_is_not_yet_a_warning(self):
        # 101-150 affects sensitive groups only; warning starts at "Unhealthy".
        assert alert_level(140) == ALERT_NONE

    def test_unhealthy_raises_warning(self):
        assert alert_level(160) == ALERT_WARNING
        assert is_hazardous(160) is True

    def test_very_unhealthy_raises_critical(self):
        assert alert_level(250) == ALERT_CRITICAL

    def test_worst_alert_wins_across_days(self):
        assert worst_alert([40, 90, 210]) == ALERT_CRITICAL
        assert worst_alert([40, 160, 90]) == ALERT_WARNING
        assert worst_alert([40, 90, 120]) == ALERT_NONE

    def test_worst_alert_ignores_missing_days(self):
        assert worst_alert([None, 40, None]) == ALERT_NONE


class TestAlertSummary:
    def test_clean_forecast_produces_no_alert(self):
        assert summarize_alert({"Mon": 40, "Tue": 80, "Wed": 120}) is None

    def test_flags_only_the_dangerous_days(self):
        summary = summarize_alert({"Mon": 40, "Tue": 165, "Wed": 180})
        assert summary["days"] == ["Tue", "Wed"]
        assert summary["level"] == ALERT_WARNING

    def test_reports_the_worst_day_and_its_advice(self):
        summary = summarize_alert({"Mon": 160, "Tue": 260, "Wed": 155})
        assert summary["worst_day"] == "Tue"
        assert summary["worst_aqi"] == 260
        assert summary["category"] == "Very Unhealthy"
        assert summary["level"] == ALERT_CRITICAL
        assert "avoid outdoor exertion" in summary["advice"].lower()

    def test_a_single_bad_day_still_alerts(self):
        # Advance notice is the entire point - one bad day in three must show.
        summary = summarize_alert({"Mon": 40, "Tue": 50, "Wed": 205})
        assert summary is not None
        assert summary["level"] == ALERT_CRITICAL
