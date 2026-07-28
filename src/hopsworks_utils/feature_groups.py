"""Feature group definitions. Schema is inferred from the first `insert()` call's
DataFrame dtypes (standard Hopsworks pattern), so there's no separate schema
declaration to keep in sync — see PROJECT_PLAN.md section 3 for the reference schema.
"""

HOURLY_RAW_FG_NAME = "aqi_hourly_raw"
HOURLY_RAW_FG_VERSION = 1

DAILY_FEATURES_FG_NAME = "aqi_daily_features"
DAILY_FEATURES_FG_VERSION = 1


def get_or_create_hourly_raw_fg(feature_store):
    return feature_store.get_or_create_feature_group(
        name=HOURLY_RAW_FG_NAME,
        version=HOURLY_RAW_FG_VERSION,
        description=(
            "Hourly raw weather + air-quality observations per city, from Open-Meteo. "
            "aqicn_live_aqi is a display-only field from AQICN and is not used for training."
        ),
        primary_key=["city", "ts"],
        event_time="ts",
        online_enabled=True,
    )


def get_or_create_daily_features_fg(feature_store):
    return feature_store.get_or_create_feature_group(
        name=DAILY_FEATURES_FG_NAME,
        version=DAILY_FEATURES_FG_VERSION,
        # Hopsworks caps entity descriptions at 256 characters.
        description=(
            "One engineered row per city per day: pollutant/weather aggregates, calendar features, "
            "and trend features (lags, rolling stats, change rate). Read by the training pipeline. "
            "Targets are derived at train time by shifting aqi_mean, not stored here."
        ),
        primary_key=["city", "date"],
        event_time="date",
        online_enabled=True,
    )
