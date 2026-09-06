"""Feature View over the daily feature group.

A Feature View is the read-side contract for training and inference: it pins
which columns a model sees and in what order, independently of how the
underlying feature group happens to evolve. Reading the feature group directly
works, but it means every consumer re-decides what "the training columns" are,
which is exactly the drift a feature store exists to prevent.

The label columns are deliberately empty. Targets are derived at train time by
shifting `aqi_mean` (see `training/data_prep.build_targets`) rather than stored,
so the newest row stays valid for live prediction even though its "tomorrow"
does not exist yet.
"""

from src.hopsworks_utils.feature_groups import get_or_create_daily_features_fg

DAILY_FEATURES_VIEW_NAME = "aqi_daily_features_view"
DAILY_FEATURES_VIEW_VERSION = 1

# Hopsworks caps entity descriptions at 256 characters.
DAILY_FEATURES_VIEW_DESCRIPTION = (
    "Read-side view over aqi_daily_features for training and inference. "
    "Selects every engineered column; forecast targets are derived at train "
    "time by shifting aqi_mean rather than stored, so the newest row stays "
    "usable for live prediction."
)


def get_or_create_daily_features_view(feature_store):
    """The Feature View the training pipeline reads through."""
    feature_group = get_or_create_daily_features_fg(feature_store)

    return feature_store.get_or_create_feature_view(
        name=DAILY_FEATURES_VIEW_NAME,
        version=DAILY_FEATURES_VIEW_VERSION,
        description=DAILY_FEATURES_VIEW_DESCRIPTION[:256],
        query=feature_group.select_all(),
        labels=[],
    )


def read_daily_features(feature_store):
    """Every stored daily row, read through the Feature View.

    Falls back to reading the feature group directly if the view cannot be
    created or queried - an older Hopsworks client, or a project where the view
    has not been provisioned yet. The nightly training run matters more than
    the read path it takes, so the fallback is loud rather than fatal: it prints
    which path served the data.
    """
    try:
        view = get_or_create_daily_features_view(feature_store)
        frame = view.get_batch_data()
        print(f"Read {len(frame)} rows via feature view {DAILY_FEATURES_VIEW_NAME}.")
        return frame
    except Exception as exc:
        print(
            f"Feature view unavailable ({type(exc).__name__}: {exc}); "
            "reading the feature group directly."
        )
        return get_or_create_daily_features_fg(feature_store).read()
