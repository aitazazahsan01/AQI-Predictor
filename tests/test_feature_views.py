import pandas as pd

from src.hopsworks_utils.feature_views import (
    DAILY_FEATURES_VIEW_DESCRIPTION,
    DAILY_FEATURES_VIEW_NAME,
    DAILY_FEATURES_VIEW_VERSION,
    read_daily_features,
)

# Hopsworks rejects entity descriptions longer than this - learned the hard way
# when creating the daily feature group.
MAX_DESCRIPTION_CHARS = 256


def make_frame(rows=3):
    return pd.DataFrame(
        [{"city": "islamabad", "date": f"2026-08-0{i + 1}", "aqi_mean": 100.0 + i} for i in range(rows)]
    )


class FakeFeatureGroup:
    def __init__(self, frame, selected="ALL"):
        self._frame = frame
        self._selected = selected
        self.read_called = False

    def select_all(self):
        return self._selected

    def read(self):
        self.read_called = True
        return self._frame


class FakeFeatureView:
    def __init__(self, frame):
        self._frame = frame
        self.batch_called = False

    def get_batch_data(self):
        self.batch_called = True
        return self._frame


class FakeFeatureStore:
    """Stands in for a Hopsworks feature store, recording how it was called."""

    def __init__(self, view_frame=None, group_frame=None, view_error=None, batch_error=None):
        self.group = FakeFeatureGroup(group_frame if group_frame is not None else make_frame(1))
        self._view_frame = view_frame if view_frame is not None else make_frame(3)
        self._view_error = view_error
        self._batch_error = batch_error
        self.view_kwargs = None

    def get_or_create_feature_group(self, **kwargs):
        return self.group

    def get_or_create_feature_view(self, **kwargs):
        self.view_kwargs = kwargs
        if self._view_error is not None:
            raise self._view_error
        if self._batch_error is not None:
            class Failing:
                def get_batch_data(inner_self):
                    raise self._batch_error

            return Failing()
        return FakeFeatureView(self._view_frame)


class TestReadDailyFeatures:
    def test_reads_through_the_view_when_it_works(self):
        store = FakeFeatureStore(view_frame=make_frame(3))

        frame = read_daily_features(store)

        assert len(frame) == 3
        assert store.group.read_called is False

    def test_view_is_built_from_the_daily_feature_group(self):
        store = FakeFeatureStore()

        read_daily_features(store)

        assert store.view_kwargs["name"] == DAILY_FEATURES_VIEW_NAME
        assert store.view_kwargs["version"] == DAILY_FEATURES_VIEW_VERSION
        assert store.view_kwargs["query"] == "ALL"

    def test_no_labels_are_declared_because_targets_are_derived_at_train_time(self):
        store = FakeFeatureStore()

        read_daily_features(store)

        assert store.view_kwargs["labels"] == []

    def test_description_fits_the_hopsworks_cap(self):
        assert len(DAILY_FEATURES_VIEW_DESCRIPTION) <= MAX_DESCRIPTION_CHARS

    def test_falls_back_to_the_feature_group_when_the_view_cannot_be_created(self):
        store = FakeFeatureStore(
            group_frame=make_frame(7), view_error=RuntimeError("unsupported client")
        )

        frame = read_daily_features(store)

        assert len(frame) == 7
        assert store.group.read_called is True

    def test_falls_back_when_the_view_exists_but_the_batch_read_fails(self):
        store = FakeFeatureStore(group_frame=make_frame(5), batch_error=ValueError("no offline data"))

        frame = read_daily_features(store)

        assert len(frame) == 5
        assert store.group.read_called is True

    def test_fallback_says_which_path_served_the_data(self, capsys):
        store = FakeFeatureStore(view_error=RuntimeError("boom"))

        read_daily_features(store)

        printed = capsys.readouterr().out
        assert "Feature view unavailable" in printed
        assert "reading the feature group directly" in printed
