"""Tests for ML feature store with point-in-time correctness."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ml_feature_store.store import (
    FeatureStore,
    FeatureTableInfo,
    FeatureStoreError,
    FeatureTableNotFoundError,
    FeatureValidationError,
)


def _make_feature_df(
    entity_ids: list,
    timestamps: list,
    feature_values: dict,
) -> pd.DataFrame:
    """Build a feature DataFrame for testing."""
    data = {
        "entity_id": entity_ids,
        "timestamp": pd.to_datetime(timestamps),
    }
    data.update(feature_values)
    return pd.DataFrame(data)


@pytest.fixture
def store() -> FeatureStore:
    """Create a fresh FeatureStore."""
    return FeatureStore(max_versions=5)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create a sample feature table with temporal data."""
    return _make_feature_df(
        entity_ids=["user_1", "user_1", "user_1", "user_2", "user_2"],
        timestamps=[
            "2024-01-01",
            "2024-01-15",
            "2024-02-01",
            "2024-01-01",
            "2024-01-20",
        ],
        feature_values={
            "feature_a": [1.0, 2.0, 3.0, 10.0, 20.0],
            "feature_b": [100, 200, 300, 1000, 2000],
        },
    )


class TestFeatureStoreInit:
    """Tests for FeatureStore initialization."""

    def test_default_init(self) -> None:
        """Default initialization creates empty store."""
        store = FeatureStore()
        assert store.table_names == []
        assert store.max_versions == 10

    def test_invalid_max_versions_raises(self) -> None:
        """max_versions < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_versions"):
            FeatureStore(max_versions=0)


class TestRegisterFeatures:
    """Tests for feature registration."""

    def test_register_new_table(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Registering a new table returns correct metadata."""
        info = store.register("user_features", sample_features)

        assert info.name == "user_features"
        assert info.version == 1
        assert info.row_count == 5
        assert "feature_a" in info.columns
        assert "feature_b" in info.columns

    def test_register_updates_version(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Re-registering a table increments the version."""
        store.register("features", sample_features)
        info = store.register("features", sample_features)

        assert info.version == 2

    def test_register_missing_columns_raises(
        self, store: FeatureStore
    ) -> None:
        """DataFrame without required columns raises validation error."""
        bad_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        with pytest.raises(FeatureValidationError, match="missing required"):
            store.register("bad", bad_df)

    def test_register_empty_df_raises(self, store: FeatureStore) -> None:
        """Empty DataFrame raises validation error."""
        empty_df = pd.DataFrame(
            columns=["entity_id", "timestamp", "feature"]
        )

        with pytest.raises(FeatureValidationError, match="empty"):
            store.register("empty", empty_df)

    def test_register_non_dataframe_raises(
        self, store: FeatureStore
    ) -> None:
        """Non-DataFrame input raises validation error."""
        with pytest.raises(FeatureValidationError, match="expected DataFrame"):
            store.register("bad", {"entity_id": [1], "timestamp": [2]})  # type: ignore


class TestPointInTimeLookup:
    """Tests for temporal feature lookups."""

    def test_get_latest_at_timestamp(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Lookup returns the latest feature at or before the timestamp."""
        store.register("features", sample_features)

        result = store.get(
            "features",
            entity_ids=["user_1"],
            timestamp=datetime(2024, 1, 20),
        )

        assert len(result) == 1
        # Should get the Jan 15 row (latest before Jan 20)
        assert result["feature_a"].iloc[0] == 2.0

    def test_no_future_data_leakage(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Features from after the query timestamp are excluded."""
        store.register("features", sample_features)

        result = store.get(
            "features",
            entity_ids=["user_1"],
            timestamp=datetime(2024, 1, 10),
        )

        assert len(result) == 1
        # Should get Jan 1 row, NOT Jan 15 or Feb 1
        assert result["feature_a"].iloc[0] == 1.0

    def test_multiple_entities(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Lookup works for multiple entities simultaneously."""
        store.register("features", sample_features)

        result = store.get(
            "features",
            entity_ids=["user_1", "user_2"],
            timestamp=datetime(2024, 1, 25),
        )

        assert len(result) == 2

    def test_entity_not_found_returns_empty(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Missing entities return empty DataFrame."""
        store.register("features", sample_features)

        result = store.get(
            "features",
            entity_ids=["nonexistent"],
            timestamp=datetime(2024, 6, 1),
        )

        assert len(result) == 0

    def test_get_nonexistent_table_raises(
        self, store: FeatureStore
    ) -> None:
        """Querying an unregistered table raises error."""
        with pytest.raises(FeatureTableNotFoundError, match="not registered"):
            store.get(
                "nonexistent",
                entity_ids=["user_1"],
                timestamp=datetime(2024, 1, 1),
            )

    def test_get_with_column_subset(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Specifying columns limits the returned features."""
        store.register("features", sample_features)

        result = store.get(
            "features",
            entity_ids=["user_1"],
            timestamp=datetime(2024, 6, 1),
            columns=["feature_a"],
        )

        assert "feature_a" in result.columns
        assert "feature_b" not in result.columns

    def test_get_invalid_columns_raises(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Requesting non-existent columns raises validation error."""
        store.register("features", sample_features)

        with pytest.raises(FeatureValidationError, match="not found"):
            store.get(
                "features",
                entity_ids=["user_1"],
                timestamp=datetime(2024, 6, 1),
                columns=["nonexistent_col"],
            )

    @pytest.mark.parametrize(
        "query_ts,expected_value",
        [
            (datetime(2024, 1, 1), 1.0),
            (datetime(2024, 1, 15), 2.0),
            (datetime(2024, 2, 1), 3.0),
            (datetime(2024, 12, 31), 3.0),
        ],
    )
    def test_point_in_time_parametrized(
        self,
        store: FeatureStore,
        sample_features: pd.DataFrame,
        query_ts: datetime,
        expected_value: float,
    ) -> None:
        """Point-in-time lookup returns correct value at various timestamps."""
        store.register("features", sample_features)

        result = store.get(
            "features",
            entity_ids=["user_1"],
            timestamp=query_ts,
        )

        assert result["feature_a"].iloc[0] == expected_value


class TestTrainingSet:
    """Tests for point-in-time correct training set generation."""

    def test_build_training_set(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Training set joins features with correct temporal semantics."""
        store.register("features", sample_features)

        entity_timestamps = pd.DataFrame({
            "entity_id": ["user_1", "user_2"],
            "timestamp": pd.to_datetime(["2024-01-20", "2024-01-15"]),
        })

        result = store.get_training_set("features", entity_timestamps)

        assert len(result) == 2


class TestVersioning:
    """Tests for feature table versioning."""

    def test_get_specific_version(
        self, store: FeatureStore
    ) -> None:
        """Specific historical versions can be retrieved."""
        df_v1 = _make_feature_df(
            ["user_1"], ["2024-01-01"], {"val": [1.0]}
        )
        df_v2 = _make_feature_df(
            ["user_1"], ["2024-01-01"], {"val": [2.0]}
        )

        store.register("features", df_v1)
        store.register("features", df_v2)

        v1 = store.get_version("features", version=1)
        v2 = store.get_version("features", version=2)

        assert v1["val"].iloc[0] == 1.0
        assert v2["val"].iloc[0] == 2.0

    def test_get_missing_version_raises(
        self, store: FeatureStore
    ) -> None:
        """Requesting a non-existent version raises error."""
        df = _make_feature_df(
            ["user_1"], ["2024-01-01"], {"val": [1.0]}
        )
        store.register("features", df)

        with pytest.raises(FeatureTableNotFoundError, match="Version"):
            store.get_version("features", version=99)


class TestTableManagement:
    """Tests for table listing and deletion."""

    def test_list_tables(self, store: FeatureStore) -> None:
        """list_tables returns info for all registered tables."""
        df = _make_feature_df(
            ["user_1"], ["2024-01-01"], {"val": [1.0]}
        )
        store.register("table_a", df)
        store.register("table_b", df)

        tables = store.list_tables()
        names = [t.name for t in tables]

        assert "table_a" in names
        assert "table_b" in names

    def test_delete_table(self, store: FeatureStore) -> None:
        """Deleting a table removes it from the store."""
        df = _make_feature_df(
            ["user_1"], ["2024-01-01"], {"val": [1.0]}
        )
        store.register("to_delete", df)

        assert store.delete("to_delete") is True
        assert "to_delete" not in store.table_names

    def test_delete_nonexistent_returns_false(
        self, store: FeatureStore
    ) -> None:
        """Deleting a non-existent table returns False."""
        assert store.delete("nonexistent") is False

    def test_get_table_info(
        self, store: FeatureStore, sample_features: pd.DataFrame
    ) -> None:
        """Table info returns correct metadata."""
        store.register("features", sample_features)
        info = store.get_table_info("features")

        assert info.name == "features"
        assert info.row_count == 5
