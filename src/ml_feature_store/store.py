"""Point-in-time correct feature store using in-memory storage.

Provides feature registration, versioned temporal joins, and train/serve
consistency. All lookups respect event-time semantics: a query at time T
only returns features known at or before T, preventing future data leakage.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum number of feature versions to retain per table
DEFAULT_MAX_VERSIONS = 10

# Required columns for feature tables
ENTITY_ID_COLUMN = "entity_id"
TIMESTAMP_COLUMN = "timestamp"
REQUIRED_COLUMNS = {ENTITY_ID_COLUMN, TIMESTAMP_COLUMN}

# Default timezone for naive timestamps
DEFAULT_TIMEZONE = timezone.utc


@dataclass(frozen=True)
class FeatureTableInfo:
    """Metadata about a registered feature table.

    Attributes:
        name: Unique table identifier.
        columns: List of feature column names (excluding entity_id and timestamp).
        version: Current version number.
        row_count: Number of rows in the current version.
        registered_at: When the table was first registered.
        updated_at: When the table was last updated.
    """

    name: str
    columns: List[str]
    version: int
    row_count: int
    registered_at: float
    updated_at: float


@dataclass
class _VersionedTable:
    """Internal representation of a versioned feature table.

    Attributes:
        name: Table identifier.
        data: DataFrame with entity_id, timestamp, and feature columns.
        version: Current version number.
        registered_at: Registration timestamp.
        updated_at: Last update timestamp.
        history: Previous versions (version -> DataFrame).
    """

    name: str
    data: pd.DataFrame
    version: int
    registered_at: float
    updated_at: float
    history: Dict[int, pd.DataFrame] = field(default_factory=dict)


class FeatureStoreError(Exception):
    """Base exception for feature store operations."""


class FeatureTableNotFoundError(FeatureStoreError):
    """Raised when a requested feature table does not exist."""


class FeatureValidationError(FeatureStoreError):
    """Raised when feature data fails validation."""


class FeatureStore:
    """Point-in-time correct feature store.

    Stores feature tables with entity_id and timestamp columns.
    Supports temporal lookups that prevent data leakage by only
    returning features known at or before the query timestamp.

    Attributes:
        max_versions: Maximum number of versions retained per table.
    """

    def __init__(
        self,
        max_versions: int = DEFAULT_MAX_VERSIONS,
    ) -> None:
        """Initialize the feature store.

        Args:
            max_versions: Maximum number of table versions to retain.

        Raises:
            ValueError: If max_versions < 1.
        """
        if max_versions < 1:
            raise ValueError(
                f"max_versions must be >= 1, got {max_versions}"
            )

        self._tables: Dict[str, _VersionedTable] = {}
        self._max_versions = max_versions

        logger.info(
            "FeatureStore initialized: max_versions=%d", max_versions
        )

    @property
    def max_versions(self) -> int:
        """Return maximum number of retained versions per table."""
        return self._max_versions

    @property
    def table_names(self) -> List[str]:
        """Return list of registered table names."""
        return list(self._tables.keys())

    def register(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
    ) -> FeatureTableInfo:
        """Register a new feature table or update an existing one.

        Args:
            name: Unique table identifier.
            df: DataFrame with entity_id, timestamp, and feature columns.
            description: Optional human-readable description.

        Returns:
            FeatureTableInfo for the registered table.

        Raises:
            FeatureValidationError: If DataFrame is missing required columns
                or contains invalid data.
        """
        self._validate_dataframe(df, name)
        prepared_df = self._prepare_dataframe(df)

        now = time.time()

        if name in self._tables:
            return self._update_existing_table(name, prepared_df, now)

        return self._register_new_table(name, prepared_df, now)

    def _register_new_table(
        self, name: str, df: pd.DataFrame, now: float
    ) -> FeatureTableInfo:
        """Register a brand new feature table.

        Args:
            name: Table identifier.
            df: Validated and prepared DataFrame.
            now: Current timestamp.

        Returns:
            FeatureTableInfo for the new table.
        """
        table = _VersionedTable(
            name=name,
            data=df.copy(),
            version=1,
            registered_at=now,
            updated_at=now,
        )
        self._tables[name] = table

        info = self._build_table_info(table)
        logger.info(
            "Registered feature table '%s': %d rows, %d features, v%d",
            name, len(df), len(info.columns), info.version,
        )
        return info

    def _update_existing_table(
        self, name: str, df: pd.DataFrame, now: float
    ) -> FeatureTableInfo:
        """Update an existing feature table, preserving history.

        Args:
            name: Table identifier.
            df: New version of the data.
            now: Current timestamp.

        Returns:
            Updated FeatureTableInfo.
        """
        table = self._tables[name]

        # Archive current version
        table.history[table.version] = table.data
        table.version += 1
        table.data = df.copy()
        table.updated_at = now

        self._prune_history(table)

        info = self._build_table_info(table)
        logger.info(
            "Updated feature table '%s': %d rows, v%d",
            name, len(df), info.version,
        )
        return info

    def get(
        self,
        name: str,
        entity_ids: List[Any],
        timestamp: datetime,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get features at a point in time (no future data leakage).

        For each entity_id, returns the most recent feature row where
        the feature timestamp is <= the query timestamp.

        Args:
            name: Feature table name.
            entity_ids: List of entity IDs to query.
            timestamp: Point-in-time cutoff (inclusive upper bound).
            columns: Optional subset of columns to return. If None, all.

        Returns:
            DataFrame with one row per entity_id (latest features at timestamp).

        Raises:
            FeatureTableNotFoundError: If table not registered.
            FeatureValidationError: If columns not found in table.
        """
        table = self._get_table(name)
        ts = _normalize_timestamp(timestamp)

        filtered = self._filter_by_time(table.data, ts)
        entity_filtered = self._filter_by_entities(filtered, entity_ids)
        latest = self._get_latest_per_entity(entity_filtered)

        if columns is not None:
            self._validate_columns(columns, table.data, name)
            keep_cols = [ENTITY_ID_COLUMN, TIMESTAMP_COLUMN] + columns
            latest = latest[keep_cols]

        logger.debug(
            "Feature lookup '%s': %d entities, cutoff=%s, returned=%d rows",
            name, len(entity_ids), ts, len(latest),
        )
        return latest.reset_index(drop=True)

    def get_training_set(
        self,
        name: str,
        entity_timestamps: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build a training set with point-in-time correct joins.

        For each (entity_id, timestamp) pair, looks up the latest feature
        row at or before that timestamp. This prevents lookahead bias.

        Args:
            name: Feature table name.
            entity_timestamps: DataFrame with entity_id and timestamp columns.
            columns: Optional subset of feature columns.

        Returns:
            DataFrame with features joined point-in-time correctly.

        Raises:
            FeatureTableNotFoundError: If table not registered.
            FeatureValidationError: If entity_timestamps is malformed.
        """
        self._validate_dataframe(entity_timestamps, "entity_timestamps")
        table = self._get_table(name)

        results: List[pd.DataFrame] = []
        for _, row in entity_timestamps.iterrows():
            entity_id = row[ENTITY_ID_COLUMN]
            ts = _normalize_timestamp(row[TIMESTAMP_COLUMN])

            match = self._point_in_time_lookup(
                table.data, entity_id, ts
            )
            if match is not None:
                results.append(match)

        if not results:
            return self._empty_result(table.data, columns)

        result_df = pd.DataFrame(results)

        if columns is not None:
            self._validate_columns(columns, table.data, name)
            keep_cols = [ENTITY_ID_COLUMN, TIMESTAMP_COLUMN] + columns
            result_df = result_df[keep_cols]

        return result_df.reset_index(drop=True)

    def _point_in_time_lookup(
        self,
        data: pd.DataFrame,
        entity_id: Any,
        timestamp: pd.Timestamp,
    ) -> Optional[Dict[str, Any]]:
        """Find the latest feature row for an entity at or before timestamp.

        Args:
            data: Feature table DataFrame.
            entity_id: Entity to look up.
            timestamp: Cutoff time.

        Returns:
            Dict of column values, or None if no match.
        """
        mask = (data[ENTITY_ID_COLUMN] == entity_id) & (
            data[TIMESTAMP_COLUMN] <= timestamp
        )
        candidates = data.loc[mask]
        if candidates.empty:
            return None

        latest_idx = candidates[TIMESTAMP_COLUMN].idxmax()
        return candidates.loc[latest_idx].to_dict()

    def list_tables(self) -> List[FeatureTableInfo]:
        """List all registered feature tables with metadata.

        Returns:
            List of FeatureTableInfo for all tables.
        """
        return [
            self._build_table_info(table)
            for table in self._tables.values()
        ]

    def get_table_info(self, name: str) -> FeatureTableInfo:
        """Get metadata for a specific feature table.

        Args:
            name: Table name.

        Returns:
            FeatureTableInfo for the table.

        Raises:
            FeatureTableNotFoundError: If table not registered.
        """
        table = self._get_table(name)
        return self._build_table_info(table)

    def delete(self, name: str) -> bool:
        """Delete a feature table and all its versions.

        Args:
            name: Table name to delete.

        Returns:
            True if deleted, False if not found.
        """
        if name not in self._tables:
            return False
        del self._tables[name]
        logger.info("Deleted feature table '%s'", name)
        return True

    def get_version(self, name: str, version: int) -> pd.DataFrame:
        """Retrieve a specific historical version of a feature table.

        Args:
            name: Table name.
            version: Version number to retrieve.

        Returns:
            DataFrame for the requested version.

        Raises:
            FeatureTableNotFoundError: If table or version not found.
        """
        table = self._get_table(name)

        if version == table.version:
            return table.data.copy()

        if version in table.history:
            return table.history[version].copy()

        raise FeatureTableNotFoundError(
            f"Version {version} not found for table '{name}'. "
            f"Available: {self._available_versions(table)}"
        )

    def _get_table(self, name: str) -> _VersionedTable:
        """Retrieve a table by name or raise.

        Args:
            name: Table name.

        Returns:
            The versioned table.

        Raises:
            FeatureTableNotFoundError: If not registered.
        """
        if name not in self._tables:
            raise FeatureTableNotFoundError(
                f"Feature table '{name}' not registered. "
                f"Available: {list(self._tables.keys())}"
            )
        return self._tables[name]

    def _validate_dataframe(
        self, df: pd.DataFrame, context: str
    ) -> None:
        """Validate that a DataFrame has required columns.

        Args:
            df: DataFrame to validate.
            context: Human-readable context for error messages.

        Raises:
            FeatureValidationError: If validation fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise FeatureValidationError(
                f"{context}: expected DataFrame, got {type(df).__name__}"
            )

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise FeatureValidationError(
                f"{context}: missing required columns: {missing}"
            )

        if df.empty:
            raise FeatureValidationError(f"{context}: DataFrame is empty")

    def _validate_columns(
        self,
        columns: List[str],
        data: pd.DataFrame,
        table_name: str,
    ) -> None:
        """Validate that requested columns exist in the table.

        Args:
            columns: Requested column names.
            data: Table data.
            table_name: Table name for error messages.

        Raises:
            FeatureValidationError: If columns not found.
        """
        feature_cols = set(data.columns) - REQUIRED_COLUMNS
        missing = set(columns) - feature_cols
        if missing:
            raise FeatureValidationError(
                f"Columns {missing} not found in table '{table_name}'. "
                f"Available: {feature_cols}"
            )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a DataFrame for storage (sort, normalize timestamps).

        Args:
            df: Raw input DataFrame.

        Returns:
            Cleaned and sorted DataFrame.
        """
        result = df.copy()
        result[TIMESTAMP_COLUMN] = pd.to_datetime(result[TIMESTAMP_COLUMN])
        result = result.sort_values(
            [ENTITY_ID_COLUMN, TIMESTAMP_COLUMN]
        ).reset_index(drop=True)
        return result

    def _filter_by_time(
        self, data: pd.DataFrame, timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Filter DataFrame to rows at or before timestamp.

        Args:
            data: Source DataFrame.
            timestamp: Upper bound cutoff.

        Returns:
            Filtered DataFrame.
        """
        return data[data[TIMESTAMP_COLUMN] <= timestamp]

    def _filter_by_entities(
        self, data: pd.DataFrame, entity_ids: List[Any]
    ) -> pd.DataFrame:
        """Filter DataFrame to specific entity IDs.

        Args:
            data: Source DataFrame.
            entity_ids: Entity IDs to include.

        Returns:
            Filtered DataFrame.
        """
        return data[data[ENTITY_ID_COLUMN].isin(entity_ids)]

    def _get_latest_per_entity(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the latest row per entity_id.

        Args:
            data: Filtered DataFrame.

        Returns:
            DataFrame with one row per entity (the most recent).
        """
        if data.empty:
            return data
        idx = data.groupby(ENTITY_ID_COLUMN)[TIMESTAMP_COLUMN].idxmax()
        return data.loc[idx]

    def _empty_result(
        self,
        reference_data: pd.DataFrame,
        columns: Optional[List[str]],
    ) -> pd.DataFrame:
        """Build an empty result DataFrame with correct columns.

        Args:
            reference_data: Table to derive column names from.
            columns: Optional column subset.

        Returns:
            Empty DataFrame with appropriate columns.
        """
        if columns is not None:
            cols = [ENTITY_ID_COLUMN, TIMESTAMP_COLUMN] + columns
        else:
            cols = list(reference_data.columns)
        return pd.DataFrame(columns=cols)

    def _build_table_info(self, table: _VersionedTable) -> FeatureTableInfo:
        """Build FeatureTableInfo from internal table state.

        Args:
            table: Internal versioned table.

        Returns:
            Public FeatureTableInfo.
        """
        feature_cols = [
            c for c in table.data.columns if c not in REQUIRED_COLUMNS
        ]
        return FeatureTableInfo(
            name=table.name,
            columns=feature_cols,
            version=table.version,
            row_count=len(table.data),
            registered_at=table.registered_at,
            updated_at=table.updated_at,
        )

    def _available_versions(self, table: _VersionedTable) -> List[int]:
        """List available version numbers for a table.

        Args:
            table: Internal versioned table.

        Returns:
            Sorted list of version numbers.
        """
        versions = list(table.history.keys()) + [table.version]
        return sorted(versions)

    def _prune_history(self, table: _VersionedTable) -> None:
        """Remove oldest history versions exceeding max_versions.

        Args:
            table: Table to prune.
        """
        all_versions = sorted(table.history.keys())
        # Keep max_versions - 1 in history (current version counts as 1)
        max_history = self._max_versions - 1
        while len(all_versions) > max_history:
            oldest = all_versions.pop(0)
            del table.history[oldest]
            logger.debug(
                "Pruned version %d from table '%s'", oldest, table.name
            )


def _normalize_timestamp(ts: Any) -> pd.Timestamp:
    """Convert various timestamp formats to pandas Timestamp.

    Args:
        ts: Timestamp as datetime, string, or pd.Timestamp.

    Returns:
        Normalized pd.Timestamp.
    """
    return pd.Timestamp(ts)
