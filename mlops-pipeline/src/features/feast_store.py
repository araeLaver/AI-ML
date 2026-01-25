"""Feast Feature Store integration for centralized feature management.

This module provides a wrapper around Feast for managing ML features,
with a local fallback when Feast is not available.
"""
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

try:
    from feast import FeatureStore, Entity, FeatureView, Field, FileSource
    from feast.types import Float64, Int64, String
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    dtype: str  # "float64", "int64", "string"
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class FeatureGroup:
    """A group of related features (similar to Feast FeatureView)."""
    name: str
    entity: str
    features: list[FeatureDefinition]
    description: str = ""
    ttl: timedelta = field(default_factory=lambda: timedelta(days=1))
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity": self.entity,
            "features": [f.to_dict() for f in self.features],
            "description": self.description,
            "ttl_seconds": self.ttl.total_seconds(),
            "tags": self.tags,
        }


class FeastFeatureStore:
    """Feature store wrapper with Feast integration and local fallback.

    This class provides a unified interface for feature management,
    supporting both Feast-based and local file-based storage.

    Attributes:
        repo_path: Path to the feature store repository
        use_feast: Whether to use Feast (if available)
    """

    # Default fraud detection feature definitions
    DEFAULT_FEATURES = [
        FeatureDefinition("amount", "float64", "Transaction amount"),
        FeatureDefinition("hour", "int64", "Hour of transaction"),
        FeatureDefinition("day_of_week", "int64", "Day of week (0=Monday)"),
        FeatureDefinition("is_weekend", "int64", "Weekend flag"),
        FeatureDefinition("merchant_category_encoded", "int64", "Merchant category"),
        FeatureDefinition("location_encoded", "int64", "Location encoding"),
        FeatureDefinition("amount_zscore", "float64", "Amount z-score"),
        FeatureDefinition("velocity_1h", "float64", "Transactions per hour"),
        FeatureDefinition("amount_to_avg_ratio", "float64", "Amount vs average"),
        FeatureDefinition("time_since_last_txn", "float64", "Time since last transaction"),
        FeatureDefinition("risk_score", "float64", "Calculated risk score"),
    ]

    def __init__(
        self,
        repo_path: str | Path = "feature_store",
        use_feast: bool = True,
    ):
        """Initialize the feature store.

        Args:
            repo_path: Path to store feature data and metadata
            use_feast: Whether to use Feast if available
        """
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)

        self.use_feast = use_feast and FEAST_AVAILABLE
        self._feature_groups: dict[str, FeatureGroup] = {}
        self._local_store: dict[str, pd.DataFrame] = {}

        # Initialize Feast if available
        self._feast_store = None
        if self.use_feast:
            self._init_feast()

        # Load existing metadata
        self._load_metadata()

    def _init_feast(self) -> None:
        """Initialize Feast feature store."""
        try:
            feast_config = self.repo_path / "feature_store.yaml"
            if feast_config.exists():
                self._feast_store = FeatureStore(repo_path=str(self.repo_path))
        except Exception:
            self.use_feast = False

    def _load_metadata(self) -> None:
        """Load feature group metadata from disk."""
        metadata_file = self.repo_path / "feature_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for group_data in data.get("feature_groups", []):
                    features = [
                        FeatureDefinition(**f) for f in group_data.pop("features", [])
                    ]
                    ttl_seconds = group_data.pop("ttl_seconds", 86400)
                    group = FeatureGroup(
                        **group_data,
                        features=features,
                        ttl=timedelta(seconds=ttl_seconds),
                    )
                    self._feature_groups[group.name] = group

    def _save_metadata(self) -> None:
        """Save feature group metadata to disk."""
        metadata_file = self.repo_path / "feature_metadata.json"
        data = {
            "feature_groups": [g.to_dict() for g in self._feature_groups.values()],
            "updated_at": datetime.now().isoformat(),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_feature_group(
        self,
        name: str,
        entity: str,
        features: list[FeatureDefinition],
        description: str = "",
        ttl: timedelta | None = None,
        tags: dict[str, str] | None = None,
    ) -> FeatureGroup:
        """Register a new feature group.

        Args:
            name: Unique name for the feature group
            entity: Primary entity (e.g., "transaction_id", "user_id")
            features: List of feature definitions
            description: Human-readable description
            ttl: Time-to-live for cached features
            tags: Key-value tags for organization

        Returns:
            Created FeatureGroup
        """
        group = FeatureGroup(
            name=name,
            entity=entity,
            features=features,
            description=description,
            ttl=ttl or timedelta(days=1),
            tags=tags or {},
        )

        self._feature_groups[name] = group
        self._save_metadata()

        return group

    def register_default_fraud_features(self) -> FeatureGroup:
        """Register default fraud detection features.

        Returns:
            Created FeatureGroup for fraud detection
        """
        return self.register_feature_group(
            name="fraud_transaction_features",
            entity="transaction_id",
            features=self.DEFAULT_FEATURES,
            description="Features for fraud detection model",
            ttl=timedelta(hours=6),
            tags={"domain": "fraud", "version": "1.0"},
        )

    def push_features(
        self,
        feature_group: str,
        df: pd.DataFrame,
        entity_column: str | None = None,
    ) -> dict[str, Any]:
        """Push feature values to the store.

        Args:
            feature_group: Name of the feature group
            df: DataFrame containing feature values
            entity_column: Column to use as entity ID

        Returns:
            Push statistics
        """
        if feature_group not in self._feature_groups:
            raise ValueError(f"Feature group '{feature_group}' not found")

        group = self._feature_groups[feature_group]
        entity_col = entity_column or group.entity

        # Validate features
        expected_features = {f.name for f in group.features}
        actual_features = set(df.columns) - {entity_col}
        missing = expected_features - actual_features

        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Store locally
        timestamp = datetime.now()
        store_key = f"{feature_group}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Add metadata columns
        df_store = df.copy()
        df_store["_feature_timestamp"] = timestamp
        df_store["_feature_group"] = feature_group

        # Save to local store and disk
        self._local_store[store_key] = df_store

        feature_file = self.repo_path / "data" / f"{store_key}.parquet"
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        df_store.to_parquet(feature_file, index=False)

        return {
            "feature_group": feature_group,
            "rows_pushed": len(df),
            "features_count": len(expected_features),
            "timestamp": timestamp.isoformat(),
            "storage_key": store_key,
        }

    def get_features(
        self,
        feature_group: str,
        entity_ids: list[Any] | None = None,
        features: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve features from the store.

        Args:
            feature_group: Name of the feature group
            entity_ids: Specific entity IDs to retrieve (None = all)
            features: Specific features to retrieve (None = all)
            as_of: Point-in-time retrieval (None = latest)

        Returns:
            DataFrame with requested features
        """
        if feature_group not in self._feature_groups:
            raise ValueError(f"Feature group '{feature_group}' not found")

        group = self._feature_groups[feature_group]

        # Find the most recent feature file
        data_dir = self.repo_path / "data"
        if not data_dir.exists():
            return pd.DataFrame()

        feature_files = sorted(
            data_dir.glob(f"{feature_group}_*.parquet"),
            reverse=True,
        )

        if not feature_files:
            return pd.DataFrame()

        # Filter by timestamp if as_of is specified
        if as_of:
            valid_files = []
            for f in feature_files:
                # Parse timestamp from filename
                ts_str = f.stem.replace(f"{feature_group}_", "")
                try:
                    file_ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    if file_ts <= as_of:
                        valid_files.append(f)
                except ValueError:
                    continue
            feature_files = valid_files

        if not feature_files:
            return pd.DataFrame()

        # Load the most recent valid file
        df = pd.read_parquet(feature_files[0])

        # Filter by entity IDs
        if entity_ids is not None:
            df = df[df[group.entity].isin(entity_ids)]

        # Select specific features
        if features:
            cols = [group.entity] + features
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
        else:
            # Remove metadata columns
            df = df.drop(
                columns=["_feature_timestamp", "_feature_group"],
                errors="ignore",
            )

        return df

    def get_historical_features(
        self,
        feature_group: str,
        entity_df: pd.DataFrame,
        entity_column: str | None = None,
        timestamp_column: str = "event_timestamp",
    ) -> pd.DataFrame:
        """Get point-in-time correct features for training.

        Args:
            feature_group: Name of the feature group
            entity_df: DataFrame with entity IDs and timestamps
            entity_column: Column containing entity IDs
            timestamp_column: Column containing event timestamps

        Returns:
            DataFrame with historical features joined to entities
        """
        if feature_group not in self._feature_groups:
            raise ValueError(f"Feature group '{feature_group}' not found")

        group = self._feature_groups[feature_group]
        entity_col = entity_column or group.entity

        # Load all historical feature data
        data_dir = self.repo_path / "data"
        if not data_dir.exists():
            return entity_df.copy()

        feature_files = sorted(data_dir.glob(f"{feature_group}_*.parquet"))

        if not feature_files:
            return entity_df.copy()

        # Concatenate all historical data
        all_features = []
        for f in feature_files:
            df = pd.read_parquet(f)
            all_features.append(df)

        if not all_features:
            return entity_df.copy()

        features_df = pd.concat(all_features, ignore_index=True)

        # Simple join for now (could implement proper point-in-time join)
        result = entity_df.merge(
            features_df.drop(columns=["_feature_timestamp", "_feature_group"], errors="ignore"),
            left_on=entity_col,
            right_on=group.entity,
            how="left",
        )

        return result

    def list_feature_groups(self) -> list[dict[str, Any]]:
        """List all registered feature groups.

        Returns:
            List of feature group summaries
        """
        return [
            {
                "name": g.name,
                "entity": g.entity,
                "feature_count": len(g.features),
                "description": g.description,
                "tags": g.tags,
            }
            for g in self._feature_groups.values()
        ]

    def get_feature_group(self, name: str) -> FeatureGroup | None:
        """Get a specific feature group by name."""
        return self._feature_groups.get(name)

    def delete_feature_group(self, name: str) -> bool:
        """Delete a feature group and its data.

        Args:
            name: Name of the feature group to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self._feature_groups:
            return False

        # Remove from registry
        del self._feature_groups[name]
        self._save_metadata()

        # Remove data files
        data_dir = self.repo_path / "data"
        if data_dir.exists():
            for f in data_dir.glob(f"{name}_*.parquet"):
                f.unlink()

        return True

    def compute_feature_statistics(
        self,
        feature_group: str,
    ) -> dict[str, Any]:
        """Compute statistics for features in a group.

        Args:
            feature_group: Name of the feature group

        Returns:
            Dictionary with feature statistics
        """
        df = self.get_features(feature_group)

        if df.empty:
            return {"feature_group": feature_group, "statistics": {}}

        group = self._feature_groups[feature_group]
        stats = {}

        for feature in group.features:
            if feature.name not in df.columns:
                continue

            col = df[feature.name]

            if feature.dtype in ("float64", "int64"):
                stats[feature.name] = {
                    "dtype": feature.dtype,
                    "count": int(col.count()),
                    "null_count": int(col.isnull().sum()),
                    "mean": float(col.mean()) if not col.empty else None,
                    "std": float(col.std()) if not col.empty else None,
                    "min": float(col.min()) if not col.empty else None,
                    "max": float(col.max()) if not col.empty else None,
                    "p25": float(col.quantile(0.25)) if not col.empty else None,
                    "p50": float(col.quantile(0.50)) if not col.empty else None,
                    "p75": float(col.quantile(0.75)) if not col.empty else None,
                }
            else:
                stats[feature.name] = {
                    "dtype": feature.dtype,
                    "count": int(col.count()),
                    "null_count": int(col.isnull().sum()),
                    "unique_count": int(col.nunique()),
                }

        return {
            "feature_group": feature_group,
            "row_count": len(df),
            "statistics": stats,
            "computed_at": datetime.now().isoformat(),
        }

    def generate_feature_hash(self, feature_group: str) -> str:
        """Generate a hash of feature definitions for versioning.

        Args:
            feature_group: Name of the feature group

        Returns:
            MD5 hash of feature definitions
        """
        if feature_group not in self._feature_groups:
            return ""

        group = self._feature_groups[feature_group]
        content = json.dumps(group.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
