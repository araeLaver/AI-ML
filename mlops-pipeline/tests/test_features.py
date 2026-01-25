"""
Feature Store Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.features import FeastFeatureStore, FeatureDefinition, FeatureGroup


class TestFeatureDefinition:
    """FeatureDefinition 테스트"""

    def test_create_feature_definition(self):
        """피처 정의 생성"""
        feature = FeatureDefinition(
            name="amount",
            dtype="float64",
            description="Transaction amount",
        )

        assert feature.name == "amount"
        assert feature.dtype == "float64"
        assert feature.description == "Transaction amount"
        assert feature.tags == {}

    def test_feature_definition_with_tags(self):
        """태그 포함 피처 정의"""
        feature = FeatureDefinition(
            name="risk_score",
            dtype="float64",
            tags={"domain": "fraud", "importance": "high"},
        )

        assert feature.tags["domain"] == "fraud"
        assert feature.tags["importance"] == "high"

    def test_feature_definition_to_dict(self):
        """피처 정의 직렬화"""
        feature = FeatureDefinition(
            name="amount",
            dtype="float64",
            description="Test",
            tags={"key": "value"},
        )

        data = feature.to_dict()

        assert data["name"] == "amount"
        assert data["dtype"] == "float64"
        assert data["description"] == "Test"
        assert data["tags"]["key"] == "value"


class TestFeatureGroup:
    """FeatureGroup 테스트"""

    def test_create_feature_group(self):
        """피처 그룹 생성"""
        features = [
            FeatureDefinition("amount", "float64"),
            FeatureDefinition("hour", "int64"),
        ]

        group = FeatureGroup(
            name="transaction_features",
            entity="transaction_id",
            features=features,
            description="Transaction features for fraud detection",
        )

        assert group.name == "transaction_features"
        assert group.entity == "transaction_id"
        assert len(group.features) == 2
        assert group.ttl == timedelta(days=1)

    def test_feature_group_custom_ttl(self):
        """커스텀 TTL 피처 그룹"""
        group = FeatureGroup(
            name="test",
            entity="id",
            features=[],
            ttl=timedelta(hours=6),
        )

        assert group.ttl == timedelta(hours=6)

    def test_feature_group_to_dict(self):
        """피처 그룹 직렬화"""
        features = [FeatureDefinition("amount", "float64")]
        group = FeatureGroup(
            name="test_group",
            entity="entity_id",
            features=features,
            ttl=timedelta(hours=12),
            tags={"version": "1.0"},
        )

        data = group.to_dict()

        assert data["name"] == "test_group"
        assert data["entity"] == "entity_id"
        assert len(data["features"]) == 1
        assert data["ttl_seconds"] == 12 * 3600
        assert data["tags"]["version"] == "1.0"


class TestFeastFeatureStore:
    """FeastFeatureStore 테스트"""

    @pytest.fixture
    def temp_repo(self):
        """임시 리포지토리 생성"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def feature_store(self, temp_repo):
        """피처 스토어 인스턴스"""
        return FeastFeatureStore(repo_path=temp_repo, use_feast=False)

    @pytest.fixture
    def sample_features_df(self):
        """샘플 피처 DataFrame"""
        return pd.DataFrame({
            "transaction_id": [f"txn_{i}" for i in range(100)],
            "amount": np.random.uniform(10, 1000, 100),
            "hour": np.random.randint(0, 24, 100),
            "day_of_week": np.random.randint(0, 7, 100),
            "is_weekend": np.random.randint(0, 2, 100),
            "merchant_category_encoded": np.random.randint(0, 10, 100),
            "location_encoded": np.random.randint(0, 50, 100),
            "amount_zscore": np.random.uniform(-3, 3, 100),
            "velocity_1h": np.random.uniform(0, 10, 100),
            "amount_to_avg_ratio": np.random.uniform(0.1, 5, 100),
            "time_since_last_txn": np.random.uniform(0, 3600, 100),
            "risk_score": np.random.uniform(0, 1, 100),
        })

    def test_init_creates_directory(self, temp_repo):
        """초기화 시 디렉토리 생성"""
        store = FeastFeatureStore(repo_path=temp_repo, use_feast=False)
        assert Path(temp_repo).exists()

    def test_register_default_fraud_features(self, feature_store):
        """기본 사기 탐지 피처 등록"""
        group = feature_store.register_default_fraud_features()

        assert group.name == "fraud_transaction_features"
        assert group.entity == "transaction_id"
        assert len(group.features) == 11
        assert group.tags["domain"] == "fraud"

    def test_register_feature_group(self, feature_store):
        """커스텀 피처 그룹 등록"""
        features = [
            FeatureDefinition("custom_feature", "float64", "Custom feature"),
        ]

        group = feature_store.register_feature_group(
            name="custom_group",
            entity="user_id",
            features=features,
            description="Custom feature group",
            ttl=timedelta(hours=2),
            tags={"env": "test"},
        )

        assert group.name == "custom_group"
        assert feature_store.get_feature_group("custom_group") is not None

    def test_list_feature_groups(self, feature_store):
        """피처 그룹 목록"""
        feature_store.register_default_fraud_features()

        groups = feature_store.list_feature_groups()

        assert len(groups) == 1
        assert groups[0]["name"] == "fraud_transaction_features"
        assert groups[0]["feature_count"] == 11

    def test_push_features(self, feature_store, sample_features_df):
        """피처 푸시"""
        feature_store.register_default_fraud_features()

        result = feature_store.push_features(
            feature_group="fraud_transaction_features",
            df=sample_features_df,
        )

        assert result["feature_group"] == "fraud_transaction_features"
        assert result["rows_pushed"] == 100
        assert result["features_count"] == 11
        assert "storage_key" in result

    def test_push_features_missing_group(self, feature_store, sample_features_df):
        """존재하지 않는 그룹에 푸시 시도"""
        with pytest.raises(ValueError, match="not found"):
            feature_store.push_features(
                feature_group="nonexistent",
                df=sample_features_df,
            )

    def test_push_features_missing_columns(self, feature_store):
        """필수 컬럼 누락 시 에러"""
        feature_store.register_default_fraud_features()

        incomplete_df = pd.DataFrame({
            "transaction_id": ["txn_1"],
            "amount": [100.0],
        })

        with pytest.raises(ValueError, match="Missing features"):
            feature_store.push_features(
                feature_group="fraud_transaction_features",
                df=incomplete_df,
            )

    def test_get_features(self, feature_store, sample_features_df):
        """피처 조회"""
        feature_store.register_default_fraud_features()
        feature_store.push_features("fraud_transaction_features", sample_features_df)

        retrieved = feature_store.get_features("fraud_transaction_features")

        assert len(retrieved) == 100
        assert "amount" in retrieved.columns
        assert "_feature_timestamp" not in retrieved.columns

    def test_get_features_with_entity_filter(self, feature_store, sample_features_df):
        """엔티티 필터로 피처 조회"""
        feature_store.register_default_fraud_features()
        feature_store.push_features("fraud_transaction_features", sample_features_df)

        entity_ids = ["txn_0", "txn_1", "txn_2"]
        retrieved = feature_store.get_features(
            "fraud_transaction_features",
            entity_ids=entity_ids,
        )

        assert len(retrieved) == 3

    def test_get_features_specific_columns(self, feature_store, sample_features_df):
        """특정 컬럼만 조회"""
        feature_store.register_default_fraud_features()
        feature_store.push_features("fraud_transaction_features", sample_features_df)

        retrieved = feature_store.get_features(
            "fraud_transaction_features",
            features=["amount", "risk_score"],
        )

        assert "amount" in retrieved.columns
        assert "risk_score" in retrieved.columns
        assert "hour" not in retrieved.columns

    def test_get_features_empty_store(self, feature_store):
        """빈 스토어에서 조회"""
        feature_store.register_default_fraud_features()

        retrieved = feature_store.get_features("fraud_transaction_features")

        assert retrieved.empty

    def test_get_historical_features(self, feature_store, sample_features_df):
        """히스토리컬 피처 조회"""
        feature_store.register_default_fraud_features()
        feature_store.push_features("fraud_transaction_features", sample_features_df)

        entity_df = pd.DataFrame({
            "transaction_id": ["txn_0", "txn_1"],
            "event_timestamp": [datetime.now(), datetime.now()],
        })

        result = feature_store.get_historical_features(
            "fraud_transaction_features",
            entity_df=entity_df,
        )

        assert len(result) == 2
        assert "amount" in result.columns

    def test_delete_feature_group(self, feature_store, sample_features_df):
        """피처 그룹 삭제"""
        feature_store.register_default_fraud_features()
        feature_store.push_features("fraud_transaction_features", sample_features_df)

        deleted = feature_store.delete_feature_group("fraud_transaction_features")

        assert deleted is True
        assert feature_store.get_feature_group("fraud_transaction_features") is None
        assert len(feature_store.list_feature_groups()) == 0

    def test_delete_nonexistent_group(self, feature_store):
        """존재하지 않는 그룹 삭제"""
        deleted = feature_store.delete_feature_group("nonexistent")
        assert deleted is False

    def test_compute_feature_statistics(self, feature_store, sample_features_df):
        """피처 통계 계산"""
        feature_store.register_default_fraud_features()
        feature_store.push_features("fraud_transaction_features", sample_features_df)

        stats = feature_store.compute_feature_statistics("fraud_transaction_features")

        assert stats["feature_group"] == "fraud_transaction_features"
        assert stats["row_count"] == 100
        assert "amount" in stats["statistics"]
        assert stats["statistics"]["amount"]["dtype"] == "float64"
        assert "mean" in stats["statistics"]["amount"]
        assert "std" in stats["statistics"]["amount"]
        assert "min" in stats["statistics"]["amount"]
        assert "max" in stats["statistics"]["amount"]

    def test_compute_statistics_empty_group(self, feature_store):
        """빈 그룹 통계"""
        feature_store.register_default_fraud_features()

        stats = feature_store.compute_feature_statistics("fraud_transaction_features")

        assert stats["statistics"] == {}

    def test_generate_feature_hash(self, feature_store):
        """피처 해시 생성"""
        feature_store.register_default_fraud_features()

        hash1 = feature_store.generate_feature_hash("fraud_transaction_features")

        assert len(hash1) == 12
        assert hash1.isalnum()

        # 동일한 그룹은 동일한 해시
        hash2 = feature_store.generate_feature_hash("fraud_transaction_features")
        assert hash1 == hash2

    def test_generate_hash_nonexistent_group(self, feature_store):
        """존재하지 않는 그룹 해시"""
        hash_val = feature_store.generate_feature_hash("nonexistent")
        assert hash_val == ""

    def test_metadata_persistence(self, temp_repo, sample_features_df):
        """메타데이터 영속성"""
        # 첫 번째 인스턴스에서 등록
        store1 = FeastFeatureStore(repo_path=temp_repo, use_feast=False)
        store1.register_default_fraud_features()
        store1.push_features("fraud_transaction_features", sample_features_df)

        # 두 번째 인스턴스에서 로드
        store2 = FeastFeatureStore(repo_path=temp_repo, use_feast=False)

        groups = store2.list_feature_groups()
        assert len(groups) == 1
        assert groups[0]["name"] == "fraud_transaction_features"

        # 데이터도 조회 가능해야 함
        retrieved = store2.get_features("fraud_transaction_features")
        assert len(retrieved) == 100

    def test_default_features_content(self):
        """기본 피처 정의 내용 확인"""
        default_features = FeastFeatureStore.DEFAULT_FEATURES

        feature_names = [f.name for f in default_features]

        assert "amount" in feature_names
        assert "hour" in feature_names
        assert "risk_score" in feature_names
        assert len(default_features) == 11
