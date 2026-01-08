"""
Data Pipeline Tests
"""

import pytest
import pandas as pd
import numpy as np

from src.data import DataIngestion, DataValidator, DataPreprocessor


class TestDataIngestion:
    """DataIngestion 테스트"""

    def test_generate_sample_data(self):
        """샘플 데이터 생성 테스트"""
        ingestion = DataIngestion(data_dir="data/test")
        df = ingestion.generate_sample_data(n_samples=1000, fraud_ratio=0.05)

        assert len(df) == 1000
        assert "is_fraud" in df.columns
        assert df["is_fraud"].mean() == pytest.approx(0.05, abs=0.02)

    def test_get_data_info(self, sample_dataframe):
        """데이터 정보 테스트"""
        ingestion = DataIngestion()
        info = ingestion.get_data_info(sample_dataframe)

        assert info["shape"] == sample_dataframe.shape
        assert "is_fraud" in info["columns"]
        assert info["fraud_ratio"] is not None


class TestDataValidator:
    """DataValidator 테스트"""

    def test_validate_valid_data(self, sample_dataframe):
        """유효한 데이터 검증"""
        validator = DataValidator()
        result = validator.validate(sample_dataframe)

        assert result.is_valid is True

    def test_validate_empty_data(self):
        """빈 데이터 검증"""
        validator = DataValidator()
        df = pd.DataFrame()
        result = validator.validate(df)

        assert result.is_valid is False
        assert "데이터가 비어있습니다" in result.errors

    def test_validate_missing_columns(self):
        """필수 컬럼 누락 검증"""
        validator = DataValidator()
        df = pd.DataFrame({"amount": [100, 200]})
        result = validator.validate(df)

        assert result.is_valid is False
        assert any("필수 컬럼 누락" in e for e in result.errors)

    def test_validate_inference_input(self, sample_transaction):
        """추론 입력 검증"""
        validator = DataValidator()
        result = validator.validate_inference_input(sample_transaction)

        assert result.is_valid is True

    def test_validate_inference_input_missing_feature(self):
        """추론 입력 - 누락된 피처"""
        validator = DataValidator()
        result = validator.validate_inference_input({"amount": 100})

        assert result.is_valid is False


class TestDataPreprocessor:
    """DataPreprocessor 테스트"""

    def test_prepare_features(self, sample_dataframe):
        """피처 준비 테스트"""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert "is_fraud" not in X.columns

    def test_fit_transform(self, sample_dataframe):
        """fit_transform 테스트"""
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.prepare_features(sample_dataframe, include_target=True)

        X_transformed = preprocessor.fit_transform(X)

        # 스케일링된 피처의 평균은 0에 가까워야 함
        numeric_features = preprocessor.NUMERIC_FEATURES
        for col in numeric_features:
            if col in X_transformed.columns:
                assert X_transformed[col].mean() == pytest.approx(0, abs=0.1)

    def test_split_data(self, sample_dataframe):
        """데이터 분할 테스트"""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)

        # 비율 확인
        assert len(X_test) == pytest.approx(len(X) * 0.2, abs=5)
