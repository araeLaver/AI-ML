# Tests for Data Module
"""
데이터 생성 및 전처리 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np


class TestDataGenerator:
    """데이터 생성기 테스트"""

    def test_generate_transaction_data(self):
        """거래 데이터 생성 테스트"""
        from src.data.generator import generate_transaction_data

        df = generate_transaction_data(n_samples=100, fraud_ratio=0.1)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        print(f"Generated {len(df)} transactions")

    def test_data_columns(self):
        """필수 컬럼 존재 테스트"""
        from src.data.generator import generate_transaction_data

        df = generate_transaction_data(n_samples=50)

        expected_columns = [
            'transaction_id', 'timestamp', 'amount', 'merchant_category',
            'location', 'distance_from_home', 'time_since_last_txn',
            'daily_txn_count', 'amount_vs_avg', 'is_international',
            'device_type', 'is_fraud'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_ratio(self):
        """이상거래 비율 테스트"""
        from src.data.generator import generate_transaction_data

        df = generate_transaction_data(n_samples=1000, fraud_ratio=0.05)

        actual_ratio = df['is_fraud'].mean()
        assert 0.03 <= actual_ratio <= 0.07, f"Fraud ratio {actual_ratio} out of range"

    def test_amount_range(self):
        """거래금액 범위 테스트"""
        from src.data.generator import generate_transaction_data

        df = generate_transaction_data(n_samples=500)

        assert df['amount'].min() >= 1000, "Amount too low"
        assert df['amount'].max() <= 50000000, "Amount too high"

    def test_unique_transaction_ids(self):
        """거래 ID 유일성 테스트"""
        from src.data.generator import generate_transaction_data

        df = generate_transaction_data(n_samples=100)

        assert df['transaction_id'].nunique() == len(df)

    def test_reproducibility(self):
        """재현성 테스트 (seed)"""
        from src.data.generator import generate_transaction_data

        df1 = generate_transaction_data(n_samples=100, seed=42)
        df2 = generate_transaction_data(n_samples=100, seed=42)

        pd.testing.assert_frame_equal(df1, df2)


class TestTransactionPreprocessor:
    """전처리기 테스트"""

    def test_preprocessor_creation(self):
        """전처리기 생성 테스트"""
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        assert preprocessor is not None

    def test_prepare_ml_data(self, sample_dataframe):
        """ML 데이터 준비 테스트"""
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    def test_transform(self, sample_dataframe):
        """데이터 변환 테스트"""
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        preprocessor.fit(sample_dataframe)
        transformed = preprocessor.transform(sample_dataframe)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_dataframe)
        # 파생 피처가 추가되었는지 확인
        assert 'hour' in transformed.columns
        assert 'log_amount' in transformed.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
