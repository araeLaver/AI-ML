"""
Pytest Fixtures
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_transaction():
    """단일 거래 데이터"""
    return {
        "amount": 150.0,
        "time_hour": 14,
        "day_of_week": 2,
        "location_distance": 5.0,
        "merchant_category": 3,
        "previous_avg_amount": 100.0,
        "transaction_count_1h": 2,
        "transaction_count_24h": 8,
        "is_weekend": 0,
        "is_night": 0,
        "device_change": 0,
    }


@pytest.fixture
def sample_fraud_transaction():
    """이상 거래 데이터"""
    return {
        "amount": 5000.0,
        "time_hour": 3,
        "day_of_week": 5,
        "location_distance": 500.0,
        "merchant_category": 8,
        "previous_avg_amount": 50.0,
        "transaction_count_1h": 10,
        "transaction_count_24h": 25,
        "is_weekend": 1,
        "is_night": 1,
        "device_change": 1,
    }


@pytest.fixture
def sample_dataframe():
    """샘플 데이터프레임"""
    np.random.seed(42)
    n = 100

    data = {
        "transaction_id": [f"TXN_{i:05d}" for i in range(n)],
        "amount": np.random.exponential(scale=100, size=n),
        "time_hour": np.random.randint(0, 24, size=n),
        "day_of_week": np.random.randint(0, 7, size=n),
        "location_distance": np.random.exponential(scale=10, size=n),
        "merchant_category": np.random.randint(0, 15, size=n),
        "previous_avg_amount": np.random.exponential(scale=80, size=n),
        "transaction_count_1h": np.random.poisson(lam=2, size=n),
        "transaction_count_24h": np.random.poisson(lam=8, size=n),
        "is_weekend": np.random.binomial(1, 0.28, size=n),
        "is_night": np.random.binomial(1, 0.25, size=n),
        "device_change": np.random.binomial(1, 0.05, size=n),
        "is_fraud": np.random.binomial(1, 0.02, size=n),
    }

    return pd.DataFrame(data)


@pytest.fixture
def trained_model(sample_dataframe):
    """학습된 모델"""
    from src.data import DataPreprocessor
    from src.training import ModelTrainer

    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

    trainer = ModelTrainer(model_type="random_forest", mlflow_tracking=False)
    trainer.train(X, y)

    return trainer, preprocessor
