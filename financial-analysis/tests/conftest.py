# Test Configuration
"""
pytest 공통 설정 및 fixtures
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_transaction():
    """샘플 거래 데이터"""
    return {
        'transaction_id': 'TXN00000001',
        'amount': 150000,
        'merchant_category': '음식점',
        'location': '서울',
        'distance_from_home': 5.0,
        'time_since_last_txn': 120.0,
        'daily_txn_count': 3,
        'amount_vs_avg': 1.2,
        'is_international': 0,
        'device_type': 'mobile',
        'hour': 14,
    }


@pytest.fixture
def fraud_transaction():
    """이상거래 샘플 데이터"""
    return {
        'transaction_id': 'TXN00000002',
        'amount': 15000000,
        'merchant_category': '해외송금',
        'location': '해외',
        'distance_from_home': 500.0,
        'time_since_last_txn': 5.0,
        'daily_txn_count': 15,
        'amount_vs_avg': 12.5,
        'is_international': 1,
        'device_type': 'unknown',
        'hour': 3,
    }


@pytest.fixture
def sample_dataframe():
    """샘플 DataFrame"""
    from src.data.generator import generate_transaction_data
    return generate_transaction_data(n_samples=100, fraud_ratio=0.1, seed=42)
