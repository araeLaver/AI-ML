# Sample Financial Transaction Data Generator
"""
학습용 금융 거래 샘플 데이터 생성
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_transaction_data(
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    금융 거래 샘플 데이터 생성

    Args:
        n_samples: 생성할 샘플 수
        fraud_ratio: 이상거래 비율 (기본 2%)
        seed: 랜덤 시드

    Returns:
        거래 데이터 DataFrame
    """
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # 정상 거래 생성
    normal_data = _generate_normal_transactions(n_normal)

    # 이상 거래 생성
    fraud_data = _generate_fraud_transactions(n_fraud)

    # 데이터 결합
    df = pd.concat([normal_data, fraud_data], ignore_index=True)

    # 셔플
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 거래 ID 추가
    df['transaction_id'] = [f'TXN{str(i).zfill(8)}' for i in range(len(df))]

    # 컬럼 순서 정리
    columns = [
        'transaction_id', 'timestamp', 'amount', 'merchant_category',
        'location', 'distance_from_home', 'time_since_last_txn',
        'daily_txn_count', 'amount_vs_avg', 'is_international',
        'device_type', 'is_fraud'
    ]

    return df[columns]


def _generate_normal_transactions(n: int) -> pd.DataFrame:
    """정상 거래 생성"""

    # 시간 (주로 낮 시간대)
    base_time = datetime(2024, 1, 1)
    timestamps = [
        base_time + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.choice(range(6, 23), p=_get_hour_distribution()),
            minutes=np.random.randint(0, 60)
        )
        for _ in range(n)
    ]

    # 금액 (로그 정규 분포)
    amounts = np.random.lognormal(mean=10, sigma=1.5, size=n)
    amounts = np.clip(amounts, 1000, 5000000)  # 1,000원 ~ 500만원

    # 가맹점 카테고리
    categories = np.random.choice(
        ['음식점', '편의점', '마트', '온라인쇼핑', '주유소', '병원', '약국', '기타'],
        size=n,
        p=[0.25, 0.15, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05]
    )

    # 위치
    locations = np.random.choice(
        ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '기타'],
        size=n,
        p=[0.35, 0.25, 0.1, 0.07, 0.08, 0.05, 0.05, 0.05]
    )

    # 집에서의 거리 (정상: 대부분 가까움)
    distance_from_home = np.random.exponential(scale=5, size=n)
    distance_from_home = np.clip(distance_from_home, 0, 100)

    # 마지막 거래 이후 시간 (분)
    time_since_last = np.random.exponential(scale=120, size=n)
    time_since_last = np.clip(time_since_last, 1, 10000)

    # 일일 거래 횟수
    daily_count = np.random.poisson(lam=3, size=n)
    daily_count = np.clip(daily_count, 1, 20)

    # 평균 대비 거래금액 비율
    amount_vs_avg = np.random.normal(loc=1.0, scale=0.3, size=n)
    amount_vs_avg = np.clip(amount_vs_avg, 0.1, 3.0)

    # 해외 거래 여부 (정상: 대부분 국내)
    is_international = np.random.choice([0, 1], size=n, p=[0.98, 0.02])

    # 디바이스 타입
    device_types = np.random.choice(
        ['mobile', 'web', 'pos', 'atm'],
        size=n,
        p=[0.4, 0.2, 0.3, 0.1]
    )

    return pd.DataFrame({
        'timestamp': timestamps,
        'amount': amounts.astype(int),
        'merchant_category': categories,
        'location': locations,
        'distance_from_home': np.round(distance_from_home, 2),
        'time_since_last_txn': np.round(time_since_last, 1),
        'daily_txn_count': daily_count,
        'amount_vs_avg': np.round(amount_vs_avg, 2),
        'is_international': is_international,
        'device_type': device_types,
        'is_fraud': 0
    })


def _generate_fraud_transactions(n: int) -> pd.DataFrame:
    """이상 거래 생성"""

    # 시간 (주로 새벽 시간대)
    base_time = datetime(2024, 1, 1)
    timestamps = [
        base_time + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.choice(range(24), p=_get_fraud_hour_distribution()),
            minutes=np.random.randint(0, 60)
        )
        for _ in range(n)
    ]

    # 금액 (더 큰 금액)
    amounts = np.random.lognormal(mean=12, sigma=2, size=n)
    amounts = np.clip(amounts, 100000, 50000000)  # 10만원 ~ 5000만원

    # 가맹점 카테고리 (보석, 전자제품 등)
    categories = np.random.choice(
        ['보석', '전자제품', '해외송금', '가상화폐', '온라인쇼핑', '카지노', '기타'],
        size=n,
        p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.1, 0.05]
    )

    # 위치 (해외 또는 먼 지역)
    locations = np.random.choice(
        ['해외', '서울', '경기', '부산', '제주', '기타'],
        size=n,
        p=[0.3, 0.2, 0.15, 0.1, 0.15, 0.1]
    )

    # 집에서의 거리 (이상: 대부분 멂)
    distance_from_home = np.random.exponential(scale=200, size=n)
    distance_from_home = np.clip(distance_from_home, 50, 1000)

    # 마지막 거래 이후 시간 (매우 짧거나 매우 김)
    time_since_last = np.where(
        np.random.random(n) > 0.5,
        np.random.uniform(1, 10, n),  # 매우 짧음
        np.random.uniform(5000, 50000, n)  # 매우 김
    )

    # 일일 거래 횟수 (많음)
    daily_count = np.random.poisson(lam=10, size=n)
    daily_count = np.clip(daily_count, 5, 50)

    # 평균 대비 거래금액 비율 (높음)
    amount_vs_avg = np.random.uniform(3, 20, size=n)

    # 해외 거래 여부 (이상: 해외 거래 많음)
    is_international = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

    # 디바이스 타입
    device_types = np.random.choice(
        ['mobile', 'web', 'pos', 'atm', 'unknown'],
        size=n,
        p=[0.2, 0.3, 0.2, 0.2, 0.1]
    )

    return pd.DataFrame({
        'timestamp': timestamps,
        'amount': amounts.astype(int),
        'merchant_category': categories,
        'location': locations,
        'distance_from_home': np.round(distance_from_home, 2),
        'time_since_last_txn': np.round(time_since_last, 1),
        'daily_txn_count': daily_count,
        'amount_vs_avg': np.round(amount_vs_avg, 2),
        'is_international': is_international,
        'device_type': device_types,
        'is_fraud': 1
    })


def _get_hour_distribution() -> list:
    """정상 거래 시간 분포 (6시~22시)"""
    # 낮 시간에 더 많은 거래
    hours = list(range(6, 23))
    weights = [1, 2, 3, 4, 5, 5, 4, 4, 5, 5, 4, 3, 4, 5, 4, 3, 2]
    total = sum(weights)
    return [w / total for w in weights]


def _get_fraud_hour_distribution() -> list:
    """이상 거래 시간 분포 (24시간)"""
    # 새벽에 더 많은 거래
    weights = [
        5, 5, 5, 5, 4, 3,  # 0-5시 (높음)
        2, 2, 2, 2, 2, 2,  # 6-11시 (낮음)
        2, 2, 2, 2, 2, 2,  # 12-17시 (낮음)
        2, 3, 3, 4, 4, 5   # 18-23시 (증가)
    ]
    total = sum(weights)
    return [w / total for w in weights]


if __name__ == "__main__":
    # 테스트 실행
    df = generate_transaction_data(n_samples=1000)
    print(f"Generated {len(df)} transactions")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample data:\n{df.head()}")
    print(f"\nFraud transactions:\n{df[df['is_fraud'] == 1].head()}")
