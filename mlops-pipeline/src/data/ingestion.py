"""
Data Ingestion Module
- 데이터 수집 및 로드
- 다양한 소스 지원 (CSV, API, Database)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataIngestion:
    """데이터 수집 및 로드 클래스"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """CSV 파일 로드"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        logger.info(f"CSV 로드: {file_path}")
        df = pd.read_csv(path)
        logger.info(f"로드 완료: {len(df)} rows, {len(df.columns)} columns")
        return df

    def generate_sample_data(
        self, n_samples: int = 10000, fraud_ratio: float = 0.02
    ) -> pd.DataFrame:
        """
        샘플 이상 거래 데이터 생성
        - 실제 Credit Card Fraud Detection 데이터셋 패턴 모방
        """
        np.random.seed(42)

        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud

        # 정상 거래 데이터
        normal_data = {
            "amount": np.random.exponential(scale=100, size=n_normal),
            "time_hour": np.random.randint(0, 24, size=n_normal),
            "day_of_week": np.random.randint(0, 7, size=n_normal),
            "location_distance": np.random.exponential(scale=10, size=n_normal),
            "merchant_category": np.random.randint(0, 15, size=n_normal),
            "previous_avg_amount": np.random.exponential(scale=80, size=n_normal),
            "transaction_count_1h": np.random.poisson(lam=2, size=n_normal),
            "transaction_count_24h": np.random.poisson(lam=8, size=n_normal),
            "is_weekend": np.random.binomial(1, 0.28, size=n_normal),
            "is_night": np.random.binomial(1, 0.25, size=n_normal),
            "device_change": np.random.binomial(1, 0.05, size=n_normal),
            "is_fraud": np.zeros(n_normal, dtype=int),
        }

        # 이상 거래 데이터 (패턴이 다름)
        fraud_data = {
            "amount": np.random.exponential(scale=500, size=n_fraud)
            + np.random.uniform(100, 1000, size=n_fraud),
            "time_hour": np.random.choice([0, 1, 2, 3, 4, 22, 23], size=n_fraud),
            "day_of_week": np.random.randint(0, 7, size=n_fraud),
            "location_distance": np.random.exponential(scale=100, size=n_fraud) + 50,
            "merchant_category": np.random.choice([3, 5, 8, 12], size=n_fraud),
            "previous_avg_amount": np.random.exponential(scale=50, size=n_fraud),
            "transaction_count_1h": np.random.poisson(lam=5, size=n_fraud) + 3,
            "transaction_count_24h": np.random.poisson(lam=15, size=n_fraud) + 5,
            "is_weekend": np.random.binomial(1, 0.5, size=n_fraud),
            "is_night": np.random.binomial(1, 0.7, size=n_fraud),
            "device_change": np.random.binomial(1, 0.4, size=n_fraud),
            "is_fraud": np.ones(n_fraud, dtype=int),
        }

        # 합치기
        normal_df = pd.DataFrame(normal_data)
        fraud_df = pd.DataFrame(fraud_data)
        df = pd.concat([normal_df, fraud_df], ignore_index=True)

        # 셔플
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # transaction_id 추가
        df.insert(0, "transaction_id", [f"TXN_{i:08d}" for i in range(len(df))])

        # timestamp 추가
        base_date = pd.Timestamp("2024-01-01")
        df["timestamp"] = [
            base_date + pd.Timedelta(hours=np.random.randint(0, 8760))
            for _ in range(len(df))
        ]

        logger.info(
            f"샘플 데이터 생성 완료: {n_samples} samples, {n_fraud} frauds ({fraud_ratio*100:.1f}%)"
        )
        return df

    def save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """데이터 저장"""
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"데이터 저장: {output_path}")
        return output_path

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """데이터셋 정보 반환"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "fraud_ratio": df["is_fraud"].mean() if "is_fraud" in df.columns else None,
        }
