"""
Data Preprocessing Module
- 피처 엔지니어링
- 데이터 변환
- 스케일링
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """데이터 전처리 클래스"""

    FEATURE_COLUMNS = [
        "amount",
        "time_hour",
        "day_of_week",
        "location_distance",
        "merchant_category",
        "previous_avg_amount",
        "transaction_count_1h",
        "transaction_count_24h",
        "is_weekend",
        "is_night",
        "device_change",
    ]

    NUMERIC_FEATURES = [
        "amount",
        "location_distance",
        "previous_avg_amount",
        "transaction_count_1h",
        "transaction_count_24h",
    ]

    TARGET_COLUMN = "is_fraud"

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_stats: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """스케일러 학습"""
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"지원하지 않는 스케일러: {self.scaler_type}")

        # 수치형 피처만 스케일링
        numeric_data = df[self.NUMERIC_FEATURES]
        self.scaler.fit(numeric_data)

        # 피처 통계 저장
        self.feature_stats = {
            "mean": dict(zip(self.NUMERIC_FEATURES, self.scaler.mean_)),
            "std": dict(
                zip(
                    self.NUMERIC_FEATURES,
                    (
                        self.scaler.scale_
                        if self.scaler_type == "standard"
                        else self.scaler.data_range_
                    ),
                )
            ),
        }

        logger.info(f"스케일러 학습 완료: {self.scaler_type}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 변환"""
        if self.scaler is None:
            raise ValueError("스케일러가 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        df_transformed = df.copy()

        # 수치형 피처 스케일링
        df_transformed[self.NUMERIC_FEATURES] = self.scaler.transform(
            df[self.NUMERIC_FEATURES]
        )

        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 및 변환"""
        self.fit(df)
        return self.transform(df)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """추가 피처 생성"""
        df = df.copy()

        # amount 관련 파생 피처
        df["amount_to_avg_ratio"] = df["amount"] / (df["previous_avg_amount"] + 1)
        df["amount_log"] = np.log1p(df["amount"])

        # 거래 빈도 관련
        df["tx_frequency_ratio"] = df["transaction_count_1h"] / (
            df["transaction_count_24h"] + 1
        )

        # 리스크 스코어 (규칙 기반)
        df["risk_score"] = (
            (df["amount_to_avg_ratio"] > 3).astype(int)
            + (df["location_distance"] > 100).astype(int)
            + (df["is_night"] == 1).astype(int)
            + (df["device_change"] == 1).astype(int)
            + (df["transaction_count_1h"] > 5).astype(int)
        )

        # 시간대 카테고리
        df["time_category"] = pd.cut(
            df["time_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True,
        )

        logger.info("피처 엔지니어링 완료")
        return df

    def prepare_features(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """학습/추론용 피처 준비"""
        # 필요한 컬럼만 선택
        available_features = [f for f in self.FEATURE_COLUMNS if f in df.columns]
        X = df[available_features].copy()

        # 결측값 처리
        X = X.fillna(0)

        if include_target and self.TARGET_COLUMN in df.columns:
            y = df[self.TARGET_COLUMN]
            return X, y

        return X, None

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """데이터 분할 (train/val/test)"""
        # 먼저 test 분리
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # train에서 validation 분리
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

        logger.info(f"데이터 분할: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, path: str) -> None:
        """전처리기 저장"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "feature_stats": self.feature_stats,
            "feature_columns": self.FEATURE_COLUMNS,
            "numeric_features": self.NUMERIC_FEATURES,
        }
        joblib.dump(data, save_path)
        logger.info(f"전처리기 저장: {save_path}")

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """전처리기 로드"""
        data = joblib.load(path)
        preprocessor = cls(scaler_type=data["scaler_type"])
        preprocessor.scaler = data["scaler"]
        preprocessor.feature_stats = data["feature_stats"]
        logger.info(f"전처리기 로드: {path}")
        return preprocessor
