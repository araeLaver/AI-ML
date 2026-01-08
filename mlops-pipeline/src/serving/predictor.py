"""
Predictor Module
- 모델 로드 및 예측
- 배치 예측 지원
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import time

logger = logging.getLogger(__name__)


class FraudPredictor:
    """이상 거래 예측 클래스"""

    REQUIRED_FEATURES = [
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

    DEFAULT_VALUES = {
        "day_of_week": 0,
        "merchant_category": 0,
        "transaction_count_1h": 1,
        "transaction_count_24h": 5,
        "is_weekend": 0,
        "is_night": 0,
        "device_change": 0,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.model = None
        self.preprocessor = None
        self.threshold = threshold
        self.model_version = "unknown"
        self.is_loaded = False

        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)

    def load_model(self, path: str) -> None:
        """모델 로드"""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

        data = joblib.load(model_path)

        if isinstance(data, dict):
            self.model = data.get("model")
            self.model_version = data.get("model_type", "unknown")
        else:
            self.model = data

        self.is_loaded = True
        logger.info(f"모델 로드 완료: {path}")

    def load_preprocessor(self, path: str) -> None:
        """전처리기 로드"""
        preprocessor_path = Path(path)
        if not preprocessor_path.exists():
            logger.warning(f"전처리기 파일을 찾을 수 없습니다: {path}")
            return

        data = joblib.load(preprocessor_path)
        self.preprocessor = data.get("scaler") if isinstance(data, dict) else data
        logger.info(f"전처리기 로드 완료: {path}")

    def _prepare_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """입력 데이터 준비"""
        # 기본값으로 채우기
        prepared = {**self.DEFAULT_VALUES, **data}

        # 필요한 피처만 선택
        features = {k: prepared[k] for k in self.REQUIRED_FEATURES}

        df = pd.DataFrame([features])
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """전처리 적용"""
        if self.preprocessor is not None:
            # 수치형 피처만 스케일링
            numeric_features = [
                "amount",
                "location_distance",
                "previous_avg_amount",
                "transaction_count_1h",
                "transaction_count_24h",
            ]
            df[numeric_features] = self.preprocessor.transform(df[numeric_features])
        return df

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """단일 예측"""
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        start_time = time.time()

        # 입력 준비
        df = self._prepare_input(data)

        # 전처리 (선택적)
        if self.preprocessor:
            df = self._preprocess(df)

        # 예측
        probability = self.model.predict_proba(df)[0][1]
        is_fraud = probability >= self.threshold

        # 리스크 레벨
        if probability >= 0.8:
            risk_level = "CRITICAL"
        elif probability >= 0.6:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
        elif probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "is_fraud": bool(is_fraud),
            "probability": round(float(probability), 4),
            "risk_level": risk_level,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2),
        }

        logger.debug(f"예측 완료: prob={probability:.4f}, fraud={is_fraud}")
        return result

    def predict_batch(
        self, data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """배치 예측"""
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")

        start_time = time.time()

        # 모든 데이터 준비
        dfs = [self._prepare_input(d) for d in data_list]
        df = pd.concat(dfs, ignore_index=True)

        # 전처리
        if self.preprocessor:
            df = self._preprocess(df)

        # 배치 예측
        probabilities = self.model.predict_proba(df)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            is_fraud = prob >= self.threshold

            if prob >= 0.8:
                risk_level = "CRITICAL"
            elif prob >= 0.6:
                risk_level = "HIGH"
            elif prob >= 0.4:
                risk_level = "MEDIUM"
            elif prob >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"

            results.append(
                {
                    "index": i,
                    "is_fraud": bool(is_fraud),
                    "probability": round(float(prob), 4),
                    "risk_level": risk_level,
                }
            )

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"배치 예측 완료: {len(data_list)}건, {total_time:.2f}ms"
        )

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "is_loaded": self.is_loaded,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "required_features": self.REQUIRED_FEATURES,
            "default_values": self.DEFAULT_VALUES,
        }

    def update_threshold(self, new_threshold: float) -> None:
        """임계값 업데이트"""
        if not 0 <= new_threshold <= 1:
            raise ValueError("임계값은 0과 1 사이여야 합니다")
        self.threshold = new_threshold
        logger.info(f"임계값 업데이트: {new_threshold}")
