"""
Data Validation Module
- 데이터 품질 검증
- 스키마 검증
- 통계적 검증
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """데이터 검증 클래스"""

    # 기대 스키마 정의
    EXPECTED_SCHEMA = {
        "transaction_id": "object",
        "amount": "float64",
        "time_hour": "int64",
        "day_of_week": "int64",
        "location_distance": "float64",
        "merchant_category": "int64",
        "previous_avg_amount": "float64",
        "transaction_count_1h": "int64",
        "transaction_count_24h": "int64",
        "is_weekend": "int64",
        "is_night": "int64",
        "device_change": "int64",
        "is_fraud": "int64",
    }

    # 유효 범위 정의
    VALID_RANGES = {
        "amount": (0, 100000),
        "time_hour": (0, 23),
        "day_of_week": (0, 6),
        "location_distance": (0, 10000),
        "merchant_category": (0, 20),
        "previous_avg_amount": (0, 100000),
        "transaction_count_1h": (0, 100),
        "transaction_count_24h": (0, 500),
        "is_weekend": (0, 1),
        "is_night": (0, 1),
        "device_change": (0, 1),
        "is_fraud": (0, 1),
    }

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """전체 데이터 검증"""
        errors = []
        warnings = []
        stats = {}

        # 1. 빈 데이터 검사
        if df.empty:
            return ValidationResult(is_valid=False, errors=["데이터가 비어있습니다"])

        stats["row_count"] = len(df)
        stats["column_count"] = len(df.columns)

        # 2. 필수 컬럼 검사
        required_columns = set(self.EXPECTED_SCHEMA.keys()) - {"timestamp"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            errors.append(f"필수 컬럼 누락: {missing_columns}")

        # 3. 결측값 검사
        missing_stats = df.isnull().sum()
        total_missing = missing_stats.sum()
        if total_missing > 0:
            missing_cols = missing_stats[missing_stats > 0].to_dict()
            warnings.append(f"결측값 발견: {missing_cols}")
            stats["missing_values"] = missing_cols

        # 4. 범위 검사
        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                n_invalid = out_of_range.sum()
                if n_invalid > 0:
                    warnings.append(
                        f"{col}: {n_invalid}개 값이 범위를 벗어남 [{min_val}, {max_val}]"
                    )

        # 5. 중복 검사
        if "transaction_id" in df.columns:
            n_duplicates = df["transaction_id"].duplicated().sum()
            if n_duplicates > 0:
                errors.append(f"중복 transaction_id 발견: {n_duplicates}개")

        # 6. 클래스 불균형 검사
        if "is_fraud" in df.columns:
            fraud_ratio = df["is_fraud"].mean()
            stats["fraud_ratio"] = fraud_ratio
            if fraud_ratio < 0.001:
                warnings.append(f"극심한 클래스 불균형: fraud ratio = {fraud_ratio:.4f}")
            elif fraud_ratio > 0.3:
                warnings.append(f"비정상적으로 높은 fraud ratio: {fraud_ratio:.4f}")

        # 7. 이상치 검사 (IQR 방식)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        for col in numeric_cols:
            if col not in ["is_fraud", "is_weekend", "is_night", "device_change"]:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_outliers > len(df) * 0.1:  # 10% 이상이 이상치
                    outlier_stats[col] = n_outliers

        if outlier_stats:
            stats["outliers"] = outlier_stats

        # 결과 집계
        is_valid = len(errors) == 0
        logger.info(
            f"검증 완료: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}"
        )

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, stats=stats
        )

    def validate_inference_input(self, data: Dict[str, Any]) -> ValidationResult:
        """추론 입력 데이터 검증"""
        errors = []

        required_features = [
            "amount",
            "time_hour",
            "location_distance",
            "previous_avg_amount",
        ]

        for feature in required_features:
            if feature not in data:
                errors.append(f"필수 피처 누락: {feature}")
            elif not isinstance(data[feature], (int, float)):
                errors.append(f"{feature}는 숫자여야 합니다")

        # 범위 검증
        for feature, value in data.items():
            if feature in self.VALID_RANGES:
                min_val, max_val = self.VALID_RANGES[feature]
                if value < min_val or value > max_val:
                    errors.append(
                        f"{feature}={value}가 범위를 벗어남 [{min_val}, {max_val}]"
                    )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
