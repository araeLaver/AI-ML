# Data Preprocessor Module
"""
금융 거래 데이터 전처리 모듈 (Step 1: Pandas/NumPy 활용)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class TransactionPreprocessor:
    """
    금융 거래 데이터 전처리 클래스

    주요 기능:
    - 결측치 처리
    - 이상치 탐지 및 처리
    - 피처 엔지니어링
    - 정규화/인코딩
    - 학습/테스트 분할
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Any] = {}
        self._is_fitted = False

        # 수치형/범주형 컬럼 정의
        self.numeric_features = [
            'amount', 'distance_from_home', 'time_since_last_txn',
            'daily_txn_count', 'amount_vs_avg'
        ]
        self.categorical_features = [
            'merchant_category', 'location', 'device_type'
        ]
        self.binary_features = ['is_international']

    def fit(self, df: pd.DataFrame) -> 'TransactionPreprocessor':
        """
        전처리기 학습 (학습 데이터 기준으로 통계치 저장)
        """
        df_copy = df.copy()

        # 수치형 피처 통계 저장
        for col in self.numeric_features:
            if col in df_copy.columns:
                self.feature_stats[col] = {
                    'mean': df_copy[col].mean(),
                    'std': df_copy[col].std(),
                    'median': df_copy[col].median(),
                    'q1': df_copy[col].quantile(0.25),
                    'q3': df_copy[col].quantile(0.75),
                }

        # 범주형 피처 인코더 학습
        for col in self.categorical_features:
            if col in df_copy.columns:
                le = LabelEncoder()
                le.fit(df_copy[col].astype(str))
                self.label_encoders[col] = le

        # 스케일러 학습 (수치형만)
        numeric_df = df_copy[self.numeric_features].fillna(0)
        self.scaler.fit(numeric_df)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        데이터 변환 (전처리 적용)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")

        df_copy = df.copy()

        # 1. 결측치 처리
        df_copy = self._handle_missing(df_copy)

        # 2. 피처 엔지니어링
        df_copy = self._engineer_features(df_copy)

        # 3. 이상치 처리
        df_copy = self._handle_outliers(df_copy)

        # 4. 범주형 인코딩
        df_copy = self._encode_categorical(df_copy)

        # 5. 수치형 정규화
        df_copy = self._scale_numeric(df_copy)

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit + transform"""
        return self.fit(df).transform(df)

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        df_copy = df.copy()

        # 수치형: 중앙값으로 대체
        for col in self.numeric_features:
            if col in df_copy.columns and col in self.feature_stats:
                df_copy[col] = df_copy[col].fillna(self.feature_stats[col]['median'])

        # 범주형: 'unknown'으로 대체
        for col in self.categorical_features:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna('unknown')

        # 이진형: 0으로 대체
        for col in self.binary_features:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(0)

        return df_copy

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링"""
        df_copy = df.copy()

        # 시간 기반 피처 추출
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
            df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
            df_copy['is_night'] = ((df_copy['hour'] >= 22) | (df_copy['hour'] <= 5)).astype(int)

        # 금액 관련 파생 피처
        if 'amount' in df_copy.columns:
            df_copy['log_amount'] = np.log1p(df_copy['amount'])

            # 금액 구간
            df_copy['amount_category'] = pd.cut(
                df_copy['amount'],
                bins=[0, 10000, 100000, 1000000, float('inf')],
                labels=['소액', '일반', '고액', '초고액']
            ).astype(str)

        # 거리 관련 파생 피처
        if 'distance_from_home' in df_copy.columns:
            df_copy['is_far'] = (df_copy['distance_from_home'] > 50).astype(int)

        # 복합 위험 지표
        if all(col in df_copy.columns for col in ['amount_vs_avg', 'is_international', 'is_night']):
            df_copy['risk_score'] = (
                (df_copy['amount_vs_avg'] > 3).astype(int) * 2 +
                df_copy['is_international'] +
                df_copy['is_night']
            )

        return df_copy

    def _handle_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """이상치 처리 (IQR 방식)"""
        df_copy = df.copy()

        for col in self.numeric_features:
            if col not in df_copy.columns or col not in self.feature_stats:
                continue

            stats = self.feature_stats[col]
            iqr = stats['q3'] - stats['q1']
            lower = stats['q1'] - 1.5 * iqr
            upper = stats['q3'] + 1.5 * iqr

            if method == 'clip':
                df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
            elif method == 'remove':
                df_copy = df_copy[(df_copy[col] >= lower) & (df_copy[col] <= upper)]

        return df_copy

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        df_copy = df.copy()

        for col in self.categorical_features:
            if col not in df_copy.columns or col not in self.label_encoders:
                continue

            le = self.label_encoders[col]

            # 학습 시 없던 카테고리는 -1로 처리
            df_copy[f'{col}_encoded'] = df_copy[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
            )

        # amount_category 인코딩 (피처 엔지니어링으로 생성된 경우)
        if 'amount_category' in df_copy.columns:
            if 'amount_category' not in self.label_encoders:
                le = LabelEncoder()
                le.fit(['소액', '일반', '고액', '초고액'])
                self.label_encoders['amount_category'] = le

            le = self.label_encoders['amount_category']
            df_copy['amount_category_encoded'] = df_copy['amount_category'].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
            )

        return df_copy

    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """수치형 변수 정규화"""
        df_copy = df.copy()

        # 원본 값 유지하고 정규화된 버전 추가
        numeric_cols = [c for c in self.numeric_features if c in df_copy.columns]
        scaled_values = self.scaler.transform(df_copy[numeric_cols].fillna(0))

        for i, col in enumerate(numeric_cols):
            df_copy[f'{col}_scaled'] = scaled_values[:, i]

        return df_copy

    def get_feature_columns(self, include_original: bool = False) -> list:
        """ML 모델 입력용 피처 컬럼 목록 반환"""
        features = []

        # 스케일된 수치형
        features.extend([f'{c}_scaled' for c in self.numeric_features])

        # 인코딩된 범주형
        features.extend([f'{c}_encoded' for c in self.categorical_features])

        # 이진형
        features.extend(self.binary_features)

        # 파생 피처
        features.extend(['hour', 'day_of_week', 'is_weekend', 'is_night', 'is_far', 'risk_score', 'log_amount'])

        if include_original:
            features.extend(self.numeric_features)

        return features

    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_fraud',
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        ML 학습용 데이터 준비

        Returns:
            X_train, X_test, y_train, y_test
        """
        # 전처리 적용
        df_processed = self.fit_transform(df)

        # 피처 선택
        feature_cols = self.get_feature_columns()
        available_features = [c for c in feature_cols if c in df_processed.columns]

        X = df_processed[available_features]
        y = df_processed[target_col]

        # 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터셋 통계 정보 반환"""
        stats = {
            'total_records': len(df),
            'fraud_count': df['is_fraud'].sum() if 'is_fraud' in df.columns else 0,
            'fraud_ratio': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': {},
            'categorical_stats': {},
        }

        # 수치형 통계
        for col in self.numeric_features:
            if col in df.columns:
                stats['numeric_stats'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                }

        # 범주형 통계
        for col in self.categorical_features:
            if col in df.columns:
                stats['categorical_stats'][col] = df[col].value_counts().to_dict()

        return stats


if __name__ == "__main__":
    # 테스트 실행
    from generator import generate_transaction_data

    # 샘플 데이터 생성
    df = generate_transaction_data(n_samples=1000)
    print(f"Original data shape: {df.shape}")

    # 전처리기 초기화 및 적용
    preprocessor = TransactionPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(df)

    print(f"\nAfter preprocessing:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Features: {X_train.columns.tolist()}")

    # 통계 출력
    stats = preprocessor.get_statistics(df)
    print(f"\nDataset statistics:")
    print(f"Total records: {stats['total_records']}")
    print(f"Fraud ratio: {stats['fraud_ratio']:.2%}")
