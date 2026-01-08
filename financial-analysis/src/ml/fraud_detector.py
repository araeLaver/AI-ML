# Fraud Detection Model
"""
금융 이상거래 탐지 ML 모델 (Step 1: scikit-learn 활용)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import pickle
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score


@dataclass
class PredictionResult:
    """예측 결과 데이터 클래스"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str  # low, medium, high, critical
    explanation: Dict[str, Any]


class FraudDetector:
    """
    금융 이상거래 탐지 모델

    지원 알고리즘:
    - RandomForest (기본)
    - GradientBoosting
    - LogisticRegression
    - IsolationForest (비지도)
    """

    ALGORITHMS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'logistic_regression': LogisticRegression,
        'isolation_forest': IsolationForest,
    }

    def __init__(
        self,
        algorithm: str = 'random_forest',
        threshold: float = 0.5,
        **model_params
    ):
        """
        Args:
            algorithm: 사용할 알고리즘
            threshold: 이상거래 판정 임계값
            **model_params: 모델 하이퍼파라미터
        """
        self.algorithm = algorithm
        self.threshold = threshold
        self.model_params = model_params
        self.model = None
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self._is_fitted = False

        # 기본 하이퍼파라미터
        self._default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            },
            'logistic_regression': {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'isolation_forest': {
                'n_estimators': 100,
                'contamination': 0.02,
                'random_state': 42,
            },
        }

    def _create_model(self):
        """모델 인스턴스 생성"""
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # 기본 파라미터 + 사용자 파라미터
        params = self._default_params.get(self.algorithm, {}).copy()
        params.update(self.model_params)

        model_class = self.ALGORITHMS[self.algorithm]
        self.model = model_class(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> 'FraudDetector':
        """
        모델 학습

        Args:
            X: 피처 데이터
            y: 타겟 (IsolationForest는 불필요)
        """
        self._create_model()
        self.feature_columns = list(X.columns)

        if self.algorithm == 'isolation_forest':
            self.model.fit(X)
        else:
            if y is None:
                raise ValueError("Target variable y is required for supervised learning")
            self.model.fit(X, y)

            # 피처 중요도 저장 (가능한 경우)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_columns,
                    self.model.feature_importances_
                ))

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        이상거래 여부 예측 (0/1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if self.algorithm == 'isolation_forest':
            # IsolationForest: -1 (이상) -> 1, 1 (정상) -> 0
            predictions = self.model.predict(X)
            return (predictions == -1).astype(int)
        else:
            # 확률 기반 예측
            probas = self.predict_proba(X)
            return (probas >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        이상거래 확률 예측
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if self.algorithm == 'isolation_forest':
            # IsolationForest는 decision_function 사용
            scores = self.model.decision_function(X)
            # 점수를 0-1 확률로 변환 (낮을수록 이상)
            probas = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            return probas
        else:
            return self.model.predict_proba(X)[:, 1]

    def predict_single(
        self,
        transaction: Dict[str, Any],
        feature_df: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """
        단일 거래 예측 (설명 포함)

        Args:
            transaction: 거래 정보 딕셔너리
            feature_df: 전처리된 피처 DataFrame (1행)

        Returns:
            PredictionResult 객체
        """
        if feature_df is None:
            raise ValueError("Preprocessed feature DataFrame is required")

        # 예측
        proba = self.predict_proba(feature_df)[0]
        is_fraud = proba >= self.threshold

        # 리스크 레벨 결정
        if proba < 0.3:
            risk_level = 'low'
        elif proba < 0.5:
            risk_level = 'medium'
        elif proba < 0.8:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        # 설명 생성
        explanation = self._generate_explanation(transaction, feature_df, proba)

        return PredictionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            is_fraud=bool(is_fraud),
            fraud_probability=float(proba),
            risk_level=risk_level,
            explanation=explanation,
        )

    def _generate_explanation(
        self,
        transaction: Dict[str, Any],
        feature_df: pd.DataFrame,
        proba: float,
    ) -> Dict[str, Any]:
        """예측 설명 생성"""
        explanation = {
            'risk_factors': [],
            'feature_contributions': {},
            'summary': '',
        }

        # 주요 위험 요소 분석
        risk_factors = []

        if transaction.get('amount', 0) > 1000000:
            risk_factors.append({
                'factor': '고액 거래',
                'value': f"{transaction['amount']:,}원",
                'weight': 0.3,
            })

        if transaction.get('is_international', 0) == 1:
            risk_factors.append({
                'factor': '해외 거래',
                'value': '예',
                'weight': 0.2,
            })

        if transaction.get('distance_from_home', 0) > 100:
            risk_factors.append({
                'factor': '원거리 거래',
                'value': f"{transaction['distance_from_home']}km",
                'weight': 0.15,
            })

        if transaction.get('amount_vs_avg', 1) > 5:
            risk_factors.append({
                'factor': '평균 대비 고액',
                'value': f"{transaction['amount_vs_avg']}배",
                'weight': 0.25,
            })

        # 시간대 분석
        hour = transaction.get('hour', 12)
        if hour >= 22 or hour <= 5:
            risk_factors.append({
                'factor': '비정상 시간대',
                'value': f"{hour}시",
                'weight': 0.1,
            })

        explanation['risk_factors'] = sorted(
            risk_factors,
            key=lambda x: x['weight'],
            reverse=True
        )

        # 피처 중요도 기반 기여도
        if self.feature_importance:
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            explanation['feature_contributions'] = dict(top_features)

        # 요약 생성
        if proba >= 0.8:
            explanation['summary'] = '매우 높은 이상거래 위험. 즉각적인 확인이 필요합니다.'
        elif proba >= 0.5:
            explanation['summary'] = '이상거래 의심. 추가 검토를 권장합니다.'
        elif proba >= 0.3:
            explanation['summary'] = '경미한 위험 징후. 모니터링을 계속하세요.'
        else:
            explanation['summary'] = '정상 거래로 판단됩니다.'

        return explanation

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        모델 평가

        Returns:
            평가 지표 딕셔너리
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, probabilities),
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
            'classification_report': classification_report(y, predictions, output_dict=True),
        }

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """교차 검증"""
        if self.algorithm == 'isolation_forest':
            raise ValueError("Cross-validation is not available for IsolationForest")

        self._create_model()

        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')

        return {
            'cv_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
        }

    def save(self, path: str):
        """모델 저장"""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        model_data = {
            'model': self.model,
            'algorithm': self.algorithm,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: str) -> 'FraudDetector':
        """모델 로드"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        detector = cls(
            algorithm=model_data['algorithm'],
            threshold=model_data['threshold'],
        )
        detector.model = model_data['model']
        detector.feature_columns = model_data['feature_columns']
        detector.feature_importance = model_data['feature_importance']
        detector._is_fitted = True

        return detector

    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """상위 N개 피처 중요도 반환"""
        if not self.feature_importance:
            return []

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]


if __name__ == "__main__":
    # 테스트 실행
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.generator import generate_transaction_data
    from data.preprocessor import TransactionPreprocessor

    # 데이터 생성
    print("Generating sample data...")
    df = generate_transaction_data(n_samples=5000, fraud_ratio=0.03)

    # 전처리
    print("Preprocessing data...")
    preprocessor = TransactionPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(df)

    # 모델 학습
    print("\nTraining RandomForest model...")
    detector = FraudDetector(algorithm='random_forest')
    detector.fit(X_train, y_train)

    # 평가
    print("\nEvaluating model...")
    metrics = detector.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    # 피처 중요도
    print("\nTop 10 Feature Importance:")
    for feature, importance in detector.get_feature_importance(10):
        print(f"  {feature}: {importance:.4f}")
