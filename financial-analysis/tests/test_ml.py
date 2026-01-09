# Tests for ML Module
"""
머신러닝 모델 테스트
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


class TestFraudDetector:
    """FraudDetector 테스트"""

    def test_detector_creation(self):
        """모델 생성 테스트"""
        from src.ml.fraud_detector import FraudDetector

        detector = FraudDetector(algorithm='random_forest')
        assert detector is not None
        assert detector.algorithm == 'random_forest'

    def test_supported_algorithms(self):
        """지원 알고리즘 테스트"""
        from src.ml.fraud_detector import FraudDetector

        algorithms = ['random_forest', 'gradient_boosting', 'logistic_regression', 'isolation_forest']

        for algo in algorithms:
            detector = FraudDetector(algorithm=algo)
            assert detector.algorithm == algo

    def test_invalid_algorithm(self):
        """유효하지 않은 알고리즘 테스트"""
        from src.ml.fraud_detector import FraudDetector

        detector = FraudDetector(algorithm='invalid_algo')
        with pytest.raises(ValueError):
            detector._create_model()

    def test_model_training(self, sample_dataframe):
        """모델 학습 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='random_forest')
        detector.fit(X_train, y_train)

        assert detector._is_fitted
        print("Model trained successfully")

    def test_model_prediction(self, sample_dataframe):
        """모델 예측 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='random_forest')
        detector.fit(X_train, y_train)

        predictions = detector.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_dataframe):
        """확률 예측 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='random_forest')
        detector.fit(X_train, y_train)

        probas = detector.predict_proba(X_test)

        assert len(probas) == len(X_test)
        assert all(0 <= p <= 1 for p in probas)

    def test_model_evaluation(self, sample_dataframe):
        """모델 평가 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='random_forest')
        detector.fit(X_train, y_train)

        metrics = detector.evaluate(X_test, y_test)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    def test_feature_importance(self, sample_dataframe):
        """피처 중요도 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='random_forest')
        detector.fit(X_train, y_train)

        importance = detector.get_feature_importance(top_n=5)

        assert len(importance) <= 5
        if importance:
            assert all(isinstance(f, tuple) and len(f) == 2 for f in importance)

    def test_model_save_load(self, sample_dataframe):
        """모델 저장/로드 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='random_forest')
        detector.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            detector.save(str(model_path))

            loaded_detector = FraudDetector.load(str(model_path))

            assert loaded_detector._is_fitted
            assert loaded_detector.algorithm == detector.algorithm

            # 동일한 예측 결과
            orig_preds = detector.predict(X_test)
            loaded_preds = loaded_detector.predict(X_test)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_predict_without_fit(self):
        """학습 전 예측 시도 테스트"""
        from src.ml.fraud_detector import FraudDetector

        detector = FraudDetector(algorithm='random_forest')

        with pytest.raises(RuntimeError):
            detector.predict(pd.DataFrame({'a': [1, 2, 3]}))

    def test_isolation_forest(self, sample_dataframe):
        """IsolationForest 알고리즘 테스트"""
        from src.ml.fraud_detector import FraudDetector
        from src.data.preprocessor import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(sample_dataframe)

        detector = FraudDetector(algorithm='isolation_forest')
        detector.fit(X_train)  # y 불필요

        predictions = detector.predict(X_test)
        assert len(predictions) == len(X_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
