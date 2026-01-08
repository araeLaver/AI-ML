"""
Training Pipeline Tests
"""

import pytest
import numpy as np
import pandas as pd

from src.training import ModelTrainer, ModelEvaluator


class TestModelTrainer:
    """ModelTrainer 테스트"""

    def test_train_random_forest(self, sample_dataframe):
        """Random Forest 학습 테스트"""
        from src.data import DataPreprocessor

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        trainer = ModelTrainer(model_type="random_forest", mlflow_tracking=False)
        trainer.train(X, y)

        assert trainer.model is not None
        assert "cv_mean" in trainer.cv_scores

    def test_train_logistic_regression(self, sample_dataframe):
        """Logistic Regression 학습 테스트"""
        from src.data import DataPreprocessor

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        trainer = ModelTrainer(model_type="logistic_regression", mlflow_tracking=False)
        trainer.train(X, y)

        assert trainer.model is not None

    def test_invalid_model_type(self):
        """잘못된 모델 타입 테스트"""
        with pytest.raises(ValueError):
            ModelTrainer(model_type="invalid_model")

    def test_predict(self, trained_model, sample_dataframe):
        """예측 테스트"""
        trainer, preprocessor = trained_model
        X, _ = preprocessor.prepare_features(sample_dataframe, include_target=True)

        predictions = trainer.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, trained_model, sample_dataframe):
        """확률 예측 테스트"""
        trainer, preprocessor = trained_model
        X, _ = preprocessor.prepare_features(sample_dataframe, include_target=True)

        probabilities = trainer.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_feature_importance(self, trained_model, sample_dataframe):
        """피처 중요도 테스트"""
        trainer, preprocessor = trained_model
        X, _ = preprocessor.prepare_features(sample_dataframe, include_target=True)

        importance = trainer.get_feature_importance(X.columns.tolist())

        assert importance is not None
        assert "feature" in importance.columns
        assert "importance" in importance.columns


class TestModelEvaluator:
    """ModelEvaluator 테스트"""

    def test_evaluate(self, trained_model, sample_dataframe):
        """평가 테스트"""
        trainer, preprocessor = trained_model
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        y_pred = trainer.predict(X)
        y_prob = trainer.predict_proba(X)[:, 1]

        evaluator = ModelEvaluator(mlflow_tracking=False)
        metrics = evaluator.evaluate(y.values, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    def test_find_optimal_threshold(self, trained_model, sample_dataframe):
        """최적 임계값 테스트"""
        trainer, preprocessor = trained_model
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        y_prob = trainer.predict_proba(X)[:, 1]

        evaluator = ModelEvaluator(mlflow_tracking=False)
        threshold, score = evaluator.find_optimal_threshold(y.values, y_prob)

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_evaluate_at_thresholds(self, trained_model, sample_dataframe):
        """다양한 임계값 평가 테스트"""
        trainer, preprocessor = trained_model
        X, y = preprocessor.prepare_features(sample_dataframe, include_target=True)

        y_prob = trainer.predict_proba(X)[:, 1]

        evaluator = ModelEvaluator(mlflow_tracking=False)
        results = evaluator.evaluate_at_thresholds(y.values, y_prob)

        assert len(results) == 9  # 0.1 ~ 0.9
        assert "threshold" in results.columns
        assert "precision" in results.columns
        assert "recall" in results.columns
