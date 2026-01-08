"""
Model Training Module
- 다양한 ML 모델 학습
- 하이퍼파라미터 튜닝
- MLflow 연동
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path
import logging
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class ModelTrainer:
    """모델 학습 클래스"""

    AVAILABLE_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }

    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "random_state": 42,
        },
        "logistic_regression": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
    }

    PARAM_GRIDS = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
        },
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        params: Optional[Dict[str, Any]] = None,
        mlflow_tracking: bool = True,
    ):
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"지원하지 않는 모델: {model_type}. 사용 가능: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS[model_type]
        self.mlflow_tracking = mlflow_tracking
        self.model = None
        self.best_params: Dict[str, Any] = {}
        self.cv_scores: Dict[str, float] = {}

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, experiment_name: str = "fraud-detection"
    ) -> "ModelTrainer":
        """모델 학습"""
        logger.info(f"모델 학습 시작: {self.model_type}")
        logger.info(f"학습 데이터: {X_train.shape}, Fraud ratio: {y_train.mean():.4f}")

        # MLflow 설정
        if self.mlflow_tracking:
            mlflow.set_experiment(experiment_name)

        # 모델 생성 및 학습
        model_class = self.AVAILABLE_MODELS[self.model_type]
        self.model = model_class(**self.params)

        if self.mlflow_tracking:
            with mlflow.start_run(run_name=f"{self.model_type}_train"):
                # 파라미터 로깅
                mlflow.log_params(self.params)
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("fraud_ratio", y_train.mean())

                # 학습
                self.model.fit(X_train, y_train)

                # Cross-validation 점수
                cv_scores = cross_val_score(
                    self.model, X_train, y_train, cv=5, scoring="roc_auc"
                )
                self.cv_scores = {
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                }

                mlflow.log_metrics(self.cv_scores)

                # 모델 저장
                mlflow.sklearn.log_model(
                    self.model, "model", registered_model_name="fraud-detector"
                )

                logger.info(
                    f"MLflow run ID: {mlflow.active_run().info.run_id}"
                )
        else:
            self.model.fit(X_train, y_train)
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=5, scoring="roc_auc"
            )
            self.cv_scores = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }

        logger.info(
            f"학습 완료. CV ROC-AUC: {self.cv_scores['cv_mean']:.4f} (+/- {self.cv_scores['cv_std']:.4f})"
        )
        return self

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, Any]:
        """하이퍼파라미터 튜닝"""
        logger.info(f"하이퍼파라미터 튜닝 시작: {self.model_type}")

        model_class = self.AVAILABLE_MODELS[self.model_type]
        base_model = model_class(random_state=42)

        param_grid = self.PARAM_GRIDS[self.model_type]

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        logger.info(f"최적 파라미터: {self.best_params}")
        logger.info(f"최고 점수: {grid_search.best_score_:.4f}")

        if self.mlflow_tracking:
            with mlflow.start_run(run_name=f"{self.model_type}_tuned"):
                mlflow.log_params(self.best_params)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                mlflow.sklearn.log_model(
                    self.model, "model", registered_model_name="fraud-detector-tuned"
                )

        return self.best_params

    def get_feature_importance(
        self, feature_names: list
    ) -> Optional[pd.DataFrame]:
        """피처 중요도 반환"""
        if self.model is None:
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            return None

        df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return df

    def save(self, path: str) -> None:
        """모델 저장"""
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "params": self.params,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
        }

        joblib.dump(model_data, save_path)
        logger.info(f"모델 저장: {save_path}")

    @classmethod
    def load(cls, path: str) -> "ModelTrainer":
        """모델 로드"""
        data = joblib.load(path)

        trainer = cls(
            model_type=data["model_type"],
            params=data["params"],
            mlflow_tracking=False,
        )
        trainer.model = data["model"]
        trainer.best_params = data["best_params"]
        trainer.cv_scores = data["cv_scores"]

        logger.info(f"모델 로드: {path}")
        return trainer

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다")
        return self.model.predict_proba(X)
