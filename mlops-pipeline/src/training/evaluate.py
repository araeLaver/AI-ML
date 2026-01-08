"""
Model Evaluation Module
- 다양한 평가 메트릭
- 시각화
- MLflow 메트릭 로깅
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from typing import Dict, Any, Optional, Tuple
import logging
import json
import mlflow

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """모델 평가 클래스"""

    def __init__(self, mlflow_tracking: bool = True):
        self.mlflow_tracking = mlflow_tracking
        self.metrics: Dict[str, float] = {}
        self.confusion_mat: Optional[np.ndarray] = None
        self.classification_rep: Optional[str] = None

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """모델 평가"""
        # 기본 메트릭
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        # 확률 기반 메트릭 (y_prob가 있는 경우)
        if y_prob is not None:
            self.metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            self.metrics["pr_auc"] = average_precision_score(y_true, y_prob)

        # 혼동 행렬
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = self.confusion_mat.ravel()

        # 추가 메트릭
        self.metrics["true_positives"] = int(tp)
        self.metrics["true_negatives"] = int(tn)
        self.metrics["false_positives"] = int(fp)
        self.metrics["false_negatives"] = int(fn)
        self.metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Classification Report
        self.classification_rep = classification_report(
            y_true, y_pred, target_names=["Normal", "Fraud"]
        )

        # 로깅
        logger.info("=== 모델 평가 결과 ===")
        logger.info(f"Accuracy:    {self.metrics['accuracy']:.4f}")
        logger.info(f"Precision:   {self.metrics['precision']:.4f}")
        logger.info(f"Recall:      {self.metrics['recall']:.4f}")
        logger.info(f"F1 Score:    {self.metrics['f1_score']:.4f}")
        if y_prob is not None:
            logger.info(f"ROC AUC:     {self.metrics['roc_auc']:.4f}")
            logger.info(f"PR AUC:      {self.metrics['pr_auc']:.4f}")

        return self.metrics

    def log_to_mlflow(self, run_name: str = "evaluation") -> None:
        """MLflow에 메트릭 로깅"""
        if not self.mlflow_tracking:
            return

        with mlflow.start_run(run_name=run_name, nested=True):
            # 메트릭 로깅
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # 혼동 행렬 저장
            if self.confusion_mat is not None:
                cm_dict = {
                    "true_negatives": int(self.confusion_mat[0, 0]),
                    "false_positives": int(self.confusion_mat[0, 1]),
                    "false_negatives": int(self.confusion_mat[1, 0]),
                    "true_positives": int(self.confusion_mat[1, 1]),
                }
                mlflow.log_dict(cm_dict, "confusion_matrix.json")

            # Classification Report 저장
            if self.classification_rep:
                mlflow.log_text(self.classification_rep, "classification_report.txt")

            logger.info("MLflow에 평가 결과 로깅 완료")

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
    ) -> Tuple[float, float]:
        """최적 임계값 찾기"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"지원하지 않는 메트릭: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(
            f"최적 임계값 ({metric}): {best_threshold:.2f}, Score: {best_score:.4f}"
        )
        return best_threshold, best_score

    def evaluate_at_thresholds(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> pd.DataFrame:
        """다양한 임계값에서 평가"""
        results = []
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            results.append(
                {
                    "threshold": threshold,
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, zero_division=0),
                    "predicted_positive": y_pred.sum(),
                }
            )

        return pd.DataFrame(results)

    def get_roc_curve_data(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """ROC 커브 데이터 반환"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    def get_pr_curve_data(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """PR 커브 데이터 반환"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return {"precision": precision, "recall": recall, "thresholds": thresholds}

    def compare_models(
        self, results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """여러 모델 비교"""
        df = pd.DataFrame(results).T
        df = df.sort_values("f1_score", ascending=False)
        return df

    def get_summary(self) -> Dict[str, Any]:
        """평가 결과 요약"""
        return {
            "metrics": self.metrics,
            "confusion_matrix": (
                self.confusion_mat.tolist() if self.confusion_mat is not None else None
            ),
            "classification_report": self.classification_rep,
        }

    def save_report(self, path: str) -> None:
        """평가 리포트 저장"""
        report = {
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "confusion_matrix": (
                self.confusion_mat.tolist() if self.confusion_mat is not None else None
            ),
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"평가 리포트 저장: {path}")
