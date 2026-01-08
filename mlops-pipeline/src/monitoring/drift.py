"""
Data/Model Drift Detection Module
- 데이터 드리프트 감지
- 모델 성능 드리프트 감지
- 알림 트리거
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """드리프트 감지 결과"""

    is_drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """드리프트 감지 클래스"""

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.1,
        psi_threshold: float = 0.2,
    ):
        self.reference_data = reference_data
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold

        if reference_data is not None:
            self._compute_reference_stats()

    def set_reference_data(self, data: pd.DataFrame) -> None:
        """레퍼런스 데이터 설정"""
        self.reference_data = data
        self._compute_reference_stats()
        logger.info(f"레퍼런스 데이터 설정: {len(data)} samples")

    def _compute_reference_stats(self) -> None:
        """레퍼런스 통계 계산"""
        if self.reference_data is None:
            return

        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = self.reference_data[col].dropna()
            self.reference_stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "q25": float(values.quantile(0.25)),
                "q50": float(values.quantile(0.50)),
                "q75": float(values.quantile(0.75)),
            }

    def detect_data_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """데이터 드리프트 감지"""
        if self.reference_data is None:
            raise ValueError("레퍼런스 데이터가 설정되지 않았습니다")

        feature_drifts = {}
        drift_details = {}

        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in self.reference_data.columns]

        for col in common_cols:
            ref_values = self.reference_data[col].dropna()
            cur_values = current_data[col].dropna()

            # KS Test (Kolmogorov-Smirnov)
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)

            # PSI (Population Stability Index)
            psi = self._calculate_psi(ref_values, cur_values)

            feature_drifts[col] = psi
            drift_details[col] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "psi": float(psi),
                "ref_mean": float(ref_values.mean()),
                "cur_mean": float(cur_values.mean()),
                "mean_shift": float(cur_values.mean() - ref_values.mean()),
            }

        # 전체 드리프트 점수 (평균 PSI)
        avg_psi = np.mean(list(feature_drifts.values())) if feature_drifts else 0
        is_drift = avg_psi > self.psi_threshold

        result = DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(avg_psi),
            feature_drifts=feature_drifts,
            details=drift_details,
        )

        if is_drift:
            logger.warning(f"데이터 드리프트 감지! PSI: {avg_psi:.4f}")
            drifted_features = [f for f, v in feature_drifts.items() if v > self.psi_threshold]
            logger.warning(f"드리프트 피처: {drifted_features}")

        return result

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """PSI (Population Stability Index) 계산"""
        # 레퍼런스 데이터 기반 버킷 생성
        try:
            _, bins = pd.qcut(reference, n_bins, retbins=True, duplicates="drop")
        except ValueError:
            # 고유값이 적은 경우
            bins = np.linspace(reference.min(), reference.max(), n_bins + 1)

        # 각 버킷의 비율 계산
        ref_counts = pd.cut(reference, bins=bins, include_lowest=True).value_counts(normalize=True)
        cur_counts = pd.cut(current, bins=bins, include_lowest=True).value_counts(normalize=True)

        # 인덱스 정렬
        ref_counts = ref_counts.sort_index()
        cur_counts = cur_counts.reindex(ref_counts.index, fill_value=0.0001)

        # PSI 계산
        ref_pct = ref_counts.values + 0.0001  # 0 방지
        cur_pct = cur_counts.values + 0.0001

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> DriftResult:
        """예측 분포 드리프트 감지"""
        # KS Test
        ks_stat, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)

        # 평균/분산 비교
        ref_mean = np.mean(reference_predictions)
        cur_mean = np.mean(current_predictions)
        ref_std = np.std(reference_predictions)
        cur_std = np.std(current_predictions)

        # PSI
        psi = self._calculate_psi(
            pd.Series(reference_predictions), pd.Series(current_predictions)
        )

        is_drift = psi > self.psi_threshold or ks_pvalue < 0.05

        result = DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(psi),
            feature_drifts={"predictions": float(psi)},
            details={
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "psi": float(psi),
                "ref_mean": float(ref_mean),
                "cur_mean": float(cur_mean),
                "ref_std": float(ref_std),
                "cur_std": float(cur_std),
            },
        )

        if is_drift:
            logger.warning(f"예측 드리프트 감지! PSI: {psi:.4f}, KS p-value: {ks_pvalue:.4f}")

        return result

    def detect_performance_drift(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        threshold_pct: float = 0.1,
    ) -> DriftResult:
        """성능 드리프트 감지"""
        metric_drifts = {}
        is_drift = False

        for metric_name in reference_metrics:
            if metric_name not in current_metrics:
                continue

            ref_value = reference_metrics[metric_name]
            cur_value = current_metrics[metric_name]

            if ref_value == 0:
                change_pct = 0 if cur_value == 0 else 1
            else:
                change_pct = abs(cur_value - ref_value) / ref_value

            metric_drifts[metric_name] = float(change_pct)

            # 성능 저하 감지 (특정 메트릭들)
            if metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                if cur_value < ref_value * (1 - threshold_pct):
                    is_drift = True
                    logger.warning(
                        f"성능 저하 감지 - {metric_name}: {ref_value:.4f} → {cur_value:.4f}"
                    )

        avg_drift = np.mean(list(metric_drifts.values())) if metric_drifts else 0

        return DriftResult(
            is_drift_detected=is_drift,
            drift_score=float(avg_drift),
            feature_drifts=metric_drifts,
            details={
                "reference_metrics": reference_metrics,
                "current_metrics": current_metrics,
            },
        )

    def get_drift_report(self, result: DriftResult) -> str:
        """드리프트 리포트 생성"""
        report = []
        report.append("=" * 50)
        report.append("드리프트 감지 리포트")
        report.append("=" * 50)
        report.append(f"시간: {result.timestamp}")
        report.append(f"드리프트 감지: {'예' if result.is_drift_detected else '아니오'}")
        report.append(f"드리프트 점수: {result.drift_score:.4f}")
        report.append("")
        report.append("피처별 드리프트:")

        for feature, score in sorted(
            result.feature_drifts.items(), key=lambda x: x[1], reverse=True
        ):
            status = "⚠️" if score > self.psi_threshold else "✓"
            report.append(f"  {status} {feature}: {score:.4f}")

        report.append("=" * 50)
        return "\n".join(report)

    def save_result(self, result: DriftResult, path: str) -> None:
        """결과 저장"""
        data = {
            "is_drift_detected": result.is_drift_detected,
            "drift_score": result.drift_score,
            "feature_drifts": result.feature_drifts,
            "timestamp": result.timestamp,
            "details": result.details,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"드리프트 결과 저장: {path}")
