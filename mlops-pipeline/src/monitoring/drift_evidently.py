"""
Evidently 기반 모델 드리프트 감지 모듈
- 데이터 드리프트 분석
- 데이터 품질 모니터링
- 모델 성능 드리프트 추적
- Evidently 미설치 시 기존 drift.py 기반 폴백
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Evidently 선택적 임포트
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

# 폴백용 기존 drift 모듈
try:
    from .drift import DriftDetector, DriftResult
    HAS_FALLBACK_DRIFT = True
except ImportError:
    HAS_FALLBACK_DRIFT = False

logger = logging.getLogger(__name__)


@dataclass
class EvidentlyDriftReport:
    """Evidently 드리프트 분석 결과"""
    is_drift_detected: bool = False
    drift_share: float = 0.0
    number_of_drifted_features: int = 0
    total_features: int = 0
    feature_drifts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dataset_drift_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_drift_detected": self.is_drift_detected,
            "drift_share": self.drift_share,
            "number_of_drifted_features": self.number_of_drifted_features,
            "total_features": self.total_features,
            "feature_drifts": self.feature_drifts,
            "dataset_drift_score": self.dataset_drift_score,
            "timestamp": self.timestamp,
        }


@dataclass
class DataQualityReport:
    """데이터 품질 리포트"""
    total_rows: int = 0
    total_columns: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_rate: Dict[str, float] = field(default_factory=dict)
    duplicated_rows: int = 0
    duplicate_rate: float = 0.0
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values": self.missing_values,
            "missing_rate": self.missing_rate,
            "duplicated_rows": self.duplicated_rows,
            "duplicate_rate": self.duplicate_rate,
            "feature_stats": self.feature_stats,
            "timestamp": self.timestamp,
        }


@dataclass
class ModelPerformanceReport:
    """모델 성능 드리프트 리포트"""
    is_performance_drift: bool = False
    reference_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    metric_changes: Dict[str, float] = field(default_factory=dict)
    degraded_metrics: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_performance_drift": self.is_performance_drift,
            "reference_metrics": self.reference_metrics,
            "current_metrics": self.current_metrics,
            "metric_changes": self.metric_changes,
            "degraded_metrics": self.degraded_metrics,
            "timestamp": self.timestamp,
        }


class EvidentlyDriftDetector:
    """Evidently 기반 드리프트 감지"""

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.5,
        performance_threshold: float = 0.1,
        report_dir: str = "reports/drift",
    ):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.report_dir = report_dir
        self._fallback_detector: Optional[Any] = None

        # 폴백 디텍터 초기화
        if HAS_FALLBACK_DRIFT and reference_data is not None:
            self._fallback_detector = DriftDetector(reference_data=reference_data)

        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

    def set_reference_data(self, data: pd.DataFrame) -> None:
        """레퍼런스 데이터 설정"""
        self.reference_data = data
        if HAS_FALLBACK_DRIFT:
            self._fallback_detector = DriftDetector(reference_data=data)
        logger.info(f"레퍼런스 데이터 설정: {len(data)} samples, {len(data.columns)} features")

    def detect_data_drift(self, current_data: pd.DataFrame) -> EvidentlyDriftReport:
        """데이터 드리프트 감지"""
        if self.reference_data is None:
            raise ValueError("레퍼런스 데이터가 설정되지 않았습니다")

        if HAS_EVIDENTLY:
            return self._detect_drift_evidently(current_data)
        return self._detect_drift_fallback(current_data)

    def _detect_drift_evidently(self, current_data: pd.DataFrame) -> EvidentlyDriftReport:
        """Evidently 기반 드리프트 감지"""
        try:
            report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
            report.run(reference_data=self.reference_data, current_data=current_data)
            result = report.as_dict()

            metrics = result.get("metrics", [])
            dataset_drift = {}
            drift_table = {}

            for metric in metrics:
                metric_id = metric.get("metric", "")
                if "DatasetDriftMetric" in metric_id:
                    dataset_drift = metric.get("result", {})
                elif "DataDriftTable" in metric_id:
                    drift_table = metric.get("result", {})

            feature_drifts = {}
            drift_by_columns = drift_table.get("drift_by_columns", {})
            for col_name, col_data in drift_by_columns.items():
                feature_drifts[col_name] = {
                    "drift_detected": col_data.get("drift_detected", False),
                    "drift_score": col_data.get("drift_score", 0.0),
                    "stattest_name": col_data.get("stattest_name", ""),
                }

            return EvidentlyDriftReport(
                is_drift_detected=dataset_drift.get("dataset_drift", False),
                drift_share=dataset_drift.get("drift_share", 0.0),
                number_of_drifted_features=dataset_drift.get("number_of_drifted_columns", 0),
                total_features=dataset_drift.get("number_of_columns", 0),
                feature_drifts=feature_drifts,
                dataset_drift_score=dataset_drift.get("drift_share", 0.0),
            )
        except Exception as e:
            logger.warning(f"Evidently 드리프트 감지 실패: {e}, 폴백 사용")
            return self._detect_drift_fallback(current_data)

    def _detect_drift_fallback(self, current_data: pd.DataFrame) -> EvidentlyDriftReport:
        """폴백 드리프트 감지 (기존 drift.py 기반)"""
        if self._fallback_detector:
            result = self._fallback_detector.detect_data_drift(current_data)
            feature_drifts = {
                feat: {"drift_detected": bool(score > self._fallback_detector.psi_threshold), "drift_score": float(score)}
                for feat, score in result.feature_drifts.items()
            }
            drifted = sum(1 for v in feature_drifts.values() if v["drift_detected"])
            return EvidentlyDriftReport(
                is_drift_detected=bool(result.is_drift_detected),
                drift_share=float(drifted / len(feature_drifts)) if feature_drifts else 0.0,
                number_of_drifted_features=drifted,
                total_features=len(feature_drifts),
                feature_drifts=feature_drifts,
                dataset_drift_score=float(result.drift_score),
            )

        # 최소 폴백: 기본 통계 비교
        return self._detect_drift_basic(current_data)

    def _detect_drift_basic(self, current_data: pd.DataFrame) -> EvidentlyDriftReport:
        """기본 통계 기반 드리프트 감지"""
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in current_data.columns]

        feature_drifts = {}
        drifted_count = 0

        for col in common_cols:
            ref_mean = self.reference_data[col].mean()
            cur_mean = current_data[col].mean()
            ref_std = self.reference_data[col].std()

            if ref_std > 0:
                drift_score = abs(cur_mean - ref_mean) / ref_std
            else:
                drift_score = 0.0 if cur_mean == ref_mean else 1.0

            is_drifted = bool(drift_score > 2.0)  # 2 sigma
            if is_drifted:
                drifted_count += 1

            feature_drifts[col] = {
                "drift_detected": is_drifted,
                "drift_score": round(float(drift_score), 4),
            }

        total = len(common_cols)
        drift_share = drifted_count / total if total > 0 else 0.0

        return EvidentlyDriftReport(
            is_drift_detected=bool(drift_share > self.drift_threshold),
            drift_share=round(drift_share, 4),
            number_of_drifted_features=drifted_count,
            total_features=total,
            feature_drifts=feature_drifts,
            dataset_drift_score=round(drift_share, 4),
        )

    def detect_feature_drift(self, current_data: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
        """특정 피처 드리프트 감지"""
        if self.reference_data is None:
            raise ValueError("레퍼런스 데이터가 설정되지 않았습니다")

        if feature_name not in self.reference_data.columns:
            raise ValueError(f"피처를 찾을 수 없습니다: {feature_name}")
        if feature_name not in current_data.columns:
            raise ValueError(f"현재 데이터에 피처가 없습니다: {feature_name}")

        if HAS_EVIDENTLY:
            try:
                report = Report(metrics=[ColumnDriftMetric(column_name=feature_name)])
                report.run(reference_data=self.reference_data, current_data=current_data)
                result = report.as_dict()
                metrics = result.get("metrics", [])
                if metrics:
                    col_result = metrics[0].get("result", {})
                    return {
                        "feature": feature_name,
                        "drift_detected": col_result.get("drift_detected", False),
                        "drift_score": col_result.get("drift_score", 0.0),
                        "stattest_name": col_result.get("stattest_name", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Evidently 피처 드리프트 감지 실패: {e}")

        # 폴백
        ref_values = self.reference_data[feature_name].dropna()
        cur_values = current_data[feature_name].dropna()
        ref_std = ref_values.std()
        drift_score = abs(cur_values.mean() - ref_values.mean()) / ref_std if ref_std > 0 else 0.0

        return {
            "feature": feature_name,
            "drift_detected": bool(drift_score > 2.0),
            "drift_score": round(float(drift_score), 4),
            "stattest_name": "z-score",
            "timestamp": datetime.now().isoformat(),
        }

    def detect_target_drift(
        self,
        reference_target: pd.Series,
        current_target: pd.Series,
    ) -> Dict[str, Any]:
        """타겟 드리프트 감지"""
        ref_mean = reference_target.mean()
        cur_mean = current_target.mean()
        ref_std = reference_target.std()

        if ref_std > 0:
            drift_score = abs(cur_mean - ref_mean) / ref_std
        else:
            drift_score = 0.0 if cur_mean == ref_mean else 1.0

        return {
            "target_drift_detected": bool(drift_score > 2.0),
            "drift_score": round(float(drift_score), 4),
            "reference_mean": round(float(ref_mean), 4),
            "current_mean": round(float(cur_mean), 4),
            "reference_std": round(float(ref_std), 4),
            "timestamp": datetime.now().isoformat(),
        }

    def analyze_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """데이터 품질 분석"""
        missing_values = data.isnull().sum().to_dict()
        missing_rate = (data.isnull().sum() / len(data)).to_dict() if len(data) > 0 else {}

        # 수치형 통계
        feature_stats = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 0:
                feature_stats[col] = {
                    "mean": round(float(values.mean()), 4),
                    "std": round(float(values.std()), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                    "median": round(float(values.median()), 4),
                }

        duplicated = int(data.duplicated().sum())

        return DataQualityReport(
            total_rows=len(data),
            total_columns=len(data.columns),
            missing_values={k: int(v) for k, v in missing_values.items()},
            missing_rate={k: round(float(v), 4) for k, v in missing_rate.items()},
            duplicated_rows=duplicated,
            duplicate_rate=round(duplicated / len(data), 4) if len(data) > 0 else 0.0,
            feature_stats=feature_stats,
        )

    def detect_model_drift(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
    ) -> ModelPerformanceReport:
        """모델 성능 드리프트 감지"""
        metric_changes = {}
        degraded = []

        for metric_name in reference_metrics:
            if metric_name not in current_metrics:
                continue
            ref_val = reference_metrics[metric_name]
            cur_val = current_metrics[metric_name]

            if ref_val != 0:
                change = (cur_val - ref_val) / abs(ref_val)
            else:
                change = 0.0 if cur_val == 0 else 1.0

            metric_changes[metric_name] = round(float(change), 4)

            # 성능 저하 감지 (높을수록 좋은 메트릭)
            if metric_name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
                if change < -self.performance_threshold:
                    degraded.append(metric_name)

        return ModelPerformanceReport(
            is_performance_drift=len(degraded) > 0,
            reference_metrics=reference_metrics,
            current_metrics=current_metrics,
            metric_changes=metric_changes,
            degraded_metrics=degraded,
        )

    def generate_drift_dashboard(self, current_data: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """드리프트 대시보드 생성"""
        if self.reference_data is None:
            raise ValueError("레퍼런스 데이터가 설정되지 않았습니다")

        if output_path is None:
            output_path = str(Path(self.report_dir) / f"drift-dashboard-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html")

        if HAS_EVIDENTLY:
            try:
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=self.reference_data, current_data=current_data)
                report.save_html(output_path)
                logger.info(f"드리프트 대시보드 생성: {output_path}")
                return output_path
            except Exception as e:
                logger.warning(f"Evidently 대시보드 생성 실패: {e}")

        # 폴백: JSON 리포트
        drift_report = self.detect_data_drift(current_data)
        json_path = output_path.replace(".html", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(drift_report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"폴백 드리프트 리포트 생성: {json_path}")
        return json_path

    def generate_data_quality_dashboard(self, data: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """데이터 품질 대시보드 생성"""
        if output_path is None:
            output_path = str(Path(self.report_dir) / f"quality-dashboard-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html")

        if HAS_EVIDENTLY:
            try:
                report = Report(metrics=[DataQualityPreset()])
                report.run(reference_data=self.reference_data, current_data=data)
                report.save_html(output_path)
                logger.info(f"데이터 품질 대시보드 생성: {output_path}")
                return output_path
            except Exception as e:
                logger.warning(f"Evidently 품질 대시보드 생성 실패: {e}")

        # 폴백: JSON 리포트
        quality_report = self.analyze_data_quality(data)
        json_path = output_path.replace(".html", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(quality_report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"폴백 품질 리포트 생성: {json_path}")
        return json_path

    def create_monitoring_suite(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """통합 모니터링 스위트 실행"""
        if self.reference_data is None:
            raise ValueError("레퍼런스 데이터가 설정되지 않았습니다")

        if HAS_EVIDENTLY:
            try:
                suite = TestSuite(tests=[DataDriftTestPreset(), DataQualityTestPreset()])
                suite.run(reference_data=self.reference_data, current_data=current_data)
                result = suite.as_dict()

                tests = result.get("tests", [])
                passed = sum(1 for t in tests if t.get("status") == "SUCCESS")
                failed = sum(1 for t in tests if t.get("status") == "FAIL")

                return {
                    "total_tests": len(tests),
                    "passed": passed,
                    "failed": failed,
                    "success_rate": round(passed / len(tests), 4) if tests else 0.0,
                    "test_results": tests,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Evidently 모니터링 스위트 실패: {e}")

        # 폴백
        drift_report = self.detect_data_drift(current_data)
        quality_report = self.analyze_data_quality(current_data)

        tests_passed = 0
        tests_total = 0

        # 드리프트 테스트
        tests_total += 1
        if not drift_report.is_drift_detected:
            tests_passed += 1

        # 품질 테스트
        for col, rate in quality_report.missing_rate.items():
            tests_total += 1
            if rate < 0.1:  # 결측 10% 미만
                tests_passed += 1

        return {
            "total_tests": tests_total,
            "passed": tests_passed,
            "failed": tests_total - tests_passed,
            "success_rate": round(tests_passed / tests_total, 4) if tests_total > 0 else 0.0,
            "drift_report": drift_report.to_dict(),
            "quality_report": quality_report.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    def save_report(self, report: Any, path: str) -> None:
        """리포트 저장"""
        data = report.to_dict() if hasattr(report, "to_dict") else report
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"리포트 저장: {path}")
