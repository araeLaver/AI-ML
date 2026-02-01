"""
Evidently 드리프트 감지 모듈 테스트
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.monitoring.drift_evidently import (
    EvidentlyDriftReport,
    DataQualityReport,
    ModelPerformanceReport,
    EvidentlyDriftDetector,
)


# --- Fixtures ---

@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "amount": np.random.normal(100, 20, 500),
        "time_hour": np.random.randint(0, 24, 500),
        "location_distance": np.random.exponential(5, 500),
        "previous_avg_amount": np.random.normal(100, 15, 500),
        "transaction_count_1h": np.random.poisson(3, 500),
    })


@pytest.fixture
def current_data_no_drift(reference_data):
    np.random.seed(43)
    return pd.DataFrame({
        "amount": np.random.normal(100, 20, 200),
        "time_hour": np.random.randint(0, 24, 200),
        "location_distance": np.random.exponential(5, 200),
        "previous_avg_amount": np.random.normal(100, 15, 200),
        "transaction_count_1h": np.random.poisson(3, 200),
    })


@pytest.fixture
def current_data_with_drift():
    np.random.seed(44)
    return pd.DataFrame({
        "amount": np.random.normal(300, 50, 200),  # big shift
        "time_hour": np.random.randint(0, 24, 200),
        "location_distance": np.random.exponential(50, 200),  # big shift
        "previous_avg_amount": np.random.normal(300, 50, 200),  # big shift
        "transaction_count_1h": np.random.poisson(20, 200),  # big shift
    })


@pytest.fixture
def detector(reference_data, tmp_path):
    return EvidentlyDriftDetector(
        reference_data=reference_data,
        report_dir=str(tmp_path / "reports"),
    )


# --- Dataclass 테스트 ---

class TestEvidentlyDriftReport:
    def test_default_values(self):
        report = EvidentlyDriftReport()
        assert report.is_drift_detected is False
        assert report.drift_share == 0.0
        assert report.feature_drifts == {}

    def test_to_dict(self):
        report = EvidentlyDriftReport(is_drift_detected=True, drift_share=0.6)
        d = report.to_dict()
        assert d["is_drift_detected"] is True
        assert d["drift_share"] == 0.6
        assert "timestamp" in d


class TestDataQualityReport:
    def test_default_values(self):
        report = DataQualityReport()
        assert report.total_rows == 0
        assert report.missing_values == {}

    def test_to_dict(self):
        report = DataQualityReport(total_rows=100, total_columns=5)
        d = report.to_dict()
        assert d["total_rows"] == 100
        assert d["total_columns"] == 5


class TestModelPerformanceReport:
    def test_default_values(self):
        report = ModelPerformanceReport()
        assert report.is_performance_drift is False
        assert report.degraded_metrics == []

    def test_to_dict(self):
        report = ModelPerformanceReport(
            is_performance_drift=True,
            degraded_metrics=["accuracy"],
        )
        d = report.to_dict()
        assert d["is_performance_drift"] is True
        assert "accuracy" in d["degraded_metrics"]


# --- EvidentlyDriftDetector 테스트 ---

class TestEvidentlyDriftDetector:
    def test_init(self, detector, reference_data):
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 500

    def test_init_no_reference(self, tmp_path):
        det = EvidentlyDriftDetector(report_dir=str(tmp_path / "reports"))
        assert det.reference_data is None

    def test_set_reference_data(self, tmp_path):
        det = EvidentlyDriftDetector(report_dir=str(tmp_path / "reports"))
        data = pd.DataFrame({"a": [1, 2, 3]})
        det.set_reference_data(data)
        assert det.reference_data is not None
        assert len(det.reference_data) == 3

    def test_detect_data_drift_no_reference(self, tmp_path):
        det = EvidentlyDriftDetector(report_dir=str(tmp_path / "reports"))
        with pytest.raises(ValueError, match="레퍼런스"):
            det.detect_data_drift(pd.DataFrame({"a": [1]}))

    def test_detect_data_drift_no_drift(self, detector, current_data_no_drift):
        report = detector.detect_data_drift(current_data_no_drift)
        assert isinstance(report, EvidentlyDriftReport)
        assert report.total_features > 0
        # With similar data, drift should be minimal
        assert report.drift_share < 1.0

    def test_detect_data_drift_with_drift(self, detector, current_data_with_drift):
        report = detector.detect_data_drift(current_data_with_drift)
        assert isinstance(report, EvidentlyDriftReport)
        # With shifted data, should detect drift
        assert report.number_of_drifted_features > 0

    def test_detect_feature_drift(self, detector, current_data_no_drift):
        result = detector.detect_feature_drift(current_data_no_drift, "amount")
        assert "feature" in result
        assert result["feature"] == "amount"
        assert "drift_score" in result
        assert "drift_detected" in result

    def test_detect_feature_drift_missing_feature(self, detector):
        with pytest.raises(ValueError, match="피처를 찾을 수 없습니다"):
            detector.detect_feature_drift(pd.DataFrame({"a": [1]}), "nonexistent")

    def test_detect_feature_drift_missing_in_current(self, detector):
        with pytest.raises(ValueError, match="현재 데이터에 피처가 없습니다"):
            detector.detect_feature_drift(pd.DataFrame({"other": [1]}), "amount")

    def test_detect_target_drift_no_drift(self, detector):
        ref_target = pd.Series(np.random.normal(0, 1, 100))
        cur_target = pd.Series(np.random.normal(0, 1, 100))
        result = detector.detect_target_drift(ref_target, cur_target)
        assert "target_drift_detected" in result
        assert "drift_score" in result

    def test_detect_target_drift_with_drift(self, detector):
        ref_target = pd.Series(np.random.normal(0, 1, 100))
        cur_target = pd.Series(np.random.normal(10, 1, 100))  # big shift
        result = detector.detect_target_drift(ref_target, cur_target)
        assert result["target_drift_detected"] is True

    def test_analyze_data_quality(self, detector, reference_data):
        report = detector.analyze_data_quality(reference_data)
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 500
        assert report.total_columns == 5
        assert "amount" in report.feature_stats
        assert "mean" in report.feature_stats["amount"]

    def test_analyze_data_quality_with_missing(self, detector):
        data = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": [None, None, 3, 4],
        })
        report = detector.analyze_data_quality(data)
        assert report.missing_values["a"] == 1
        assert report.missing_values["b"] == 2
        assert report.missing_rate["b"] == 0.5

    def test_analyze_data_quality_with_duplicates(self, detector):
        data = pd.DataFrame({
            "a": [1, 1, 2, 3],
            "b": [4, 4, 5, 6],
        })
        report = detector.analyze_data_quality(data)
        assert report.duplicated_rows == 1

    def test_detect_model_drift_no_degradation(self, detector):
        ref_metrics = {"accuracy": 0.95, "f1_score": 0.90}
        cur_metrics = {"accuracy": 0.94, "f1_score": 0.89}
        report = detector.detect_model_drift(ref_metrics, cur_metrics)
        assert isinstance(report, ModelPerformanceReport)
        assert report.is_performance_drift is False

    def test_detect_model_drift_with_degradation(self, detector):
        ref_metrics = {"accuracy": 0.95, "f1_score": 0.90}
        cur_metrics = {"accuracy": 0.70, "f1_score": 0.60}  # big drop
        report = detector.detect_model_drift(ref_metrics, cur_metrics)
        assert report.is_performance_drift is True
        assert len(report.degraded_metrics) > 0

    def test_detect_model_drift_partial_metrics(self, detector):
        ref_metrics = {"accuracy": 0.95, "custom": 100}
        cur_metrics = {"accuracy": 0.90}
        report = detector.detect_model_drift(ref_metrics, cur_metrics)
        assert "accuracy" in report.metric_changes
        assert "custom" not in report.metric_changes

    def test_generate_drift_dashboard(self, detector, current_data_no_drift, tmp_path):
        output = str(tmp_path / "dashboard.html")
        result = detector.generate_drift_dashboard(current_data_no_drift, output_path=output)
        assert Path(result).exists()

    def test_generate_drift_dashboard_no_reference(self, tmp_path):
        det = EvidentlyDriftDetector(report_dir=str(tmp_path))
        with pytest.raises(ValueError):
            det.generate_drift_dashboard(pd.DataFrame({"a": [1]}))

    def test_generate_data_quality_dashboard(self, detector, reference_data, tmp_path):
        output = str(tmp_path / "quality.html")
        result = detector.generate_data_quality_dashboard(reference_data, output_path=output)
        assert Path(result).exists()

    def test_create_monitoring_suite(self, detector, current_data_no_drift):
        result = detector.create_monitoring_suite(current_data_no_drift)
        assert "total_tests" in result
        assert "passed" in result
        assert "failed" in result
        assert "success_rate" in result
        assert result["total_tests"] > 0

    def test_create_monitoring_suite_no_reference(self, tmp_path):
        det = EvidentlyDriftDetector(report_dir=str(tmp_path))
        with pytest.raises(ValueError):
            det.create_monitoring_suite(pd.DataFrame({"a": [1]}))

    def test_save_report(self, detector, reference_data, tmp_path):
        report = detector.analyze_data_quality(reference_data)
        path = str(tmp_path / "report.json")
        detector.save_report(report, path)
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert data["total_rows"] == 500
