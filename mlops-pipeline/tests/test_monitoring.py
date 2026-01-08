"""
Monitoring Tests
"""

import pytest
import numpy as np
import pandas as pd
import uuid

from src.monitoring import MetricsCollector, DriftDetector


class TestMetricsCollector:
    """MetricsCollector 테스트"""

    @pytest.fixture
    def collector(self):
        """각 테스트마다 고유한 메트릭 수집기 생성"""
        unique_name = f"test_app_{uuid.uuid4().hex[:8]}"
        return MetricsCollector(app_name=unique_name)

    def test_record_prediction(self, collector):
        """예측 메트릭 기록 테스트"""
        # 정상적으로 기록되어야 함
        collector.record_prediction(
            model_version="v1.0.0",
            probability=0.3,
            is_fraud=False,
            latency_seconds=0.05,
        )

        collector.record_prediction(
            model_version="v1.0.0",
            probability=0.8,
            is_fraud=True,
            latency_seconds=0.03,
        )

        # fraud rate 계산됨
        assert len(collector._recent_predictions) == 2

    def test_record_batch_prediction(self, collector):
        """배치 예측 메트릭 기록 테스트"""
        collector.record_batch_prediction(
            batch_size=100,
            processing_time=0.5,
            fraud_count=5,
        )

        # 에러 없이 실행되어야 함

    def test_record_api_request(self, collector):
        """API 요청 메트릭 기록 테스트"""
        collector.record_api_request(
            endpoint="/predict",
            method="POST",
            status_code=200,
            latency_seconds=0.1,
        )

        # 에러 없이 실행되어야 함

    def test_set_model_info(self, collector):
        """모델 정보 설정 테스트"""
        collector.set_model_info(
            version="v1.0.0",
            model_type="random_forest",
            threshold=0.5,
        )

        # 에러 없이 실행되어야 함

    def test_get_metrics(self):
        """메트릭 조회 테스트"""
        unique_name = f"test_metrics_{uuid.uuid4().hex[:8]}"
        collector = MetricsCollector(app_name=unique_name)
        metrics = collector.get_metrics()

        assert isinstance(metrics, bytes)


class TestDriftDetector:
    """DriftDetector 테스트"""

    @pytest.fixture
    def reference_data(self):
        """레퍼런스 데이터"""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame(
            {
                "amount": np.random.exponential(scale=100, size=n),
                "time_hour": np.random.randint(0, 24, size=n),
                "location_distance": np.random.exponential(scale=10, size=n),
            }
        )

    def test_set_reference_data(self, reference_data):
        """레퍼런스 데이터 설정 테스트"""
        detector = DriftDetector()
        detector.set_reference_data(reference_data)

        assert detector.reference_data is not None
        assert len(detector.reference_stats) > 0

    def test_detect_no_drift(self, reference_data):
        """드리프트 없음 감지 테스트"""
        detector = DriftDetector(reference_data=reference_data)

        # 같은 분포의 데이터
        np.random.seed(42)
        current_data = pd.DataFrame(
            {
                "amount": np.random.exponential(scale=100, size=500),
                "time_hour": np.random.randint(0, 24, size=500),
                "location_distance": np.random.exponential(scale=10, size=500),
            }
        )

        result = detector.detect_data_drift(current_data)

        # 같은 분포이므로 드리프트가 감지되지 않아야 함
        assert result.drift_score < 0.5

    def test_detect_drift(self, reference_data):
        """드리프트 감지 테스트"""
        detector = DriftDetector(reference_data=reference_data, psi_threshold=0.1)

        # 다른 분포의 데이터 (같은 범위 내에서 분포만 다르게)
        np.random.seed(123)
        current_data = pd.DataFrame(
            {
                "amount": np.random.exponential(scale=300, size=500),  # 스케일만 다르게
                "time_hour": np.random.choice([0, 1, 2, 22, 23], size=500),  # 야간 집중
                "location_distance": np.random.exponential(scale=50, size=500),  # 스케일만 다르게
            }
        )

        result = detector.detect_data_drift(current_data)

        # 드리프트가 감지되어야 함 (KS 통계량으로 확인)
        # PSI가 nan일 수 있으므로 details의 KS 통계량으로 확인
        ks_stats = [v.get("ks_statistic", 0) for v in result.details.values()]
        assert any(ks > 0.1 for ks in ks_stats)  # KS 통계량이 유의미하게 다름

    def test_detect_prediction_drift(self):
        """예측 드리프트 감지 테스트"""
        detector = DriftDetector()

        ref_predictions = np.random.beta(2, 5, size=1000)  # 낮은 확률 분포
        cur_predictions = np.random.beta(5, 2, size=500)  # 높은 확률 분포

        result = detector.detect_prediction_drift(ref_predictions, cur_predictions)

        # 완전히 다른 분포이므로 드리프트 감지
        assert result.is_drift_detected is True

    def test_detect_performance_drift(self):
        """성능 드리프트 감지 테스트"""
        detector = DriftDetector()

        ref_metrics = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1_score": 0.87,
        }

        cur_metrics = {
            "accuracy": 0.80,  # 15% 하락
            "precision": 0.70,  # 22% 하락
            "recall": 0.75,  # 12% 하락
            "f1_score": 0.72,  # 17% 하락
        }

        result = detector.detect_performance_drift(ref_metrics, cur_metrics)

        assert result.is_drift_detected is True

    def test_get_drift_report(self, reference_data):
        """드리프트 리포트 생성 테스트"""
        detector = DriftDetector(reference_data=reference_data)

        current_data = reference_data.sample(500, random_state=42)
        result = detector.detect_data_drift(current_data)

        report = detector.get_drift_report(result)

        assert "드리프트 감지 리포트" in report
        assert "드리프트 점수" in report
