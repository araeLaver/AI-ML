"""
Prometheus Metrics Module
- 예측 메트릭 수집
- API 성능 모니터링
- 모델 성능 추적
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Prometheus 메트릭 수집기"""

    def __init__(self, app_name: str = "fraud_detection"):
        self.app_name = app_name

        # 예측 관련 메트릭
        self.prediction_total = Counter(
            f"{app_name}_prediction_total",
            "Total number of predictions",
            ["model_version", "result"],
        )

        self.prediction_latency = Histogram(
            f"{app_name}_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model_version"],
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5],
        )

        self.prediction_probability = Histogram(
            f"{app_name}_prediction_probability",
            "Distribution of prediction probabilities",
            ["model_version"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # 배치 처리 메트릭
        self.batch_size = Histogram(
            f"{app_name}_batch_size",
            "Batch prediction sizes",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        )

        self.batch_processing_time = Histogram(
            f"{app_name}_batch_processing_seconds",
            "Batch processing time in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # API 메트릭
        self.api_requests_total = Counter(
            f"{app_name}_api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"],
        )

        self.api_request_latency = Histogram(
            f"{app_name}_api_request_latency_seconds",
            "API request latency",
            ["endpoint", "method"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        # 모델 메트릭
        self.model_info = Info(
            f"{app_name}_model",
            "Model information",
        )

        self.fraud_rate = Gauge(
            f"{app_name}_fraud_rate",
            "Current fraud rate (rolling)",
            ["window"],
        )

        # 시스템 메트릭
        self.active_connections = Gauge(
            f"{app_name}_active_connections",
            "Number of active connections",
        )

        # 내부 상태
        self._recent_predictions = []
        self._max_recent = 1000

    def record_prediction(
        self,
        model_version: str,
        probability: float,
        is_fraud: bool,
        latency_seconds: float,
    ) -> None:
        """예측 메트릭 기록"""
        result = "fraud" if is_fraud else "normal"

        # 카운터 증가
        self.prediction_total.labels(
            model_version=model_version, result=result
        ).inc()

        # 레이턴시 기록
        self.prediction_latency.labels(model_version=model_version).observe(
            latency_seconds
        )

        # 확률 분포 기록
        self.prediction_probability.labels(model_version=model_version).observe(
            probability
        )

        # 최근 예측 저장 (fraud rate 계산용)
        self._recent_predictions.append(is_fraud)
        if len(self._recent_predictions) > self._max_recent:
            self._recent_predictions.pop(0)

        # Fraud rate 업데이트
        if self._recent_predictions:
            rate = sum(self._recent_predictions) / len(self._recent_predictions)
            self.fraud_rate.labels(window="last_1000").set(rate)

    def record_batch_prediction(
        self,
        batch_size: int,
        processing_time: float,
        fraud_count: int,
    ) -> None:
        """배치 예측 메트릭 기록"""
        self.batch_size.observe(batch_size)
        self.batch_processing_time.observe(processing_time)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """API 요청 메트릭 기록"""
        self.api_requests_total.labels(
            endpoint=endpoint, method=method, status=str(status_code)
        ).inc()

        self.api_request_latency.labels(endpoint=endpoint, method=method).observe(
            latency_seconds
        )

    def set_model_info(
        self, version: str, model_type: str, threshold: float
    ) -> None:
        """모델 정보 설정"""
        self.model_info.info(
            {
                "version": version,
                "type": model_type,
                "threshold": str(threshold),
            }
        )

    def increment_connections(self) -> None:
        """활성 연결 증가"""
        self.active_connections.inc()

    def decrement_connections(self) -> None:
        """활성 연결 감소"""
        self.active_connections.dec()

    def get_metrics(self) -> bytes:
        """Prometheus 형식 메트릭 반환"""
        return generate_latest()

    def get_content_type(self) -> str:
        """Prometheus 컨텐츠 타입"""
        return CONTENT_TYPE_LATEST


# 싱글톤 인스턴스
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(app_name: str = "fraud_detection") -> MetricsCollector:
    """메트릭 수집기 싱글톤"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(app_name)
    return _metrics_collector
