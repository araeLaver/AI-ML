# -*- coding: utf-8 -*-
"""
Prometheus 메트릭 수집 모듈

[기능]
- RAG 파이프라인 메트릭
- LLM 호출 메트릭
- 벡터DB 메트릭
- 시스템 리소스 메트릭
- 메트릭 익스포트

[사용 예시]
>>> metrics = MetricsCollector()
>>> metrics.rag.record_query_latency(0.5)
>>> metrics.rag.increment_query_count()
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """메트릭 값"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """히스토그램 버킷"""
    le: float  # less than or equal
    count: int = 0


class MetricRegistry:
    """메트릭 레지스트리"""

    _instance: Optional["MetricRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._metrics: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, List[HistogramBucket]] = {}
        self._lock = threading.Lock()
        self._initialized = True

    def register(self, name: str, metric_type: MetricType, labels: Dict[str, str] = None):
        """메트릭 등록"""
        key = self._make_key(name, labels or {})
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = MetricValue(
                    name=name,
                    value=0,
                    labels=labels or {},
                    metric_type=metric_type,
                )

    def increment(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """카운터 증가"""
        key = self._make_key(name, labels or {})
        with self._lock:
            if key in self._metrics:
                self._metrics[key].value += value
                self._metrics[key].timestamp = time.time()
            else:
                self._metrics[key] = MetricValue(
                    name=name,
                    value=value,
                    labels=labels or {},
                    metric_type=MetricType.COUNTER,
                )

    def set(self, name: str, value: float, labels: Dict[str, str] = None):
        """게이지 설정"""
        key = self._make_key(name, labels or {})
        with self._lock:
            if key in self._metrics:
                self._metrics[key].value = value
                self._metrics[key].timestamp = time.time()
            else:
                self._metrics[key] = MetricValue(
                    name=name,
                    value=value,
                    labels=labels or {},
                    metric_type=MetricType.GAUGE,
                )

    def observe(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        buckets: List[float] = None,
    ):
        """히스토그램 관측"""
        key = self._make_key(name, labels or {})
        buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]

        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = [
                    HistogramBucket(le=b) for b in buckets
                ] + [HistogramBucket(le=float("inf"))]

            for bucket in self._histograms[key]:
                if value <= bucket.le:
                    bucket.count += 1

            # sum과 count 저장
            sum_key = f"{key}_sum"
            count_key = f"{key}_count"

            if sum_key not in self._metrics:
                self._metrics[sum_key] = MetricValue(
                    name=f"{name}_sum",
                    value=0,
                    labels=labels or {},
                )
            if count_key not in self._metrics:
                self._metrics[count_key] = MetricValue(
                    name=f"{name}_count",
                    value=0,
                    labels=labels or {},
                )

            self._metrics[sum_key].value += value
            self._metrics[count_key].value += 1

    def get(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """메트릭 값 조회"""
        key = self._make_key(name, labels or {})
        with self._lock:
            metric = self._metrics.get(key)
            return metric.value if metric else None

    def get_all(self) -> Dict[str, MetricValue]:
        """모든 메트릭 조회"""
        with self._lock:
            return dict(self._metrics)

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """메트릭 키 생성"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def clear(self):
        """메트릭 초기화"""
        with self._lock:
            self._metrics.clear()
            self._histograms.clear()


class RAGMetrics:
    """
    RAG 파이프라인 메트릭

    [메트릭]
    - rag_query_total: 총 쿼리 수
    - rag_query_latency_seconds: 쿼리 레이턴시
    - rag_retrieval_latency_seconds: 검색 레이턴시
    - rag_generation_latency_seconds: 생성 레이턴시
    - rag_documents_retrieved: 검색된 문서 수
    - rag_cache_hits_total: 캐시 히트 수
    - rag_cache_misses_total: 캐시 미스 수
    """

    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or MetricRegistry()
        self._prefix = "rag"

    def record_query(
        self,
        latency: float,
        status: str = "success",
        cached: bool = False,
    ):
        """쿼리 기록"""
        labels = {"status": status}
        self.registry.increment(f"{self._prefix}_query_total", labels=labels)
        self.registry.observe(
            f"{self._prefix}_query_latency_seconds",
            latency,
            labels=labels,
        )

        if cached:
            self.registry.increment(f"{self._prefix}_cache_hits_total")
        else:
            self.registry.increment(f"{self._prefix}_cache_misses_total")

    def record_retrieval(self, latency: float, doc_count: int):
        """검색 기록"""
        self.registry.observe(
            f"{self._prefix}_retrieval_latency_seconds",
            latency,
        )
        self.registry.observe(
            f"{self._prefix}_documents_retrieved",
            doc_count,
            buckets=[1, 3, 5, 10, 20, 50],
        )

    def record_generation(self, latency: float, tokens: int = 0):
        """생성 기록"""
        self.registry.observe(
            f"{self._prefix}_generation_latency_seconds",
            latency,
        )
        if tokens > 0:
            self.registry.increment(
                f"{self._prefix}_tokens_generated_total",
                tokens,
            )

    def set_active_queries(self, count: int):
        """활성 쿼리 수 설정"""
        self.registry.set(f"{self._prefix}_active_queries", count)

    @contextmanager
    def query_timer(self, status: str = "success"):
        """쿼리 타이머 컨텍스트 매니저"""
        start = time.time()
        try:
            yield
        finally:
            self.record_query(time.time() - start, status=status)


class LLMMetrics:
    """
    LLM 호출 메트릭

    [메트릭]
    - llm_requests_total: 총 요청 수
    - llm_request_latency_seconds: 요청 레이턴시
    - llm_tokens_total: 토큰 사용량
    - llm_errors_total: 에러 수
    """

    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or MetricRegistry()
        self._prefix = "llm"

    def record_request(
        self,
        latency: float,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        status: str = "success",
    ):
        """LLM 요청 기록"""
        labels = {"model": model, "status": status}

        self.registry.increment(f"{self._prefix}_requests_total", labels=labels)
        self.registry.observe(
            f"{self._prefix}_request_latency_seconds",
            latency,
            labels={"model": model},
        )

        if input_tokens > 0:
            self.registry.increment(
                f"{self._prefix}_tokens_total",
                input_tokens,
                labels={"model": model, "type": "input"},
            )
        if output_tokens > 0:
            self.registry.increment(
                f"{self._prefix}_tokens_total",
                output_tokens,
                labels={"model": model, "type": "output"},
            )

    def record_error(self, model: str, error_type: str):
        """에러 기록"""
        self.registry.increment(
            f"{self._prefix}_errors_total",
            labels={"model": model, "error_type": error_type},
        )

    @contextmanager
    def request_timer(self, model: str):
        """요청 타이머"""
        start = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.record_request(time.time() - start, model, status=status)


class VectorDBMetrics:
    """
    벡터DB 메트릭

    [메트릭]
    - vectordb_queries_total: 총 쿼리 수
    - vectordb_query_latency_seconds: 쿼리 레이턴시
    - vectordb_documents_total: 문서 수
    - vectordb_index_size_bytes: 인덱스 크기
    """

    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or MetricRegistry()
        self._prefix = "vectordb"

    def record_query(
        self,
        latency: float,
        collection: str,
        results_count: int,
    ):
        """쿼리 기록"""
        labels = {"collection": collection}

        self.registry.increment(f"{self._prefix}_queries_total", labels=labels)
        self.registry.observe(
            f"{self._prefix}_query_latency_seconds",
            latency,
            labels=labels,
        )
        self.registry.observe(
            f"{self._prefix}_results_count",
            results_count,
            labels=labels,
            buckets=[1, 3, 5, 10, 20],
        )

    def record_insert(self, collection: str, doc_count: int):
        """삽입 기록"""
        self.registry.increment(
            f"{self._prefix}_inserts_total",
            doc_count,
            labels={"collection": collection},
        )

    def set_document_count(self, collection: str, count: int):
        """문서 수 설정"""
        self.registry.set(
            f"{self._prefix}_documents_total",
            count,
            labels={"collection": collection},
        )

    def set_index_size(self, collection: str, size_bytes: int):
        """인덱스 크기 설정"""
        self.registry.set(
            f"{self._prefix}_index_size_bytes",
            size_bytes,
            labels={"collection": collection},
        )


class SystemMetrics:
    """시스템 리소스 메트릭"""

    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or MetricRegistry()
        self._prefix = "system"

    def record_cpu_usage(self, percent: float):
        """CPU 사용률 기록"""
        self.registry.set(f"{self._prefix}_cpu_usage_percent", percent)

    def record_memory_usage(self, used_bytes: int, total_bytes: int):
        """메모리 사용량 기록"""
        self.registry.set(f"{self._prefix}_memory_used_bytes", used_bytes)
        self.registry.set(f"{self._prefix}_memory_total_bytes", total_bytes)
        self.registry.set(
            f"{self._prefix}_memory_usage_percent",
            (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0,
        )

    def record_disk_usage(self, path: str, used_bytes: int, total_bytes: int):
        """디스크 사용량 기록"""
        labels = {"path": path}
        self.registry.set(f"{self._prefix}_disk_used_bytes", used_bytes, labels)
        self.registry.set(f"{self._prefix}_disk_total_bytes", total_bytes, labels)


class MetricsCollector:
    """
    통합 메트릭 수집기

    모든 메트릭을 중앙에서 관리
    """

    def __init__(self):
        self.registry = MetricRegistry()
        self.rag = RAGMetrics(self.registry)
        self.llm = LLMMetrics(self.registry)
        self.vectordb = VectorDBMetrics(self.registry)
        self.system = SystemMetrics(self.registry)

    def get_all_metrics(self) -> Dict[str, float]:
        """모든 메트릭 조회"""
        metrics = self.registry.get_all()
        return {k: v.value for k, v in metrics.items()}

    def clear(self):
        """메트릭 초기화"""
        self.registry.clear()


class PrometheusExporter:
    """
    Prometheus 포맷 익스포터

    메트릭을 Prometheus 텍스트 포맷으로 변환
    """

    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or MetricRegistry()

    def export(self) -> str:
        """Prometheus 포맷으로 익스포트"""
        lines = []
        metrics = self.registry.get_all()

        # 메트릭 그룹화
        metric_groups: Dict[str, List[MetricValue]] = {}
        for key, metric in metrics.items():
            base_name = metric.name.split("{")[0]
            if base_name not in metric_groups:
                metric_groups[base_name] = []
            metric_groups[base_name].append(metric)

        for name, group in sorted(metric_groups.items()):
            # TYPE 주석
            metric_type = group[0].metric_type.value
            lines.append(f"# TYPE {name} {metric_type}")

            # 메트릭 값
            for metric in group:
                if metric.labels:
                    label_str = ",".join(
                        f'{k}="{v}"' for k, v in sorted(metric.labels.items())
                    )
                    lines.append(f"{metric.name}{{{label_str}}} {metric.value}")
                else:
                    lines.append(f"{metric.name} {metric.value}")

            lines.append("")

        return "\n".join(lines)

    def get_content_type(self) -> str:
        """Content-Type 헤더"""
        return "text/plain; version=0.0.4; charset=utf-8"


# =============================================================================
# 편의 함수
# =============================================================================

_default_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """기본 메트릭 수집기 조회"""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def record_query_latency(latency: float, status: str = "success"):
    """쿼리 레이턴시 기록 (편의 함수)"""
    get_metrics_collector().rag.record_query(latency, status)
