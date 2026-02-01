"""
Jaeger 분산 추적 모듈
- 스팬/트레이스 관리
- ML 파이프라인 추적
- 성능 병목 분석
- Jaeger 미설치 시 로컬 파일 폴백
"""

import json
import uuid
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

# OpenTelemetry 선택적 임포트
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    HAS_JAEGER_EXPORTER = True
except ImportError:
    HAS_JAEGER_EXPORTER = False

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """스팬 데이터"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    trace_id: str = ""
    parent_span_id: str = ""
    operation_name: str = ""
    service_name: str = "fraud-detection"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "OK"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
        }


@dataclass
class Trace:
    """트레이스 데이터"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()).replace("-", ""))
    spans: List[Span] = field(default_factory=list)
    service_name: str = "fraud-detection"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
        }

    def get_critical_path(self) -> List[Span]:
        """크리티컬 패스 분석 - 가장 오래 걸린 스팬 경로"""
        if not self.spans:
            return []
        sorted_spans = sorted(self.spans, key=lambda s: s.duration_ms, reverse=True)
        return sorted_spans


@dataclass
class TracingStats:
    """추적 통계"""
    total_traces: int = 0
    total_spans: int = 0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    operations: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "total_spans": self.total_spans,
            "avg_duration_ms": self.avg_duration_ms,
            "p50_duration_ms": self.p50_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "operations": self.operations,
        }


class JaegerTracer:
    """Jaeger 기반 분산 추적 시스템"""

    def __init__(
        self,
        service_name: str = "fraud-detection",
        jaeger_host: Optional[str] = None,
        jaeger_port: int = 6831,
        fallback_dir: str = "logs/traces",
    ):
        self.service_name = service_name
        self.fallback_dir = fallback_dir
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}
        self._otel_tracer = None
        self._max_traces = 5000

        # Jaeger/OpenTelemetry 설정
        jaeger_host = jaeger_host or os.getenv("JAEGER_HOST")
        if jaeger_host and HAS_OPENTELEMETRY and HAS_JAEGER_EXPORTER:
            try:
                exporter = JaegerExporter(
                    agent_host_name=jaeger_host,
                    agent_port=jaeger_port,
                )
                provider = TracerProvider()
                provider.add_span_processor(SimpleSpanProcessor(exporter))
                self._otel_tracer = provider.get_tracer(service_name)
                logger.info(f"Jaeger 연결 성공: {jaeger_host}:{jaeger_port}")
            except Exception as e:
                logger.warning(f"Jaeger 연결 실패: {e}, 로컬 폴백 사용")

        # 폴백 디렉토리 생성
        Path(self.fallback_dir).mkdir(parents=True, exist_ok=True)

    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: str = "",
        tags: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """스팬 시작"""
        if trace_id is None:
            trace_id = str(uuid.uuid4()).replace("-", "")

        span = Span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=time.time(),
            tags=tags or {},
        )

        # 트레이스 생성 또는 조회
        if trace_id not in self.traces:
            self.traces[trace_id] = Trace(
                trace_id=trace_id,
                service_name=self.service_name,
                start_time=time.time(),
            )
            # 오래된 트레이스 정리
            if len(self.traces) > self._max_traces:
                oldest_keys = sorted(self.traces.keys(), key=lambda k: self.traces[k].start_time)
                for key in oldest_keys[:len(oldest_keys) // 2]:
                    del self.traces[key]

        self.active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span, status: str = "OK", error: Optional[str] = None) -> Span:
        """스팬 종료"""
        span.end_time = time.time()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status = status

        if error:
            span.status = "ERROR"
            span.logs.append({
                "timestamp": datetime.now().isoformat(),
                "event": "error",
                "message": error,
            })

        # 트레이스에 스팬 추가
        if span.trace_id in self.traces:
            trace = self.traces[span.trace_id]
            trace.spans.append(span)
            trace.end_time = span.end_time
            trace.duration_ms = (trace.end_time - trace.start_time) * 1000

        # 활성 스팬에서 제거
        self.active_spans.pop(span.span_id, None)

        return span

    @contextmanager
    def trace(self, operation_name: str, trace_id: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
        """컨텍스트 매니저 기반 트레이싱"""
        span = self.start_span(operation_name, trace_id=trace_id, tags=tags)
        try:
            yield span
            self.finish_span(span, status="OK")
        except Exception as e:
            self.finish_span(span, status="ERROR", error=str(e))
            raise

    @contextmanager
    def trace_prediction(self, model_version: str = "", trace_id: Optional[str] = None):
        """예측 트레이싱"""
        tags = {"ml.component": "prediction", "ml.model_version": model_version}
        with self.trace("ml.prediction", trace_id=trace_id, tags=tags) as span:
            yield span

    @contextmanager
    def trace_preprocessing(self, trace_id: Optional[str] = None):
        """전처리 트레이싱"""
        tags = {"ml.component": "preprocessing"}
        with self.trace("ml.preprocessing", trace_id=trace_id, tags=tags) as span:
            yield span

    @contextmanager
    def trace_model_inference(self, model_version: str = "", trace_id: Optional[str] = None):
        """모델 추론 트레이싱"""
        tags = {"ml.component": "inference", "ml.model_version": model_version}
        with self.trace("ml.inference", trace_id=trace_id, tags=tags) as span:
            yield span

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """트레이스 조회"""
        return self.traces.get(trace_id)

    def query_traces(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """트레이스 검색"""
        results = []
        for trace in sorted(self.traces.values(), key=lambda t: t.start_time, reverse=True):
            if service and trace.service_name != service:
                continue
            if operation:
                ops = [s.operation_name for s in trace.spans]
                if operation not in ops:
                    continue
            if min_duration_ms is not None and trace.duration_ms < min_duration_ms:
                continue
            if max_duration_ms is not None and trace.duration_ms > max_duration_ms:
                continue
            results.append(trace.to_dict())
            if len(results) >= limit:
                break
        return results

    def find_slow_traces(self, threshold_ms: float = 1000.0, limit: int = 20) -> List[Dict[str, Any]]:
        """느린 트레이스 조회"""
        slow = []
        for trace in sorted(self.traces.values(), key=lambda t: t.duration_ms, reverse=True):
            if trace.duration_ms >= threshold_ms:
                slow.append(trace.to_dict())
                if len(slow) >= limit:
                    break
        return slow

    def get_tracing_stats(self) -> TracingStats:
        """추적 통계"""
        if not self.traces:
            return TracingStats()

        durations = [t.duration_ms for t in self.traces.values() if t.duration_ms > 0]
        total_spans = sum(len(t.spans) for t in self.traces.values())
        error_count = sum(
            1 for t in self.traces.values()
            for s in t.spans if s.status == "ERROR"
        )

        operations: Dict[str, int] = {}
        for t in self.traces.values():
            for s in t.spans:
                operations[s.operation_name] = operations.get(s.operation_name, 0) + 1

        if durations:
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            avg = sum(durations_sorted) / n
            p50 = durations_sorted[int(n * 0.5)] if n > 0 else 0
            p95 = durations_sorted[min(int(n * 0.95), n - 1)] if n > 0 else 0
            p99 = durations_sorted[min(int(n * 0.99), n - 1)] if n > 0 else 0
        else:
            avg = p50 = p95 = p99 = 0.0

        return TracingStats(
            total_traces=len(self.traces),
            total_spans=total_spans,
            avg_duration_ms=round(avg, 2),
            p50_duration_ms=round(p50, 2),
            p95_duration_ms=round(p95, 2),
            p99_duration_ms=round(p99, 2),
            error_count=error_count,
            error_rate=round(error_count / total_spans, 4) if total_spans > 0 else 0.0,
            operations=operations,
        )

    def save_traces(self, path: Optional[str] = None) -> str:
        """트레이스 저장"""
        if path is None:
            path = str(Path(self.fallback_dir) / f"traces-{datetime.now().strftime('%Y-%m-%d')}.json")

        data = [t.to_dict() for t in self.traces.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"트레이스 저장: {path} ({len(data)}개)")
        return path


# 싱글톤
_jaeger_tracer: Optional[JaegerTracer] = None


def get_jaeger_tracer(service_name: str = "fraud-detection") -> JaegerTracer:
    """Jaeger 트레이서 싱글톤"""
    global _jaeger_tracer
    if _jaeger_tracer is None:
        _jaeger_tracer = JaegerTracer(service_name=service_name)
    return _jaeger_tracer
