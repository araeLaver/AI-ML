# -*- coding: utf-8 -*-
"""
분산 추적 모듈

[기능]
- 요청 트레이싱
- 스팬 관리
- 컨텍스트 전파
- 추적 익스포트

[사용 예시]
>>> tracer = TracingManager()
>>> with tracer.start_span("rag_query") as span:
...     span.set_attribute("query", "삼성전자")
...     result = rag.query("삼성전자")
"""

import logging
import time
import uuid
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """스팬 상태"""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """스팬 컨텍스트"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    sampled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sampled": self.sampled,
        }


@dataclass
class SpanEvent:
    """스팬 이벤트"""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """추적 스팬"""
    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_attribute(self, key: str, value: Any):
        """속성 설정"""
        with self._lock:
            self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]):
        """여러 속성 설정"""
        with self._lock:
            self.attributes.update(attributes)

    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """이벤트 추가"""
        with self._lock:
            self.events.append(SpanEvent(
                name=name,
                timestamp=time.time(),
                attributes=attributes or {},
            ))

    def set_status(self, status: SpanStatus, description: str = None):
        """상태 설정"""
        with self._lock:
            self.status = status
            if description:
                self.attributes["status_description"] = description

    def end(self):
        """스팬 종료"""
        with self._lock:
            if self.end_time is None:
                self.end_time = time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        """지속 시간 (밀리초)"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


class SpanProcessor:
    """스팬 처리기 인터페이스"""

    def on_start(self, span: Span):
        """스팬 시작 시 호출"""
        pass

    def on_end(self, span: Span):
        """스팬 종료 시 호출"""
        pass


class SimpleSpanProcessor(SpanProcessor):
    """간단한 스팬 처리기"""

    def __init__(self, exporter: "TraceExporter"):
        self.exporter = exporter

    def on_end(self, span: Span):
        """스팬 종료 시 익스포트"""
        self.exporter.export([span])


class BatchSpanProcessor(SpanProcessor):
    """배치 스팬 처리기"""

    def __init__(
        self,
        exporter: "TraceExporter",
        max_batch_size: int = 100,
        schedule_delay_ms: float = 5000,
    ):
        self.exporter = exporter
        self.max_batch_size = max_batch_size
        self.schedule_delay_ms = schedule_delay_ms
        self._batch: List[Span] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def on_end(self, span: Span):
        """스팬 배치에 추가"""
        with self._lock:
            self._batch.append(span)

            if len(self._batch) >= self.max_batch_size:
                self._flush()
            elif self._timer is None:
                self._schedule_flush()

    def _schedule_flush(self):
        """플러시 예약"""
        self._timer = threading.Timer(
            self.schedule_delay_ms / 1000,
            self._flush
        )
        self._timer.daemon = True
        self._timer.start()

    def _flush(self):
        """배치 플러시"""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

            if self._batch:
                self.exporter.export(self._batch.copy())
                self._batch.clear()


class TraceExporter:
    """추적 익스포터 인터페이스"""

    def export(self, spans: List[Span]):
        """스팬 익스포트"""
        pass


class ConsoleTraceExporter(TraceExporter):
    """콘솔 추적 익스포터"""

    def export(self, spans: List[Span]):
        """콘솔에 스팬 출력"""
        for span in spans:
            duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "N/A"
            logger.info(
                f"[TRACE] {span.name} "
                f"trace_id={span.context.trace_id[:8]}... "
                f"duration={duration} "
                f"status={span.status.value}"
            )


class JSONTraceExporter(TraceExporter):
    """JSON 추적 익스포터"""

    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self._spans: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def export(self, spans: List[Span]):
        """JSON으로 스팬 저장"""
        with self._lock:
            for span in spans:
                self._spans.append(span.to_dict())

        if self.file_path:
            import json
            with open(self.file_path, "w") as f:
                json.dump(self._spans, f, indent=2, default=str)

    def get_spans(self) -> List[Dict[str, Any]]:
        """저장된 스팬 조회"""
        with self._lock:
            return self._spans.copy()


class TracingManager:
    """
    추적 관리자

    [특징]
    - 스팬 생성 및 관리
    - 컨텍스트 전파
    - 샘플링
    """

    _context_var = threading.local()

    def __init__(
        self,
        service_name: str = "finance-rag",
        processor: SpanProcessor = None,
        sample_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.processor = processor or SimpleSpanProcessor(ConsoleTraceExporter())

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
    ):
        """
        새 스팬 시작

        Args:
            name: 스팬 이름
            attributes: 초기 속성

        Yields:
            Span: 시작된 스팬
        """
        # 샘플링 확인
        import random
        sampled = random.random() < self.sample_rate

        # 부모 스팬 컨텍스트 조회
        parent_context = getattr(self._context_var, "current_span", None)
        parent_span_id = parent_context.context.span_id if parent_context else None
        trace_id = (
            parent_context.context.trace_id
            if parent_context
            else uuid.uuid4().hex
        )

        # 새 스팬 생성
        context = SpanContext(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            sampled=sampled,
        )

        span = Span(
            name=name,
            context=context,
            attributes={"service.name": self.service_name},
        )

        if attributes:
            span.set_attributes(attributes)

        # 프로세서에 시작 알림
        self.processor.on_start(span)

        # 컨텍스트 설정
        previous_span = getattr(self._context_var, "current_span", None)
        self._context_var.current_span = span

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            span.end()
            self._context_var.current_span = previous_span

            # 프로세서에 종료 알림
            if sampled:
                self.processor.on_end(span)

    def get_current_span(self) -> Optional[Span]:
        """현재 활성 스팬 조회"""
        return getattr(self._context_var, "current_span", None)

    def get_current_context(self) -> Optional[SpanContext]:
        """현재 스팬 컨텍스트 조회"""
        span = self.get_current_span()
        return span.context if span else None

    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """HTTP 헤더에 컨텍스트 주입"""
        context = self.get_current_context()
        if context:
            headers["traceparent"] = (
                f"00-{context.trace_id}-{context.span_id}-"
                f"{'01' if context.sampled else '00'}"
            )
        return headers

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """HTTP 헤더에서 컨텍스트 추출"""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) == 4:
                return SpanContext(
                    trace_id=parts[1],
                    span_id=parts[2],
                    sampled=parts[3] == "01",
                )
        except Exception:
            pass

        return None


class RAGTracer:
    """
    RAG 파이프라인 전용 추적기

    RAG 워크플로우에 특화된 추적 기능
    """

    def __init__(self, tracer: TracingManager = None):
        self.tracer = tracer or TracingManager()

    @contextmanager
    def trace_query(self, query: str):
        """RAG 쿼리 추적"""
        with self.tracer.start_span("rag.query") as span:
            span.set_attribute("rag.query", query[:100])
            span.set_attribute("rag.query_length", len(query))
            yield span

    @contextmanager
    def trace_retrieval(self, query: str, top_k: int):
        """검색 단계 추적"""
        with self.tracer.start_span("rag.retrieval") as span:
            span.set_attribute("rag.top_k", top_k)
            yield span

    @contextmanager
    def trace_reranking(self, doc_count: int):
        """리랭킹 단계 추적"""
        with self.tracer.start_span("rag.reranking") as span:
            span.set_attribute("rag.input_docs", doc_count)
            yield span

    @contextmanager
    def trace_generation(self, model: str):
        """생성 단계 추적"""
        with self.tracer.start_span("rag.generation") as span:
            span.set_attribute("llm.model", model)
            yield span

    @contextmanager
    def trace_embedding(self, text_count: int):
        """임베딩 단계 추적"""
        with self.tracer.start_span("rag.embedding") as span:
            span.set_attribute("rag.text_count", text_count)
            yield span


# =============================================================================
# 편의 함수
# =============================================================================

_default_tracer: Optional[TracingManager] = None


def get_tracer() -> TracingManager:
    """기본 추적기 조회"""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = TracingManager()
    return _default_tracer


def trace(name: str, attributes: Dict[str, Any] = None):
    """데코레이터로 함수 추적"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with get_tracer().start_span(name, attributes) as span:
                return func(*args, **kwargs)
        return wrapper
    return decorator
