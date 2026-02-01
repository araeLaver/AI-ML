"""
Jaeger 분산 추적 모듈 테스트
"""

import json
import time
import pytest
from pathlib import Path

from src.monitoring.tracing import (
    Span,
    Trace,
    TracingStats,
    JaegerTracer,
    get_jaeger_tracer,
)


# --- Span 테스트 ---

class TestSpan:
    def test_default_values(self):
        span = Span()
        assert span.span_id != ""
        assert span.trace_id == ""
        assert span.operation_name == ""
        assert span.status == "OK"
        assert span.tags == {}
        assert span.logs == []

    def test_to_dict(self):
        span = Span(operation_name="test-op", trace_id="t1")
        d = span.to_dict()
        assert d["operation_name"] == "test-op"
        assert d["trace_id"] == "t1"
        assert "span_id" in d
        assert "duration_ms" in d


class TestTrace:
    def test_default_values(self):
        trace = Trace()
        assert trace.trace_id != ""
        assert trace.spans == []
        assert trace.duration_ms == 0.0

    def test_to_dict(self):
        span = Span(operation_name="op1")
        trace = Trace(spans=[span])
        d = trace.to_dict()
        assert len(d["spans"]) == 1
        assert d["spans"][0]["operation_name"] == "op1"

    def test_get_critical_path(self):
        s1 = Span(operation_name="fast", duration_ms=10)
        s2 = Span(operation_name="slow", duration_ms=100)
        s3 = Span(operation_name="medium", duration_ms=50)
        trace = Trace(spans=[s1, s2, s3])
        path = trace.get_critical_path()
        assert path[0].operation_name == "slow"
        assert path[-1].operation_name == "fast"

    def test_get_critical_path_empty(self):
        trace = Trace()
        assert trace.get_critical_path() == []


class TestTracingStats:
    def test_default_values(self):
        stats = TracingStats()
        assert stats.total_traces == 0
        assert stats.p50_duration_ms == 0.0
        assert stats.error_rate == 0.0

    def test_to_dict(self):
        stats = TracingStats(total_traces=10, p95_duration_ms=150.0)
        d = stats.to_dict()
        assert d["total_traces"] == 10
        assert d["p95_duration_ms"] == 150.0


# --- JaegerTracer 테스트 ---

class TestJaegerTracer:
    @pytest.fixture
    def tracer(self, tmp_path):
        return JaegerTracer(
            service_name="test-service",
            fallback_dir=str(tmp_path / "traces"),
        )

    def test_init(self, tracer):
        assert tracer.service_name == "test-service"
        assert tracer._otel_tracer is None
        assert tracer.traces == {}

    def test_start_span(self, tracer):
        span = tracer.start_span("test-op")
        assert span.operation_name == "test-op"
        assert span.trace_id != ""
        assert span.start_time > 0
        assert span.span_id in tracer.active_spans

    def test_start_span_with_trace_id(self, tracer):
        span = tracer.start_span("op", trace_id="my-trace")
        assert span.trace_id == "my-trace"
        assert "my-trace" in tracer.traces

    def test_finish_span(self, tracer):
        span = tracer.start_span("op")
        time.sleep(0.01)
        finished = tracer.finish_span(span)
        assert finished.end_time > finished.start_time
        assert finished.duration_ms > 0
        assert finished.status == "OK"
        assert span.span_id not in tracer.active_spans

    def test_finish_span_with_error(self, tracer):
        span = tracer.start_span("op")
        finished = tracer.finish_span(span, status="ERROR", error="something broke")
        assert finished.status == "ERROR"
        assert len(finished.logs) == 1
        assert finished.logs[0]["event"] == "error"

    def test_trace_context_manager(self, tracer):
        with tracer.trace("test-op") as span:
            time.sleep(0.01)
        assert span.duration_ms > 0
        assert span.status == "OK"
        assert span.trace_id in tracer.traces

    def test_trace_context_manager_error(self, tracer):
        with pytest.raises(ValueError):
            with tracer.trace("fail-op") as span:
                raise ValueError("test error")
        assert span.status == "ERROR"

    def test_trace_prediction(self, tracer):
        with tracer.trace_prediction(model_version="v1.0") as span:
            time.sleep(0.01)
        assert span.tags["ml.component"] == "prediction"
        assert span.tags["ml.model_version"] == "v1.0"

    def test_trace_preprocessing(self, tracer):
        with tracer.trace_preprocessing() as span:
            pass
        assert span.tags["ml.component"] == "preprocessing"

    def test_trace_model_inference(self, tracer):
        with tracer.trace_model_inference(model_version="v2.0") as span:
            pass
        assert span.tags["ml.component"] == "inference"

    def test_get_trace(self, tracer):
        with tracer.trace("op1") as span:
            pass
        trace = tracer.get_trace(span.trace_id)
        assert trace is not None
        assert len(trace.spans) == 1

    def test_get_trace_not_found(self, tracer):
        assert tracer.get_trace("nonexistent") is None

    def test_query_traces(self, tracer):
        with tracer.trace("op1"):
            pass
        with tracer.trace("op2"):
            pass
        results = tracer.query_traces()
        assert len(results) == 2

    def test_query_traces_by_operation(self, tracer):
        with tracer.trace("op1"):
            pass
        with tracer.trace("op2"):
            pass
        results = tracer.query_traces(operation="op1")
        assert len(results) == 1

    def test_query_traces_limit(self, tracer):
        for i in range(5):
            with tracer.trace(f"op{i}"):
                pass
        results = tracer.query_traces(limit=2)
        assert len(results) == 2

    def test_find_slow_traces(self, tracer):
        with tracer.trace("fast-op"):
            time.sleep(0.001)
        # Manually create a slow trace
        trace_id = "slow-trace-id"
        span = tracer.start_span("slow-op", trace_id=trace_id)
        span.start_time = time.time() - 2.0
        tracer.finish_span(span)
        # Also update the trace duration
        trace = tracer.get_trace(trace_id)
        trace.start_time = span.start_time
        trace.duration_ms = span.duration_ms

        slow = tracer.find_slow_traces(threshold_ms=1000.0)
        assert len(slow) >= 1

    def test_get_tracing_stats(self, tracer):
        with tracer.trace("op1"):
            time.sleep(0.01)
        with tracer.trace("op2"):
            time.sleep(0.01)
        stats = tracer.get_tracing_stats()
        assert stats.total_traces == 2
        assert stats.total_spans == 2
        assert stats.avg_duration_ms > 0

    def test_get_tracing_stats_empty(self, tracer):
        stats = tracer.get_tracing_stats()
        assert stats.total_traces == 0

    def test_get_tracing_stats_with_errors(self, tracer):
        with tracer.trace("ok-op"):
            pass
        try:
            with tracer.trace("err-op"):
                raise RuntimeError("fail")
        except RuntimeError:
            pass
        stats = tracer.get_tracing_stats()
        assert stats.error_count == 1

    def test_save_traces(self, tracer, tmp_path):
        with tracer.trace("op1"):
            pass
        path = tracer.save_traces(str(tmp_path / "test-traces.json"))
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_multiple_spans_in_trace(self, tracer):
        trace_id = "shared-trace"
        with tracer.trace("op1", trace_id=trace_id):
            pass
        with tracer.trace("op2", trace_id=trace_id):
            pass
        trace = tracer.get_trace(trace_id)
        assert len(trace.spans) == 2

    def test_max_traces_cleanup(self, tracer):
        tracer._max_traces = 10
        for i in range(15):
            with tracer.trace(f"op{i}"):
                pass
        assert len(tracer.traces) <= 10
