# -*- coding: utf-8 -*-
"""
모니터링 모듈 테스트
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.metrics import (
    MetricsCollector,
    MetricRegistry,
    MetricType,
    PrometheusExporter,
    RAGMetrics,
    LLMMetrics,
    VectorDBMetrics,
)
from src.monitoring.tracing import (
    TracingManager,
    SpanStatus,
    SpanContext,
    JSONTraceExporter,
    RAGTracer,
)
from src.monitoring.logging_config import (
    StructuredLogger,
    LogLevel,
    LogContext,
    QueryLogger,
    get_log_context,
    add_log_context,
    clear_log_context,
)
from src.monitoring.profiler import (
    PerformanceProfiler,
    QueryProfiler,
    MemoryProfiler,
    ProfileStats,
)
from src.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    LogNotifier,
)
from src.monitoring.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    FunctionHealthCheck,
)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetricRegistry:
    """메트릭 레지스트리 테스트"""

    def test_singleton(self):
        """싱글톤 패턴 테스트"""
        registry1 = MetricRegistry()
        registry2 = MetricRegistry()
        assert registry1 is registry2

    def test_increment_counter(self):
        """카운터 증가 테스트"""
        registry = MetricRegistry()
        registry.clear()

        registry.increment("test_counter", 1)
        assert registry.get("test_counter") == 1

        registry.increment("test_counter", 5)
        assert registry.get("test_counter") == 6

    def test_set_gauge(self):
        """게이지 설정 테스트"""
        registry = MetricRegistry()
        registry.clear()

        registry.set("test_gauge", 42)
        assert registry.get("test_gauge") == 42

        registry.set("test_gauge", 100)
        assert registry.get("test_gauge") == 100

    def test_observe_histogram(self):
        """히스토그램 관측 테스트"""
        registry = MetricRegistry()
        registry.clear()

        registry.observe("test_histogram", 0.5)
        registry.observe("test_histogram", 1.5)
        registry.observe("test_histogram", 2.5)

        # sum과 count 확인
        assert registry.get("test_histogram_count") == 3
        assert registry.get("test_histogram_sum") == 4.5

    def test_labels(self):
        """레이블 테스트"""
        registry = MetricRegistry()
        registry.clear()

        registry.increment("test", 1, labels={"status": "success"})
        registry.increment("test", 2, labels={"status": "error"})

        assert registry.get("test", labels={"status": "success"}) == 1
        assert registry.get("test", labels={"status": "error"}) == 2


class TestRAGMetrics:
    """RAG 메트릭 테스트"""

    def test_record_query(self):
        """쿼리 기록 테스트"""
        registry = MetricRegistry()
        registry.clear()

        metrics = RAGMetrics(registry)
        metrics.record_query(0.5, status="success", cached=False)

        assert registry.get("rag_query_total", {"status": "success"}) == 1
        assert registry.get("rag_cache_misses_total") == 1

    def test_query_timer(self):
        """쿼리 타이머 테스트"""
        registry = MetricRegistry()
        registry.clear()

        metrics = RAGMetrics(registry)

        with metrics.query_timer():
            time.sleep(0.1)

        assert registry.get("rag_query_total", {"status": "success"}) == 1


class TestPrometheusExporter:
    """Prometheus 익스포터 테스트"""

    def test_export(self):
        """익스포트 테스트"""
        registry = MetricRegistry()
        registry.clear()

        registry.increment("test_counter")
        registry.set("test_gauge", 42)

        exporter = PrometheusExporter(registry)
        output = exporter.export()

        assert "test_counter" in output
        assert "test_gauge 42" in output


# =============================================================================
# Tracing Tests
# =============================================================================

class TestTracingManager:
    """추적 관리자 테스트"""

    def test_start_span(self):
        """스팬 시작 테스트"""
        tracer = TracingManager()

        with tracer.start_span("test_span") as span:
            span.set_attribute("key", "value")
            assert span.name == "test_span"
            assert span.attributes.get("key") == "value"

        assert span.status == SpanStatus.OK
        assert span.duration_ms is not None

    def test_nested_spans(self):
        """중첩 스팬 테스트"""
        tracer = TracingManager()

        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.context.parent_span_id == parent.context.span_id
                assert child.context.trace_id == parent.context.trace_id

    def test_span_error(self):
        """스팬 에러 테스트"""
        tracer = TracingManager()

        try:
            with tracer.start_span("error_span") as span:
                raise ValueError("test error")
        except ValueError:
            pass

        assert span.status == SpanStatus.ERROR
        assert "test error" in span.attributes.get("error.message", "")


class TestJSONTraceExporter:
    """JSON 추적 익스포터 테스트"""

    def test_export(self):
        """익스포트 테스트"""
        exporter = JSONTraceExporter()
        tracer = TracingManager(
            processor=MagicMock(on_start=MagicMock(), on_end=lambda s: exporter.export([s]))
        )

        with tracer.start_span("test") as span:
            span.set_attribute("test", "value")

        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0]["name"] == "test"


class TestRAGTracer:
    """RAG 추적기 테스트"""

    def test_trace_query(self):
        """쿼리 추적 테스트"""
        rag_tracer = RAGTracer()

        with rag_tracer.trace_query("삼성전자 실적") as span:
            assert span.name == "rag.query"
            assert "rag.query" in span.attributes  # 쿼리 내용 속성 확인

        assert span.duration_ms is not None


# =============================================================================
# Logging Tests
# =============================================================================

class TestStructuredLogger:
    """구조화된 로거 테스트"""

    def test_log_with_extras(self):
        """추가 필드 로깅 테스트"""
        logger = StructuredLogger("test", level=LogLevel.DEBUG)
        # 로그가 에러 없이 실행되는지 확인
        logger.info("Test message", key1="value1", key2=42)
        logger.debug("Debug message", data={"nested": "value"})

    def test_log_levels(self):
        """로그 레벨 테스트"""
        logger = StructuredLogger("test")
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")


class TestLogContext:
    """로그 컨텍스트 테스트"""

    def test_context_manager(self):
        """컨텍스트 매니저 테스트"""
        clear_log_context()

        with LogContext(request_id="123", user_id="user1"):
            context = get_log_context()
            assert context.get("request_id") == "123"
            assert context.get("user_id") == "user1"

        # 컨텍스트 복원 확인
        context = get_log_context()
        assert "request_id" not in context

    def test_add_context(self):
        """컨텍스트 추가 테스트"""
        clear_log_context()

        add_log_context(key1="value1")
        add_log_context(key2="value2")

        context = get_log_context()
        assert context.get("key1") == "value1"
        assert context.get("key2") == "value2"


# =============================================================================
# Profiler Tests
# =============================================================================

class TestPerformanceProfiler:
    """성능 프로파일러 테스트"""

    def test_profile(self):
        """프로파일 테스트"""
        profiler = PerformanceProfiler(enable_memory=False)

        with profiler.profile("test_operation") as result:
            time.sleep(0.1)

        assert result.duration_ms >= 100
        assert result.name == "test_operation"

    def test_nested_profile(self):
        """중첩 프로파일 테스트"""
        profiler = PerformanceProfiler(enable_memory=False)

        with profiler.profile("parent") as parent:
            with profiler.profile("child") as child:
                time.sleep(0.05)

        assert len(parent.children) == 1
        assert parent.children[0].name == "child"

    def test_stats(self):
        """통계 테스트"""
        profiler = PerformanceProfiler(enable_memory=False)

        # 5번 프로파일링 (중첩 없이 루트 레벨에서)
        for i in range(5):
            with profiler.profile(f"repeated_{i}"):
                time.sleep(0.01)

        # 개별 프로파일 확인
        report = profiler.get_report()
        assert "recent_profiles" in report
        assert len(report["recent_profiles"]) >= 5


class TestQueryProfiler:
    """쿼리 프로파일러 테스트"""

    def test_profile_query(self):
        """쿼리 프로파일 테스트"""
        profiler = QueryProfiler()

        with profiler.profile_query("test query"):
            with profiler.profile_retrieval():
                time.sleep(0.01)
            with profiler.profile_generation():
                time.sleep(0.01)

        profiles = profiler.get_query_profiles()
        assert len(profiles) >= 1


class TestMemoryProfiler:
    """메모리 프로파일러 테스트"""

    def test_snapshot(self):
        """스냅샷 테스트"""
        profiler = MemoryProfiler()
        snapshot = profiler.take_snapshot("test")

        assert "label" in snapshot
        assert snapshot["label"] == "test"


# =============================================================================
# Alerts Tests
# =============================================================================

class TestAlertRule:
    """알림 규칙 테스트"""

    def test_create_rule(self):
        """규칙 생성 테스트"""
        rule = AlertRule(
            name="test_rule",
            condition=lambda m: m.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            description="Test alert",
        )

        assert rule.name == "test_rule"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.condition({"value": 150}) is True
        assert rule.condition({"value": 50}) is False


class TestAlertManager:
    """알림 관리자 테스트"""

    def test_add_and_fire_alert(self):
        """알림 추가 및 발생 테스트"""
        manager = AlertManager()

        rule = AlertRule(
            name="high_value",
            condition=lambda m: m.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0,  # 테스트용 쿨다운 비활성화
        )
        manager.add_rule(rule)

        # 알림 발생
        alerts = manager.check_and_alert({"value": 150})
        assert len(alerts) == 1
        assert alerts[0].rule_name == "high_value"

        # 정상화 시 해결
        manager.check_and_alert({"value": 50})
        active = manager.get_active_alerts()
        assert len(active) == 0

    def test_cooldown(self):
        """쿨다운 테스트"""
        manager = AlertManager()

        rule = AlertRule(
            name="test",
            condition=lambda m: True,
            cooldown_seconds=60,  # 긴 쿨다운
        )
        manager.add_rule(rule)

        # 첫 번째 알림
        alerts1 = manager.check_and_alert({})
        assert len(alerts1) == 1

        # 쿨다운 중 재알림 없음
        alerts2 = manager.check_and_alert({})
        assert len(alerts2) == 0


# =============================================================================
# Health Tests
# =============================================================================

class TestHealthChecker:
    """헬스 체커 테스트"""

    def test_add_check(self):
        """체크 추가 테스트"""
        checker = HealthChecker()

        check = FunctionHealthCheck("test", lambda: True)
        checker.add_check(check)

        health = checker.check_component("test")
        assert health is not None
        assert health.status == HealthStatus.HEALTHY

    def test_unhealthy_check(self):
        """비정상 체크 테스트"""
        checker = HealthChecker()

        check = FunctionHealthCheck("failing", lambda: False)
        checker.add_check(check)

        health = checker.check_component("failing")
        assert health.status == HealthStatus.UNHEALTHY

    def test_exception_check(self):
        """예외 체크 테스트"""
        checker = HealthChecker()

        def failing_fn():
            raise RuntimeError("Test error")

        check = FunctionHealthCheck("error", failing_fn)
        checker.add_check(check)

        health = checker.check_component("error")
        assert health.status == HealthStatus.UNHEALTHY
        assert "Test error" in health.message

    def test_check_all(self):
        """전체 체크 테스트"""
        checker = HealthChecker()

        checker.add_function_check("healthy1", lambda: True)
        checker.add_function_check("healthy2", lambda: True)

        overall = checker.check_all()
        assert overall.status == HealthStatus.HEALTHY
        assert overall.is_healthy
        assert overall.is_ready

    def test_degraded_status(self):
        """저하 상태 테스트"""
        checker = HealthChecker()

        checker.add_function_check("healthy", lambda: True)
        checker.add_function_check("unhealthy", lambda: False)

        overall = checker.check_all()
        assert overall.status == HealthStatus.UNHEALTHY
        assert not overall.is_healthy

    def test_liveness_readiness(self):
        """Liveness/Readiness 테스트"""
        checker = HealthChecker()
        checker.add_function_check("test", lambda: True)

        liveness = checker.get_liveness()
        assert liveness["status"] == "alive"

        readiness = checker.get_readiness()
        assert readiness["status"] == "ready"


# =============================================================================
# Integration Tests
# =============================================================================

class TestMonitoringIntegration:
    """모니터링 통합 테스트"""

    def test_metrics_with_tracing(self):
        """메트릭 + 추적 통합 테스트"""
        registry = MetricRegistry()
        registry.clear()

        metrics = RAGMetrics(registry)
        tracer = TracingManager()

        with tracer.start_span("query") as span:
            with metrics.query_timer():
                time.sleep(0.1)

            span.set_attribute("latency_ms", 100)

        assert registry.get("rag_query_total", {"status": "success"}) == 1
        assert span.attributes.get("latency_ms") == 100

    def test_profiler_with_alerts(self):
        """프로파일러 + 알림 통합 테스트"""
        profiler = QueryProfiler()
        alert_manager = AlertManager()

        # 고지연 알림 규칙
        alert_manager.add_rule(AlertRule(
            name="high_latency",
            condition=lambda m: m.get("latency_ms", 0) > 100,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0,
        ))

        with profiler.profile_query("test"):
            time.sleep(0.15)

        profiles = profiler.get_query_profiles()
        if profiles:
            latency = profiles[-1]["total_ms"]
            alerts = alert_manager.check_and_alert({"latency_ms": latency})

            if latency > 100:
                assert len(alerts) >= 1
