"""
ELK Stack 중앙 로깅 모듈 테스트
"""

import json
import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path

from src.monitoring.logging import (
    LogEntry,
    LogAggregationStats,
    LogLevel,
    ELKLogger,
    get_elk_logger,
)


# --- LogEntry 테스트 ---

class TestLogEntry:
    def test_default_values(self):
        entry = LogEntry()
        assert entry.level == "INFO"
        assert entry.message == ""
        assert entry.service == "fraud-detection"
        assert entry.trace_id == ""
        assert entry.metadata == {}
        assert entry.timestamp is not None
        assert entry.request_id != ""

    def test_custom_values(self):
        entry = LogEntry(
            level="ERROR",
            message="test error",
            service="test-service",
            request_id="req-123",
            trace_id="trace-456",
            metadata={"key": "value"},
        )
        assert entry.level == "ERROR"
        assert entry.message == "test error"
        assert entry.service == "test-service"
        assert entry.request_id == "req-123"
        assert entry.trace_id == "trace-456"
        assert entry.metadata == {"key": "value"}

    def test_to_dict(self):
        entry = LogEntry(level="WARNING", message="test")
        d = entry.to_dict()
        assert d["level"] == "WARNING"
        assert d["message"] == "test"
        assert "timestamp" in d
        assert "request_id" in d
        assert "trace_id" in d
        assert "metadata" in d
        assert "service" in d


class TestLogAggregationStats:
    def test_default_values(self):
        stats = LogAggregationStats()
        assert stats.total_logs == 0
        assert stats.level_counts == {}
        assert stats.error_rate == 0.0

    def test_to_dict(self):
        stats = LogAggregationStats(total_logs=100, error_rate=0.05)
        d = stats.to_dict()
        assert d["total_logs"] == 100
        assert d["error_rate"] == 0.05


# --- ELKLogger 테스트 ---

class TestELKLogger:
    @pytest.fixture
    def tmp_log_dir(self, tmp_path):
        return str(tmp_path / "logs")

    @pytest.fixture
    def elk_logger(self, tmp_log_dir):
        return ELKLogger(
            service_name="test-service",
            fallback_dir=tmp_log_dir,
        )

    def test_init(self, elk_logger):
        assert elk_logger.service_name == "test-service"
        assert elk_logger.es_client is None
        assert elk_logger.logs == []

    def test_log_basic(self, elk_logger):
        entry = elk_logger.log("INFO", "test message")
        assert entry.level == "INFO"
        assert entry.message == "test message"
        assert entry.service == "test-service"
        assert len(elk_logger.logs) == 1

    def test_debug(self, elk_logger):
        entry = elk_logger.debug("debug msg")
        assert entry.level == "DEBUG"
        assert entry.message == "debug msg"

    def test_info(self, elk_logger):
        entry = elk_logger.info("info msg")
        assert entry.level == "INFO"

    def test_warning(self, elk_logger):
        entry = elk_logger.warning("warning msg")
        assert entry.level == "WARNING"

    def test_error(self, elk_logger):
        entry = elk_logger.error("error msg")
        assert entry.level == "ERROR"

    def test_critical(self, elk_logger):
        entry = elk_logger.critical("critical msg")
        assert entry.level == "CRITICAL"

    def test_log_with_trace_id(self, elk_logger):
        entry = elk_logger.info("test", trace_id="trace-123")
        assert entry.trace_id == "trace-123"

    def test_log_with_metadata(self, elk_logger):
        entry = elk_logger.info("test", metadata={"key": "val"})
        assert entry.metadata == {"key": "val"}

    def test_log_prediction(self, elk_logger):
        entry = elk_logger.log_prediction(
            model_version="v1.0",
            probability=0.85,
            is_fraud=True,
            latency_ms=12.5,
            trace_id="trace-001",
        )
        assert entry.metadata["event_type"] == "prediction"
        assert entry.metadata["probability"] == 0.85
        assert entry.metadata["is_fraud"] is True
        assert entry.trace_id == "trace-001"

    def test_log_model_load_success(self, elk_logger):
        entry = elk_logger.log_model_load(
            model_version="v2.0",
            model_path="/models/model.pkl",
            load_time_ms=150.0,
            success=True,
        )
        assert entry.level == "INFO"
        assert entry.metadata["event_type"] == "model_load"
        assert entry.metadata["success"] is True

    def test_log_model_load_failure(self, elk_logger):
        entry = elk_logger.log_model_load(
            model_version="v2.0",
            model_path="/models/model.pkl",
            load_time_ms=0,
            success=False,
            error_message="File not found",
        )
        assert entry.level == "ERROR"
        assert entry.metadata["error_message"] == "File not found"

    def test_log_drift_detection_no_drift(self, elk_logger):
        entry = elk_logger.log_drift_detection(
            drift_score=0.05,
            is_drift_detected=False,
        )
        assert entry.level == "INFO"
        assert entry.metadata["drift_score"] == 0.05

    def test_log_drift_detection_with_drift(self, elk_logger):
        entry = elk_logger.log_drift_detection(
            drift_score=0.8,
            is_drift_detected=True,
            feature_drifts={"amount": 0.9},
        )
        assert entry.level == "WARNING"
        assert entry.metadata["is_drift_detected"] is True

    def test_query_logs_all(self, elk_logger):
        elk_logger.info("msg1")
        elk_logger.error("msg2")
        elk_logger.warning("msg3")
        results = elk_logger.query_logs()
        assert len(results) == 3

    def test_query_logs_by_level(self, elk_logger):
        elk_logger.info("msg1")
        elk_logger.error("msg2")
        elk_logger.info("msg3")
        results = elk_logger.query_logs(level="ERROR")
        assert len(results) == 1
        assert results[0]["level"] == "ERROR"

    def test_query_logs_by_trace_id(self, elk_logger):
        elk_logger.info("msg1", trace_id="t1")
        elk_logger.info("msg2", trace_id="t2")
        results = elk_logger.query_logs(trace_id="t1")
        assert len(results) == 1

    def test_query_logs_limit(self, elk_logger):
        for i in range(10):
            elk_logger.info(f"msg{i}")
        results = elk_logger.query_logs(limit=3)
        assert len(results) == 3

    def test_get_aggregation_stats(self, elk_logger):
        elk_logger.info("msg1")
        elk_logger.error("msg2")
        elk_logger.error("msg3")
        elk_logger.warning("msg4")

        stats = elk_logger.get_aggregation_stats()
        assert stats.total_logs == 4
        assert stats.level_counts["INFO"] == 1
        assert stats.level_counts["ERROR"] == 2
        assert stats.error_rate == 0.5

    def test_get_aggregation_stats_empty(self, elk_logger):
        stats = elk_logger.get_aggregation_stats()
        assert stats.total_logs == 0

    def test_get_error_logs(self, elk_logger):
        elk_logger.info("ok")
        elk_logger.error("err1")
        elk_logger.critical("crit1")
        errors = elk_logger.get_error_logs()
        assert len(errors) == 2

    def test_fallback_file_write(self, elk_logger, tmp_log_dir):
        elk_logger.info("fallback test")
        # 폴백 파일이 생성되었는지 확인
        log_files = list(Path(tmp_log_dir).glob("logs-*.json"))
        assert len(log_files) == 1
        with open(log_files[0], "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["message"] == "fallback test"

    def test_max_local_logs(self, elk_logger):
        elk_logger._max_local_logs = 5
        for i in range(10):
            elk_logger.info(f"msg{i}")
        assert len(elk_logger.logs) == 5

    def test_log_level_enum(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.CRITICAL.value == "CRITICAL"
