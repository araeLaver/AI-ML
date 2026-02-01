"""
ELK Stack 중앙 로깅 모듈
- Elasticsearch 기반 로그 수집/검색
- 구조화된 JSON 로깅
- ML 이벤트 전용 로깅
- Elasticsearch 미연결 시 로컬 파일 폴백
"""

import json
import uuid
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# Elasticsearch 선택적 임포트
try:
    from elasticsearch import Elasticsearch
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """로그 엔트리"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    level: str = "INFO"
    message: str = ""
    service: str = "fraud-detection"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "service": self.service,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


@dataclass
class LogAggregationStats:
    """로그 집계 통계"""
    total_logs: int = 0
    level_counts: Dict[str, int] = field(default_factory=dict)
    service_counts: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    time_range_start: str = ""
    time_range_end: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_logs": self.total_logs,
            "level_counts": self.level_counts,
            "service_counts": self.service_counts,
            "error_rate": self.error_rate,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
        }


class ELKLogger:
    """ELK Stack 기반 중앙 로깅 시스템"""

    def __init__(
        self,
        service_name: str = "fraud-detection",
        elasticsearch_url: Optional[str] = None,
        index_prefix: str = "mlops-logs",
        fallback_dir: str = "logs",
    ):
        self.service_name = service_name
        self.index_prefix = index_prefix
        self.fallback_dir = fallback_dir
        self.es_client: Optional[Any] = None
        self.logs: List[LogEntry] = []
        self._max_local_logs = 10000

        # Elasticsearch 연결 시도
        es_url = elasticsearch_url or os.getenv("ELASTICSEARCH_URL")
        if es_url and HAS_ELASTICSEARCH:
            try:
                self.es_client = Elasticsearch([es_url])
                if self.es_client.ping():
                    logger.info(f"Elasticsearch 연결 성공: {es_url}")
                else:
                    logger.warning("Elasticsearch ping 실패, 로컬 폴백 사용")
                    self.es_client = None
            except Exception as e:
                logger.warning(f"Elasticsearch 연결 실패: {e}, 로컬 폴백 사용")
                self.es_client = None

        # 폴백 디렉토리 생성
        Path(self.fallback_dir).mkdir(parents=True, exist_ok=True)

    def _get_index_name(self) -> str:
        """일별 인덱스 이름 생성"""
        return f"{self.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"

    def log(
        self,
        level: str,
        message: str,
        trace_id: str = "",
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """로그 기록"""
        entry = LogEntry(
            level=level,
            message=message,
            service=self.service_name,
            request_id=request_id or str(uuid.uuid4()),
            trace_id=trace_id,
            metadata=metadata or {},
        )

        # 로컬 저장
        self.logs.append(entry)
        if len(self.logs) > self._max_local_logs:
            self.logs = self.logs[-self._max_local_logs:]

        # Elasticsearch 전송
        if self.es_client:
            try:
                self.es_client.index(
                    index=self._get_index_name(),
                    body=entry.to_dict(),
                )
            except Exception as e:
                logger.warning(f"Elasticsearch 전송 실패: {e}")
                self._write_fallback(entry)
        else:
            self._write_fallback(entry)

        return entry

    def debug(self, message: str, **kwargs) -> LogEntry:
        """DEBUG 로그"""
        return self.log(LogLevel.DEBUG.value, message, **kwargs)

    def info(self, message: str, **kwargs) -> LogEntry:
        """INFO 로그"""
        return self.log(LogLevel.INFO.value, message, **kwargs)

    def warning(self, message: str, **kwargs) -> LogEntry:
        """WARNING 로그"""
        return self.log(LogLevel.WARNING.value, message, **kwargs)

    def error(self, message: str, **kwargs) -> LogEntry:
        """ERROR 로그"""
        return self.log(LogLevel.ERROR.value, message, **kwargs)

    def critical(self, message: str, **kwargs) -> LogEntry:
        """CRITICAL 로그"""
        return self.log(LogLevel.CRITICAL.value, message, **kwargs)

    def log_prediction(
        self,
        model_version: str,
        probability: float,
        is_fraud: bool,
        latency_ms: float,
        trace_id: str = "",
        request_id: str = "",
    ) -> LogEntry:
        """예측 이벤트 로깅"""
        metadata = {
            "event_type": "prediction",
            "model_version": model_version,
            "probability": probability,
            "is_fraud": is_fraud,
            "latency_ms": latency_ms,
        }
        message = f"Prediction: fraud={is_fraud}, prob={probability:.4f}, model={model_version}"
        return self.log("INFO", message, trace_id=trace_id, request_id=request_id, metadata=metadata)

    def log_model_load(
        self,
        model_version: str,
        model_path: str,
        load_time_ms: float,
        success: bool = True,
        error_message: str = "",
    ) -> LogEntry:
        """모델 로드 이벤트 로깅"""
        metadata = {
            "event_type": "model_load",
            "model_version": model_version,
            "model_path": model_path,
            "load_time_ms": load_time_ms,
            "success": success,
        }
        level = "INFO" if success else "ERROR"
        message = f"Model load: version={model_version}, success={success}"
        if error_message:
            metadata["error_message"] = error_message
            message += f", error={error_message}"
        return self.log(level, message, metadata=metadata)

    def log_drift_detection(
        self,
        drift_score: float,
        is_drift_detected: bool,
        feature_drifts: Optional[Dict[str, float]] = None,
        trace_id: str = "",
    ) -> LogEntry:
        """드리프트 감지 이벤트 로깅"""
        metadata = {
            "event_type": "drift_detection",
            "drift_score": drift_score,
            "is_drift_detected": is_drift_detected,
            "feature_drifts": feature_drifts or {},
        }
        level = "WARNING" if is_drift_detected else "INFO"
        message = f"Drift detection: score={drift_score:.4f}, detected={is_drift_detected}"
        return self.log(level, message, trace_id=trace_id, metadata=metadata)

    def query_logs(
        self,
        level: Optional[str] = None,
        service: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """로그 조회"""
        # Elasticsearch 조회
        if self.es_client:
            try:
                return self._query_elasticsearch(level, service, trace_id, start_time, end_time, limit)
            except Exception as e:
                logger.warning(f"Elasticsearch 조회 실패: {e}, 로컬 조회 사용")

        # 로컬 조회
        results = []
        for entry in reversed(self.logs):
            if level and entry.level != level:
                continue
            if service and entry.service != service:
                continue
            if trace_id and entry.trace_id != trace_id:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            results.append(entry.to_dict())
            if len(results) >= limit:
                break
        return results

    def _query_elasticsearch(
        self,
        level: Optional[str],
        service: Optional[str],
        trace_id: Optional[str],
        start_time: Optional[str],
        end_time: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Elasticsearch 쿼리"""
        must = []
        if level:
            must.append({"match": {"level": level}})
        if service:
            must.append({"match": {"service": service}})
        if trace_id:
            must.append({"match": {"trace_id": trace_id}})

        time_range = {}
        if start_time:
            time_range["gte"] = start_time
        if end_time:
            time_range["lte"] = end_time
        if time_range:
            must.append({"range": {"timestamp": time_range}})

        query = {"bool": {"must": must}} if must else {"match_all": {}}

        result = self.es_client.search(
            index=f"{self.index_prefix}-*",
            body={
                "query": query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit,
            },
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]

    def get_aggregation_stats(self) -> LogAggregationStats:
        """로그 집계 통계"""
        if not self.logs:
            return LogAggregationStats()

        level_counts: Dict[str, int] = {}
        service_counts: Dict[str, int] = {}
        error_count = 0

        for entry in self.logs:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
            service_counts[entry.service] = service_counts.get(entry.service, 0) + 1
            if entry.level in ("ERROR", "CRITICAL"):
                error_count += 1

        total = len(self.logs)
        error_rate = error_count / total if total > 0 else 0.0

        return LogAggregationStats(
            total_logs=total,
            level_counts=level_counts,
            service_counts=service_counts,
            error_rate=error_rate,
            time_range_start=self.logs[0].timestamp,
            time_range_end=self.logs[-1].timestamp,
        )

    def get_error_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """에러 로그 조회"""
        return self.query_logs(level="ERROR", limit=limit) + self.query_logs(level="CRITICAL", limit=limit)

    def _write_fallback(self, entry: LogEntry) -> None:
        """폴백 파일 기록"""
        try:
            fallback_file = Path(self.fallback_dir) / f"logs-{datetime.now().strftime('%Y-%m-%d')}.json"
            entries = []
            if fallback_file.exists():
                with open(fallback_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            entries.append(entry.to_dict())
            with open(fallback_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"폴백 파일 기록 실패: {e}")


# 싱글톤
_elk_logger: Optional[ELKLogger] = None


def get_elk_logger(service_name: str = "fraud-detection") -> ELKLogger:
    """ELK 로거 싱글톤"""
    global _elk_logger
    if _elk_logger is None:
        _elk_logger = ELKLogger(service_name=service_name)
    return _elk_logger
