# -*- coding: utf-8 -*-
"""
구조화된 로깅 모듈

[기능]
- JSON 구조화 로깅
- 컨텍스트 정보 자동 추가
- 로그 레벨 관리
- 로그 필터링

[사용 예시]
>>> logger = StructuredLogger("rag")
>>> logger.info("Query processed", query="삼성전자", latency_ms=150)
"""

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

# 컨텍스트 저장
_context = threading.local()


class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """로그 설정"""
    level: LogLevel = LogLevel.INFO
    format: str = "json"  # "json" or "text"
    include_timestamp: bool = True
    include_caller: bool = True
    include_context: bool = True
    output: str = "stdout"  # "stdout", "stderr", or file path


class JSONFormatter(logging.Formatter):
    """JSON 로그 포매터"""

    def __init__(self, include_timestamp: bool = True, include_caller: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        if self.include_caller:
            log_data["caller"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # 추가 필드
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # 컨텍스트 정보
        context = get_log_context()
        if context:
            log_data["context"] = context

        # 예외 정보
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """텍스트 로그 포매터"""

    def __init__(self, include_timestamp: bool = True):
        fmt = ""
        if include_timestamp:
            fmt = "%(asctime)s "
        fmt += "%(levelname)s [%(name)s] %(message)s"
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        # 추가 필드를 메시지에 포함
        if hasattr(record, "extra_fields") and record.extra_fields:
            extras = " ".join(f"{k}={v}" for k, v in record.extra_fields.items())
            record.msg = f"{record.msg} | {extras}"

        return super().format(record)


class StructuredLogger:
    """
    구조화된 로거

    [특징]
    - 키-값 쌍으로 로그 기록
    - 자동 컨텍스트 포함
    - JSON 또는 텍스트 포맷
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        format: str = "json",
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        # 핸들러 설정
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)

            if format == "json":
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(TextFormatter())

            self._logger.addHandler(handler)

    def _log(self, level: int, message: str, **kwargs):
        """로그 기록"""
        extra = {"extra_fields": kwargs}
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        """DEBUG 로그"""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """INFO 로그"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """WARNING 로그"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """ERROR 로그"""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """CRITICAL 로그"""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """예외와 함께 ERROR 로그"""
        extra = {"extra_fields": kwargs}
        self._logger.exception(message, extra=extra)


class LogContext:
    """
    로그 컨텍스트 관리자

    with 문으로 컨텍스트 정보 자동 추가
    """

    def __init__(self, **context):
        self.context = context
        self._previous_context = None

    def __enter__(self):
        self._previous_context = get_log_context()
        current = dict(self._previous_context) if self._previous_context else {}
        current.update(self.context)
        set_log_context(current)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_log_context(self._previous_context or {})


def get_log_context() -> Dict[str, Any]:
    """현재 로그 컨텍스트 조회"""
    return getattr(_context, "data", {})


def set_log_context(context: Dict[str, Any]):
    """로그 컨텍스트 설정"""
    _context.data = context


def add_log_context(**kwargs):
    """로그 컨텍스트에 값 추가"""
    current = get_log_context()
    current.update(kwargs)
    set_log_context(current)


def clear_log_context():
    """로그 컨텍스트 초기화"""
    _context.data = {}


class RequestLogger:
    """
    HTTP 요청 로거

    요청/응답 자동 로깅
    """

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or StructuredLogger("request")

    def log_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str] = None,
        body_size: int = 0,
    ):
        """요청 로깅"""
        self.logger.info(
            "Request received",
            http_method=method,
            http_path=path,
            body_size=body_size,
        )

    def log_response(
        self,
        status_code: int,
        latency_ms: float,
        body_size: int = 0,
    ):
        """응답 로깅"""
        self.logger.info(
            "Response sent",
            http_status=status_code,
            latency_ms=round(latency_ms, 2),
            body_size=body_size,
        )


class QueryLogger:
    """
    RAG 쿼리 로거

    쿼리 처리 과정 로깅
    """

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or StructuredLogger("rag.query")

    def log_query_start(self, query: str, session_id: str = None):
        """쿼리 시작 로깅"""
        self.logger.info(
            "Query started",
            query=query[:100],
            query_length=len(query),
            session_id=session_id,
        )

    def log_retrieval(
        self,
        doc_count: int,
        latency_ms: float,
        top_k: int,
    ):
        """검색 결과 로깅"""
        self.logger.info(
            "Documents retrieved",
            doc_count=doc_count,
            latency_ms=round(latency_ms, 2),
            top_k=top_k,
        )

    def log_generation(
        self,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """생성 결과 로깅"""
        self.logger.info(
            "Answer generated",
            model=model,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def log_query_complete(
        self,
        total_latency_ms: float,
        success: bool = True,
        error: str = None,
    ):
        """쿼리 완료 로깅"""
        if success:
            self.logger.info(
                "Query completed",
                total_latency_ms=round(total_latency_ms, 2),
            )
        else:
            self.logger.error(
                "Query failed",
                total_latency_ms=round(total_latency_ms, 2),
                error=error,
            )


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    format: str = "json",
    log_file: str = None,
):
    """
    전역 로깅 설정

    Args:
        level: 로그 레벨
        format: 로그 포맷 ("json" or "text")
        log_file: 로그 파일 경로 (None이면 stdout)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 새 핸들러 설정
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)

    if format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # 라이브러리 로거 레벨 조정
    for lib in ["urllib3", "chromadb", "httpx"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


class AuditLogger:
    """
    감사 로거

    보안 관련 이벤트 로깅
    """

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or StructuredLogger("audit", level=LogLevel.INFO)

    def log_auth_success(self, user_id: str, method: str):
        """인증 성공"""
        self.logger.info(
            "Authentication successful",
            user_id=user_id,
            auth_method=method,
            event_type="auth.success",
        )

    def log_auth_failure(self, user_id: str, reason: str):
        """인증 실패"""
        self.logger.warning(
            "Authentication failed",
            user_id=user_id,
            reason=reason,
            event_type="auth.failure",
        )

    def log_access(self, user_id: str, resource: str, action: str):
        """리소스 접근"""
        self.logger.info(
            "Resource accessed",
            user_id=user_id,
            resource=resource,
            action=action,
            event_type="access",
        )

    def log_admin_action(self, admin_id: str, action: str, target: str):
        """관리자 작업"""
        self.logger.info(
            "Admin action",
            admin_id=admin_id,
            action=action,
            target=target,
            event_type="admin",
        )
