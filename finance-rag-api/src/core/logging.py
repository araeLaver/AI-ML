# -*- coding: utf-8 -*-
"""
구조화된 로깅 시스템

[백엔드 개발자 관점]
- Logback/Log4j와 유사한 패턴
- JSON 포맷 지원 (프로덕션)
- 콘솔 포맷 지원 (개발)
- 요청 ID 추적
"""

import logging
import sys
import json
from datetime import datetime
from typing import Optional
from contextvars import ContextVar
from functools import lru_cache

from src.core.config import get_settings

# Context variable for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포매터 (프로덕션용)"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 요청 ID 추가
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # 예외 정보 추가
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 추가 필드
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """컬러 콘솔 포매터 (개발용)"""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)

        # 요청 ID
        request_id = request_id_var.get()
        req_id_str = f"[{request_id[:8]}] " if request_id else ""

        # 포맷
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = record.getMessage()

        formatted = (
            f"{color}{timestamp} | {record.levelname:8} | "
            f"{req_id_str}{record.name} | {message}{self.RESET}"
        )

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class LoggerAdapter(logging.LoggerAdapter):
    """추가 컨텍스트를 포함하는 로거 어댑터"""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


@lru_cache()
def setup_logging() -> None:
    """로깅 시스템 초기화"""
    settings = get_settings()

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # 기존 핸들러 제거
    root_logger.handlers.clear()

    # 핸들러 생성
    handler = logging.StreamHandler(sys.stdout)

    # 환경에 따른 포매터 선택
    if settings.app_env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())

    root_logger.addHandler(handler)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 반환

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing request", extra={"user_id": 123})
    """
    setup_logging()
    return logging.getLogger(name)


def set_request_id(request_id: str) -> None:
    """요청 ID 설정 (미들웨어에서 호출)"""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """현재 요청 ID 반환"""
    return request_id_var.get()
