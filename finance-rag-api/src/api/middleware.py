# -*- coding: utf-8 -*-
"""
API 미들웨어

[백엔드 개발자 관점]
- Spring의 Filter/Interceptor와 유사
- 요청/응답 로깅
- 요청 ID 추적
- 실행 시간 측정
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger, set_request_id

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    요청/응답 로깅 미들웨어

    - 고유 요청 ID 생성 및 추적
    - 요청/응답 시간 측정
    - 요청 메타데이터 로깅
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 요청 ID 생성
        request_id = str(uuid.uuid4())
        set_request_id(request_id)

        # 시작 시간
        start_time = time.perf_counter()

        # 요청 정보 로깅
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.query_params),
                    "client_ip": request.client.host if request.client else "unknown",
                }
            }
        )

        # 응답 처리
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(
                f"Unhandled exception: {str(e)}",
                exc_info=True
            )
            raise

        # 처리 시간 계산
        process_time = (time.perf_counter() - start_time) * 1000  # ms

        # 응답 헤더에 메타데이터 추가
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        # 응답 정보 로깅
        log_method = logger.info if response.status_code < 400 else logger.warning
        log_method(
            f"Request completed: {response.status_code} ({process_time:.2f}ms)",
            extra={
                "extra_fields": {
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time, 2),
                }
            }
        )

        return response


class CORSHeaderMiddleware(BaseHTTPMiddleware):
    """CORS 헤더 미들웨어 (간단한 CORS 지원)"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # CORS 헤더 추가
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Request-ID"

        return response
