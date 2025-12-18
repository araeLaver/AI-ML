# -*- coding: utf-8 -*-
"""
전역 예외 핸들러

[백엔드 개발자 관점]
- Spring의 @ExceptionHandler와 유사
- 일관된 에러 응답 포맷
- 예외별 적절한 HTTP 상태 코드
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from src.core.exceptions import RAGException, ErrorCode
from src.core.logging import get_logger, get_request_id

logger = get_logger(__name__)


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    detail: str = None
) -> JSONResponse:
    """표준 에러 응답 생성"""
    content = {
        "error": True,
        "code": error_code,
        "message": message,
    }

    # 요청 ID 추가
    request_id = get_request_id()
    if request_id:
        content["request_id"] = request_id

    if detail:
        content["detail"] = detail

    return JSONResponse(status_code=status_code, content=content)


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """RAG 커스텀 예외 핸들러"""
    logger.warning(
        f"RAG Exception: {exc.error_code.value} - {exc.message}",
        extra={"extra_fields": {"detail": exc.detail}}
    )

    return create_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code.value,
        message=exc.message,
        detail=exc.detail
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Pydantic 유효성 검사 예외 핸들러"""
    errors = exc.errors()

    # 첫 번째 에러 메시지 추출
    if errors:
        first_error = errors[0]
        field = " -> ".join(str(loc) for loc in first_error.get("loc", []))
        msg = first_error.get("msg", "유효성 검사 실패")
        detail = f"{field}: {msg}"
    else:
        detail = "입력값 유효성 검사에 실패했습니다."

    logger.warning(f"Validation error: {detail}")

    return create_error_response(
        status_code=422,
        error_code=ErrorCode.VALIDATION_ERROR.value,
        message="입력값이 올바르지 않습니다.",
        detail=detail
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """일반 예외 핸들러 (예상치 못한 에러)"""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )

    return create_error_response(
        status_code=500,
        error_code=ErrorCode.INTERNAL_ERROR.value,
        message="서버 내부 오류가 발생했습니다.",
        detail=str(exc) if str(exc) else None
    )
