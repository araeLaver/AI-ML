# -*- coding: utf-8 -*-
"""
커스텀 예외 클래스

[백엔드 개발자 관점]
- Spring의 @ControllerAdvice 패턴과 유사
- 계층화된 예외 구조
- 에러 코드 및 메시지 표준화
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """에러 코드 열거형"""
    # General (1xxx)
    INTERNAL_ERROR = "E1000"
    VALIDATION_ERROR = "E1001"
    NOT_FOUND = "E1002"
    RATE_LIMITED = "E1003"

    # LLM Related (2xxx)
    LLM_CONNECTION_ERROR = "E2000"
    LLM_TIMEOUT = "E2001"
    LLM_INVALID_RESPONSE = "E2002"
    LLM_MODEL_NOT_FOUND = "E2003"

    # Document Related (3xxx)
    DOCUMENT_PARSE_ERROR = "E3000"
    DOCUMENT_TOO_LARGE = "E3001"
    UNSUPPORTED_FORMAT = "E3002"
    EMPTY_DOCUMENT = "E3003"

    # Vector Store Related (4xxx)
    VECTORSTORE_ERROR = "E4000"
    EMBEDDING_ERROR = "E4001"
    SEARCH_ERROR = "E4002"

    # File Related (5xxx)
    FILE_NOT_FOUND = "E5000"
    FILE_READ_ERROR = "E5001"
    FILE_WRITE_ERROR = "E5002"


class RAGException(Exception):
    """
    RAG 시스템 기본 예외

    Attributes:
        error_code: 에러 코드
        message: 사용자 친화적 메시지
        detail: 상세 정보 (디버깅용)
        status_code: HTTP 상태 코드
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        detail: Optional[str] = None,
        status_code: int = 500
    ):
        self.error_code = error_code
        self.message = message
        self.detail = detail
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """API 응답용 딕셔너리 변환"""
        result = {
            "error": True,
            "code": self.error_code.value,
            "message": self.message,
        }
        if self.detail:
            result["detail"] = self.detail
        return result


class LLMException(RAGException):
    """LLM 관련 예외"""

    def __init__(
        self,
        error_code: ErrorCode = ErrorCode.LLM_CONNECTION_ERROR,
        message: str = "LLM 서비스 연결에 실패했습니다.",
        detail: Optional[str] = None
    ):
        super().__init__(error_code, message, detail, status_code=503)


class LLMTimeoutException(LLMException):
    """LLM 타임아웃 예외"""

    def __init__(self, detail: Optional[str] = None):
        super().__init__(
            ErrorCode.LLM_TIMEOUT,
            "LLM 응답 대기 시간이 초과되었습니다.",
            detail
        )


class LLMModelNotFoundException(LLMException):
    """LLM 모델 미발견 예외"""

    def __init__(self, model_name: str):
        super().__init__(
            ErrorCode.LLM_MODEL_NOT_FOUND,
            f"모델 '{model_name}'을(를) 찾을 수 없습니다.",
            "ollama pull 명령으로 모델을 다운로드하세요."
        )


class DocumentException(RAGException):
    """문서 처리 관련 예외"""

    def __init__(
        self,
        error_code: ErrorCode = ErrorCode.DOCUMENT_PARSE_ERROR,
        message: str = "문서 처리 중 오류가 발생했습니다.",
        detail: Optional[str] = None
    ):
        super().__init__(error_code, message, detail, status_code=400)


class UnsupportedFormatException(DocumentException):
    """지원하지 않는 파일 형식 예외"""

    def __init__(self, extension: str, supported: list):
        super().__init__(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"지원하지 않는 파일 형식입니다: {extension}",
            f"지원 형식: {', '.join(supported)}"
        )


class DocumentTooLargeException(DocumentException):
    """문서 크기 초과 예외"""

    def __init__(self, size_mb: float, max_size_mb: float):
        super().__init__(
            ErrorCode.DOCUMENT_TOO_LARGE,
            f"파일이 너무 큽니다: {size_mb:.1f}MB",
            f"최대 크기: {max_size_mb}MB"
        )


class VectorStoreException(RAGException):
    """벡터 스토어 관련 예외"""

    def __init__(
        self,
        error_code: ErrorCode = ErrorCode.VECTORSTORE_ERROR,
        message: str = "벡터 스토어 작업 중 오류가 발생했습니다.",
        detail: Optional[str] = None
    ):
        super().__init__(error_code, message, detail, status_code=500)


class EmbeddingException(VectorStoreException):
    """임베딩 생성 예외"""

    def __init__(self, detail: Optional[str] = None):
        super().__init__(
            ErrorCode.EMBEDDING_ERROR,
            "임베딩 생성 중 오류가 발생했습니다.",
            detail
        )


class SearchException(VectorStoreException):
    """검색 예외"""

    def __init__(self, detail: Optional[str] = None):
        super().__init__(
            ErrorCode.SEARCH_ERROR,
            "문서 검색 중 오류가 발생했습니다.",
            detail
        )


class ValidationException(RAGException):
    """입력 유효성 검사 예외"""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            ErrorCode.VALIDATION_ERROR,
            message,
            detail,
            status_code=422
        )


class NotFoundException(RAGException):
    """리소스 미발견 예외"""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            ErrorCode.NOT_FOUND,
            f"{resource}을(를) 찾을 수 없습니다: {identifier}",
            status_code=404
        )
