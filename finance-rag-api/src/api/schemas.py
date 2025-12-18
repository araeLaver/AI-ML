# -*- coding: utf-8 -*-
"""
API 스키마 정의 (Pydantic Models)

[백엔드 개발자 관점]
- DTO (Data Transfer Object) 패턴
- 요청/응답 데이터 검증
- OpenAPI 문서 자동 생성

[면접 포인트]
Q: "Pydantic을 왜 사용하나요?"
A: "타입 검증, 직렬화/역직렬화, OpenAPI 스키마 자동 생성,
    IDE 자동완성 지원 등 백엔드 개발 생산성을 높입니다."
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum


# ============================================================
# 공통 스키마
# ============================================================

class ConfidenceLevel(str, Enum):
    """신뢰도 수준"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceInfo(BaseModel):
    """출처 정보"""
    source: str = Field(..., description="문서 출처명")
    content_preview: str = Field(..., description="내용 미리보기")
    relevance_score: float = Field(..., description="관련도 점수 (높을수록 관련성 높음)")


# ============================================================
# 질의 API 스키마
# ============================================================

class QueryRequest(BaseModel):
    """RAG 질의 요청"""
    question: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="질문 내용",
        examples=["ETF가 뭔가요?", "초보자 투자 방법 알려주세요"]
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="검색할 문서 수"
    )
    filter_source: Optional[str] = Field(
        default=None,
        description="특정 출처로 필터링"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "ETF와 주식의 차이점이 뭔가요?",
                "top_k": 3,
                "filter_source": None
            }
        }
    )


class QueryResponse(BaseModel):
    """RAG 질의 응답"""
    question: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="AI 생성 답변")
    sources: List[SourceInfo] = Field(..., description="참조 문서 목록")
    confidence: ConfidenceLevel = Field(..., description="답변 신뢰도")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "ETF가 뭔가요?",
                "answer": "ETF(Exchange Traded Fund)는 주식처럼 거래되는 펀드입니다...",
                "sources": [
                    {
                        "source": "투자 가이드 2024",
                        "content_preview": "ETF는 Exchange Traded Fund의 약자로...",
                        "relevance_score": 0.85
                    }
                ],
                "confidence": "high"
            }
        }
    )


# ============================================================
# 문서 관리 API 스키마
# ============================================================

class DocumentAddRequest(BaseModel):
    """문서 추가 요청"""
    documents: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="추가할 문서 내용 리스트"
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="문서 출처명"
    )
    category: Optional[str] = Field(
        default=None,
        description="문서 카테고리"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    "ETF는 Exchange Traded Fund의 약자입니다.",
                    "채권은 정부나 기업이 발행하는 빚 증서입니다."
                ],
                "source": "투자 기초 가이드",
                "category": "투자 상품"
            }
        }
    )


class DocumentAddResponse(BaseModel):
    """문서 추가 응답"""
    success: bool = Field(..., description="성공 여부")
    added_count: int = Field(..., description="추가된 문서 수")
    total_count: int = Field(..., description="전체 문서 수")
    message: str = Field(..., description="결과 메시지")


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    total_count: int = Field(..., description="전체 문서 수")
    documents: List[Dict[str, Any]] = Field(..., description="문서 목록")


class FileUploadResponse(BaseModel):
    """파일 업로드 응답"""
    success: bool = Field(..., description="성공 여부")
    filename: str = Field(..., description="업로드된 파일명")
    source_name: str = Field(..., description="문서 출처명")
    chunks_created: int = Field(..., description="생성된 청크 수")
    total_documents: int = Field(..., description="전체 문서 수")
    message: str = Field(..., description="결과 메시지")


class ChunkingConfigRequest(BaseModel):
    """청킹 설정 요청"""
    chunk_size: int = Field(default=500, ge=100, le=2000, description="청크 크기")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="오버랩 크기")


# ============================================================
# 시스템 API 스키마
# ============================================================

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    llm_model: str = Field(..., description="사용 중인 LLM 모델")


class StatsResponse(BaseModel):
    """통계 응답"""
    total_documents: int = Field(..., description="전체 문서 수")
    llm_model: str = Field(..., description="LLM 모델")
    top_k: int = Field(..., description="기본 검색 문서 수")
    temperature: float = Field(..., description="LLM temperature")
