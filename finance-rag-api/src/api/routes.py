# -*- coding: utf-8 -*-
"""
API 라우트 정의

[백엔드 개발자 관점]
- Controller Layer 패턴
- RESTful 엔드포인트 설계
- 의존성 주입 활용

[API 설계 원칙]
- GET: 조회 (멱등성 O)
- POST: 생성/복잡한 조회
- DELETE: 삭제

[보안]
- 공개 API: health, stats, documents 조회
- 보호 API: query, documents 추가/삭제, upload (Rate Limited)
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from typing import Optional
import io

from .schemas import (
    QueryRequest, QueryResponse,
    DocumentAddRequest, DocumentAddResponse, DocumentListResponse,
    HealthResponse, StatsResponse, FileUploadResponse
)
from .security import check_rate_limit, optional_api_key, APIKeyInfo
from ..rag.rag_service import RAGService
from ..rag.vectorstore import VectorStoreService
from ..rag.document_loader import DocumentLoaderFactory, ChunkingConfig
from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# 라우터 생성
router = APIRouter()

# 전역 서비스 인스턴스 (싱글톤)
_vectorstore: Optional[VectorStoreService] = None
_rag_service: Optional[RAGService] = None


def get_vectorstore() -> VectorStoreService:
    """VectorStore 의존성"""
    global _vectorstore
    if _vectorstore is None:
        settings = get_settings()
        _vectorstore = VectorStoreService(
            persist_dir=settings.chroma_persist_dir,
            collection_name="finance_docs"
        )
    return _vectorstore


def get_rag_service() -> RAGService:
    """RAGService 의존성"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            vectorstore=get_vectorstore(),
            llm_model="llama3.2",
            top_k=3,
            temperature=0.2
        )
    return _rag_service


# ============================================================
# 시스템 API
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스체크",
    description="서비스 상태를 확인합니다."
)
async def health_check():
    """
    서비스 헬스체크

    Kubernetes Liveness/Readiness Probe 용도로 사용
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_model="llama3.2"
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="시스템 통계",
    description="RAG 시스템 통계를 조회합니다."
)
async def get_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """시스템 통계 조회"""
    stats = rag_service.get_stats()
    return StatsResponse(**stats)


# ============================================================
# 질의 API (핵심!)
# ============================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG 질의",
    description="금융 관련 질문에 대해 문서 기반으로 답변합니다. (Rate Limited)",
    responses={
        200: {"description": "질의 성공"},
        400: {"description": "잘못된 요청"},
        401: {"description": "인증 필요"},
        429: {"description": "요청 한도 초과"},
        500: {"description": "서버 오류"}
    }
)
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    api_key_info: APIKeyInfo = Depends(check_rate_limit)
):
    """
    RAG 질의 처리

    1. 벡터 DB에서 관련 문서 검색
    2. LLM으로 답변 생성
    3. 출처와 함께 반환

    면접 포인트:
    - 비동기 처리 (async/await)
    - 의존성 주입 (Depends)
    - 응답 스키마 명시
    """
    try:
        # 필터 구성
        filter_metadata = None
        if request.filter_source:
            filter_metadata = {"source": request.filter_source}

        # RAG 질의 실행
        result = rag_service.query(
            question=request.question,
            filter_metadata=filter_metadata
        )

        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                {
                    "source": s["source"],
                    "content_preview": s["content_preview"],
                    "relevance_score": s["relevance_score"]
                }
                for s in result.sources
            ],
            confidence=result.confidence
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"질의 처리 중 오류 발생: {str(e)}"
        )


# ============================================================
# 문서 관리 API
# ============================================================

@router.post(
    "/documents",
    response_model=DocumentAddResponse,
    summary="문서 추가",
    description="RAG 시스템에 새 문서를 추가합니다. (Rate Limited)",
    status_code=status.HTTP_201_CREATED
)
async def add_documents(
    request: DocumentAddRequest,
    rag_service: RAGService = Depends(get_rag_service),
    api_key_info: APIKeyInfo = Depends(check_rate_limit)
):
    """
    문서 추가

    금융 문서를 벡터 DB에 저장합니다.
    추가된 문서는 질의 시 검색 대상이 됩니다.
    """
    try:
        # 추가 메타데이터
        additional = {}
        if request.category:
            additional["category"] = request.category

        # 문서 추가
        added = rag_service.add_documents(
            documents=request.documents,
            source=request.source,
            additional_metadata=additional if additional else None
        )

        stats = rag_service.get_stats()

        return DocumentAddResponse(
            success=True,
            added_count=added,
            total_count=stats["total_documents"],
            message=f"{added}개 문서가 추가되었습니다."
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 추가 중 오류 발생: {str(e)}"
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="문서 목록",
    description="저장된 문서 목록을 조회합니다."
)
async def list_documents(
    limit: int = 100,
    vectorstore: VectorStoreService = Depends(get_vectorstore)
):
    """문서 목록 조회"""
    try:
        result = vectorstore.list_all_documents(limit=limit)

        documents = []
        for i, doc_id in enumerate(result["ids"]):
            documents.append({
                "id": doc_id,
                "content_preview": result["documents"][i][:100] + "..."
                    if len(result["documents"][i]) > 100
                    else result["documents"][i],
                "metadata": result["metadatas"][i] if result["metadatas"] else {}
            })

        return DocumentListResponse(
            total_count=len(documents),
            documents=documents
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 조회 중 오류 발생: {str(e)}"
        )


@router.delete(
    "/documents",
    summary="전체 문서 삭제",
    description="모든 문서를 삭제합니다. (주의: 복구 불가, Rate Limited)"
)
async def delete_all_documents(
    vectorstore: VectorStoreService = Depends(get_vectorstore),
    api_key_info: APIKeyInfo = Depends(check_rate_limit)
):
    """전체 문서 삭제 (리셋)"""
    try:
        vectorstore.delete_collection()
        return {"success": True, "message": "모든 문서가 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"삭제 중 오류 발생: {str(e)}"
        )


# ============================================================
# 파일 업로드 API (포트폴리오 핵심!)
# ============================================================

@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="PDF/텍스트 파일 업로드",
    description="PDF 또는 텍스트 파일을 업로드하여 RAG 시스템에 추가합니다. (Rate Limited)",
    status_code=status.HTTP_201_CREATED
)
async def upload_file(
    file: UploadFile = File(..., description="업로드할 파일 (PDF, TXT, MD)"),
    source_name: Optional[str] = Form(None, description="문서 출처명 (미입력시 파일명 사용)"),
    chunk_size: int = Form(500, ge=100, le=2000, description="청크 크기"),
    chunk_overlap: int = Form(100, ge=0, le=500, description="청크 오버랩"),
    vectorstore: VectorStoreService = Depends(get_vectorstore),
    api_key_info: APIKeyInfo = Depends(check_rate_limit)
):
    """
    파일 업로드 및 RAG 시스템 등록

    [지원 형식]
    - PDF (.pdf): 금융 보고서, 투자 설명서 등
    - 텍스트 (.txt, .md): 일반 문서

    [처리 과정]
    1. 파일 형식 검증
    2. 텍스트 추출 (PDF → 텍스트)
    3. 청킹 (의미 단위 분할)
    4. 임베딩 생성 및 벡터 DB 저장

    [면접 포인트]
    - 대용량 파일 스트림 처리
    - 청킹 전략 설명 가능
    - 메타데이터 관리
    """
    # 파일 확장자 검증
    supported = DocumentLoaderFactory.get_supported_extensions()
    file_ext = f".{file.filename.split('.')[-1].lower()}" if file.filename else ""

    if file_ext not in supported:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported)}"
        )

    try:
        # 파일 내용 읽기
        content = await file.read()
        file_stream = io.BytesIO(content)

        # 청킹 설정
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 로더 선택 및 문서 로드
        loader = DocumentLoaderFactory.get_loader(file.filename, chunking_config)

        # PDF vs 텍스트 처리
        if file_ext == ".pdf":
            documents = loader.load_from_bytes(
                file_stream,
                file.filename,
                source_name
            )
        else:
            text_content = content.decode("utf-8")
            documents = loader.load_from_string(
                text_content,
                source_name or file.filename.rsplit(".", 1)[0]
            )

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="파일에서 텍스트를 추출할 수 없습니다."
            )

        # 벡터 DB에 저장
        doc_contents = [doc.content for doc in documents]
        doc_ids = [doc.id for doc in documents]
        doc_metadatas = [doc.metadata for doc in documents]

        vectorstore.add_documents(
            documents=doc_contents,
            ids=doc_ids,
            metadatas=doc_metadatas
        )

        total_count = vectorstore.get_document_count()
        actual_source = documents[0].metadata.get("source", file.filename)

        return FileUploadResponse(
            success=True,
            filename=file.filename,
            source_name=actual_source,
            chunks_created=len(documents),
            total_documents=total_count,
            message=f"'{file.filename}'에서 {len(documents)}개 청크가 생성되어 저장되었습니다."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 처리 중 오류 발생: {str(e)}"
        )
