# -*- coding: utf-8 -*-
"""
Finance RAG API - 메인 애플리케이션

금융 문서 기반 RAG (Retrieval-Augmented Generation) API 서비스

[포트폴리오 핵심]
- FastAPI로 구축한 프로덕션 레디 API
- Ollama 로컬 LLM 활용
- ChromaDB 벡터 검색
- 환각 방지 프롬프트 엔지니어링

실행:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

문서:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from .api.routes import router, get_rag_service
from .api.realtime_routes import realtime_router
from .api.multimodal_routes import multimodal_router
from .api.performance_routes import performance_router
from .api.security_routes import security_router
from .api.pipeline_routes import pipeline_router
from .api.middleware import RequestLoggingMiddleware
from .api.exception_handlers import (
    rag_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from .core.config import get_settings
from .core.exceptions import RAGException
from .core.logging import get_logger, setup_logging
from .realtime import (
    start_sync_scheduler,
    stop_sync_scheduler,
    SyncConfig,
    get_websocket_manager,
    broadcast_disclosure,
)

# 로깅 초기화
setup_logging()
logger = get_logger(__name__)


# 샘플 금융 문서 (초기 데이터)
SAMPLE_DOCUMENTS = [
    {
        "source": "투자 기초 가이드",
        "documents": [
            "ETF(Exchange Traded Fund)는 주식처럼 거래소에서 실시간 매매가 가능한 펀드입니다. 여러 종목에 분산 투자되어 있어 개별 주식보다 리스크가 낮습니다.",
            "채권은 정부나 기업이 자금을 조달하기 위해 발행하는 빚 증서입니다. 만기에 원금과 이자를 받을 수 있어 안정적인 투자 수단입니다.",
            "주식은 기업의 소유권 일부를 나타내며, 주주는 배당금과 시세차익을 기대할 수 있습니다. 기업 실적에 따라 가격이 변동합니다.",
            "분산 투자는 여러 자산에 나누어 투자하는 전략입니다. '달걀을 한 바구니에 담지 말라'는 격언처럼 리스크를 줄일 수 있습니다.",
            "복리 효과는 이자에 이자가 붙는 것을 말합니다. 장기 투자에서 복리 효과는 매우 강력하며, 이를 '눈덩이 효과'라고도 합니다."
        ]
    },
    {
        "source": "초보자 투자 가이드",
        "documents": [
            "투자를 시작하기 전 비상금을 먼저 확보하세요. 최소 3-6개월 생활비를 예금으로 보유한 후 여유 자금으로만 투자해야 합니다.",
            "초보자에게는 인덱스 ETF를 추천합니다. S&P 500이나 코스피 200 추종 ETF는 시장 전체에 투자하는 효과가 있어 개별 종목 선정 부담이 없습니다.",
            "적립식 투자(DCA, Dollar Cost Averaging)는 정기적으로 일정 금액을 투자하는 방법입니다. 시점 선택의 부담을 줄이고 평균 매입 단가를 낮출 수 있습니다.",
            "투자 전 자신의 투자 성향을 파악하세요. 공격적, 중립적, 보수적 성향에 따라 주식과 채권의 비율을 조절해야 합니다."
        ]
    },
    {
        "source": "리스크 관리 매뉴얼",
        "documents": [
            "레버리지 ETF는 기초 지수 수익률의 2배, 3배를 추종합니다. 손실도 확대되므로 초보자에게 적합하지 않습니다. 장기 보유 시 복리 손실이 발생할 수 있습니다.",
            "손절매(Stop Loss)는 손실을 제한하기 위해 미리 정한 가격에 매도하는 전략입니다. 감정적 판단을 배제하고 원칙에 따라 실행해야 합니다.",
            "투자 금액은 전체 자산의 일부만 배분하세요. 생활에 필요한 자금이나 빚을 내서 투자하는 것은 매우 위험합니다.",
            "고수익 상품은 고위험을 동반합니다. '원금 보장 고수익' 같은 광고는 사기일 가능성이 높으니 주의하세요."
        ]
    },
    {
        "source": "2024년 시장 전망",
        "documents": [
            "2024년 금리 인하 기대로 채권 가격 상승이 예상됩니다. 채권 ETF나 채권형 펀드 비중 확대를 고려할 수 있습니다.",
            "AI 반도체 수요 증가로 관련 기업들의 실적 개선이 기대됩니다. 다만 밸류에이션이 높아 단기 조정 가능성에 유의해야 합니다.",
            "미국 대선과 지정학적 리스크로 시장 변동성이 클 수 있습니다. 분산 투자와 현금 비중 유지가 중요합니다."
        ]
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리

    시작 시: 샘플 데이터 로드, 실시간 스케줄러 시작
    종료 시: 리소스 정리, 스케줄러 종료
    """
    # 시작 시 실행
    logger.info("=" * 50)
    logger.info("Finance RAG API 시작")
    logger.info("=" * 50)

    # 샘플 문서 로드
    rag_service = get_rag_service()
    stats = rag_service.get_stats()

    if stats["total_documents"] == 0:
        logger.info("초기 데이터 로드 중...")
        for doc_set in SAMPLE_DOCUMENTS:
            count = rag_service.add_documents(
                documents=doc_set["documents"],
                source=doc_set["source"]
            )
            logger.info(f"  {doc_set['source']}: {count}개 문서 추가")

        final_stats = rag_service.get_stats()
        logger.info(f"초기 데이터 로드 완료: 총 {final_stats['total_documents']}개 문서")
    else:
        logger.info(f"기존 데이터 발견: {stats['total_documents']}개 문서")

    # 실시간 동기화 스케줄러 시작
    settings = get_settings()
    if settings.sync_enabled:
        logger.info("DART 동기화 스케줄러 시작...")
        sync_config = SyncConfig(
            interval_hours=settings.sync_interval_hours,
            daily_time=settings.sync_daily_time,
            max_disclosures_per_sync=settings.sync_max_disclosures,
            lookback_days=settings.sync_lookback_days,
        )
        scheduler = start_sync_scheduler(sync_config)

        # 새 공시 발생 시 WebSocket 브로드캐스트 콜백 등록
        def on_new_disclosure(disclosure):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(broadcast_disclosure({
                        "rcept_no": getattr(disclosure, "rcept_no", ""),
                        "corp_name": getattr(disclosure, "corp_name", ""),
                        "corp_code": getattr(disclosure, "corp_code", ""),
                        "report_nm": getattr(disclosure, "report_nm", ""),
                        "rcept_dt": getattr(disclosure, "rcept_dt", ""),
                    }))
            except Exception as e:
                logger.error(f"Failed to broadcast disclosure: {e}")

        scheduler.on_new_disclosure(on_new_disclosure)
        logger.info(f"동기화 스케줄러 활성화 (주기: {settings.sync_interval_hours}시간)")
    else:
        logger.info("DART 동기화 스케줄러 비활성화 (sync_enabled=False)")

    logger.info("=" * 50)
    logger.info("API 준비 완료!")
    logger.info(f"Swagger UI: http://localhost:{settings.api_port}/docs")
    logger.info(f"WebSocket: ws://localhost:{settings.api_port}/api/v1/ws/notifications")
    logger.info(f"SSE Stream: http://localhost:{settings.api_port}/api/v1/stream/query")
    logger.info("=" * 50)

    yield  # 애플리케이션 실행

    # 종료 시 실행
    logger.info("서버 종료 중...")

    # 동기화 스케줄러 종료
    if settings.sync_enabled:
        logger.info("DART 동기화 스케줄러 종료...")
        stop_sync_scheduler()

    logger.info("서버 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="Finance RAG API",
    description="""
## 금융 문서 기반 RAG API

LLM과 벡터 검색을 결합한 금융 Q&A 시스템입니다.

### 주요 기능
- **RAG 질의**: 금융 문서를 검색하여 LLM이 답변 생성
- **문서 관리**: 금융 문서 추가/조회/삭제
- **출처 제공**: 답변의 근거 문서 명시
- **멀티모달**: PDF 표 추출, OCR, 차트 분석

### 기술 스택
- **LLM**: Ollama (llama3.2)
- **벡터 DB**: ChromaDB
- **프레임워크**: FastAPI
- **임베딩**: Ollama Embeddings

### 포트폴리오 하이라이트
- 환각 방지를 위한 프롬프트 엔지니어링
- 벡터 유사도 기반 문서 검색
- 신뢰도 점수 제공
- RESTful API 설계
    """,
    version="2.5.0",
    contact={
        "name": "김다운",
        "email": "your-email@example.com"
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

# ===== 미들웨어 등록 (순서 중요: 먼저 등록된 것이 나중에 실행) =====

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 로깅 미들웨어
app.add_middleware(RequestLoggingMiddleware)

# ===== 예외 핸들러 등록 =====
app.add_exception_handler(RAGException, rag_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ===== 라우터 등록 =====
app.include_router(router, prefix="/api/v1", tags=["RAG API"])
app.include_router(realtime_router, prefix="/api/v1", tags=["Realtime API"])
app.include_router(multimodal_router, prefix="/api/v1/multimodal", tags=["Multimodal API"])
app.include_router(performance_router, prefix="/api/v1/performance", tags=["Performance API"])
app.include_router(security_router, prefix="/api/v1/security", tags=["Security API"])
app.include_router(pipeline_router, prefix="/api/v1/pipeline", tags=["Pipeline API"])


# 루트 엔드포인트
@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트 - API 정보 반환"""
    return {
        "name": "Finance RAG API",
        "version": "2.5.0",
        "description": "금융 문서 기반 RAG Q&A 시스템 (실시간 + 멀티모달 + 성능최적화 + 보안 + 파이프라인)",
        "docs": "/docs",
        "health": "/api/v1/health",
        "realtime": {
            "websocket": "/api/v1/ws/notifications",
            "sync_status": "/api/v1/sync/status",
            "stream_query": "/api/v1/stream/query"
        },
        "multimodal": {
            "extract_tables": "/api/v1/multimodal/extract-tables",
            "ocr": "/api/v1/multimodal/ocr/image",
            "chart_analysis": "/api/v1/multimodal/analyze-chart",
            "stats": "/api/v1/multimodal/stats"
        },
        "performance": {
            "cache_stats": "/api/v1/performance/cache/stats",
            "metrics": "/api/v1/performance/metrics",
            "batch_process": "/api/v1/performance/batch/process"
        },
        "security": {
            "login": "/api/v1/security/auth/login",
            "api_keys": "/api/v1/security/api-keys",
            "audit_logs": "/api/v1/security/audit/logs",
            "health": "/api/v1/security/health"
        },
        "pipeline": {
            "create": "/api/v1/pipeline/create",
            "run": "/api/v1/pipeline/run",
            "results": "/api/v1/pipeline/results",
            "stats": "/api/v1/pipeline/stats",
            "health": "/api/v1/pipeline/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
