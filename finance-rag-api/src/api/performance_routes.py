# -*- coding: utf-8 -*-
"""
성능 모니터링 API 라우터

캐시, 배치 처리, 성능 메트릭 API 엔드포인트
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.cache import get_cache_manager, CacheConfig, QueryCache, EmbeddingCache
from ..core.batch import BatchProcessor, BatchConfig, AsyncJobQueue
from ..core.async_utils import get_performance_tracker

logger = get_logger(__name__)

# 라우터 생성
performance_router = APIRouter(tags=["Performance"])


# ============================================================
# 스키마 정의
# ============================================================

class CacheStatsResponse(BaseModel):
    """캐시 통계 응답"""
    enabled: bool = Field(..., description="캐시 활성화 여부")
    backend: str = Field(..., description="캐시 백엔드 (redis/local)")
    stats: dict = Field(..., description="캐시 통계")


class CacheClearRequest(BaseModel):
    """캐시 클리어 요청"""
    pattern: str = Field("*", description="클리어할 키 패턴")
    cache_type: Optional[str] = Field(None, description="캐시 유형 (query/embedding/session)")


class CacheClearResponse(BaseModel):
    """캐시 클리어 응답"""
    success: bool = Field(..., description="성공 여부")
    cleared_count: int = Field(..., description="클리어된 키 수")
    pattern: str = Field(..., description="적용된 패턴")


class PerformanceMetricsResponse(BaseModel):
    """성능 메트릭 응답"""
    metrics: dict = Field(..., description="성능 메트릭")
    cache_stats: dict = Field(..., description="캐시 통계")
    summary: dict = Field(..., description="요약")


class BatchJobRequest(BaseModel):
    """배치 작업 요청"""
    documents: list[str] = Field(..., description="처리할 문서 목록")
    source: str = Field("api", description="문서 출처")
    batch_size: int = Field(50, description="배치 크기", ge=1, le=500)


class BatchJobResponse(BaseModel):
    """배치 작업 응답"""
    batch_id: str = Field(..., description="배치 ID")
    total_items: int = Field(..., description="총 아이템 수")
    successful: int = Field(..., description="성공 수")
    failed: int = Field(..., description="실패 수")
    duration: float = Field(..., description="처리 시간 (초)")
    success_rate: float = Field(..., description="성공률")


class JobQueueStatsResponse(BaseModel):
    """작업 큐 통계 응답"""
    total_jobs: int = Field(..., description="총 작업 수")
    completed: int = Field(..., description="완료")
    failed: int = Field(..., description="실패")
    pending: int = Field(..., description="대기 중")
    workers: int = Field(..., description="워커 수")


# ============================================================
# 캐시 API
# ============================================================

@performance_router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="캐시 통계 조회",
    description="캐시 적중률, 미스율 등 통계를 조회합니다.",
)
async def get_cache_stats():
    """캐시 통계 조회"""
    try:
        cache = await get_cache_manager()
        stats = cache.get_stats()

        # 백엔드 타입 확인
        backend = "disabled"
        if cache._backend:
            backend = "redis" if hasattr(cache._backend, "_client") else "local"

        return CacheStatsResponse(
            enabled=cache.config.enabled,
            backend=backend,
            stats=stats,
        )
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@performance_router.post(
    "/cache/clear",
    response_model=CacheClearResponse,
    summary="캐시 클리어",
    description="지정된 패턴의 캐시를 삭제합니다.",
)
async def clear_cache(request: CacheClearRequest):
    """캐시 클리어"""
    try:
        cache = await get_cache_manager()

        # 캐시 유형별 클리어
        if request.cache_type == "query":
            query_cache = QueryCache(cache)
            cleared = await query_cache.clear_all()
            pattern = "query:*"
        elif request.cache_type == "embedding":
            embedding_cache = EmbeddingCache(cache)
            cleared = await cache.clear("embedding:*")
            pattern = "embedding:*"
        elif request.cache_type == "session":
            cleared = await cache.clear("session:*")
            pattern = "session:*"
        else:
            cleared = await cache.clear(request.pattern)
            pattern = request.pattern

        logger.info(f"Cache cleared: {cleared} keys with pattern '{pattern}'")

        return CacheClearResponse(
            success=True,
            cleared_count=cleared,
            pattern=pattern,
        )
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@performance_router.get(
    "/cache/get",
    summary="캐시 값 조회",
    description="특정 키의 캐시 값을 조회합니다 (디버깅용).",
)
async def get_cache_value(key: str = Query(..., description="캐시 키")):
    """캐시 값 조회"""
    try:
        cache = await get_cache_manager()
        value = await cache.get(key)

        return {
            "key": key,
            "exists": value is not None,
            "value": value,
        }
    except Exception as e:
        logger.error(f"Failed to get cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 성능 메트릭 API
# ============================================================

@performance_router.get(
    "/metrics",
    response_model=PerformanceMetricsResponse,
    summary="성능 메트릭 조회",
    description="API 성능 메트릭을 조회합니다.",
)
async def get_performance_metrics():
    """성능 메트릭 조회"""
    try:
        tracker = get_performance_tracker()
        metrics = tracker.get_metrics()

        cache = await get_cache_manager()
        cache_stats = cache.get_stats()

        # 요약 계산
        total_requests = sum(m.get("count", 0) for m in metrics.values())
        total_errors = sum(m.get("errors", 0) for m in metrics.values())
        avg_response_time = (
            sum(m.get("avg_time", 0) * m.get("count", 0) for m in metrics.values())
            / total_requests if total_requests > 0 else 0
        )

        summary = {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "avg_response_time": round(avg_response_time, 4),
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
        }

        return PerformanceMetricsResponse(
            metrics=metrics,
            cache_stats=cache_stats,
            summary=summary,
        )
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@performance_router.post(
    "/metrics/reset",
    summary="성능 메트릭 초기화",
    description="성능 메트릭을 초기화합니다.",
)
async def reset_performance_metrics():
    """성능 메트릭 초기화"""
    try:
        tracker = get_performance_tracker()
        tracker.reset()

        return {"success": True, "message": "Performance metrics reset"}
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 배치 처리 API
# ============================================================

@performance_router.post(
    "/batch/process",
    response_model=BatchJobResponse,
    summary="문서 배치 처리",
    description="여러 문서를 배치로 처리합니다.",
)
async def process_batch(request: BatchJobRequest):
    """배치 처리"""
    try:
        config = BatchConfig(
            batch_size=request.batch_size,
            max_concurrent=10,
        )
        processor = BatchProcessor(config)

        # 더미 처리 함수 (실제로는 임베딩 + 인덱싱)
        async def process_doc(doc: str) -> dict:
            # 실제 처리 로직 대신 시뮬레이션
            import asyncio
            await asyncio.sleep(0.01)  # 처리 시뮬레이션
            return {
                "length": len(doc),
                "preview": doc[:50],
            }

        result = await processor.process_batch(
            request.documents,
            process_doc,
        )

        return BatchJobResponse(
            batch_id=result.batch_id,
            total_items=result.total_items,
            successful=result.successful,
            failed=result.failed,
            duration=result.duration,
            success_rate=result.success_rate,
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 헬스체크 & 상태
# ============================================================

@performance_router.get(
    "/health",
    summary="성능 모듈 헬스체크",
    description="성능 모니터링 모듈 상태를 확인합니다.",
)
async def performance_health():
    """성능 모듈 헬스체크"""
    try:
        cache = await get_cache_manager()
        cache_healthy = cache._backend is not None if cache.config.enabled else True

        return {
            "status": "ok",
            "cache": {
                "enabled": cache.config.enabled,
                "healthy": cache_healthy,
            },
            "features": {
                "caching": True,
                "batch_processing": True,
                "performance_tracking": True,
            },
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }


@performance_router.get(
    "/config",
    summary="성능 설정 조회",
    description="현재 성능 관련 설정을 조회합니다.",
)
async def get_performance_config():
    """성능 설정 조회"""
    try:
        cache = await get_cache_manager()

        return {
            "cache": {
                "enabled": cache.config.enabled,
                "redis_url": cache.config.redis_url.split("@")[-1] if "@" in cache.config.redis_url else cache.config.redis_url,
                "default_ttl": cache.config.default_ttl,
                "prefix": cache.config.prefix,
            },
            "batch": {
                "default_batch_size": 100,
                "max_concurrent": 5,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
