# -*- coding: utf-8 -*-
"""
성능 최적화 모듈 테스트 (Phase 12)

캐싱, 배치 처리, 비동기 유틸리티 API 테스트
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


# ============================================================
# API 엔드포인트 테스트
# ============================================================

class TestPerformanceAPI:
    """성능 API 테스트"""

    def test_health_endpoint(self):
        """헬스체크"""
        response = client.get("/api/v1/performance/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert "cache" in data
        assert "features" in data

    def test_cache_stats(self):
        """캐시 통계 조회"""
        response = client.get("/api/v1/performance/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "backend" in data
        assert "stats" in data

    def test_performance_metrics(self):
        """성능 메트릭 조회"""
        response = client.get("/api/v1/performance/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "cache_stats" in data
        assert "summary" in data

    def test_performance_config(self):
        """성능 설정 조회"""
        response = client.get("/api/v1/performance/config")
        assert response.status_code == 200
        data = response.json()
        assert "cache" in data
        assert "batch" in data

    def test_cache_clear(self):
        """캐시 클리어"""
        response = client.post(
            "/api/v1/performance/cache/clear",
            json={"pattern": "test:*"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cleared_count" in data

    def test_cache_get_nonexistent(self):
        """존재하지 않는 캐시 키 조회"""
        response = client.get(
            "/api/v1/performance/cache/get",
            params={"key": "nonexistent_key_12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is False

    def test_metrics_reset(self):
        """메트릭 초기화"""
        response = client.post("/api/v1/performance/metrics/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_batch_process(self):
        """배치 처리"""
        response = client.post(
            "/api/v1/performance/batch/process",
            json={
                "documents": ["문서 1", "문서 2", "문서 3"],
                "source": "test",
                "batch_size": 10
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert data["total_items"] == 3
        assert data["successful"] == 3


# ============================================================
# 캐시 모듈 단위 테스트
# ============================================================

class TestCacheModule:
    """캐시 모듈 테스트"""

    def test_import_cache_module(self):
        """캐시 모듈 임포트"""
        from src.core.cache import (
            CacheConfig,
            CacheStats,
            LocalCache,
            CacheManager,
            QueryCache,
            EmbeddingCache,
        )
        assert CacheConfig is not None

    def test_cache_config_defaults(self):
        """캐시 설정 기본값"""
        from src.core.cache import CacheConfig

        config = CacheConfig()
        assert config.enabled is True
        assert config.default_ttl == 3600
        assert config.prefix == "finance_rag:"

    def test_cache_stats(self):
        """캐시 통계"""
        from src.core.cache import CacheStats

        stats = CacheStats()
        assert stats.hit_rate == 0.0

        stats.hits = 7
        stats.misses = 3
        assert stats.hit_rate == 0.7

    @pytest.mark.asyncio
    async def test_local_cache_operations(self):
        """로컬 캐시 기본 연산"""
        from src.core.cache import LocalCache, CacheConfig

        config = CacheConfig(prefix="test:")
        cache = LocalCache(config)

        # Set
        await cache.set("key1", {"value": 123})
        assert cache.stats.sets == 1

        # Get
        value = await cache.get("key1")
        assert value == {"value": 123}
        assert cache.stats.hits == 1

        # Get miss
        value = await cache.get("nonexistent")
        assert value is None
        assert cache.stats.misses == 1

        # Delete
        await cache.delete("key1")
        assert cache.stats.deletes == 1

        # Verify deleted
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_local_cache_ttl(self):
        """로컬 캐시 TTL"""
        from src.core.cache import LocalCache, CacheConfig

        config = CacheConfig(prefix="test:")
        cache = LocalCache(config)

        # 짧은 TTL로 설정
        await cache.set("expire_key", "value", ttl=1)

        # 즉시 조회 - 존재
        value = await cache.get("expire_key")
        assert value == "value"

        # 대기 후 조회 - 만료
        await asyncio.sleep(1.1)
        value = await cache.get("expire_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_manager_key_generation(self):
        """캐시 키 생성"""
        from src.core.cache import CacheManager

        key1 = CacheManager.make_key("query", top_k=5)
        key2 = CacheManager.make_key("query", top_k=5)
        key3 = CacheManager.make_key("query", top_k=10)

        assert key1 == key2  # 같은 인자는 같은 키
        assert key1 != key3  # 다른 인자는 다른 키


# ============================================================
# 배치 처리 모듈 테스트
# ============================================================

class TestBatchModule:
    """배치 처리 모듈 테스트"""

    def test_import_batch_module(self):
        """배치 모듈 임포트"""
        from src.core.batch import (
            BatchConfig,
            BatchResult,
            BatchProcessor,
            JobStatus,
            AsyncJobQueue,
        )
        assert BatchConfig is not None

    def test_batch_config_defaults(self):
        """배치 설정 기본값"""
        from src.core.batch import BatchConfig

        config = BatchConfig()
        assert config.batch_size == 100
        assert config.max_concurrent == 5
        assert config.retry_count == 3

    def test_job_status_enum(self):
        """작업 상태 열거형"""
        from src.core.batch import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

    @pytest.mark.asyncio
    async def test_batch_processor_single_item(self):
        """단일 아이템 처리"""
        from src.core.batch import BatchProcessor

        processor = BatchProcessor()

        async def process(item):
            return item * 2

        result = await processor.process_item(5, process)
        assert result.status.value == "completed"
        assert result.result == 10

    @pytest.mark.asyncio
    async def test_batch_processor_multiple_items(self):
        """여러 아이템 배치 처리"""
        from src.core.batch import BatchProcessor, BatchConfig

        config = BatchConfig(max_concurrent=3)
        processor = BatchProcessor(config)

        async def process(item):
            await asyncio.sleep(0.01)
            return item ** 2

        items = [1, 2, 3, 4, 5]
        result = await processor.process_batch(items, process)

        assert result.total_items == 5
        assert result.successful == 5
        assert result.failed == 0
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_batch_processor_with_failure(self):
        """실패가 포함된 배치 처리"""
        from src.core.batch import BatchProcessor, BatchConfig

        config = BatchConfig(retry_count=0)  # 재시도 없음
        processor = BatchProcessor(config)

        async def process(item):
            if item == 3:
                raise ValueError("Test error")
            return item

        items = [1, 2, 3, 4, 5]
        result = await processor.process_batch(items, process)

        assert result.total_items == 5
        assert result.successful == 4
        assert result.failed == 1


# ============================================================
# 비동기 유틸리티 테스트
# ============================================================

class TestAsyncUtils:
    """비동기 유틸리티 테스트"""

    def test_import_async_utils(self):
        """비동기 유틸리티 임포트"""
        from src.core.async_utils import (
            HttpConnectionPool,
            gather_with_concurrency,
            run_in_parallel,
            run_with_timeout,
            retry_async,
            PerformanceTracker,
        )
        assert HttpConnectionPool is not None

    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self):
        """동시성 제한 gather"""
        from src.core.async_utils import gather_with_concurrency

        results = []

        async def task(n):
            results.append(n)
            await asyncio.sleep(0.01)
            return n * 2

        output = await gather_with_concurrency(
            2,  # 동시 2개만
            task(1), task(2), task(3), task(4)
        )

        assert len(output) == 4
        assert set(output) == {2, 4, 6, 8}

    @pytest.mark.asyncio
    async def test_run_in_parallel(self):
        """병렬 처리"""
        from src.core.async_utils import run_in_parallel

        async def double(x):
            return x * 2

        results = await run_in_parallel(double, [1, 2, 3, 4], max_concurrent=2)
        assert results == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_run_with_timeout_success(self):
        """타임아웃 - 성공"""
        from src.core.async_utils import run_with_timeout

        async def quick_task():
            return "done"

        result = await run_with_timeout(quick_task(), timeout=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_run_with_timeout_exceeded(self):
        """타임아웃 - 초과"""
        from src.core.async_utils import run_with_timeout

        async def slow_task():
            await asyncio.sleep(10)
            return "done"

        result = await run_with_timeout(slow_task(), timeout=0.1, default="timeout")
        assert result == "timeout"

    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """재시도 - 성공"""
        from src.core.async_utils import retry_async

        call_count = 0

        async def flaky_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = await retry_async(flaky_task, max_retries=5, delay=0.01)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_performance_tracker(self):
        """성능 추적기"""
        from src.core.async_utils import PerformanceTracker

        tracker = PerformanceTracker()

        async with tracker.track("test_op"):
            await asyncio.sleep(0.01)

        metrics = tracker.get_metrics("test_op")
        assert metrics["count"] == 1
        assert metrics["avg_time"] > 0


# ============================================================
# 라우터 통합 테스트
# ============================================================

class TestRouterIntegration:
    """라우터 통합 테스트"""

    def test_performance_endpoints_registered(self):
        """성능 엔드포인트 등록 확인"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/api/v1/performance/cache/stats" in paths
        assert "/api/v1/performance/metrics" in paths
        assert "/api/v1/performance/health" in paths
        assert "/api/v1/performance/batch/process" in paths

    def test_root_endpoint_includes_performance(self):
        """루트 엔드포인트에 성능 정보 포함"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "2.5.0"
        assert "performance" in data
        assert "cache_stats" in data["performance"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
