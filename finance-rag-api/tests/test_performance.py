# -*- coding: utf-8 -*-
"""
성능 최적화 모듈 테스트
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.performance.batch_processor import (
    BatchConfig,
    BatchProcessor,
    BatchResult,
    EmbeddingBatchProcessor,
    QueryBatchProcessor,
    StreamingBatchProcessor,
)
from src.performance.connection_pool import (
    ConnectionManager,
    ConnectionPool,
    PoolConfig,
    PooledConnection,
    PoolStats,
)
from src.performance.async_executor import (
    AsyncExecutor,
    BackgroundTaskManager,
    ParallelQueryExecutor,
    TaskStatus,
)
from src.performance.index_optimizer import (
    ChromaIndexOptimizer,
    IndexStats,
    OptimizationRecommendation,
)
from src.performance.multilevel_cache import (
    CacheConfig,
    L1MemoryCache,
    L2RedisCache,
    MultiLevelCache,
    SemanticCache,
)


# =============================================================================
# Batch Processor Tests
# =============================================================================

class TestBatchConfig:
    """배치 설정 테스트"""

    def test_default_config(self):
        config = BatchConfig()
        assert config.batch_size == 100
        assert config.max_workers == 4
        assert config.retry_count == 3

    def test_custom_config(self):
        config = BatchConfig(batch_size=50, max_workers=8)
        assert config.batch_size == 50
        assert config.max_workers == 8


class TestEmbeddingBatchProcessor:
    """임베딩 배치 처리기 테스트"""

    def test_process_batch(self):
        """배치 처리 테스트"""
        # Mock 임베딩 함수
        def mock_embed(texts):
            return [[0.1] * 384 for _ in texts]

        config = BatchConfig(batch_size=5)
        processor = EmbeddingBatchProcessor(
            embedding_fn=mock_embed,
            config=config,
        )

        texts = ["text1", "text2", "text3", "text4", "text5", "text6"]
        result = processor.process(texts)

        assert isinstance(result, BatchResult)
        assert len(result.results) == 6
        assert result.batch_count == 2  # 6 items / 5 batch_size = 2 batches
        assert len(result.failed_indices) == 0

    def test_process_with_progress(self):
        """진행 상황 콜백 테스트"""
        def mock_embed(texts):
            return [[0.1] * 384 for _ in texts]

        processor = EmbeddingBatchProcessor(
            embedding_fn=mock_embed,
            config=BatchConfig(batch_size=2),
        )

        progress_calls = []
        def progress_callback(processed, total):
            progress_calls.append((processed, total))

        texts = ["a", "b", "c", "d"]
        result = processor.process(texts, progress_callback=progress_callback)

        assert len(result.results) == 4
        assert len(progress_calls) == 2  # 2 batches

    def test_get_stats(self):
        """통계 조회 테스트"""
        def mock_embed(texts):
            time.sleep(0.01)
            return [[0.1] * 384 for _ in texts]

        processor = EmbeddingBatchProcessor(
            embedding_fn=mock_embed,
            config=BatchConfig(batch_size=2),
        )

        texts = ["a", "b", "c", "d"]
        processor.process(texts)

        stats = processor.get_stats()
        assert stats.processed_items == 4
        assert stats.avg_batch_time_ms > 0


class TestQueryBatchProcessor:
    """쿼리 배치 처리기 테스트"""

    def test_process_queries(self):
        """쿼리 처리 테스트"""
        def mock_query(q):
            return {"query": q, "answer": f"Answer to {q}"}

        processor = QueryBatchProcessor(
            query_fn=mock_query,
            config=BatchConfig(batch_size=2, max_workers=2),
        )

        queries = ["Q1", "Q2", "Q3"]
        result = processor.process(queries)

        assert len(result.results) == 3
        assert result.results[0]["query"] == "Q1"


class TestStreamingBatchProcessor:
    """스트리밍 배치 처리기 테스트"""

    def test_stream_processing(self):
        """스트리밍 처리 테스트"""
        def mock_process(batch):
            return [x * 2 for x in batch]

        processor = StreamingBatchProcessor(
            process_fn=mock_process,
            batch_size=3,
        )

        items = iter([1, 2, 3, 4, 5])
        results = list(processor.process_stream(items))

        assert results == [2, 4, 6, 8, 10]


# =============================================================================
# Connection Pool Tests
# =============================================================================

class TestPoolConfig:
    """풀 설정 테스트"""

    def test_default_config(self):
        config = PoolConfig()
        assert config.max_size == 10
        assert config.min_size == 2
        assert config.timeout == 5.0


class TestPooledConnection:
    """풀링된 연결 테스트"""

    def test_connection_age(self):
        """연결 나이 테스트"""
        mock_pool = MagicMock()
        conn = PooledConnection("test_conn", mock_pool)

        time.sleep(0.1)
        assert conn.age >= 0.1

    def test_connection_touch(self):
        """사용 시간 갱신 테스트"""
        mock_pool = MagicMock()
        conn = PooledConnection("test_conn", mock_pool)

        initial_count = conn._use_count
        conn.touch()
        assert conn._use_count == initial_count + 1


class TestConnectionManager:
    """연결 관리자 테스트"""

    def test_singleton(self):
        """싱글톤 패턴 테스트"""
        manager1 = ConnectionManager()
        manager2 = ConnectionManager()
        assert manager1 is manager2


# =============================================================================
# Async Executor Tests
# =============================================================================

class TestAsyncExecutor:
    """비동기 실행기 테스트"""

    @pytest.mark.asyncio
    async def test_execute(self):
        """단일 실행 테스트"""
        executor = AsyncExecutor()

        def sync_func(x):
            return x * 2

        result = await executor.execute(sync_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        """병렬 실행 테스트"""
        executor = AsyncExecutor(max_workers=4)

        tasks = [lambda i=i: i * 2 for i in range(5)]
        results = await executor.execute_parallel(tasks)

        assert len(results) == 5
        assert sorted(results) == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """타임아웃 테스트"""
        executor = AsyncExecutor()

        def slow_func():
            time.sleep(2)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await executor.execute(slow_func, timeout=0.1)


class TestBackgroundTaskManager:
    """백그라운드 태스크 관리자 테스트"""

    def test_submit_task(self):
        """태스크 제출 테스트"""
        manager = BackgroundTaskManager()

        def simple_task():
            return "result"

        task_id = manager.submit("test_task", simple_task)
        assert task_id is not None

        result = manager.get_result(task_id, timeout=5)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "result"

    def test_task_failure(self):
        """태스크 실패 테스트"""
        manager = BackgroundTaskManager()

        def failing_task():
            raise ValueError("Test error")

        task_id = manager.submit("failing", failing_task)
        result = manager.get_result(task_id, timeout=5)

        assert result.status == TaskStatus.FAILED
        assert "Test error" in result.error

    def test_list_tasks(self):
        """태스크 목록 테스트"""
        manager = BackgroundTaskManager()

        manager.submit("task1", lambda: "a")
        manager.submit("task2", lambda: "b")

        tasks = manager.list_tasks()
        assert len(tasks) >= 2


# =============================================================================
# Index Optimizer Tests
# =============================================================================

class TestIndexStats:
    """인덱스 통계 테스트"""

    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        stats = IndexStats(
            collection_name="test",
            document_count=1000,
            avg_query_time_ms=50.0,
        )

        data = stats.to_dict()
        assert data["collection_name"] == "test"
        assert data["document_count"] == 1000
        assert data["performance"]["avg_query_time_ms"] == 50.0


class TestChromaIndexOptimizer:
    """ChromaDB 인덱스 최적화기 테스트"""

    def test_analyze_with_mock_collection(self):
        """분석 테스트 (모킹)"""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 500
        mock_collection.metadata = {}
        mock_collection.get.return_value = {
            "embeddings": [[0.1] * 384]
        }
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
        }

        optimizer = ChromaIndexOptimizer(
            collection=mock_collection,
            sample_queries=["test query"],
        )

        stats = optimizer.analyze()
        assert stats.collection_name == "test_collection"
        assert stats.document_count == 500

    def test_recommendations(self):
        """권장 사항 테스트"""
        mock_collection = MagicMock()
        mock_collection.name = "large_collection"
        mock_collection.count.return_value = 150000  # 대규모
        mock_collection.metadata = {}
        mock_collection.get.return_value = {"embeddings": [[0.1] * 384]}
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
        }

        optimizer = ChromaIndexOptimizer(mock_collection)
        optimizer.analyze()

        recommendations = optimizer.get_recommendations()
        # 대규모 컬렉션에 대해 M 파라미터 권장 있어야 함
        param_names = [r.parameter for r in recommendations]
        assert "hnsw:M" in param_names


# =============================================================================
# Multi-level Cache Tests
# =============================================================================

class TestL1MemoryCache:
    """L1 인메모리 캐시 테스트"""

    def test_set_and_get(self):
        """저장 및 조회 테스트"""
        cache = L1MemoryCache()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_lru_eviction(self):
        """LRU 퇴거 테스트"""
        cache = L1MemoryCache(max_size=2)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # 'a' 퇴거됨

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_ttl_expiration(self):
        """TTL 만료 테스트"""
        cache = L1MemoryCache(default_ttl=1)

        cache.set("key", "value", ttl=1)
        assert cache.get("key") == "value"

        time.sleep(1.1)
        assert cache.get("key") is None

    def test_stats(self):
        """통계 테스트"""
        cache = L1MemoryCache()

        cache.set("key", "value")
        cache.get("key")  # hit
        cache.get("missing")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestSemanticCache:
    """시맨틱 캐시 테스트"""

    def test_cache_and_find_similar(self):
        """캐싱 및 유사 검색 테스트"""
        # Mock 임베딩 함수
        def mock_embed(text):
            if "삼성" in text:
                return [0.9, 0.1, 0.0]
            elif "SK" in text:
                return [0.1, 0.9, 0.0]
            else:
                return [0.5, 0.5, 0.0]

        cache = SemanticCache(
            embedding_fn=mock_embed,
            threshold=0.8,
        )

        # 캐싱
        cache.cache_query("삼성전자 실적", {"answer": "삼성 답변"})

        # 유사 쿼리로 검색
        result = cache.find_similar("삼성 영업이익")
        assert result is not None
        original_query, cached_result, similarity = result
        assert "삼성" in original_query
        assert similarity > 0.8

        # 다른 쿼리는 캐시 미스
        result = cache.find_similar("SK하이닉스 실적")
        assert result is None

    def test_threshold(self):
        """임계값 테스트"""
        def mock_embed(text):
            # 매우 다른 임베딩 생성
            if text == "positive":
                return [1.0, 0.0, 0.0]
            elif text == "negative":
                return [-1.0, 0.0, 0.0]  # 완전 반대 방향
            else:
                return [0.0, 1.0, 0.0]  # 직교

        cache = SemanticCache(
            embedding_fn=mock_embed,
            threshold=0.95,  # 높은 임계값
        )

        cache.cache_query("positive", {"answer": "test"})

        # 반대 방향 벡터는 유사도 낮음
        result = cache.find_similar("negative")
        assert result is None  # 코사인 유사도 = -1

        # 직교 벡터도 유사도 낮음
        result = cache.find_similar("other")
        assert result is None  # 코사인 유사도 = 0


class TestMultiLevelCache:
    """멀티레벨 캐시 테스트"""

    def test_l1_only(self):
        """L1만 사용하는 테스트"""
        config = CacheConfig(enable_l2=False, enable_semantic=False)
        cache = MultiLevelCache(config=config)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        stats = cache.get_stats()
        assert stats["l1_hits"] == 1

    def test_cache_promotion(self):
        """L2 → L1 승격 테스트 (L2 사용 가능 시)"""
        config = CacheConfig(enable_l2=True, enable_semantic=False)
        cache = MultiLevelCache(config=config)

        # L1과 L2에 모두 저장
        cache.set("key", "value")

        # L1 클리어
        cache._l1.clear()

        # L2에서 조회 시 L1으로 승격
        if cache._l2 and cache._l2.is_available:
            value = cache.get("key")
            # L1에도 있어야 함
            assert cache._l1.get("key") == value

    def test_clear_all_levels(self):
        """전체 레벨 삭제 테스트"""
        config = CacheConfig(enable_l2=False, enable_semantic=False)
        cache = MultiLevelCache(config=config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        result = cache.clear()
        assert result["l1"] >= 2

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheStats:
    """캐시 통계 테스트"""

    def test_hit_rates(self):
        """히트율 계산 테스트"""
        config = CacheConfig(enable_l2=False, enable_semantic=False)
        cache = MultiLevelCache(config=config)

        cache.set("key", "value")

        # 3 hits
        cache.get("key")
        cache.get("key")
        cache.get("key")

        # 2 misses
        cache.get("missing1")
        cache.get("missing2")

        stats = cache.get_stats()
        assert stats["l1_hits"] == 3
        assert stats["l1_misses"] == 2
        assert stats["l1_hit_rate"] == 0.6


# =============================================================================
# Integration Tests
# =============================================================================

class TestPerformanceIntegration:
    """성능 모듈 통합 테스트"""

    def test_batch_with_cache(self):
        """배치 처리 + 캐싱 통합 테스트"""
        # 캐시 설정
        config = CacheConfig(enable_l2=False, enable_semantic=False)
        cache = MultiLevelCache(config=config)

        # 배치 처리기 설정
        def mock_embed(texts):
            return [[0.1] * 384 for _ in texts]

        processor = EmbeddingBatchProcessor(
            embedding_fn=mock_embed,
            config=BatchConfig(batch_size=2),
        )

        # 처리 및 캐싱
        texts = ["a", "b", "c"]
        result = processor.process(texts)

        for i, text in enumerate(texts):
            cache.set(f"embed:{text}", result.results[i])

        # 캐시에서 조회
        cached = cache.get("embed:a")
        assert cached is not None
        assert len(cached) == 384

    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """비동기 배치 처리 테스트"""
        executor = AsyncExecutor(max_workers=2)

        def slow_embed(texts):
            time.sleep(0.05)
            return [[0.1] * 10 for _ in texts]

        processor = EmbeddingBatchProcessor(
            embedding_fn=slow_embed,
            config=BatchConfig(batch_size=2),
        )

        # 비동기로 배치 처리
        result = await executor.execute(
            processor.process,
            ["a", "b", "c", "d"]
        )

        assert len(result.results) == 4
