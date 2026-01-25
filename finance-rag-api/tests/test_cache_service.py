# -*- coding: utf-8 -*-
"""
캐시 서비스 테스트

테스트 대상:
- CacheEntry: 캐시 엔트리 데이터클래스
- CacheStats: 캐시 통계
- InMemoryCache: 인메모리 LRU 캐시
- CacheService: 메인 캐시 서비스
- CachedRAGService: RAG 캐싱 래퍼
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.cache_service import (
    CacheEntry,
    CacheStats,
    InMemoryCache,
    RedisCache,
    CacheService,
    CachedRAGService,
    get_cache_service,
    cache_result,
    get_cached,
)


# =============================================================================
# CacheEntry 테스트
# =============================================================================

class TestCacheEntry:
    """CacheEntry 테스트"""

    def test_create_cache_entry(self):
        """캐시 엔트리 생성"""
        entry = CacheEntry(key="test", value={"data": "value"})

        assert entry.key == "test"
        assert entry.value == {"data": "value"}
        assert entry.ttl == 3600
        assert entry.hits == 0
        assert entry.created_at > 0

    def test_is_expired_false(self):
        """만료되지 않은 엔트리"""
        entry = CacheEntry(key="test", value="value", ttl=3600)

        assert entry.is_expired is False

    def test_is_expired_true(self):
        """만료된 엔트리"""
        entry = CacheEntry(key="test", value="value", ttl=0)
        entry.created_at = time.time() - 10  # 10초 전 생성

        assert entry.is_expired is True

    def test_remaining_ttl(self):
        """남은 TTL 계산"""
        entry = CacheEntry(key="test", value="value", ttl=3600)

        remaining = entry.remaining_ttl
        assert 3590 <= remaining <= 3600

    def test_remaining_ttl_expired(self):
        """만료된 엔트리의 남은 TTL"""
        entry = CacheEntry(key="test", value="value", ttl=0)
        entry.created_at = time.time() - 10

        assert entry.remaining_ttl == 0


# =============================================================================
# CacheStats 테스트
# =============================================================================

class TestCacheStats:
    """CacheStats 테스트"""

    def test_create_stats(self):
        """통계 생성"""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_entries == 0
        assert stats.memory_usage_bytes == 0

    def test_hit_rate_zero(self):
        """히트율 - 조회 없음"""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """히트율 계산"""
        stats = CacheStats(hits=80, misses=20)

        assert stats.hit_rate == 0.8

    def test_hit_rate_all_hits(self):
        """100% 히트율"""
        stats = CacheStats(hits=100, misses=0)

        assert stats.hit_rate == 1.0

    def test_to_dict(self):
        """딕셔너리 변환"""
        stats = CacheStats(hits=75, misses=25, total_entries=10, memory_usage_bytes=1024)

        result = stats.to_dict()

        assert result["hits"] == 75
        assert result["misses"] == 25
        assert result["hit_rate"] == 0.75
        assert result["total_entries"] == 10
        assert result["memory_usage_bytes"] == 1024


# =============================================================================
# InMemoryCache 테스트
# =============================================================================

class TestInMemoryCache:
    """InMemoryCache 테스트"""

    def test_create_cache(self):
        """캐시 생성"""
        cache = InMemoryCache(max_size=100)

        assert cache.max_size == 100

    def test_set_and_get(self):
        """저장 및 조회"""
        cache = InMemoryCache()

        cache.set("key1", {"value": 1})
        result = cache.get("key1")

        assert result == {"value": 1}

    def test_get_nonexistent(self):
        """존재하지 않는 키 조회"""
        cache = InMemoryCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_get_expired(self):
        """만료된 항목 조회"""
        cache = InMemoryCache()
        cache.set("key1", "value", ttl=1)

        # TTL 만료 대기
        time.sleep(1.1)

        result = cache.get("key1")
        assert result is None

    def test_delete(self):
        """삭제"""
        cache = InMemoryCache()
        cache.set("key1", "value")

        result = cache.delete("key1")

        assert result is True
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        """존재하지 않는 키 삭제"""
        cache = InMemoryCache()

        result = cache.delete("nonexistent")

        assert result is False

    def test_exists(self):
        """존재 여부 확인"""
        cache = InMemoryCache()
        cache.set("key1", "value")

        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_exists_expired(self):
        """만료된 항목 존재 여부"""
        cache = InMemoryCache()
        cache.set("key1", "value", ttl=1)

        time.sleep(1.1)

        assert cache.exists("key1") is False

    def test_clear(self):
        """전체 삭제"""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        count = cache.clear()

        assert count == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_eviction(self):
        """LRU 퇴거"""
        cache = InMemoryCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # key1 조회하여 LRU 업데이트
        cache.get("key1")

        # 새 항목 추가 - key2가 제거되어야 함
        cache.set("key4", "value4")

        assert cache.exists("key1") is True
        assert cache.exists("key2") is False
        assert cache.exists("key3") is True
        assert cache.exists("key4") is True

    def test_cleanup_expired(self):
        """만료된 항목 정리"""
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=3600)

        time.sleep(1.1)

        count = cache.cleanup_expired()

        assert count == 1
        assert cache.exists("key1") is False
        assert cache.exists("key2") is True

    def test_get_stats(self):
        """통계 조회"""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()

        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_entries == 2

    def test_hit_increments_entry_hits(self):
        """조회 시 엔트리 히트 카운트 증가"""
        cache = InMemoryCache()
        cache.set("key1", "value")

        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        # 내부 엔트리 확인
        entry = cache._cache.get("key1")
        assert entry.hits == 3


# =============================================================================
# RedisCache 테스트 (모킹)
# =============================================================================

class TestRedisCache:
    """RedisCache 테스트 (Redis 연결 모킹)"""

    def test_create_without_redis(self):
        """Redis 없이 생성"""
        # Redis 모듈이 없거나 연결 실패 시
        with patch.dict('sys.modules', {'redis': None}):
            cache = RedisCache()

            assert cache.is_available is False

    def test_make_key_with_prefix(self):
        """키 프리픽스 추가"""
        cache = RedisCache()
        cache.prefix = "test:"

        key = cache._make_key("mykey")

        assert key == "test:mykey"

    def test_get_unavailable(self):
        """Redis 미사용 시 조회"""
        cache = RedisCache()
        cache._available = False

        result = cache.get("key")

        assert result is None

    def test_set_unavailable(self):
        """Redis 미사용 시 저장"""
        cache = RedisCache()
        cache._available = False

        result = cache.set("key", "value")

        assert result is False

    def test_delete_unavailable(self):
        """Redis 미사용 시 삭제"""
        cache = RedisCache()
        cache._available = False

        result = cache.delete("key")

        assert result is False

    def test_exists_unavailable(self):
        """Redis 미사용 시 존재 확인"""
        cache = RedisCache()
        cache._available = False

        result = cache.exists("key")

        assert result is False

    def test_clear_unavailable(self):
        """Redis 미사용 시 전체 삭제"""
        cache = RedisCache()
        cache._available = False

        result = cache.clear()

        assert result == 0

    def test_get_stats_unavailable(self):
        """Redis 미사용 시 통계"""
        cache = RedisCache()
        cache._available = False

        stats = cache.get_stats()

        assert isinstance(stats, CacheStats)


# =============================================================================
# CacheService 테스트
# =============================================================================

class TestCacheService:
    """CacheService 테스트"""

    def test_create_service_memory_only(self):
        """인메모리 전용 서비스"""
        service = CacheService(use_redis=False)

        assert service._redis is None
        assert isinstance(service._backend, InMemoryCache)

    def test_fallback_to_memory(self):
        """Redis 실패 시 인메모리 폴백"""
        # Redis 연결 실패 상황
        service = CacheService(use_redis=True, redis_host="invalid-host")

        # 인메모리로 폴백
        assert isinstance(service._backend, InMemoryCache)

    def test_generate_key(self):
        """캐시 키 생성"""
        key1 = CacheService.generate_key("삼성전자 실적", top_k=5)
        key2 = CacheService.generate_key("삼성전자 실적", top_k=5)
        key3 = CacheService.generate_key("삼성전자 실적", top_k=10)

        # 같은 쿼리+파라미터 = 같은 키
        assert key1 == key2
        # 다른 파라미터 = 다른 키
        assert key1 != key3
        # 키 형식
        assert key1.startswith("query:")

    def test_generate_key_with_params(self):
        """파라미터 포함 키 생성"""
        key1 = CacheService.generate_key("query", a=1, b=2)
        key2 = CacheService.generate_key("query", b=2, a=1)

        # 파라미터 순서와 무관하게 같은 키
        assert key1 == key2

    def test_set_and_get(self):
        """저장 및 조회"""
        service = CacheService(use_redis=False)

        service.set("key1", {"answer": "test"})
        result = service.get("key1")

        assert result == {"answer": "test"}

    def test_delete(self):
        """삭제"""
        service = CacheService(use_redis=False)
        service.set("key1", "value")

        result = service.delete("key1")

        assert result is True
        assert service.get("key1") is None

    def test_exists(self):
        """존재 여부"""
        service = CacheService(use_redis=False)
        service.set("key1", "value")

        assert service.exists("key1") is True
        assert service.exists("nonexistent") is False

    def test_clear(self):
        """전체 삭제"""
        service = CacheService(use_redis=False)
        service.set("key1", "value1")
        service.set("key2", "value2")

        count = service.clear()

        assert count == 2

    def test_get_stats(self):
        """통계 조회"""
        service = CacheService(use_redis=False)
        service.set("key1", "value")
        service.get("key1")
        service.get("nonexistent")

        stats = service.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["backend"] == "InMemoryCache"

    def test_cache_query_result(self):
        """쿼리 결과 캐싱"""
        service = CacheService(use_redis=False)
        result = {"answer": "삼성전자의 2024년 실적은...", "sources": []}

        key = service.cache_query_result("삼성전자 실적", result, top_k=5)

        assert key.startswith("query:")
        cached = service.get(key)
        assert cached == result

    def test_get_cached_result_hit(self):
        """캐시된 결과 조회 (히트)"""
        service = CacheService(use_redis=False)
        result = {"answer": "test", "sources": []}
        service.cache_query_result("테스트 쿼리", result, top_k=5)

        cached = service.get_cached_result("테스트 쿼리", top_k=5)

        assert cached == result

    def test_get_cached_result_miss(self):
        """캐시된 결과 조회 (미스)"""
        service = CacheService(use_redis=False)

        cached = service.get_cached_result("없는 쿼리", top_k=5)

        assert cached is None

    def test_custom_ttl(self):
        """커스텀 TTL"""
        service = CacheService(use_redis=False, default_ttl=60)

        service.set("key1", "value")  # 기본 TTL
        service.set("key2", "value", ttl=120)  # 커스텀 TTL

        # 둘 다 존재
        assert service.exists("key1")
        assert service.exists("key2")


# =============================================================================
# CachedRAGService 테스트
# =============================================================================

@dataclass
class MockRAGResponse:
    """테스트용 RAG 응답 (실제 RAGResponse 구조와 동일)"""
    question: str
    answer: str
    sources: List[dict]
    confidence: str


class TestCachedRAGService:
    """CachedRAGService 테스트"""

    def create_mock_rag_service(self):
        """모의 RAG 서비스 생성"""
        mock = Mock()
        mock.query.return_value = MockRAGResponse(
            question="테스트 쿼리",
            answer="테스트 응답입니다.",
            sources=[
                {"content": "문서1 내용", "source": "doc1"},
                {"content": "문서2 내용", "source": "doc2"},
            ],
            confidence="high"
        )
        return mock

    def test_create_cached_service(self):
        """캐시된 서비스 생성"""
        mock_rag = self.create_mock_rag_service()

        service = CachedRAGService(mock_rag, enable_cache=True)

        assert service.rag_service == mock_rag
        assert service.enable_cache is True

    def test_query_cache_miss(self):
        """쿼리 - 캐시 미스"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        result = service.query("테스트 쿼리", top_k=5)

        # RAG 서비스 호출됨
        mock_rag.query.assert_called_once()
        assert result.answer == "테스트 응답입니다."

    def test_query_cache_hit(self):
        """쿼리 - 캐시 히트"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        # 첫 번째 쿼리 (캐시 미스)
        service.query("테스트 쿼리", top_k=5)

        # 두 번째 쿼리 (캐시 히트)
        result = service.query("테스트 쿼리", top_k=5)

        # RAG 서비스는 한 번만 호출
        assert mock_rag.query.call_count == 1
        assert result.answer == "테스트 응답입니다."

    def test_query_cache_disabled(self):
        """캐시 비활성화"""
        mock_rag = self.create_mock_rag_service()
        service = CachedRAGService(mock_rag, enable_cache=False)

        service.query("쿼리1", top_k=5)
        service.query("쿼리1", top_k=5)

        # 캐시 없이 매번 호출
        assert mock_rag.query.call_count == 2

    def test_query_use_cache_false(self):
        """use_cache=False로 캐시 우회"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        service.query("쿼리1", top_k=5, use_cache=True)
        service.query("쿼리1", top_k=5, use_cache=False)

        # 두 번째는 캐시 우회
        assert mock_rag.query.call_count == 2

    def test_invalidate(self):
        """캐시 무효화"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        # 캐싱
        service.query("테스트 쿼리", top_k=5)

        # 무효화
        result = service.invalidate("테스트 쿼리", top_k=5)

        assert result is True

        # 다시 쿼리하면 캐시 미스
        service.query("테스트 쿼리", top_k=5)
        assert mock_rag.query.call_count == 2

    def test_clear_cache(self):
        """전체 캐시 삭제"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        service.query("쿼리1", top_k=5)
        service.query("쿼리2", top_k=5)

        count = service.clear_cache()

        assert count == 2

    def test_get_cache_stats(self):
        """캐시 통계"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        service.query("쿼리1", top_k=5)  # miss
        service.query("쿼리1", top_k=5)  # hit

        stats = service.get_cache_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_response_to_dict_conversion(self):
        """응답 딕셔너리 변환"""
        mock_rag = self.create_mock_rag_service()
        cache = CacheService(use_redis=False)
        service = CachedRAGService(mock_rag, cache_service=cache)

        response = MockRAGResponse(
            question="질문",
            answer="답변",
            sources=[{"content": "내용", "key": "val"}],
            confidence="high"
        )

        result = service._response_to_dict(response)

        assert result["answer"] == "답변"
        assert result["question"] == "질문"
        assert result["confidence"] == "high"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["content"] == "내용"
        assert "cached_at" in result


# =============================================================================
# 편의 함수 테스트
# =============================================================================

class TestConvenienceFunctions:
    """편의 함수 테스트"""

    def test_get_cache_service_singleton(self):
        """기본 캐시 서비스 싱글톤"""
        # 전역 캐시 초기화
        import rag.cache_service as cache_module
        cache_module._default_cache = None

        service1 = get_cache_service()
        service2 = get_cache_service()

        assert service1 is service2

    def test_cache_result(self):
        """결과 캐싱 편의 함수"""
        import rag.cache_service as cache_module
        cache_module._default_cache = CacheService(use_redis=False)

        key = cache_result("쿼리", {"answer": "답변"}, ttl=60)

        assert key.startswith("query:")

    def test_get_cached(self):
        """캐시 조회 편의 함수"""
        import rag.cache_service as cache_module
        cache_module._default_cache = CacheService(use_redis=False)

        # 캐싱
        cache_result("테스트", {"answer": "결과"})

        # 조회
        result = get_cached("테스트", top_k=5)

        assert result["answer"] == "결과"


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_cache_workflow(self):
        """전체 캐시 워크플로우"""
        # 1. 서비스 생성
        cache = CacheService(use_redis=False, max_memory_entries=100)

        # 2. 여러 결과 캐싱
        for i in range(10):
            cache.cache_query_result(
                f"쿼리 {i}",
                {"answer": f"답변 {i}", "sources": []},
                top_k=5
            )

        # 3. 통계 확인
        stats = cache.get_stats()
        assert stats["total_entries"] == 10

        # 4. 캐시 조회
        result = cache.get_cached_result("쿼리 5", top_k=5)
        assert result["answer"] == "답변 5"

        # 5. 전체 삭제
        count = cache.clear()
        assert count == 10

    def test_cache_with_complex_data(self):
        """복잡한 데이터 캐싱"""
        cache = CacheService(use_redis=False)

        complex_data = {
            "answer": "삼성전자의 2024년 실적 분석",
            "sources": [
                {"content": "문서1", "metadata": {"date": "2024-01-01"}},
                {"content": "문서2", "metadata": {"date": "2024-02-01"}},
            ],
            "metadata": {
                "query_time": 0.5,
                "reranked": True,
                "model": "gpt-4",
            }
        }

        cache.cache_query_result("삼성전자 실적", complex_data, top_k=5)
        result = cache.get_cached_result("삼성전자 실적", top_k=5)

        assert result["answer"] == complex_data["answer"]
        assert len(result["sources"]) == 2
        assert result["metadata"]["reranked"] is True

    def test_concurrent_access_safety(self):
        """동시 접근 안전성 (간단 테스트)"""
        import threading

        cache = InMemoryCache(max_size=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{threading.current_thread().name}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{threading.current_thread().name}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t1 = threading.Thread(target=writer, name=f"writer_{i}")
            t2 = threading.Thread(target=reader, name=f"reader_{i}")
            threads.extend([t1, t2])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
