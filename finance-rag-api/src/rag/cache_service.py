# -*- coding: utf-8 -*-
"""
RAG 캐싱 서비스

[기능]
- 쿼리 결과 캐싱으로 응답 속도 개선
- Redis 기반 분산 캐싱 (선택)
- 인메모리 캐싱 (기본/폴백)
- TTL 기반 자동 만료
- 캐시 통계 및 모니터링

[사용 예시]
>>> cache = CacheService()
>>> cache.set("query:삼성전자 실적", {"answer": "...", "sources": [...]})
>>> result = cache.get("query:삼성전자 실적")
"""

import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600  # 초 단위
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """만료 여부"""
        return time.time() > self.created_at + self.ttl

    @property
    def remaining_ttl(self) -> int:
        """남은 TTL (초)"""
        remaining = int(self.created_at + self.ttl - time.time())
        return max(0, remaining)


@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "total_entries": self.total_entries,
            "memory_usage_bytes": self.memory_usage_bytes,
        }


class CacheBackend(ABC):
    """캐시 백엔드 인터페이스"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> int:
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        pass


class InMemoryCache(CacheBackend):
    """
    인메모리 LRU 캐시

    [특징]
    - 스레드 안전
    - LRU 퇴거 정책
    - TTL 기반 만료
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                return None

            # LRU 업데이트
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        with self._lock:
            # 최대 크기 초과 시 가장 오래된 항목 제거
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(key=key, value=value, ttl=ttl)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats = CacheStats()
            return count

    def cleanup_expired(self) -> int:
        """만료된 항목 정리"""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        with self._lock:
            self._stats.total_entries = len(self._cache)
            # 대략적인 메모리 사용량 추정
            self._stats.memory_usage_bytes = sum(
                len(str(e.value)) for e in self._cache.values()
            )
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                total_entries=self._stats.total_entries,
                memory_usage_bytes=self._stats.memory_usage_bytes,
            )


class RedisCache(CacheBackend):
    """
    Redis 기반 분산 캐시

    [특징]
    - 분산 환경 지원
    - 영속성 옵션
    - 클러스터 지원 가능
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag:",
    ):
        self.prefix = prefix
        self._stats = CacheStats()

        try:
            import redis
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            # 연결 테스트
            self._client.ping()
            self._available = True
            logger.info(f"Redis connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self._client = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return None

        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        if not self._available:
            return False

        try:
            data = json.dumps(value, ensure_ascii=False, default=str)
            self._client.setex(self._make_key(key), ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._available:
            return False

        try:
            return self._client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        if not self._available:
            return False

        try:
            return self._client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def clear(self) -> int:
        if not self._available:
            return 0

        try:
            keys = self._client.keys(f"{self.prefix}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    def get_stats(self) -> CacheStats:
        if not self._available:
            return self._stats

        try:
            info = self._client.info("memory")
            keys = self._client.keys(f"{self.prefix}*")
            self._stats.total_entries = len(keys)
            self._stats.memory_usage_bytes = info.get("used_memory", 0)
        except Exception:
            pass

        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            total_entries=self._stats.total_entries,
            memory_usage_bytes=self._stats.memory_usage_bytes,
        )


class CacheService:
    """
    RAG 캐싱 서비스

    [특징]
    - Redis 우선, 인메모리 폴백
    - 쿼리 결과 자동 캐싱
    - 캐시 키 자동 생성
    """

    def __init__(
        self,
        use_redis: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,
        max_memory_entries: int = 1000,
    ):
        self.default_ttl = default_ttl

        # Redis 시도
        self._redis: Optional[RedisCache] = None
        if use_redis:
            self._redis = RedisCache(
                host=redis_host,
                port=redis_port,
                password=redis_password,
            )
            if not self._redis.is_available:
                self._redis = None
                logger.info("Falling back to in-memory cache")

        # 인메모리 캐시 (폴백 또는 기본)
        self._memory = InMemoryCache(max_size=max_memory_entries)

        # 활성 백엔드
        self._backend: CacheBackend = self._redis or self._memory

        logger.info(f"Cache service initialized: {type(self._backend).__name__}")

    @staticmethod
    def generate_key(query: str, **params) -> str:
        """쿼리 기반 캐시 키 생성"""
        # 파라미터 정렬하여 일관된 키 생성
        key_parts = [query]
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")

        key_string = "|".join(key_parts)
        # MD5 해시로 짧은 키 생성
        hash_key = hashlib.md5(key_string.encode()).hexdigest()

        return f"query:{hash_key}"

    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        return self._backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시 저장"""
        ttl = ttl or self.default_ttl
        return self._backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        return self._backend.delete(key)

    def exists(self, key: str) -> bool:
        """캐시 존재 여부"""
        return self._backend.exists(key)

    def clear(self) -> int:
        """전체 캐시 삭제"""
        return self._backend.clear()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        stats = self._backend.get_stats()
        return {
            **stats.to_dict(),
            "backend": type(self._backend).__name__,
        }

    def cache_query_result(
        self,
        query: str,
        result: Dict[str, Any],
        top_k: int = 5,
        ttl: Optional[int] = None,
    ) -> str:
        """쿼리 결과 캐싱"""
        key = self.generate_key(query, top_k=top_k)
        self.set(key, result, ttl)
        logger.debug(f"Cached query result: {key}")
        return key

    def get_cached_result(
        self,
        query: str,
        top_k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """캐시된 쿼리 결과 조회"""
        key = self.generate_key(query, top_k=top_k)
        result = self.get(key)

        if result:
            logger.debug(f"Cache hit: {key}")
        else:
            logger.debug(f"Cache miss: {key}")

        return result


class CachedRAGService:
    """
    캐싱이 적용된 RAG 서비스 래퍼

    기존 RAGService를 래핑하여 자동 캐싱 적용
    """

    def __init__(
        self,
        rag_service,
        cache_service: Optional[CacheService] = None,
        cache_ttl: int = 3600,
        enable_cache: bool = True,
    ):
        self.rag_service = rag_service
        self.cache = cache_service or CacheService()
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        **kwargs
    ):
        """
        캐싱이 적용된 RAG 쿼리

        Args:
            query: 검색 쿼리
            top_k: 상위 K개 결과
            use_cache: 캐시 사용 여부
            **kwargs: 추가 파라미터

        Returns:
            RAG 응답
        """
        # 캐시 확인
        if self.enable_cache and use_cache:
            cached = self.cache.get_cached_result(query, top_k)
            if cached:
                logger.info(f"Returning cached result for: {query[:50]}...")
                # 캐시된 결과를 RAGResponse 형식으로 반환
                return self._dict_to_response(cached)

        # RAG 실행
        response = self.rag_service.query(query, top_k=top_k, **kwargs)

        # 결과 캐싱
        if self.enable_cache and use_cache:
            result_dict = self._response_to_dict(response)
            self.cache.cache_query_result(
                query, result_dict, top_k, self.cache_ttl
            )

        return response

    def _response_to_dict(self, response) -> Dict[str, Any]:
        """RAGResponse를 딕셔너리로 변환"""
        return {
            "question": getattr(response, "question", ""),
            "answer": response.answer,
            "sources": response.sources if hasattr(response, "sources") else [],
            "confidence": getattr(response, "confidence", "medium"),
            "cached_at": datetime.now().isoformat(),
        }

    def _dict_to_response(self, data: Dict[str, Any]):
        """딕셔너리를 RAGResponse로 변환"""
        from .rag_service import RAGResponse

        # sources에 cached 플래그 추가
        sources = []
        for s in data.get("sources", []):
            source_copy = dict(s)
            source_copy["cached"] = True
            sources.append(source_copy)

        return RAGResponse(
            question=data.get("question", ""),
            answer=data["answer"],
            sources=sources,
            confidence=data.get("confidence", "medium"),
        )

    def invalidate(self, query: str, top_k: int = 5) -> bool:
        """특정 쿼리 캐시 무효화"""
        key = self.cache.generate_key(query, top_k=top_k)
        return self.cache.delete(key)

    def clear_cache(self) -> int:
        """전체 캐시 삭제"""
        return self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        return self.cache.get_stats()


# =============================================================================
# 편의 함수
# =============================================================================

_default_cache: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """기본 캐시 서비스 조회"""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheService()
    return _default_cache


def cache_result(query: str, result: Dict[str, Any], ttl: int = 3600) -> str:
    """결과 캐싱 (편의 함수)"""
    return get_cache_service().cache_query_result(query, result, ttl=ttl)


def get_cached(query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """캐시 조회 (편의 함수)"""
    return get_cache_service().get_cached_result(query, top_k)
