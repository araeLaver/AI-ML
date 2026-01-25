# -*- coding: utf-8 -*-
"""
멀티레벨 캐싱 모듈

[기능]
- L1: 인메모리 캐시 (초고속)
- L2: Redis 캐시 (분산)
- 시맨틱 캐싱 (유사 쿼리 재사용)
- 캐시 워밍
- 자동 무효화

[사용 예시]
>>> cache = MultiLevelCache()
>>> cache.set("query:삼성전자", result)
>>> cached = cache.get("query:삼성전자")

>>> semantic_cache = SemanticCache(embedding_fn)
>>> semantic_cache.cache_query("삼성전자 실적", result)
>>> similar = semantic_cache.find_similar("삼성 영업이익")  # 캐시 히트!
"""

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """캐시 설정"""
    l1_max_size: int = 1000
    l1_ttl: int = 300  # 5분
    l2_ttl: int = 3600  # 1시간
    semantic_threshold: float = 0.85  # 유사도 임계값
    enable_l2: bool = True
    enable_semantic: bool = True
    warmup_on_start: bool = False


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600
    hits: int = 0
    level: str = "L1"
    embedding: Optional[List[float]] = None

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """캐시 통계"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    semantic_hits: int = 0
    total_entries: int = 0
    avg_hit_latency_ms: float = 0.0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        hits = self.l1_hits + self.l2_hits + self.semantic_hits
        misses = self.l1_misses  # L2 miss는 L1 miss 이후에만 발생
        total = hits + misses
        return hits / total if total > 0 else 0.0


class CacheLevel(ABC):
    """캐시 레벨 인터페이스"""

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
    def clear(self) -> int:
        pass


class L1MemoryCache(CacheLevel):
    """
    L1 인메모리 캐시

    [특징]
    - 나노초 수준 접근
    - LRU 퇴거
    - 스레드 안전
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # LRU 업데이트
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats["hits"] += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        ttl = ttl or self.default_ttl

        with self._lock:
            # 최대 크기 초과 시 LRU 퇴거
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                level="L1"
            )
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())


class L2RedisCache(CacheLevel):
    """
    L2 Redis 캐시

    [특징]
    - 분산 환경 지원
    - 영속성
    - 클러스터 확장 가능
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag:l2:",
        default_ttl: int = 3600,
    ):
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._client = None
        self._available = False
        self._stats = {"hits": 0, "misses": 0}

        try:
            import redis
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            self._client.ping()
            self._available = True
            logger.info(f"L2 Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"L2 Redis not available: {e}")

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return None

        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return json.loads(data)
        except Exception as e:
            logger.error(f"L2 get error: {e}")
            self._stats["misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self._available:
            return False

        ttl = ttl or self.default_ttl

        try:
            data = json.dumps(value, ensure_ascii=False, default=str)
            self._client.setex(self._make_key(key), ttl, data)
            return True
        except Exception as e:
            logger.error(f"L2 set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._available:
            return False

        try:
            return self._client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"L2 delete error: {e}")
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
            logger.error(f"L2 clear error: {e}")
            return 0

    @property
    def is_available(self) -> bool:
        return self._available

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


class SemanticCache:
    """
    시맨틱 캐시

    [특징]
    - 임베딩 기반 유사도 매칭
    - 유사한 쿼리 결과 재사용
    - 동적 임계값 조정
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], List[float]],
        threshold: float = 0.85,
        max_entries: int = 500,
    ):
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: List[CacheEntry] = []
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """코사인 유사도 계산"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def cache_query(
        self,
        query: str,
        result: Any,
        ttl: int = 3600,
    ) -> bool:
        """
        쿼리 결과 캐싱

        Args:
            query: 쿼리 문자열
            result: 결과
            ttl: TTL

        Returns:
            캐싱 성공 여부
        """
        try:
            embedding = self.embedding_fn(query)

            with self._lock:
                # 최대 크기 초과 시 오래된 항목 제거
                while len(self._entries) >= self.max_entries:
                    self._entries.pop(0)

                # 만료된 항목 정리
                self._entries = [
                    e for e in self._entries if not e.is_expired
                ]

                key = hashlib.md5(query.encode()).hexdigest()
                entry = CacheEntry(
                    key=key,
                    value={"query": query, "result": result},
                    ttl=ttl,
                    embedding=embedding,
                    level="semantic",
                )
                self._entries.append(entry)

            return True
        except Exception as e:
            logger.error(f"Semantic cache set error: {e}")
            return False

    def find_similar(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, Any, float]]:
        """
        유사한 캐시 항목 찾기

        Args:
            query: 쿼리 문자열
            threshold: 유사도 임계값 (None이면 기본값 사용)

        Returns:
            (원본 쿼리, 결과, 유사도) 또는 None
        """
        threshold = threshold or self.threshold

        try:
            query_embedding = self.embedding_fn(query)

            with self._lock:
                best_match = None
                best_similarity = 0.0

                for entry in self._entries:
                    if entry.is_expired:
                        continue

                    if entry.embedding is None:
                        continue

                    similarity = self._cosine_similarity(
                        query_embedding, entry.embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry

                if best_match and best_similarity >= threshold:
                    best_match.hits += 1
                    self._stats["hits"] += 1

                    data = best_match.value
                    return (
                        data["query"],
                        data["result"],
                        best_similarity,
                    )

                self._stats["misses"] += 1
                return None

        except Exception as e:
            logger.error(f"Semantic cache find error: {e}")
            self._stats["misses"] += 1
            return None

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries = []
            return count

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "entries": len(self._entries),
            "threshold": self.threshold,
        }


class MultiLevelCache:
    """
    멀티레벨 캐시

    [구조]
    L1 (메모리) → L2 (Redis) → 시맨틱 캐시

    [특징]
    - 계층적 조회
    - 자동 승격 (L2 → L1)
    - 시맨틱 폴백
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
    ):
        self.config = config or CacheConfig()

        # L1 인메모리 캐시
        self._l1 = L1MemoryCache(
            max_size=self.config.l1_max_size,
            default_ttl=self.config.l1_ttl,
        )

        # L2 Redis 캐시
        self._l2: Optional[L2RedisCache] = None
        if self.config.enable_l2:
            self._l2 = L2RedisCache(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                default_ttl=self.config.l2_ttl,
            )

        # 시맨틱 캐시
        self._semantic: Optional[SemanticCache] = None
        if self.config.enable_semantic and embedding_fn:
            self._semantic = SemanticCache(
                embedding_fn=embedding_fn,
                threshold=self.config.semantic_threshold,
            )

        self._stats = CacheStats()
        self._hit_latencies: List[float] = []

        logger.info(
            f"MultiLevelCache initialized: "
            f"L1={True}, L2={self._l2 is not None and self._l2.is_available}, "
            f"Semantic={self._semantic is not None}"
        )

    def get(
        self,
        key: str,
        query: Optional[str] = None,
    ) -> Optional[Any]:
        """
        캐시 조회

        Args:
            key: 캐시 키
            query: 원본 쿼리 (시맨틱 폴백용)

        Returns:
            캐시된 값 또는 None
        """
        start = time.time()

        # L1 조회
        value = self._l1.get(key)
        if value is not None:
            self._stats.l1_hits += 1
            self._record_latency(start)
            return value

        self._stats.l1_misses += 1

        # L2 조회
        if self._l2 and self._l2.is_available:
            value = self._l2.get(key)
            if value is not None:
                self._stats.l2_hits += 1
                # L1에 승격
                self._l1.set(key, value)
                self._record_latency(start)
                return value

            self._stats.l2_misses += 1

        # 시맨틱 폴백
        if self._semantic and query:
            result = self._semantic.find_similar(query)
            if result:
                _, cached_result, similarity = result
                self._stats.semantic_hits += 1
                logger.debug(f"Semantic cache hit (similarity={similarity:.3f})")
                self._record_latency(start)
                return cached_result

        return None

    def set(
        self,
        key: str,
        value: Any,
        query: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        캐시 저장

        Args:
            key: 캐시 키
            value: 저장할 값
            query: 원본 쿼리 (시맨틱 캐싱용)
            ttl: TTL

        Returns:
            저장 성공 여부
        """
        # L1 저장
        self._l1.set(key, value, ttl or self.config.l1_ttl)

        # L2 저장
        if self._l2 and self._l2.is_available:
            self._l2.set(key, value, ttl or self.config.l2_ttl)

        # 시맨틱 캐싱
        if self._semantic and query:
            self._semantic.cache_query(query, value)

        return True

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        self._l1.delete(key)
        if self._l2:
            self._l2.delete(key)
        return True

    def clear(self) -> Dict[str, int]:
        """전체 캐시 삭제"""
        result = {"l1": self._l1.clear()}
        if self._l2:
            result["l2"] = self._l2.clear()
        if self._semantic:
            result["semantic"] = self._semantic.clear()
        return result

    def _record_latency(self, start: float):
        """레이턴시 기록"""
        latency = (time.time() - start) * 1000
        self._hit_latencies.append(latency)
        if len(self._hit_latencies) > 100:
            self._hit_latencies = self._hit_latencies[-100:]
        self._stats.avg_hit_latency_ms = np.mean(self._hit_latencies)

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        stats = {
            "l1_hits": self._stats.l1_hits,
            "l1_misses": self._stats.l1_misses,
            "l1_hit_rate": round(self._stats.l1_hit_rate, 4),
            "l2_hits": self._stats.l2_hits,
            "l2_misses": self._stats.l2_misses,
            "l2_hit_rate": round(self._stats.l2_hit_rate, 4),
            "semantic_hits": self._stats.semantic_hits,
            "overall_hit_rate": round(self._stats.overall_hit_rate, 4),
            "avg_hit_latency_ms": round(self._stats.avg_hit_latency_ms, 3),
        }

        if self._semantic:
            stats["semantic"] = self._semantic.get_stats()

        return stats

    def warmup(self, queries: List[Tuple[str, Any]]):
        """
        캐시 워밍

        Args:
            queries: (query, result) 튜플 리스트
        """
        for query, result in queries:
            key = hashlib.md5(query.encode()).hexdigest()
            self.set(key, result, query=query)

        logger.info(f"Cache warmed with {len(queries)} entries")


class CacheWarmer:
    """
    캐시 워머

    자주 사용되는 쿼리를 미리 캐싱
    """

    def __init__(
        self,
        cache: MultiLevelCache,
        query_fn: Callable[[str], Any],
    ):
        self.cache = cache
        self.query_fn = query_fn

    def warmup_from_history(
        self,
        queries: List[str],
        parallel: bool = True,
        max_workers: int = 4,
    ):
        """
        쿼리 히스토리로 캐시 워밍

        Args:
            queries: 쿼리 리스트
            parallel: 병렬 처리 여부
            max_workers: 병렬 워커 수
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def warm_query(query: str):
            try:
                result = self.query_fn(query)
                key = hashlib.md5(query.encode()).hexdigest()
                self.cache.set(key, result, query=query)
                return query, True
            except Exception as e:
                logger.error(f"Warmup failed for query: {query[:50]}... - {e}")
                return query, False

        success_count = 0

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(warm_query, q)
                    for q in queries
                ]
                for future in as_completed(futures):
                    _, success = future.result()
                    if success:
                        success_count += 1
        else:
            for query in queries:
                _, success = warm_query(query)
                if success:
                    success_count += 1

        logger.info(f"Cache warmup complete: {success_count}/{len(queries)} queries")
        return success_count

    def warmup_popular(self, top_n: int = 100):
        """
        인기 쿼리 워밍

        Args:
            top_n: 상위 N개 쿼리
        """
        # 실제 구현에서는 쿼리 로그에서 인기 쿼리 추출
        logger.info(f"Popular query warmup: top {top_n}")
        pass
