# -*- coding: utf-8 -*-
"""
캐싱 모듈

Redis 기반 캐싱으로 성능 최적화
- 쿼리 결과 캐싱
- 임베딩 캐싱
- 세션/Rate Limit 캐싱
"""

import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

try:
    import redis
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None  # type: ignore

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheConfig:
    """캐시 설정

    Attributes:
        enabled: 캐시 활성화 여부
        redis_url: Redis 연결 URL
        default_ttl: 기본 TTL (초)
        max_memory: 최대 메모리 (MB, 로컬 캐시용)
        prefix: 캐시 키 접두사
        serializer: 직렬화 방식 ('json' or 'pickle')
    """
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600  # 1시간
    max_memory: int = 100  # MB
    prefix: str = "finance_rag:"
    serializer: str = "json"


@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheBackend(ABC):
    """캐시 백엔드 추상 클래스"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self, pattern: str = "*") -> int:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class LocalCache(CacheBackend):
    """로컬 메모리 캐시 (Redis 없을 때 폴백)"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expire_time)
        self._max_size = config.max_memory * 1024 * 1024  # bytes
        self.stats = CacheStats()

    def _is_expired(self, expire_time: float) -> bool:
        return time.time() > expire_time if expire_time > 0 else False

    def _cleanup_expired(self) -> None:
        """만료된 항목 정리"""
        now = time.time()
        expired_keys = [
            k for k, (_, exp) in self._cache.items()
            if exp > 0 and now > exp
        ]
        for key in expired_keys:
            del self._cache[key]

    async def get(self, key: str) -> Optional[Any]:
        full_key = f"{self.config.prefix}{key}"

        if full_key in self._cache:
            value, expire_time = self._cache[full_key]
            if not self._is_expired(expire_time):
                self.stats.hits += 1
                return value
            else:
                del self._cache[full_key]

        self.stats.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        full_key = f"{self.config.prefix}{key}"
        ttl = ttl or self.config.default_ttl
        expire_time = time.time() + ttl if ttl > 0 else 0

        self._cache[full_key] = (value, expire_time)
        self.stats.sets += 1

        # 주기적으로 만료 항목 정리
        if self.stats.sets % 100 == 0:
            self._cleanup_expired()

        return True

    async def delete(self, key: str) -> bool:
        full_key = f"{self.config.prefix}{key}"
        if full_key in self._cache:
            del self._cache[full_key]
            self.stats.deletes += 1
            return True
        return False

    async def exists(self, key: str) -> bool:
        full_key = f"{self.config.prefix}{key}"
        if full_key in self._cache:
            _, expire_time = self._cache[full_key]
            return not self._is_expired(expire_time)
        return False

    async def clear(self, pattern: str = "*") -> int:
        if pattern == "*":
            count = len(self._cache)
            self._cache.clear()
            return count

        # 패턴 매칭 (간단한 와일드카드)
        import fnmatch
        full_pattern = f"{self.config.prefix}{pattern}"
        keys_to_delete = [
            k for k in self._cache.keys()
            if fnmatch.fnmatch(k, full_pattern)
        ]
        for key in keys_to_delete:
            del self._cache[key]
        return len(keys_to_delete)

    async def close(self) -> None:
        self._cache.clear()


class RedisCache(CacheBackend):
    """Redis 캐시 백엔드"""

    def __init__(self, config: CacheConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisCache")

        self.config = config
        self._client: Optional[Any] = None
        self.stats = CacheStats()

    async def _get_client(self) -> Any:
        if self._client is None:
            self._client = await aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=False,  # bytes로 받음
            )
        return self._client

    def _serialize(self, value: Any) -> bytes:
        if self.config.serializer == "json":
            return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        if self.config.serializer == "json":
            return json.loads(data.decode("utf-8"))
        return pickle.loads(data)

    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"

            data = await client.get(full_key)
            if data is not None:
                self.stats.hits += 1
                return self._deserialize(data)

            self.stats.misses += 1
            return None

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"
            ttl = ttl or self.config.default_ttl

            data = self._serialize(value)

            if ttl > 0:
                await client.setex(full_key, ttl, data)
            else:
                await client.set(full_key, data)

            self.stats.sets += 1
            return True

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"

            result = await client.delete(full_key)
            if result:
                self.stats.deletes += 1
            return bool(result)

        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self.stats.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        try:
            client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"
            return bool(await client.exists(full_key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def clear(self, pattern: str = "*") -> int:
        try:
            client = await self._get_client()
            full_pattern = f"{self.config.prefix}{pattern}"

            keys = []
            async for key in client.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                return await client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class CacheManager:
    """캐시 매니저

    Redis 사용 가능 시 Redis, 아니면 로컬 메모리 캐시 사용
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._backend: Optional[CacheBackend] = None
        self._initialized = False

    async def initialize(self) -> None:
        """캐시 초기화"""
        if self._initialized:
            return

        if not self.config.enabled:
            logger.info("Cache disabled")
            self._initialized = True
            return

        # Redis 시도
        if REDIS_AVAILABLE:
            try:
                self._backend = RedisCache(self.config)
                # 연결 테스트
                client = await self._backend._get_client()
                await client.ping()
                logger.info(f"Redis cache initialized: {self.config.redis_url}")
                self._initialized = True
                return
            except Exception as e:
                logger.warning(f"Redis connection failed, falling back to local cache: {e}")

        # 폴백: 로컬 캐시
        self._backend = LocalCache(self.config)
        logger.info("Local memory cache initialized")
        self._initialized = True

    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if not self.config.enabled or not self._backend:
            return None
        return await self._backend.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        if not self.config.enabled or not self._backend:
            return False
        return await self._backend.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        if not self.config.enabled or not self._backend:
            return False
        return await self._backend.delete(key)

    async def exists(self, key: str) -> bool:
        """캐시에 키 존재 여부"""
        if not self.config.enabled or not self._backend:
            return False
        return await self._backend.exists(key)

    async def clear(self, pattern: str = "*") -> int:
        """캐시 클리어"""
        if not self.config.enabled or not self._backend:
            return 0
        return await self._backend.clear(pattern)

    async def close(self) -> None:
        """캐시 종료"""
        if self._backend:
            await self._backend.close()

    def get_stats(self) -> dict:
        """캐시 통계"""
        if self._backend:
            return self._backend.stats.to_dict()
        return {}

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()


# 글로벌 캐시 매니저
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """캐시 매니저 싱글톤"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
        await _cache_manager.initialize()
    return _cache_manager


def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
):
    """캐시 데코레이터

    Args:
        ttl: 캐시 TTL (초)
        key_prefix: 캐시 키 접두사
        key_builder: 커스텀 키 생성 함수

    Example:
        @cached(ttl=300, key_prefix="query:")
        async def get_answer(question: str):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            cache = await get_cache_manager()

            # 캐시 키 생성
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = CacheManager.make_key(*args, **kwargs)

            full_key = f"{key_prefix}{cache_key}"

            # 캐시 조회
            cached_value = await cache.get(full_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {full_key}")
                return cached_value

            # 실행 및 캐싱
            result = await func(*args, **kwargs)
            await cache.set(full_key, result, ttl)
            logger.debug(f"Cache set: {full_key}")

            return result

        return wrapper
    return decorator


# ============================================================
# 특화 캐시 유틸리티
# ============================================================

class QueryCache:
    """RAG 쿼리 결과 캐시"""

    PREFIX = "query:"
    DEFAULT_TTL = 1800  # 30분

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    @staticmethod
    def make_query_key(query: str, top_k: int = 5) -> str:
        """쿼리 캐시 키 생성"""
        normalized = query.lower().strip()
        return hashlib.md5(f"{normalized}:{top_k}".encode()).hexdigest()

    async def get(self, query: str, top_k: int = 5) -> Optional[dict]:
        key = f"{self.PREFIX}{self.make_query_key(query, top_k)}"
        return await self.cache.get(key)

    async def set(self, query: str, result: dict, top_k: int = 5, ttl: Optional[int] = None) -> bool:
        key = f"{self.PREFIX}{self.make_query_key(query, top_k)}"
        return await self.cache.set(key, result, ttl or self.DEFAULT_TTL)

    async def invalidate(self, query: str, top_k: int = 5) -> bool:
        key = f"{self.PREFIX}{self.make_query_key(query, top_k)}"
        return await self.cache.delete(key)

    async def clear_all(self) -> int:
        return await self.cache.clear(f"{self.PREFIX}*")


class EmbeddingCache:
    """임베딩 벡터 캐시"""

    PREFIX = "embedding:"
    DEFAULT_TTL = 86400  # 24시간

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    @staticmethod
    def make_embedding_key(text: str, model: str = "default") -> str:
        """임베딩 캐시 키 생성"""
        return hashlib.md5(f"{model}:{text}".encode()).hexdigest()

    async def get(self, text: str, model: str = "default") -> Optional[list[float]]:
        key = f"{self.PREFIX}{self.make_embedding_key(text, model)}"
        return await self.cache.get(key)

    async def set(self, text: str, embedding: list[float], model: str = "default", ttl: Optional[int] = None) -> bool:
        key = f"{self.PREFIX}{self.make_embedding_key(text, model)}"
        return await self.cache.set(key, embedding, ttl or self.DEFAULT_TTL)

    async def get_many(self, texts: list[str], model: str = "default") -> dict[str, Optional[list[float]]]:
        """여러 텍스트의 임베딩 일괄 조회"""
        results = {}
        for text in texts:
            results[text] = await self.get(text, model)
        return results

    async def set_many(self, embeddings: dict[str, list[float]], model: str = "default", ttl: Optional[int] = None) -> int:
        """여러 임베딩 일괄 저장"""
        count = 0
        for text, embedding in embeddings.items():
            if await self.set(text, embedding, model, ttl):
                count += 1
        return count


class SessionCache:
    """세션 캐시"""

    PREFIX = "session:"
    DEFAULT_TTL = 3600  # 1시간

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    async def get(self, session_id: str) -> Optional[dict]:
        key = f"{self.PREFIX}{session_id}"
        return await self.cache.get(key)

    async def set(self, session_id: str, data: dict, ttl: Optional[int] = None) -> bool:
        key = f"{self.PREFIX}{session_id}"
        return await self.cache.set(key, data, ttl or self.DEFAULT_TTL)

    async def delete(self, session_id: str) -> bool:
        key = f"{self.PREFIX}{session_id}"
        return await self.cache.delete(key)

    async def extend(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """세션 TTL 연장"""
        data = await self.get(session_id)
        if data:
            return await self.set(session_id, data, ttl or self.DEFAULT_TTL)
        return False
