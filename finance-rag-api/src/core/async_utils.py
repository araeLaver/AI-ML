# -*- coding: utf-8 -*-
"""
비동기 유틸리티 모듈

커넥션 풀링, 병렬 처리, 비동기 최적화
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Coroutine, Optional, TypeVar

import httpx

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ============================================================
# HTTP 커넥션 풀
# ============================================================

@dataclass
class HttpPoolConfig:
    """HTTP 풀 설정"""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0


class HttpConnectionPool:
    """HTTP 커넥션 풀 관리자"""

    def __init__(self, config: Optional[HttpPoolConfig] = None):
        self.config = config or HttpPoolConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 획득"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_connections=self.config.max_connections,
                            max_keepalive_connections=self.config.max_keepalive_connections,
                            keepalive_expiry=self.config.keepalive_expiry,
                        ),
                        timeout=httpx.Timeout(
                            connect=self.config.connect_timeout,
                            read=self.config.read_timeout,
                            write=self.config.write_timeout,
                            pool=self.config.pool_timeout,
                        ),
                    )
        return self._client

    async def close(self):
        """풀 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(self, url: str, **kwargs) -> httpx.Response:
        client = await self.get_client()
        return await client.get(url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        client = await self.get_client()
        return await client.post(url, **kwargs)


# 글로벌 HTTP 풀
_http_pool: Optional[HttpConnectionPool] = None


async def get_http_pool(config: Optional[HttpPoolConfig] = None) -> HttpConnectionPool:
    """HTTP 풀 싱글톤"""
    global _http_pool
    if _http_pool is None:
        _http_pool = HttpConnectionPool(config)
    return _http_pool


# ============================================================
# 병렬 처리 유틸리티
# ============================================================

async def gather_with_concurrency(
    n: int,
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = False,
) -> list[T]:
    """동시성 제한이 있는 gather

    Args:
        n: 최대 동시 실행 수
        *coros: 코루틴들
        return_exceptions: 예외를 결과로 반환할지 여부

    Returns:
        결과 리스트
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(sem_coro(c) for c in coros),
        return_exceptions=return_exceptions,
    )


async def run_in_parallel(
    func: Callable[..., Coroutine[Any, Any, T]],
    items: list[Any],
    max_concurrent: int = 10,
    return_exceptions: bool = False,
) -> list[T]:
    """아이템들을 병렬로 처리

    Args:
        func: 각 아이템에 적용할 비동기 함수
        items: 처리할 아이템 리스트
        max_concurrent: 최대 동시 실행 수
        return_exceptions: 예외를 결과로 반환할지 여부

    Returns:
        결과 리스트 (입력 순서 유지)
    """
    coros = [func(item) for item in items]
    return await gather_with_concurrency(
        max_concurrent,
        *coros,
        return_exceptions=return_exceptions,
    )


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: Optional[T] = None,
) -> Optional[T]:
    """타임아웃이 있는 코루틴 실행

    Args:
        coro: 실행할 코루틴
        timeout: 타임아웃 (초)
        default: 타임아웃 시 반환값

    Returns:
        실행 결과 또는 기본값
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout}s")
        return default


async def retry_async(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    """재시도 로직이 있는 비동기 실행

    Args:
        func: 실행할 비동기 함수
        max_retries: 최대 재시도 횟수
        delay: 초기 대기 시간
        backoff: 백오프 계수
        exceptions: 재시도할 예외 타입들
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

    raise last_exception


# ============================================================
# 데코레이터
# ============================================================

def async_timed(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
    """비동기 함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.4f}s")

    return wrapper


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """재시도 데코레이터"""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                func, *args,
                max_retries=max_retries,
                delay=delay,
                backoff=backoff,
                exceptions=exceptions,
                **kwargs,
            )
        return wrapper
    return decorator


def async_timeout(timeout: float, default: Any = None):
    """타임아웃 데코레이터"""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, Optional[T]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"{func.__name__} timed out after {timeout}s")
                return default
        return wrapper
    return decorator


# ============================================================
# 리소스 풀
# ============================================================

@dataclass
class PooledResource:
    """풀링된 리소스"""
    resource: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0


class ResourcePool:
    """범용 리소스 풀

    데이터베이스 연결, API 클라이언트 등의 리소스 풀링
    """

    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, T]],
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: float = 300.0,  # 5분
    ):
        self._factory = factory
        self._max_size = max_size
        self._min_size = min_size
        self._max_idle_time = max_idle_time
        self._pool: asyncio.Queue[PooledResource] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def _create_resource(self) -> PooledResource:
        """새 리소스 생성"""
        resource = await self._factory()
        return PooledResource(resource=resource)

    async def acquire(self) -> T:
        """리소스 획득"""
        if self._closed:
            raise RuntimeError("Pool is closed")

        # 풀에서 가져오기 시도
        try:
            pooled = self._pool.get_nowait()
            pooled.last_used = datetime.now()
            pooled.use_count += 1
            return pooled.resource
        except asyncio.QueueEmpty:
            pass

        # 새로 생성 가능한지 확인
        async with self._lock:
            if self._size < self._max_size:
                self._size += 1
                pooled = await self._create_resource()
                pooled.use_count = 1
                return pooled.resource

        # 대기
        pooled = await self._pool.get()
        pooled.last_used = datetime.now()
        pooled.use_count += 1
        return pooled.resource

    async def release(self, resource: T) -> None:
        """리소스 반환"""
        if self._closed:
            return

        pooled = PooledResource(resource=resource, last_used=datetime.now())

        try:
            self._pool.put_nowait(pooled)
        except asyncio.QueueFull:
            # 풀이 가득 찼으면 버림
            async with self._lock:
                self._size -= 1

    @asynccontextmanager
    async def connection(self):
        """컨텍스트 매니저로 리소스 사용"""
        resource = await self.acquire()
        try:
            yield resource
        finally:
            await self.release(resource)

    async def close(self) -> None:
        """풀 종료"""
        self._closed = True

        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                if hasattr(pooled.resource, "close"):
                    if asyncio.iscoroutinefunction(pooled.resource.close):
                        await pooled.resource.close()
                    else:
                        pooled.resource.close()
            except asyncio.QueueEmpty:
                break

    def stats(self) -> dict:
        """풀 통계"""
        return {
            "size": self._size,
            "available": self._pool.qsize(),
            "max_size": self._max_size,
            "closed": self._closed,
        }


# ============================================================
# 성능 측정
# ============================================================

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    operation: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    errors: int = 0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    def record(self, duration: float, error: bool = False) -> None:
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        if error:
            self.errors += 1

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "count": self.count,
            "total_time": round(self.total_time, 4),
            "avg_time": round(self.avg_time, 4),
            "min_time": round(self.min_time, 4) if self.min_time != float("inf") else 0,
            "max_time": round(self.max_time, 4),
            "errors": self.errors,
            "error_rate": round(self.errors / self.count, 4) if self.count > 0 else 0,
        }


class PerformanceTracker:
    """성능 추적기"""

    def __init__(self):
        self._metrics: dict[str, PerformanceMetrics] = {}
        self._lock = asyncio.Lock()

    async def record(self, operation: str, duration: float, error: bool = False) -> None:
        """메트릭 기록"""
        async with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = PerformanceMetrics(operation=operation)
            self._metrics[operation].record(duration, error)

    @asynccontextmanager
    async def track(self, operation: str):
        """컨텍스트 매니저로 성능 추적"""
        start = time.perf_counter()
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            duration = time.perf_counter() - start
            await self.record(operation, duration, error)

    def get_metrics(self, operation: Optional[str] = None) -> dict:
        """메트릭 조회"""
        if operation:
            metrics = self._metrics.get(operation)
            return metrics.to_dict() if metrics else {}

        return {
            name: metrics.to_dict()
            for name, metrics in self._metrics.items()
        }

    def reset(self) -> None:
        """메트릭 초기화"""
        self._metrics.clear()


# 글로벌 성능 추적기
_performance_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """성능 추적기 싱글톤"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker
