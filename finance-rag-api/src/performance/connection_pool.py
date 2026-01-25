# -*- coding: utf-8 -*-
"""
연결 풀링 모듈

[기능]
- Redis 연결 풀
- ChromaDB 클라이언트 풀
- Ollama 클라이언트 풀
- 헬스 체크 및 자동 복구
- 연결 통계

[사용 예시]
>>> pool = RedisPool(host="localhost", max_connections=10)
>>> with pool.get_connection() as conn:
...     conn.set("key", "value")
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PoolStats:
    """연결 풀 통계"""
    max_size: int = 10
    current_size: int = 0
    available: int = 0
    in_use: int = 0
    total_connections_created: int = 0
    total_connections_closed: int = 0
    total_checkouts: int = 0
    total_checkins: int = 0
    avg_checkout_time_ms: float = 0.0
    failed_checkouts: int = 0


@dataclass
class PoolConfig:
    """연결 풀 설정"""
    max_size: int = 10
    min_size: int = 2
    timeout: float = 5.0
    max_lifetime: float = 3600.0
    health_check_interval: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0


class PooledConnection(Generic[T]):
    """풀링된 연결 래퍼"""

    def __init__(self, connection: T, pool: "ConnectionPool"):
        self.connection = connection
        self._pool = pool
        self._created_at = time.time()
        self._last_used_at = time.time()
        self._use_count = 0
        self._healthy = True

    @property
    def age(self) -> float:
        """연결 생성 후 경과 시간 (초)"""
        return time.time() - self._created_at

    @property
    def idle_time(self) -> float:
        """마지막 사용 후 경과 시간 (초)"""
        return time.time() - self._last_used_at

    def touch(self):
        """사용 시간 갱신"""
        self._last_used_at = time.time()
        self._use_count += 1


class ConnectionPool(ABC, Generic[T]):
    """
    연결 풀 베이스 클래스

    [특징]
    - 제네릭 타입 지원
    - 스레드 안전
    - 자동 헬스 체크
    - 연결 수명 관리
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._pool: Queue[PooledConnection[T]] = Queue(maxsize=self.config.max_size)
        self._lock = threading.Lock()
        self._stats = PoolStats(max_size=self.config.max_size)
        self._checkout_times: list = []
        self._closed = False

        # 최소 연결 수 만큼 사전 생성
        self._initialize_pool()

        # 헬스 체크 스레드 시작
        self._start_health_check()

    @abstractmethod
    def _create_connection(self) -> T:
        """연결 생성 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    def _close_connection(self, connection: T):
        """연결 종료 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    def _validate_connection(self, connection: T) -> bool:
        """연결 검증 (하위 클래스에서 구현)"""
        pass

    def _initialize_pool(self):
        """초기 연결 생성"""
        for _ in range(self.config.min_size):
            try:
                conn = self._create_pooled_connection()
                self._pool.put(conn)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")

    def _create_pooled_connection(self) -> PooledConnection[T]:
        """풀링된 연결 생성"""
        connection = self._create_connection()
        pooled = PooledConnection(connection, self)
        self._stats.total_connections_created += 1
        self._stats.current_size += 1
        return pooled

    def _start_health_check(self):
        """헬스 체크 스레드 시작"""
        def health_check_loop():
            while not self._closed:
                try:
                    self._check_connections()
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                time.sleep(self.config.health_check_interval)

        thread = threading.Thread(target=health_check_loop, daemon=True)
        thread.start()

    def _check_connections(self):
        """연결 상태 확인 및 정리"""
        checked: list = []

        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()

                # 수명 초과 연결 정리
                if pooled.age > self.config.max_lifetime:
                    self._close_pooled_connection(pooled)
                    continue

                # 연결 검증
                if not self._validate_connection(pooled.connection):
                    self._close_pooled_connection(pooled)
                    continue

                checked.append(pooled)
            except Empty:
                break

        # 정상 연결 다시 풀에 추가
        for pooled in checked:
            self._pool.put(pooled)

        # 최소 연결 수 유지
        current = self._pool.qsize()
        if current < self.config.min_size:
            for _ in range(self.config.min_size - current):
                try:
                    conn = self._create_pooled_connection()
                    self._pool.put(conn)
                except Exception as e:
                    logger.warning(f"Failed to replenish pool: {e}")

    def _close_pooled_connection(self, pooled: PooledConnection[T]):
        """풀링된 연결 종료"""
        try:
            self._close_connection(pooled.connection)
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            self._stats.total_connections_closed += 1
            self._stats.current_size -= 1

    @contextmanager
    def get_connection(self):
        """
        연결 체크아웃 (컨텍스트 매니저)

        사용 후 자동으로 풀에 반환됨
        """
        start_time = time.time()
        pooled = None

        try:
            # 기존 연결 가져오기 시도
            try:
                pooled = self._pool.get(timeout=self.config.timeout)
                pooled.touch()
            except Empty:
                # 풀이 비어있으면 새 연결 생성
                with self._lock:
                    if self._stats.current_size < self.config.max_size:
                        pooled = self._create_pooled_connection()
                    else:
                        # 최대 연결 수 초과
                        self._stats.failed_checkouts += 1
                        raise TimeoutError("Connection pool exhausted")

            # 연결 검증
            if not pooled._healthy or not self._validate_connection(pooled.connection):
                self._close_pooled_connection(pooled)
                pooled = self._create_pooled_connection()

            self._stats.total_checkouts += 1
            checkout_time = (time.time() - start_time) * 1000
            self._checkout_times.append(checkout_time)

            # 최근 100개만 유지
            if len(self._checkout_times) > 100:
                self._checkout_times = self._checkout_times[-100:]

            yield pooled.connection

        finally:
            # 연결 반환
            if pooled is not None:
                pooled.touch()
                try:
                    self._pool.put_nowait(pooled)
                    self._stats.total_checkins += 1
                except Exception:
                    # 풀이 가득 찬 경우 연결 종료
                    self._close_pooled_connection(pooled)

    def get_stats(self) -> PoolStats:
        """통계 조회"""
        self._stats.available = self._pool.qsize()
        self._stats.in_use = self._stats.current_size - self._stats.available
        if self._checkout_times:
            self._stats.avg_checkout_time_ms = sum(self._checkout_times) / len(self._checkout_times)
        return self._stats

    def close(self):
        """풀 종료"""
        self._closed = True
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                self._close_pooled_connection(pooled)
            except Empty:
                break


class RedisPool(ConnectionPool):
    """
    Redis 연결 풀

    [특징]
    - 자동 재연결
    - 파이프라인 지원
    - 클러스터 지원 가능
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        config: Optional[PoolConfig] = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis_module = None

        # Redis 모듈 로드
        try:
            import redis
            self._redis_module = redis
        except ImportError:
            logger.warning("Redis module not installed")

        super().__init__(config)

    def _create_connection(self):
        if self._redis_module is None:
            raise RuntimeError("Redis module not available")

        return self._redis_module.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

    def _close_connection(self, connection):
        try:
            connection.close()
        except Exception:
            pass

    def _validate_connection(self, connection) -> bool:
        try:
            connection.ping()
            return True
        except Exception:
            return False


class ChromaPool(ConnectionPool):
    """
    ChromaDB 클라이언트 풀

    [특징]
    - HTTP/gRPC 클라이언트 풀링
    - 컬렉션 캐싱
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        config: Optional[PoolConfig] = None,
    ):
        self.host = host
        self.port = port
        self._chroma_module = None

        try:
            import chromadb
            self._chroma_module = chromadb
        except ImportError:
            logger.warning("ChromaDB module not installed")

        super().__init__(config)

    def _create_connection(self):
        if self._chroma_module is None:
            raise RuntimeError("ChromaDB module not available")

        return self._chroma_module.HttpClient(
            host=self.host,
            port=self.port,
        )

    def _close_connection(self, connection):
        # ChromaDB HttpClient doesn't need explicit close
        pass

    def _validate_connection(self, connection) -> bool:
        try:
            connection.heartbeat()
            return True
        except Exception:
            return False


class OllamaPool(ConnectionPool):
    """
    Ollama 클라이언트 풀

    [특징]
    - 모델 로딩 상태 관리
    - Keep-alive 연결 유지
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        config: Optional[PoolConfig] = None,
    ):
        self.host = host
        self._ollama_module = None

        try:
            import ollama
            self._ollama_module = ollama
        except ImportError:
            logger.warning("Ollama module not installed")

        super().__init__(config)

    def _create_connection(self):
        if self._ollama_module is None:
            raise RuntimeError("Ollama module not available")

        return self._ollama_module.Client(host=self.host)

    def _close_connection(self, connection):
        # Ollama Client doesn't need explicit close
        pass

    def _validate_connection(self, connection) -> bool:
        try:
            connection.list()
            return True
        except Exception:
            return False


class ConnectionManager:
    """
    통합 연결 관리자

    모든 연결 풀을 중앙에서 관리
    """

    _instance: Optional["ConnectionManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._pools: Dict[str, ConnectionPool] = {}
        self._initialized = True

    def register_pool(self, name: str, pool: ConnectionPool):
        """연결 풀 등록"""
        self._pools[name] = pool
        logger.info(f"Registered connection pool: {name}")

    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """연결 풀 조회"""
        return self._pools.get(name)

    @contextmanager
    def get_connection(self, pool_name: str):
        """연결 체크아웃"""
        pool = self._pools.get(pool_name)
        if pool is None:
            raise ValueError(f"Unknown pool: {pool_name}")

        with pool.get_connection() as conn:
            yield conn

    def get_all_stats(self) -> Dict[str, PoolStats]:
        """모든 풀 통계 조회"""
        return {
            name: pool.get_stats()
            for name, pool in self._pools.items()
        }

    def close_all(self):
        """모든 풀 종료"""
        for name, pool in self._pools.items():
            logger.info(f"Closing pool: {name}")
            pool.close()
        self._pools.clear()
