# -*- coding: utf-8 -*-
"""
헬스 체크 모듈

[기능]
- 컴포넌트별 헬스 체크
- 통합 헬스 상태
- Readiness/Liveness 프로브
- 헬스 이력

[사용 예시]
>>> checker = HealthChecker()
>>> checker.add_check("redis", redis_health_check)
>>> status = checker.check_all()
>>> print(status.is_healthy)
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """헬스 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """컴포넌트 헬스"""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check,
            "details": self.details,
        }


@dataclass
class OverallHealth:
    """전체 헬스 상태"""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """서비스 준비 상태 (degraded도 준비됨으로 간주)"""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            },
            "timestamp": self.timestamp,
        }


class HealthCheck(ABC):
    """헬스 체크 인터페이스"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def check(self) -> ComponentHealth:
        pass


class FunctionHealthCheck(HealthCheck):
    """함수 기반 헬스 체크"""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], bool],
        timeout: float = 5.0,
    ):
        self._name = name
        self._check_fn = check_fn
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> ComponentHealth:
        start = time.time()
        try:
            result = self._check_fn()
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                message="OK" if result else "Check failed",
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
            )


class RedisHealthCheck(HealthCheck):
    """Redis 헬스 체크"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.password = password
        self._name = "redis"

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> ComponentHealth:
        start = time.time()
        try:
            import redis
            client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                socket_timeout=5.0,
            )
            client.ping()
            latency = (time.time() - start) * 1000

            info = client.info("memory")
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="OK",
                latency_ms=latency,
                details={
                    "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                },
            )

        except ImportError:
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNKNOWN,
                message="Redis module not installed",
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
            )


class ChromaDBHealthCheck(HealthCheck):
    """ChromaDB 헬스 체크"""

    def __init__(self, client=None, persist_dir: str = None):
        self._client = client
        self._persist_dir = persist_dir
        self._name = "chromadb"

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> ComponentHealth:
        start = time.time()
        try:
            import chromadb

            client = self._client
            if client is None:
                if self._persist_dir:
                    client = chromadb.PersistentClient(path=self._persist_dir)
                else:
                    client = chromadb.Client()

            # 하트비트 확인
            client.heartbeat()

            latency = (time.time() - start) * 1000

            # 컬렉션 수 확인
            collections = client.list_collections()

            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="OK",
                latency_ms=latency,
                details={
                    "collections_count": len(collections),
                },
            )

        except ImportError:
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNKNOWN,
                message="ChromaDB module not installed",
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
            )


class OllamaHealthCheck(HealthCheck):
    """Ollama 헬스 체크"""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self._name = "ollama"

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> ComponentHealth:
        start = time.time()
        try:
            import requests

            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()

            latency = (time.time() - start) * 1000
            data = response.json()

            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="OK",
                latency_ms=latency,
                details={
                    "models_count": len(data.get("models", [])),
                },
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
            )


class HealthChecker:
    """
    헬스 체커

    [특징]
    - 다중 컴포넌트 체크
    - 병렬 체크
    - 캐싱
    - 이력 관리
    """

    def __init__(
        self,
        cache_ttl: float = 10.0,
        check_timeout: float = 5.0,
    ):
        self.cache_ttl = cache_ttl
        self.check_timeout = check_timeout
        self._checks: Dict[str, HealthCheck] = {}
        self._cache: Dict[str, ComponentHealth] = {}
        self._history: List[OverallHealth] = []
        self._lock = threading.Lock()

    def add_check(self, check: HealthCheck):
        """헬스 체크 추가"""
        with self._lock:
            self._checks[check.name] = check
        logger.info(f"Added health check: {check.name}")

    def add_function_check(
        self,
        name: str,
        check_fn: Callable[[], bool],
    ):
        """함수 기반 헬스 체크 추가"""
        self.add_check(FunctionHealthCheck(name, check_fn))

    def remove_check(self, name: str):
        """헬스 체크 제거"""
        with self._lock:
            self._checks.pop(name, None)

    def check_component(self, name: str, use_cache: bool = True) -> Optional[ComponentHealth]:
        """단일 컴포넌트 체크"""
        with self._lock:
            check = self._checks.get(name)
            if check is None:
                return None

            # 캐시 확인
            if use_cache and name in self._cache:
                cached = self._cache[name]
                if time.time() - cached.last_check < self.cache_ttl:
                    return cached

        # 체크 실행
        health = check.check()

        with self._lock:
            self._cache[name] = health

        return health

    def check_all(self, use_cache: bool = True) -> OverallHealth:
        """전체 컴포넌트 체크"""
        components: Dict[str, ComponentHealth] = {}

        with self._lock:
            check_names = list(self._checks.keys())

        # 병렬 체크
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.check_component, name, use_cache): name
                for name in check_names
            }

            for future in as_completed(futures, timeout=self.check_timeout):
                name = futures[future]
                try:
                    health = future.result()
                    if health:
                        components[name] = health
                except Exception as e:
                    components[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Check failed: {e}",
                    )

        # 전체 상태 결정
        statuses = [c.status for c in components.values()]

        if not statuses:
            overall_status = HealthStatus.UNKNOWN
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        overall = OverallHealth(
            status=overall_status,
            components=components,
        )

        # 이력 저장
        with self._lock:
            self._history.append(overall)
            # 최근 100개만 유지
            if len(self._history) > 100:
                self._history = self._history[-100:]

        return overall

    def get_liveness(self) -> Dict[str, Any]:
        """Liveness 프로브 응답"""
        return {
            "status": "alive",
            "timestamp": time.time(),
        }

    def get_readiness(self) -> Dict[str, Any]:
        """Readiness 프로브 응답"""
        overall = self.check_all()
        return {
            "status": "ready" if overall.is_ready else "not_ready",
            "healthy": overall.is_healthy,
            "details": overall.to_dict(),
        }

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """헬스 이력"""
        with self._lock:
            history = self._history[-limit:]
        return [h.to_dict() for h in history]

    def get_stats(self) -> Dict[str, Any]:
        """헬스 통계"""
        with self._lock:
            history = list(self._history)

        if not history:
            return {}

        # 최근 상태별 카운트
        status_counts = {}
        for h in history[-20:]:
            status = h.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # 컴포넌트별 가용성
        component_uptime = {}
        for name in self._checks:
            healthy_count = sum(
                1 for h in history[-20:]
                if name in h.components and h.components[name].status == HealthStatus.HEALTHY
            )
            component_uptime[name] = healthy_count / min(len(history), 20)

        return {
            "total_checks": len(history),
            "recent_status_counts": status_counts,
            "component_uptime": component_uptime,
        }


# =============================================================================
# 편의 함수
# =============================================================================

_default_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """기본 헬스 체커 조회"""
    global _default_checker
    if _default_checker is None:
        _default_checker = HealthChecker()
    return _default_checker


def setup_default_checks(checker: HealthChecker = None):
    """기본 헬스 체크 설정"""
    checker = checker or get_health_checker()

    # Redis 체크
    checker.add_check(RedisHealthCheck())

    # ChromaDB 체크
    checker.add_check(ChromaDBHealthCheck())

    # Ollama 체크
    checker.add_check(OllamaHealthCheck())

    logger.info("Set up default health checks")
