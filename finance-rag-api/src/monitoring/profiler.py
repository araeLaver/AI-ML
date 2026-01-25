# -*- coding: utf-8 -*-
"""
성능 프로파일링 모듈

[기능]
- 쿼리 프로파일링
- 메모리 프로파일링
- CPU 프로파일링
- 병목 지점 분석

[사용 예시]
>>> profiler = QueryProfiler()
>>> with profiler.profile("my_query"):
...     result = rag.query("삼성전자")
>>> print(profiler.get_report())
"""

import gc
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """프로파일 결과"""
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["ProfileResult"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ProfileStats:
    """프로파일 통계"""
    total_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    p50_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0


class PerformanceProfiler:
    """
    성능 프로파일러

    [특징]
    - 계층적 프로파일링
    - 시간, 메모리, CPU 측정
    - 통계 집계
    """

    _context = threading.local()

    def __init__(self, enable_memory: bool = True, enable_cpu: bool = False):
        self.enable_memory = enable_memory
        self.enable_cpu = enable_cpu
        self._results: List[ProfileResult] = []
        self._stats: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def profile(self, name: str, **metadata):
        """
        프로파일 컨텍스트

        Args:
            name: 프로파일 이름
            **metadata: 추가 메타데이터
        """
        # 메모리 측정
        memory_start = 0.0
        if self.enable_memory:
            memory_start = self._get_memory_mb()

        start_time = time.time()

        # 부모 결과 확인
        parent_result = getattr(self._context, "current_result", None)

        result = ProfileResult(
            name=name,
            start_time=start_time,
            end_time=0,
            duration_ms=0,
            memory_start_mb=memory_start,
            metadata=metadata,
        )

        # 현재 결과 설정
        self._context.current_result = result

        try:
            yield result
        finally:
            result.end_time = time.time()
            result.duration_ms = (result.end_time - result.start_time) * 1000

            if self.enable_memory:
                result.memory_end_mb = self._get_memory_mb()
                result.memory_delta_mb = result.memory_end_mb - result.memory_start_mb

            # 부모에 추가 또는 루트로 저장
            if parent_result:
                parent_result.children.append(result)
                self._context.current_result = parent_result
            else:
                with self._lock:
                    self._results.append(result)
                    # 통계 업데이트
                    if name not in self._stats:
                        self._stats[name] = []
                    self._stats[name].append(result.duration_ms)
                # 루트 레벨 프로파일 완료 시 current_result 초기화
                self._context.current_result = None

    def _get_memory_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_stats(self, name: str) -> ProfileStats:
        """특정 프로파일 통계"""
        import numpy as np

        with self._lock:
            times = self._stats.get(name, [])

        if not times:
            return ProfileStats()

        return ProfileStats(
            total_calls=len(times),
            total_time_ms=sum(times),
            avg_time_ms=np.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=np.percentile(times, 50),
            p95_time_ms=np.percentile(times, 95),
            p99_time_ms=np.percentile(times, 99),
        )

    def get_all_stats(self) -> Dict[str, ProfileStats]:
        """모든 프로파일 통계"""
        with self._lock:
            names = list(self._stats.keys())
        return {name: self.get_stats(name) for name in names}

    def get_report(self) -> Dict[str, Any]:
        """프로파일 리포트"""
        stats = self.get_all_stats()
        return {
            "stats": {
                name: {
                    "calls": s.total_calls,
                    "avg_ms": round(s.avg_time_ms, 2),
                    "min_ms": round(s.min_time_ms, 2),
                    "max_ms": round(s.max_time_ms, 2),
                    "p95_ms": round(s.p95_time_ms, 2),
                    "p99_ms": round(s.p99_time_ms, 2),
                }
                for name, s in stats.items()
            },
            "recent_profiles": [r.to_dict() for r in self._results[-10:]],
        }

    def clear(self):
        """결과 초기화"""
        with self._lock:
            self._results.clear()
            self._stats.clear()


class QueryProfiler:
    """
    RAG 쿼리 프로파일러

    쿼리 단계별 프로파일링
    """

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self._query_profiles: List[Dict[str, Any]] = []

    @contextmanager
    def profile_query(self, query: str):
        """전체 쿼리 프로파일"""
        with self.profiler.profile("query", query=query[:50]) as result:
            yield result

        # 쿼리 프로파일 저장
        profile = {
            "query": query[:100],
            "total_ms": result.duration_ms,
            "stages": {},
            "timestamp": result.start_time,
        }

        for child in result.children:
            profile["stages"][child.name] = child.duration_ms

        self._query_profiles.append(profile)

        # 최근 100개만 유지
        if len(self._query_profiles) > 100:
            self._query_profiles = self._query_profiles[-100:]

    @contextmanager
    def profile_retrieval(self):
        """검색 단계 프로파일"""
        with self.profiler.profile("retrieval") as result:
            yield result

    @contextmanager
    def profile_reranking(self):
        """리랭킹 단계 프로파일"""
        with self.profiler.profile("reranking") as result:
            yield result

    @contextmanager
    def profile_generation(self):
        """생성 단계 프로파일"""
        with self.profiler.profile("generation") as result:
            yield result

    @contextmanager
    def profile_embedding(self):
        """임베딩 단계 프로파일"""
        with self.profiler.profile("embedding") as result:
            yield result

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """병목 지점 분석"""
        stats = self.profiler.get_all_stats()

        bottlenecks = []
        for name, stat in stats.items():
            if stat.avg_time_ms > 100:  # 100ms 이상
                bottlenecks.append({
                    "stage": name,
                    "avg_ms": round(stat.avg_time_ms, 2),
                    "p95_ms": round(stat.p95_time_ms, 2),
                    "severity": "high" if stat.avg_time_ms > 500 else "medium",
                })

        return sorted(bottlenecks, key=lambda x: x["avg_ms"], reverse=True)

    def get_query_profiles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 쿼리 프로파일"""
        return self._query_profiles[-limit:]

    def get_report(self) -> Dict[str, Any]:
        """쿼리 프로파일 리포트"""
        return {
            "bottlenecks": self.get_bottlenecks(),
            "stats": self.profiler.get_report()["stats"],
            "recent_queries": self.get_query_profiles(5),
        }


class MemoryProfiler:
    """
    메모리 프로파일러

    메모리 사용량 및 누수 탐지
    """

    def __init__(self):
        self._snapshots: List[Dict[str, Any]] = []
        self._baseline: Optional[float] = None

    def take_snapshot(self, label: str = None) -> Dict[str, Any]:
        """메모리 스냅샷"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            snapshot = {
                "label": label or f"snapshot_{len(self._snapshots)}",
                "timestamp": time.time(),
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
            }

            self._snapshots.append(snapshot)
            return snapshot

        except ImportError:
            return {"error": "psutil not installed"}

    def set_baseline(self):
        """기준 메모리 설정"""
        snapshot = self.take_snapshot("baseline")
        self._baseline = snapshot.get("rss_mb", 0)

    def check_growth(self, threshold_mb: float = 100) -> Dict[str, Any]:
        """메모리 증가 확인"""
        if self._baseline is None:
            self.set_baseline()

        current = self.take_snapshot("current")
        current_mb = current.get("rss_mb", 0)
        growth = current_mb - self._baseline

        return {
            "baseline_mb": round(self._baseline, 2),
            "current_mb": round(current_mb, 2),
            "growth_mb": round(growth, 2),
            "exceeded_threshold": growth > threshold_mb,
        }

    def get_object_counts(self) -> Dict[str, int]:
        """객체 타입별 개수"""
        gc.collect()
        counts: Dict[str, int] = {}

        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1

        # 상위 20개만 반환
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])

    def get_report(self) -> Dict[str, Any]:
        """메모리 리포트"""
        return {
            "snapshots": self._snapshots[-10:],
            "growth": self.check_growth() if self._baseline else None,
            "object_counts": self.get_object_counts(),
        }


class CPUProfiler:
    """CPU 프로파일러"""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """샘플링 시작"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """샘플링 중지"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _sample_loop(self):
        """샘플링 루프"""
        try:
            import psutil
            process = psutil.Process(os.getpid())

            while self._running:
                cpu_percent = process.cpu_percent(interval=self.interval)
                self._samples.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                })

                # 최근 1000개만 유지
                if len(self._samples) > 1000:
                    self._samples = self._samples[-1000:]

        except ImportError:
            logger.warning("psutil not installed, CPU profiling disabled")

    def get_stats(self) -> Dict[str, Any]:
        """CPU 통계"""
        if not self._samples:
            return {}

        import numpy as np
        cpu_values = [s["cpu_percent"] for s in self._samples]

        return {
            "avg_percent": round(np.mean(cpu_values), 2),
            "max_percent": round(max(cpu_values), 2),
            "min_percent": round(min(cpu_values), 2),
            "samples": len(cpu_values),
        }
