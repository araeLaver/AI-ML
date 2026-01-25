# -*- coding: utf-8 -*-
"""
Rate Limiting 모듈

[기능]
- Token Bucket 알고리즘
- Sliding Window 알고리즘
- Leaky Bucket 알고리즘
- 분산 Rate Limiting 지원
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Rate Limit 초과 예외"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        remaining: int = 0,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


@dataclass
class RateLimitResult:
    """Rate Limit 체크 결과"""
    allowed: bool
    limit: int
    remaining: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_headers(self) -> Dict[str, str]:
        """HTTP 헤더로 변환"""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Rate Limiter 기본 클래스"""

    @abstractmethod
    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Rate limit 체크"""
        pass

    @abstractmethod
    def reset(self, key: str) -> None:
        """특정 키 리셋"""
        pass

    def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        """토큰 획득 (실패시 예외)"""
        result = self.check(key, cost)
        if not result.allowed:
            raise RateLimitExceeded(
                retry_after=result.retry_after,
                limit=result.limit,
                remaining=result.remaining,
            )
        return result


class TokenBucket(RateLimiter):
    """
    Token Bucket 알고리즘

    일정 속도로 토큰이 채워지고, 요청 시 토큰 소비
    버스트 트래픽 허용
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._time_func = time_func or time.time

        # 키별 버킷 상태: {key: (tokens, last_update)}
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def _get_bucket(self, key: str) -> Tuple[float, float]:
        """버킷 상태 조회 (토큰 리필 적용)"""
        now = self._time_func()

        if key not in self._buckets:
            return (self.capacity, now)

        tokens, last_update = self._buckets[key]

        # 경과 시간에 따른 토큰 리필
        elapsed = now - last_update
        refilled = tokens + elapsed * self.refill_rate
        tokens = min(refilled, self.capacity)

        return (tokens, now)

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Rate limit 체크 및 토큰 소비"""
        tokens, now = self._get_bucket(key)

        if tokens >= cost:
            # 토큰 소비
            new_tokens = tokens - cost
            self._buckets[key] = (new_tokens, now)

            return RateLimitResult(
                allowed=True,
                limit=self.capacity,
                remaining=int(new_tokens),
                reset_at=now + (self.capacity - new_tokens) / self.refill_rate,
            )
        else:
            # 토큰 부족
            wait_time = (cost - tokens) / self.refill_rate
            self._buckets[key] = (tokens, now)

            return RateLimitResult(
                allowed=False,
                limit=self.capacity,
                remaining=0,
                reset_at=now + wait_time,
                retry_after=wait_time,
            )

    def reset(self, key: str) -> None:
        """버킷 리셋"""
        if key in self._buckets:
            del self._buckets[key]


class SlidingWindowLog(RateLimiter):
    """
    Sliding Window Log 알고리즘

    요청 타임스탬프를 로그로 저장
    정확하지만 메모리 사용량 높음
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._time_func = time_func or time.time

        # 키별 타임스탬프 로그
        self._logs: Dict[str, deque] = {}

    def _cleanup(self, key: str, now: float) -> deque:
        """만료된 타임스탬프 정리"""
        if key not in self._logs:
            self._logs[key] = deque()
            return self._logs[key]

        log = self._logs[key]
        cutoff = now - self.window_seconds

        # 오래된 타임스탬프 제거
        while log and log[0] <= cutoff:
            log.popleft()

        return log

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Rate limit 체크"""
        now = self._time_func()
        log = self._cleanup(key, now)

        current_count = len(log)
        remaining = max(0, self.limit - current_count - cost)

        if current_count + cost <= self.limit:
            # 허용: 타임스탬프 추가
            for _ in range(cost):
                log.append(now)

            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=remaining,
                reset_at=now + self.window_seconds,
            )
        else:
            # 거부: 가장 오래된 타임스탬프 기준 대기 시간
            if log:
                oldest = log[0]
                retry_after = oldest + self.window_seconds - now
            else:
                retry_after = 0

            return RateLimitResult(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_at=now + retry_after,
                retry_after=max(0, retry_after),
            )

    def reset(self, key: str) -> None:
        """로그 리셋"""
        if key in self._logs:
            del self._logs[key]


class SlidingWindowCounter(RateLimiter):
    """
    Sliding Window Counter 알고리즘

    이전 윈도우와 현재 윈도우의 가중 평균 사용
    메모리 효율적, 근사값
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._time_func = time_func or time.time

        # 키별 카운터: {key: {window_id: count}}
        self._counters: Dict[str, Dict[int, int]] = {}

    def _get_window_id(self, timestamp: float) -> int:
        """윈도우 ID 계산"""
        return int(timestamp // self.window_seconds)

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Rate limit 체크"""
        now = self._time_func()
        current_window = self._get_window_id(now)
        prev_window = current_window - 1

        if key not in self._counters:
            self._counters[key] = {}

        counters = self._counters[key]

        # 현재 윈도우 내 위치 (0~1)
        window_progress = (now % self.window_seconds) / self.window_seconds

        # 가중 카운트 계산
        prev_count = counters.get(prev_window, 0)
        curr_count = counters.get(current_window, 0)
        weighted_count = prev_count * (1 - window_progress) + curr_count

        remaining = max(0, int(self.limit - weighted_count - cost))

        if weighted_count + cost <= self.limit:
            # 허용: 현재 윈도우 카운터 증가
            counters[current_window] = curr_count + cost

            # 오래된 윈도우 정리
            old_windows = [w for w in counters if w < prev_window]
            for w in old_windows:
                del counters[w]

            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=remaining,
                reset_at=(current_window + 1) * self.window_seconds,
            )
        else:
            # 거부
            retry_after = self.window_seconds * (1 - window_progress)

            return RateLimitResult(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_at=now + retry_after,
                retry_after=retry_after,
            )

    def reset(self, key: str) -> None:
        """카운터 리셋"""
        if key in self._counters:
            del self._counters[key]


class LeakyBucket(RateLimiter):
    """
    Leaky Bucket 알고리즘

    일정한 속도로 요청 처리
    버스트 제한, 균일한 처리율
    """

    def __init__(
        self,
        capacity: int,
        leak_rate: float,  # requests per second
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.capacity = capacity
        self.leak_rate = leak_rate
        self._time_func = time_func or time.time

        # 키별 버킷 상태: {key: (water_level, last_update)}
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def _get_bucket(self, key: str) -> Tuple[float, float]:
        """버킷 상태 조회 (누수 적용)"""
        now = self._time_func()

        if key not in self._buckets:
            return (0, now)

        level, last_update = self._buckets[key]

        # 경과 시간에 따른 누수
        elapsed = now - last_update
        leaked = elapsed * self.leak_rate
        level = max(0, level - leaked)

        return (level, now)

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Rate limit 체크"""
        level, now = self._get_bucket(key)

        if level + cost <= self.capacity:
            # 허용: 물 추가
            new_level = level + cost
            self._buckets[key] = (new_level, now)

            return RateLimitResult(
                allowed=True,
                limit=self.capacity,
                remaining=int(self.capacity - new_level),
                reset_at=now + new_level / self.leak_rate,
            )
        else:
            # 거부: 오버플로우
            overflow = level + cost - self.capacity
            wait_time = overflow / self.leak_rate
            self._buckets[key] = (level, now)

            return RateLimitResult(
                allowed=False,
                limit=self.capacity,
                remaining=0,
                reset_at=now + wait_time,
                retry_after=wait_time,
            )

    def reset(self, key: str) -> None:
        """버킷 리셋"""
        if key in self._buckets:
            del self._buckets[key]


class FixedWindow(RateLimiter):
    """
    Fixed Window 알고리즘

    고정된 시간 윈도우 내 요청 수 제한
    간단하지만 경계 시점에 버스트 가능
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._time_func = time_func or time.time

        # 키별 카운터: {key: (count, window_start)}
        self._counters: Dict[str, Tuple[int, float]] = {}

    def _get_window_start(self, timestamp: float) -> float:
        """윈도우 시작 시간 계산"""
        return (timestamp // self.window_seconds) * self.window_seconds

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Rate limit 체크"""
        now = self._time_func()
        window_start = self._get_window_start(now)
        window_end = window_start + self.window_seconds

        if key not in self._counters:
            count, stored_window = 0, window_start
        else:
            count, stored_window = self._counters[key]

            # 새 윈도우면 리셋
            if stored_window != window_start:
                count = 0
                stored_window = window_start

        remaining = max(0, self.limit - count - cost)

        if count + cost <= self.limit:
            # 허용
            self._counters[key] = (count + cost, stored_window)

            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=remaining,
                reset_at=window_end,
            )
        else:
            # 거부
            retry_after = window_end - now

            return RateLimitResult(
                allowed=False,
                limit=self.limit,
                remaining=0,
                reset_at=window_end,
                retry_after=retry_after,
            )

    def reset(self, key: str) -> None:
        """카운터 리셋"""
        if key in self._counters:
            del self._counters[key]


class TieredRateLimiter(RateLimiter):
    """
    계층형 Rate Limiter

    여러 시간 단위의 제한 적용
    예: 초당 10, 분당 100, 시간당 1000
    """

    def __init__(self, limiters: List[Tuple[str, RateLimiter]]):
        """
        Args:
            limiters: [(name, limiter), ...] 리스트
        """
        self.limiters = limiters

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """모든 계층 체크"""
        results = []

        for name, limiter in self.limiters:
            result = limiter.check(key, cost)
            results.append((name, result))

            if not result.allowed:
                # 하나라도 거부되면 전체 거부
                # 다른 리미터들 롤백
                for prev_name, _ in results[:-1]:
                    for n, l in self.limiters:
                        if n == prev_name:
                            l.reset(key)
                            break

                return RateLimitResult(
                    allowed=False,
                    limit=result.limit,
                    remaining=result.remaining,
                    reset_at=result.reset_at,
                    retry_after=result.retry_after,
                    metadata={"tier": name},
                )

        # 모두 허용
        min_remaining = min(r.remaining for _, r in results)
        earliest_reset = min(r.reset_at for _, r in results)

        return RateLimitResult(
            allowed=True,
            limit=results[0][1].limit,
            remaining=min_remaining,
            reset_at=earliest_reset,
            metadata={"tiers": [name for name, _ in results]},
        )

    def reset(self, key: str) -> None:
        """모든 계층 리셋"""
        for _, limiter in self.limiters:
            limiter.reset(key)


class RateLimiterFactory:
    """Rate Limiter 팩토리"""

    @staticmethod
    def create_token_bucket(
        requests_per_second: float,
        burst_size: Optional[int] = None,
    ) -> TokenBucket:
        """Token Bucket 생성"""
        capacity = burst_size or int(requests_per_second * 2)
        return TokenBucket(capacity=capacity, refill_rate=requests_per_second)

    @staticmethod
    def create_sliding_window(
        limit: int,
        window_seconds: float,
        use_counter: bool = True,
    ) -> RateLimiter:
        """Sliding Window 생성"""
        if use_counter:
            return SlidingWindowCounter(limit=limit, window_seconds=window_seconds)
        return SlidingWindowLog(limit=limit, window_seconds=window_seconds)

    @staticmethod
    def create_leaky_bucket(
        requests_per_second: float,
        queue_size: Optional[int] = None,
    ) -> LeakyBucket:
        """Leaky Bucket 생성"""
        capacity = queue_size or int(requests_per_second * 5)
        return LeakyBucket(capacity=capacity, leak_rate=requests_per_second)

    @staticmethod
    def create_tiered(
        per_second: int = 10,
        per_minute: int = 100,
        per_hour: int = 1000,
    ) -> TieredRateLimiter:
        """계층형 Rate Limiter 생성"""
        return TieredRateLimiter([
            ("second", FixedWindow(limit=per_second, window_seconds=1)),
            ("minute", FixedWindow(limit=per_minute, window_seconds=60)),
            ("hour", FixedWindow(limit=per_hour, window_seconds=3600)),
        ])


# 사전 정의된 플랜별 Rate Limiter
class RateLimitPlan(Enum):
    """Rate Limit 플랜"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


PLAN_LIMITS = {
    RateLimitPlan.FREE: {
        "requests_per_minute": 10,
        "requests_per_day": 100,
        "burst_size": 5,
    },
    RateLimitPlan.BASIC: {
        "requests_per_minute": 60,
        "requests_per_day": 1000,
        "burst_size": 20,
    },
    RateLimitPlan.PRO: {
        "requests_per_minute": 300,
        "requests_per_day": 10000,
        "burst_size": 50,
    },
    RateLimitPlan.ENTERPRISE: {
        "requests_per_minute": 1000,
        "requests_per_day": 100000,
        "burst_size": 200,
    },
}


def create_plan_limiter(plan: RateLimitPlan) -> TieredRateLimiter:
    """플랜별 Rate Limiter 생성"""
    limits = PLAN_LIMITS[plan]

    return TieredRateLimiter([
        ("burst", TokenBucket(
            capacity=limits["burst_size"],
            refill_rate=limits["requests_per_minute"] / 60,
        )),
        ("minute", FixedWindow(
            limit=limits["requests_per_minute"],
            window_seconds=60,
        )),
        ("day", FixedWindow(
            limit=limits["requests_per_day"],
            window_seconds=86400,
        )),
    ])
