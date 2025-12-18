# -*- coding: utf-8 -*-
"""
API 보안 모듈

[백엔드 개발자 관점]
- Spring Security의 Filter와 유사한 패턴
- API Key 기반 인증
- Rate Limiting
- 보안 유틸리티

[포트폴리오 포인트]
- 실무에서 자주 사용하는 인증 패턴
- 확장 가능한 구조 (JWT, OAuth 등으로 확장 가능)
"""

import time
import secrets
from typing import Optional, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

from fastapi import HTTPException, Security, status, Request, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# API Key 인증
# ============================================================

# API Key 헤더/쿼리 파라미터 정의
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


class APIKeyInfo(BaseModel):
    """API Key 정보"""
    key: str
    name: str
    created_at: datetime
    rate_limit: int = 100  # 분당 요청 수
    is_active: bool = True


# 데모용 API Keys (실제로는 DB에서 관리)
DEMO_API_KEYS: Dict[str, APIKeyInfo] = {
    "demo-api-key-2024": APIKeyInfo(
        key="demo-api-key-2024",
        name="Demo Key",
        created_at=datetime.now(),
        rate_limit=100,
        is_active=True
    ),
    "test-key-for-development": APIKeyInfo(
        key="test-key-for-development",
        name="Development Key",
        created_at=datetime.now(),
        rate_limit=1000,
        is_active=True
    ),
}


def generate_api_key() -> str:
    """새 API Key 생성"""
    return secrets.token_urlsafe(32)


async def get_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query)
) -> Optional[str]:
    """
    API Key 추출 (헤더 또는 쿼리 파라미터)

    우선순위: 헤더 > 쿼리 파라미터
    """
    return api_key_header or api_key_query


async def validate_api_key(
    api_key: Optional[str] = Depends(get_api_key)
) -> APIKeyInfo:
    """
    API Key 유효성 검증

    Raises:
        HTTPException: 인증 실패 시
    """
    settings = get_settings()

    # 개발 모드에서는 인증 건너뛰기 (선택적)
    if settings.app_env == "development" and not api_key:
        # 개발 환경에서 API key 없이도 접근 허용
        return APIKeyInfo(
            key="dev-bypass",
            name="Development Bypass",
            created_at=datetime.now(),
            rate_limit=10000,
            is_active=True
        )

    if not api_key:
        logger.warning("API key missing in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "AUTH001",
                "message": "API Key가 필요합니다.",
                "hint": "X-API-Key 헤더 또는 api_key 쿼리 파라미터를 사용하세요."
            }
        )

    key_info = DEMO_API_KEYS.get(api_key)

    if not key_info:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "AUTH002",
                "message": "유효하지 않은 API Key입니다."
            }
        )

    if not key_info.is_active:
        logger.warning(f"Inactive API key used: {key_info.name}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": True,
                "code": "AUTH003",
                "message": "비활성화된 API Key입니다."
            }
        )

    logger.info(f"API key validated: {key_info.name}")
    return key_info


# ============================================================
# Rate Limiting
# ============================================================

class RateLimiter:
    """
    간단한 인메모리 Rate Limiter

    [주의] 프로덕션에서는 Redis 등 외부 저장소 사용 권장
    """

    def __init__(self, default_limit: int = 60, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)

    def _clean_old_requests(self, key: str) -> None:
        """오래된 요청 기록 정리"""
        cutoff = time.time() - self.window_seconds
        self.requests[key] = [
            ts for ts in self.requests[key] if ts > cutoff
        ]

    def is_allowed(self, key: str, limit: Optional[int] = None) -> bool:
        """요청 허용 여부 확인"""
        limit = limit or self.default_limit
        self._clean_old_requests(key)

        if len(self.requests[key]) >= limit:
            return False

        self.requests[key].append(time.time())
        return True

    def get_remaining(self, key: str, limit: Optional[int] = None) -> int:
        """남은 요청 수 반환"""
        limit = limit or self.default_limit
        self._clean_old_requests(key)
        return max(0, limit - len(self.requests[key]))

    def get_reset_time(self, key: str) -> int:
        """리셋까지 남은 시간(초)"""
        if not self.requests[key]:
            return 0
        oldest = min(self.requests[key])
        return max(0, int(self.window_seconds - (time.time() - oldest)))


# 전역 Rate Limiter 인스턴스
rate_limiter = RateLimiter(default_limit=60, window_seconds=60)


async def check_rate_limit(
    request: Request,
    api_key_info: APIKeyInfo = Depends(validate_api_key)
) -> APIKeyInfo:
    """
    Rate Limit 체크 의존성

    응답 헤더에 rate limit 정보 추가
    """
    # IP + API Key 조합으로 식별
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{api_key_info.key}:{client_ip}"

    limit = api_key_info.rate_limit

    if not rate_limiter.is_allowed(rate_key, limit):
        remaining = rate_limiter.get_remaining(rate_key, limit)
        reset_time = rate_limiter.get_reset_time(rate_key)

        logger.warning(
            f"Rate limit exceeded for {api_key_info.name}",
            extra={"extra_fields": {"client_ip": client_ip}}
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": True,
                "code": "RATE001",
                "message": "요청 한도를 초과했습니다.",
                "limit": limit,
                "reset_in_seconds": reset_time
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(reset_time)
            }
        )

    return api_key_info


# ============================================================
# 선택적 인증 (공개 API와 보호 API 구분)
# ============================================================

async def optional_api_key(
    api_key: Optional[str] = Depends(get_api_key)
) -> Optional[APIKeyInfo]:
    """
    선택적 API Key 검증 (없어도 됨)

    공개 엔드포인트에서 사용
    """
    if not api_key:
        return None

    key_info = DEMO_API_KEYS.get(api_key)
    if key_info and key_info.is_active:
        return key_info

    return None
