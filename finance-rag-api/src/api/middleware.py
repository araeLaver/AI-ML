# -*- coding: utf-8 -*-
"""
API 미들웨어

[기능]
- 요청/응답 로깅
- 요청 ID 추적
- 인증 미들웨어
- Rate Limiting 미들웨어
- 사용량 추적 미들웨어
- 권한 검사 미들웨어
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# FastAPI 미들웨어 (기존 호환성)
try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from src.core.logging import get_logger, set_request_id
    _fastapi_logger = get_logger(__name__)
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .auth import (
    AuthManager,
    AuthenticationError,
    AuthorizationError,
    Permission,
    User,
)
from .billing import (
    BillingService,
    QuotaExceededError,
    ResourceType,
)
from .rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitResult,
    create_plan_limiter,
    RateLimitPlan,
)

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    """요청 컨텍스트"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user: Optional[User] = None
    start_time: float = field(default_factory=time.time)
    path: str = ""
    method: str = ""
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """요청 처리 시간 (밀리초)"""
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user.id if self.user else None,
            "path": self.path,
            "method": self.method,
            "client_ip": self.client_ip,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class MiddlewareResponse:
    """미들웨어 응답"""
    allowed: bool = True
    status_code: int = 200
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    context: Optional[RequestContext] = None

    def to_error_dict(self) -> Dict[str, Any]:
        """에러 응답 딕셔너리"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.error_message,
            },
            "request_id": self.context.request_id if self.context else None,
        }


class AuthMiddleware:
    """인증 미들웨어"""

    def __init__(
        self,
        auth_manager: AuthManager,
        exclude_paths: Optional[List[str]] = None,
    ):
        self.auth_manager = auth_manager
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    def _should_skip(self, path: str) -> bool:
        """인증 건너뛰기 여부"""
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return True
        return False

    def process(
        self,
        authorization: Optional[str],
        path: str,
        context: RequestContext,
    ) -> MiddlewareResponse:
        """인증 처리"""
        if self._should_skip(path):
            return MiddlewareResponse(allowed=True, context=context)

        if not authorization:
            return MiddlewareResponse(
                allowed=False,
                status_code=401,
                error_message="Missing authorization header",
                error_code="MISSING_AUTH",
                headers={"WWW-Authenticate": "Bearer"},
                context=context,
            )

        try:
            auth_type, credentials = self.auth_manager.parse_authorization_header(
                authorization
            )
            user = self.auth_manager.authenticate(credentials, auth_type)
            context.user = user

            return MiddlewareResponse(allowed=True, context=context)

        except AuthenticationError as e:
            return MiddlewareResponse(
                allowed=False,
                status_code=401,
                error_message=str(e),
                error_code=e.code,
                headers={"WWW-Authenticate": "Bearer"},
                context=context,
            )


class RateLimitMiddleware:
    """Rate Limit 미들웨어"""

    def __init__(
        self,
        default_limiter: Optional[RateLimiter] = None,
        plan_limiters: Optional[Dict[str, RateLimiter]] = None,
    ):
        self.default_limiter = default_limiter
        self.plan_limiters = plan_limiters or {}

        # 기본 플랜별 리미터 생성
        if not self.plan_limiters:
            for plan in RateLimitPlan:
                self.plan_limiters[plan.value] = create_plan_limiter(plan)

    def _get_limiter(self, user: Optional[User]) -> RateLimiter:
        """사용자에 맞는 리미터 반환"""
        if user and "rate_limit_plan" in user.metadata:
            plan = user.metadata["rate_limit_plan"]
            if plan in self.plan_limiters:
                return self.plan_limiters[plan]

        return self.default_limiter or self.plan_limiters.get(
            RateLimitPlan.FREE.value
        )

    def _get_key(self, user: Optional[User], client_ip: Optional[str]) -> str:
        """Rate limit 키 생성"""
        if user:
            return f"user:{user.id}"
        if client_ip:
            return f"ip:{client_ip}"
        return "anonymous"

    def process(
        self,
        context: RequestContext,
        cost: int = 1,
    ) -> MiddlewareResponse:
        """Rate limit 처리"""
        limiter = self._get_limiter(context.user)
        if not limiter:
            return MiddlewareResponse(allowed=True, context=context)

        key = self._get_key(context.user, context.client_ip)

        try:
            result = limiter.acquire(key, cost)

            return MiddlewareResponse(
                allowed=True,
                headers=result.to_headers(),
                context=context,
            )

        except RateLimitExceeded as e:
            headers = {
                "X-RateLimit-Limit": str(e.limit),
                "X-RateLimit-Remaining": str(e.remaining),
            }
            if e.retry_after:
                headers["Retry-After"] = str(int(e.retry_after))

            return MiddlewareResponse(
                allowed=False,
                status_code=429,
                error_message="Rate limit exceeded",
                error_code="RATE_LIMIT_EXCEEDED",
                headers=headers,
                context=context,
            )


class QuotaMiddleware:
    """쿼터 미들웨어"""

    def __init__(
        self,
        billing_service: BillingService,
        resource_mapping: Optional[Dict[str, ResourceType]] = None,
    ):
        self.billing_service = billing_service
        # 경로 패턴 -> 리소스 타입 매핑
        self.resource_mapping = resource_mapping or {
            "/api/rag/query": ResourceType.RAG_QUERY,
            "/api/rag/index": ResourceType.DOCUMENT_INDEX,
            "/api/": ResourceType.API_CALL,
        }

    def _get_resource_type(self, path: str) -> ResourceType:
        """경로에 맞는 리소스 타입"""
        for prefix, resource_type in self.resource_mapping.items():
            if path.startswith(prefix):
                return resource_type
        return ResourceType.API_CALL

    def process(
        self,
        context: RequestContext,
    ) -> MiddlewareResponse:
        """쿼터 확인"""
        if not context.user:
            return MiddlewareResponse(allowed=True, context=context)

        resource_type = self._get_resource_type(context.path)

        allowed = self.billing_service.check_quota(
            user_id=context.user.id,
            resource_type=resource_type,
        )

        if not allowed:
            status = self.billing_service.get_quota_status(context.user.id)
            quota_info = status["quotas"].get(resource_type.value, {})

            return MiddlewareResponse(
                allowed=False,
                status_code=402,
                error_message=f"Quota exceeded for {resource_type.value}",
                error_code="QUOTA_EXCEEDED",
                headers={
                    "X-Quota-Limit": str(quota_info.get("limit", 0)),
                    "X-Quota-Used": str(quota_info.get("used", 0)),
                    "X-Quota-Remaining": str(quota_info.get("remaining", 0)),
                },
                context=context,
            )

        return MiddlewareResponse(allowed=True, context=context)


class PermissionMiddleware:
    """권한 미들웨어"""

    def __init__(
        self,
        permission_mapping: Optional[Dict[str, List[Permission]]] = None,
    ):
        # 경로 패턴 -> 필요 권한 매핑
        self.permission_mapping = permission_mapping or {
            "/api/rag/query": [Permission.RAG_QUERY],
            "/api/rag/index": [Permission.RAG_INDEX],
            "/api/rag/delete": [Permission.RAG_DELETE],
            "/api/admin/": [Permission.ADMIN],
            "/api/keys/": [Permission.API_KEY_CREATE],
        }

    def _get_required_permissions(self, path: str, method: str) -> List[Permission]:
        """필요한 권한 목록"""
        permissions = []

        for prefix, perms in self.permission_mapping.items():
            if path.startswith(prefix):
                permissions.extend(perms)

        # HTTP 메서드별 추가 권한
        if method in ["POST", "PUT", "PATCH"]:
            permissions.append(Permission.WRITE)
        elif method == "DELETE":
            permissions.append(Permission.DELETE)

        return list(set(permissions))

    def process(
        self,
        context: RequestContext,
    ) -> MiddlewareResponse:
        """권한 확인"""
        if not context.user:
            return MiddlewareResponse(allowed=True, context=context)

        required = self._get_required_permissions(context.path, context.method)

        if not required:
            return MiddlewareResponse(allowed=True, context=context)

        if not context.user.has_any_permission(required):
            return MiddlewareResponse(
                allowed=False,
                status_code=403,
                error_message="Insufficient permissions",
                error_code="FORBIDDEN",
                headers={
                    "X-Required-Permissions": ",".join(p.value for p in required),
                },
                context=context,
            )

        return MiddlewareResponse(allowed=True, context=context)


class RequestLogger:
    """요청 로거"""

    def __init__(
        self,
        log_headers: bool = False,
        log_body: bool = False,
        sensitive_fields: Optional[List[str]] = None,
    ):
        self.log_headers = log_headers
        self.log_body = log_body
        self.sensitive_fields = sensitive_fields or [
            "authorization", "api_key", "password", "token"
        ]

    def _mask_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """민감 정보 마스킹"""
        masked = {}
        for key, value in data.items():
            if key.lower() in self.sensitive_fields:
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        return masked

    def log_request(
        self,
        context: RequestContext,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
    ) -> None:
        """요청 로깅"""
        log_data = {
            "type": "request",
            "request_id": context.request_id,
            "method": context.method,
            "path": context.path,
            "client_ip": context.client_ip,
            "user_id": context.user.id if context.user else None,
        }

        if self.log_headers and headers:
            log_data["headers"] = self._mask_sensitive(headers)

        if self.log_body and body:
            log_data["body"] = self._mask_sensitive(body) if isinstance(body, dict) else str(body)[:500]

        logger.info(f"API Request: {log_data}")

    def log_response(
        self,
        context: RequestContext,
        status_code: int,
        response_size: Optional[int] = None,
    ) -> None:
        """응답 로깅"""
        log_data = {
            "type": "response",
            "request_id": context.request_id,
            "status_code": status_code,
            "duration_ms": round(context.duration_ms, 2),
            "user_id": context.user.id if context.user else None,
        }

        if response_size:
            log_data["response_size"] = response_size

        logger.info(f"API Response: {log_data}")


class APIMiddlewareChain:
    """API 미들웨어 체인"""

    def __init__(
        self,
        auth_middleware: Optional[AuthMiddleware] = None,
        rate_limit_middleware: Optional[RateLimitMiddleware] = None,
        quota_middleware: Optional[QuotaMiddleware] = None,
        permission_middleware: Optional[PermissionMiddleware] = None,
        request_logger: Optional[RequestLogger] = None,
    ):
        self.auth_middleware = auth_middleware
        self.rate_limit_middleware = rate_limit_middleware
        self.quota_middleware = quota_middleware
        self.permission_middleware = permission_middleware
        self.request_logger = request_logger or RequestLogger()

    def process(
        self,
        authorization: Optional[str],
        path: str,
        method: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> MiddlewareResponse:
        """미들웨어 체인 처리"""
        context = RequestContext(
            path=path,
            method=method,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        # 요청 로깅
        self.request_logger.log_request(context, headers)

        # 1. 인증
        if self.auth_middleware:
            response = self.auth_middleware.process(authorization, path, context)
            if not response.allowed:
                return response

        # 2. Rate Limiting
        if self.rate_limit_middleware:
            response = self.rate_limit_middleware.process(context)
            if not response.allowed:
                return response

        # 3. 쿼터 확인
        if self.quota_middleware:
            response = self.quota_middleware.process(context)
            if not response.allowed:
                return response

        # 4. 권한 확인
        if self.permission_middleware:
            response = self.permission_middleware.process(context)
            if not response.allowed:
                return response

        return MiddlewareResponse(allowed=True, context=context)

    def post_process(
        self,
        context: RequestContext,
        status_code: int,
        response_size: Optional[int] = None,
    ) -> None:
        """후처리 (응답 로깅 등)"""
        self.request_logger.log_response(context, status_code, response_size)


def create_middleware_chain(
    auth_manager: Optional[AuthManager] = None,
    billing_service: Optional[BillingService] = None,
    exclude_auth_paths: Optional[List[str]] = None,
) -> APIMiddlewareChain:
    """미들웨어 체인 생성 헬퍼"""
    auth_middleware = None
    quota_middleware = None

    if auth_manager:
        auth_middleware = AuthMiddleware(
            auth_manager=auth_manager,
            exclude_paths=exclude_auth_paths or ["/health", "/docs", "/openapi.json"],
        )

    if billing_service:
        quota_middleware = QuotaMiddleware(billing_service=billing_service)

    return APIMiddlewareChain(
        auth_middleware=auth_middleware,
        rate_limit_middleware=RateLimitMiddleware(),
        quota_middleware=quota_middleware,
        permission_middleware=PermissionMiddleware(),
        request_logger=RequestLogger(),
    )


# =============================================================================
# FastAPI 미들웨어 (기존 호환성)
# =============================================================================

if HAS_FASTAPI:
    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        """
        요청/응답 로깅 미들웨어

        - 고유 요청 ID 생성 및 추적
        - 요청/응답 시간 측정
        - 요청 메타데이터 로깅
        """

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            # 요청 ID 생성
            request_id = str(uuid.uuid4())
            set_request_id(request_id)

            # 시작 시간
            start_time = time.perf_counter()

            # 요청 정보 로깅
            _fastapi_logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "query": str(request.query_params),
                        "client_ip": request.client.host if request.client else "unknown",
                    }
                }
            )

            # 응답 처리
            try:
                response = await call_next(request)
            except Exception as e:
                _fastapi_logger.error(
                    f"Unhandled exception: {str(e)}",
                    exc_info=True
                )
                raise

            # 처리 시간 계산
            process_time = (time.perf_counter() - start_time) * 1000  # ms

            # 응답 헤더에 메타데이터 추가
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

            # 응답 정보 로깅
            log_method = _fastapi_logger.info if response.status_code < 400 else _fastapi_logger.warning
            log_method(
                f"Request completed: {response.status_code} ({process_time:.2f}ms)",
                extra={
                    "extra_fields": {
                        "status_code": response.status_code,
                        "process_time_ms": round(process_time, 2),
                    }
                }
            )

            return response

    class CORSHeaderMiddleware(BaseHTTPMiddleware):
        """CORS 헤더 미들웨어 (간단한 CORS 지원)"""

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            response = await call_next(request)

            # CORS 헤더 추가
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Request-ID"

            return response
else:
    # FastAPI 없이 실행시 더미 클래스
    class RequestLoggingMiddleware:
        """FastAPI 없이 실행시 더미 클래스"""
        pass

    class CORSHeaderMiddleware:
        """FastAPI 없이 실행시 더미 클래스"""
        pass
