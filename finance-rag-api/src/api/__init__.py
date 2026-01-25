# -*- coding: utf-8 -*-
"""
API 모듈

[기능]
- REST API 라우트
- 인증 및 권한 관리
- Rate Limiting
- 과금 및 쿼터 관리
- 미들웨어 체인
"""

from .routes import router
from .schemas import (
    QueryRequest, QueryResponse,
    DocumentAddRequest, DocumentAddResponse, DocumentListResponse,
    HealthResponse, StatsResponse
)

# Rate Limiting
from .rate_limiter import (
    RateLimiter,
    RateLimitResult,
    RateLimitExceeded,
    TokenBucket,
    SlidingWindowLog,
    SlidingWindowCounter,
    LeakyBucket,
    FixedWindow,
    TieredRateLimiter,
    RateLimiterFactory,
    RateLimitPlan,
    PLAN_LIMITS,
    create_plan_limiter,
)

# Authentication
from .auth import (
    AuthenticationError,
    AuthorizationError,
    Permission,
    User,
    APIKey,
    JWTPayload,
    Authenticator,
    APIKeyAuthenticator,
    JWTAuthenticator,
    OAuthToken,
    OAuthProvider,
    MockOAuthProvider,
    AuthManager,
    require_permissions,
)

# Billing
from .billing import (
    QuotaExceededError,
    ResourceType,
    BillingPlan,
    PlanLimits,
    PLAN_LIMITS as BILLING_PLAN_LIMITS,
    UsageRecord,
    UsageSummary,
    UsageTracker,
    QuotaManager,
    Invoice,
    BillingCalculator,
    BillingService,
    UsageContext,
    track_usage,
)

# Middleware
from .middleware import (
    RequestContext,
    MiddlewareResponse,
    AuthMiddleware,
    RateLimitMiddleware,
    QuotaMiddleware,
    PermissionMiddleware,
    RequestLogger,
    APIMiddlewareChain,
    create_middleware_chain,
)

__all__ = [
    # Routes
    "router",

    # Schemas
    "QueryRequest",
    "QueryResponse",
    "DocumentAddRequest",
    "DocumentAddResponse",
    "DocumentListResponse",
    "HealthResponse",
    "StatsResponse",

    # Rate Limiting
    "RateLimiter",
    "RateLimitResult",
    "RateLimitExceeded",
    "TokenBucket",
    "SlidingWindowLog",
    "SlidingWindowCounter",
    "LeakyBucket",
    "FixedWindow",
    "TieredRateLimiter",
    "RateLimiterFactory",
    "RateLimitPlan",
    "PLAN_LIMITS",
    "create_plan_limiter",

    # Authentication
    "AuthenticationError",
    "AuthorizationError",
    "Permission",
    "User",
    "APIKey",
    "JWTPayload",
    "Authenticator",
    "APIKeyAuthenticator",
    "JWTAuthenticator",
    "OAuthToken",
    "OAuthProvider",
    "MockOAuthProvider",
    "AuthManager",
    "require_permissions",

    # Billing
    "QuotaExceededError",
    "ResourceType",
    "BillingPlan",
    "PlanLimits",
    "BILLING_PLAN_LIMITS",
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "QuotaManager",
    "Invoice",
    "BillingCalculator",
    "BillingService",
    "UsageContext",
    "track_usage",

    # Middleware
    "RequestContext",
    "MiddlewareResponse",
    "AuthMiddleware",
    "RateLimitMiddleware",
    "QuotaMiddleware",
    "PermissionMiddleware",
    "RequestLogger",
    "APIMiddlewareChain",
    "create_middleware_chain",
]
