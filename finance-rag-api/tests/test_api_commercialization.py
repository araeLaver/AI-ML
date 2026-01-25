# -*- coding: utf-8 -*-
"""
API 상용화 테스트 (Phase 8)

Rate Limiting, 인증, 과금 테스트
"""

import time
import pytest

from src.api.rate_limiter import (
    TokenBucket,
    SlidingWindowLog,
    SlidingWindowCounter,
    LeakyBucket,
    FixedWindow,
    TieredRateLimiter,
    RateLimiterFactory,
    RateLimitExceeded,
    RateLimitPlan,
    create_plan_limiter,
)
from src.api.auth import (
    Permission,
    User,
    APIKey,
    APIKeyAuthenticator,
    JWTAuthenticator,
    MockOAuthProvider,
    AuthManager,
    AuthenticationError,
    AuthorizationError,
    require_permissions,
)
from src.api.billing import (
    ResourceType,
    BillingPlan,
    UsageTracker,
    QuotaManager,
    BillingCalculator,
    BillingService,
    QuotaExceededError,
    UsageContext,
)
from src.api.middleware import (
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


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestTokenBucket:
    """Token Bucket 테스트"""

    def test_allow_within_capacity(self):
        """용량 내 허용"""
        mock_time = [0.0]
        bucket = TokenBucket(
            capacity=10,
            refill_rate=1.0,
            time_func=lambda: mock_time[0],
        )

        for _ in range(10):
            result = bucket.check("test_key")
            assert result.allowed

    def test_deny_over_capacity(self):
        """용량 초과 거부"""
        mock_time = [0.0]
        bucket = TokenBucket(
            capacity=5,
            refill_rate=1.0,
            time_func=lambda: mock_time[0],
        )

        # 5개 소비
        for _ in range(5):
            bucket.check("test_key")

        # 6번째는 거부
        result = bucket.check("test_key")
        assert not result.allowed
        assert result.retry_after > 0

    def test_refill_over_time(self):
        """시간에 따른 리필"""
        mock_time = [0.0]
        bucket = TokenBucket(
            capacity=5,
            refill_rate=2.0,  # 초당 2토큰
            time_func=lambda: mock_time[0],
        )

        # 모든 토큰 소비
        for _ in range(5):
            bucket.check("test_key")

        # 2초 후 4토큰 리필
        mock_time[0] = 2.0
        result = bucket.check("test_key")
        assert result.allowed
        assert result.remaining >= 3

    def test_reset(self):
        """리셋"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)

        for _ in range(5):
            bucket.check("test_key")

        bucket.reset("test_key")
        result = bucket.check("test_key")
        assert result.allowed
        assert result.remaining == 4


class TestSlidingWindowLog:
    """Sliding Window Log 테스트"""

    def test_allow_within_limit(self):
        """제한 내 허용"""
        mock_time = [0.0]
        window = SlidingWindowLog(
            limit=10,
            window_seconds=60.0,
            time_func=lambda: mock_time[0],
        )

        for _ in range(10):
            result = window.check("test_key")
            assert result.allowed

    def test_deny_over_limit(self):
        """제한 초과 거부"""
        mock_time = [0.0]
        window = SlidingWindowLog(
            limit=5,
            window_seconds=60.0,
            time_func=lambda: mock_time[0],
        )

        for _ in range(5):
            window.check("test_key")

        result = window.check("test_key")
        assert not result.allowed

    def test_window_slides(self):
        """윈도우 슬라이딩"""
        mock_time = [0.0]
        window = SlidingWindowLog(
            limit=3,
            window_seconds=10.0,
            time_func=lambda: mock_time[0],
        )

        # 3개 소비
        for _ in range(3):
            window.check("test_key")

        # 거부
        result = window.check("test_key")
        assert not result.allowed

        # 11초 후 윈도우 밖으로
        mock_time[0] = 11.0
        result = window.check("test_key")
        assert result.allowed


class TestSlidingWindowCounter:
    """Sliding Window Counter 테스트"""

    def test_weighted_count(self):
        """가중 카운트"""
        mock_time = [0.0]
        window = SlidingWindowCounter(
            limit=10,
            window_seconds=60.0,
            time_func=lambda: mock_time[0],
        )

        # 이전 윈도우에서 8개 사용
        for _ in range(8):
            window.check("test_key")

        # 새 윈도우 중간 (50%)으로 이동
        mock_time[0] = 90.0  # 1.5 윈도우 (30초 진행)

        # 가중 카운트: 8 * 0.5 = 4
        # 새로 6개 추가 가능
        for _ in range(6):
            result = window.check("test_key")
            assert result.allowed


class TestLeakyBucket:
    """Leaky Bucket 테스트"""

    def test_allow_within_capacity(self):
        """용량 내 허용"""
        mock_time = [0.0]
        bucket = LeakyBucket(
            capacity=5,
            leak_rate=1.0,
            time_func=lambda: mock_time[0],
        )

        for _ in range(5):
            result = bucket.check("test_key")
            assert result.allowed

    def test_leak_over_time(self):
        """시간에 따른 누수"""
        mock_time = [0.0]
        bucket = LeakyBucket(
            capacity=5,
            leak_rate=2.0,  # 초당 2
            time_func=lambda: mock_time[0],
        )

        # 5개 채움
        for _ in range(5):
            bucket.check("test_key")

        # 거부
        result = bucket.check("test_key")
        assert not result.allowed

        # 2초 후 4개 누수
        mock_time[0] = 2.0
        result = bucket.check("test_key")
        assert result.allowed


class TestFixedWindow:
    """Fixed Window 테스트"""

    def test_reset_on_new_window(self):
        """새 윈도우에서 리셋"""
        mock_time = [0.0]
        window = FixedWindow(
            limit=5,
            window_seconds=60.0,
            time_func=lambda: mock_time[0],
        )

        for _ in range(5):
            window.check("test_key")

        result = window.check("test_key")
        assert not result.allowed

        # 새 윈도우로 이동
        mock_time[0] = 61.0
        result = window.check("test_key")
        assert result.allowed


class TestTieredRateLimiter:
    """계층형 Rate Limiter 테스트"""

    def test_all_tiers_check(self):
        """모든 계층 확인"""
        mock_time = [0.0]
        limiter = TieredRateLimiter([
            ("second", FixedWindow(limit=2, window_seconds=1, time_func=lambda: mock_time[0])),
            ("minute", FixedWindow(limit=10, window_seconds=60, time_func=lambda: mock_time[0])),
        ])

        # 초당 2개까지 허용
        result = limiter.check("test_key")
        assert result.allowed
        result = limiter.check("test_key")
        assert result.allowed

        # 3번째는 초 제한에 걸림
        result = limiter.check("test_key")
        assert not result.allowed
        assert result.metadata.get("tier") == "second"


class TestRateLimiterFactory:
    """Rate Limiter 팩토리 테스트"""

    def test_create_token_bucket(self):
        """Token Bucket 생성"""
        limiter = RateLimiterFactory.create_token_bucket(
            requests_per_second=10.0,
            burst_size=20,
        )
        assert isinstance(limiter, TokenBucket)

    def test_create_sliding_window(self):
        """Sliding Window 생성"""
        limiter = RateLimiterFactory.create_sliding_window(
            limit=100,
            window_seconds=60.0,
        )
        assert isinstance(limiter, SlidingWindowCounter)

    def test_create_tiered(self):
        """계층형 생성"""
        limiter = RateLimiterFactory.create_tiered(
            per_second=10,
            per_minute=100,
            per_hour=1000,
        )
        assert isinstance(limiter, TieredRateLimiter)


class TestPlanLimiter:
    """플랜별 Rate Limiter 테스트"""

    def test_create_plan_limiter(self):
        """플랜 리미터 생성"""
        for plan in RateLimitPlan:
            limiter = create_plan_limiter(plan)
            assert isinstance(limiter, TieredRateLimiter)


# =============================================================================
# Authentication Tests
# =============================================================================

class TestUser:
    """User 테스트"""

    def test_has_permission(self):
        """권한 확인"""
        user = User(
            id="user1",
            permissions={Permission.RAG_QUERY, Permission.READ},
        )

        assert user.has_permission(Permission.RAG_QUERY)
        assert user.has_permission(Permission.READ)
        assert not user.has_permission(Permission.ADMIN)

    def test_admin_has_all_permissions(self):
        """관리자 모든 권한"""
        admin = User(
            id="admin1",
            permissions={Permission.ADMIN},
        )

        assert admin.has_permission(Permission.RAG_QUERY)
        assert admin.has_permission(Permission.BILLING_MANAGE)

    def test_has_any_permission(self):
        """권한 중 하나"""
        user = User(
            id="user1",
            permissions={Permission.READ},
        )

        assert user.has_any_permission([Permission.READ, Permission.WRITE])
        assert not user.has_any_permission([Permission.ADMIN, Permission.DELETE])


class TestAPIKeyAuthenticator:
    """API Key 인증 테스트"""

    def test_generate_key(self):
        """키 생성"""
        plain_key, key_hash = APIKeyAuthenticator.generate_key("sk")

        assert plain_key.startswith("sk_")
        assert len(key_hash) == 64  # SHA-256

    def test_register_and_authenticate(self):
        """등록 및 인증"""
        auth = APIKeyAuthenticator()
        plain_key, api_key = auth.register_key(
            user_id="user1",
            name="Test Key",
            permissions={Permission.RAG_QUERY},
        )

        user = auth.authenticate(plain_key)
        assert user.id == "user1"
        assert Permission.RAG_QUERY in user.permissions

    def test_invalid_key(self):
        """잘못된 키"""
        auth = APIKeyAuthenticator()

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate("invalid_key")
        assert exc_info.value.code == "INVALID_KEY"

    def test_revoked_key(self):
        """폐기된 키"""
        auth = APIKeyAuthenticator()
        plain_key, api_key = auth.register_key(
            user_id="user1",
            name="Test Key",
        )

        auth.revoke_key(api_key.id)

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate(plain_key)
        assert exc_info.value.code == "KEY_REVOKED"


class TestJWTAuthenticator:
    """JWT 인증 테스트"""

    def test_create_and_verify_token(self):
        """토큰 생성 및 검증"""
        auth = JWTAuthenticator(secret_key="test_secret")

        token = auth.create_token(
            user_id="user1",
            permissions=[Permission.RAG_QUERY],
        )

        payload = auth.verify_token(token)
        assert payload.sub == "user1"
        assert "rag:query" in payload.permissions

    def test_authenticate(self):
        """JWT 인증"""
        auth = JWTAuthenticator(secret_key="test_secret")

        token = auth.create_token(
            user_id="user1",
            permissions=[Permission.RAG_QUERY, Permission.READ],
        )

        user = auth.authenticate(token)
        assert user.id == "user1"

    def test_expired_token(self):
        """만료된 토큰"""
        auth = JWTAuthenticator(
            secret_key="test_secret",
            token_ttl_seconds=1,  # 1초
        )

        token = auth.create_token(
            user_id="user1",
            permissions=[Permission.RAG_QUERY],
        )

        time.sleep(1.5)

        with pytest.raises(AuthenticationError) as exc_info:
            auth.verify_token(token)
        assert exc_info.value.code == "TOKEN_EXPIRED"

    def test_invalid_signature(self):
        """잘못된 서명"""
        auth1 = JWTAuthenticator(secret_key="secret1")
        auth2 = JWTAuthenticator(secret_key="secret2")

        token = auth1.create_token(
            user_id="user1",
            permissions=[Permission.RAG_QUERY],
        )

        with pytest.raises(AuthenticationError) as exc_info:
            auth2.verify_token(token)
        assert exc_info.value.code == "INVALID_SIGNATURE"

    def test_revoke_token(self):
        """토큰 폐기"""
        auth = JWTAuthenticator(secret_key="test_secret")

        token = auth.create_token(
            user_id="user1",
            permissions=[Permission.RAG_QUERY],
        )

        payload = auth.verify_token(token)
        auth.revoke_token(payload.jti)

        with pytest.raises(AuthenticationError) as exc_info:
            auth.verify_token(token)
        assert exc_info.value.code == "TOKEN_REVOKED"


class TestMockOAuthProvider:
    """OAuth 제공자 테스트"""

    def test_authorization_flow(self):
        """인증 흐름"""
        provider = MockOAuthProvider(
            client_id="test_client",
            client_secret="test_secret",
            redirect_uri="http://localhost/callback",
        )

        # 1. 인증 URL 생성
        auth_url = provider.get_authorization_url("state123")
        assert "client_id=test_client" in auth_url

        # 2. 인증 시뮬레이션
        code = provider.simulate_authorization("state123", "user1")

        # 3. 코드 교환
        token = provider.exchange_code(code)
        assert token.access_token
        assert token.refresh_token

        # 4. 사용자 정보 조회
        user_info = provider.get_user_info(token.access_token)
        assert user_info["id"] == "user1"


class TestAuthManager:
    """AuthManager 테스트"""

    def test_api_key_auth(self):
        """API 키 인증"""
        api_key_auth = APIKeyAuthenticator()
        plain_key, _ = api_key_auth.register_key(
            user_id="user1",
            name="Test Key",
        )

        manager = AuthManager(api_key_auth=api_key_auth)
        user = manager.authenticate(plain_key, "api_key")
        assert user.id == "user1"

    def test_jwt_auth(self):
        """JWT 인증"""
        jwt_auth = JWTAuthenticator(secret_key="test_secret")
        token = jwt_auth.create_token(
            user_id="user1",
            permissions=[Permission.RAG_QUERY],
        )

        manager = AuthManager(jwt_auth=jwt_auth)
        user = manager.authenticate(token, "jwt")
        assert user.id == "user1"

    def test_parse_authorization_header(self):
        """Authorization 헤더 파싱"""
        manager = AuthManager()

        # Bearer JWT
        auth_type, cred = manager.parse_authorization_header("Bearer a.b.c")
        assert auth_type == "jwt"

        # Bearer OAuth
        auth_type, cred = manager.parse_authorization_header("Bearer simpletoken")
        assert auth_type == "oauth:default"

        # API Key
        auth_type, cred = manager.parse_authorization_header("ApiKey sk_test_key")
        assert auth_type == "api_key"


class TestRequirePermissions:
    """권한 데코레이터 테스트"""

    def test_allowed(self):
        """허용"""
        @require_permissions(Permission.RAG_QUERY)
        def protected_func(user: User):
            return "success"

        user = User(id="user1", permissions={Permission.RAG_QUERY})
        result = protected_func(user)
        assert result == "success"

    def test_denied(self):
        """거부"""
        @require_permissions(Permission.ADMIN)
        def admin_func(user: User):
            return "success"

        user = User(id="user1", permissions={Permission.RAG_QUERY})
        with pytest.raises(AuthorizationError):
            admin_func(user)


# =============================================================================
# Billing Tests
# =============================================================================

class TestUsageTracker:
    """사용량 추적 테스트"""

    def test_record_usage(self):
        """사용량 기록"""
        tracker = UsageTracker()
        tracker.set_plan("user1", BillingPlan.STARTER)

        record = tracker.record(
            user_id="user1",
            resource_type=ResourceType.RAG_QUERY,
            quantity=1,
        )

        assert record.user_id == "user1"
        assert record.resource_type == ResourceType.RAG_QUERY

    def test_get_usage(self):
        """사용량 조회"""
        tracker = UsageTracker()

        for _ in range(5):
            tracker.record("user1", ResourceType.RAG_QUERY)
        for _ in range(3):
            tracker.record("user1", ResourceType.API_CALL)

        assert tracker.get_usage("user1", ResourceType.RAG_QUERY) == 5
        assert tracker.get_usage("user1", ResourceType.API_CALL) == 3
        assert tracker.get_usage("user1") == 8

    def test_monthly_summary(self):
        """월간 요약"""
        tracker = UsageTracker()
        tracker.set_plan("user1", BillingPlan.STARTER)

        for _ in range(10):
            tracker.record("user1", ResourceType.RAG_QUERY)

        summary = tracker.get_monthly_summary("user1")
        assert summary.usage[ResourceType.RAG_QUERY] == 10
        assert summary.limits[ResourceType.RAG_QUERY] == 1000  # STARTER plan


class TestQuotaManager:
    """쿼터 관리 테스트"""

    def test_check_quota(self):
        """쿼터 확인"""
        tracker = UsageTracker()
        tracker.set_plan("user1", BillingPlan.FREE)
        manager = QuotaManager(tracker)

        # FREE 플랜: 100 RAG 쿼리
        allowed, remaining = manager.check_quota(
            "user1",
            ResourceType.RAG_QUERY,
        )
        assert allowed
        assert remaining == 100

    def test_quota_exceeded(self):
        """쿼터 초과"""
        tracker = UsageTracker()
        tracker.set_plan("user1", BillingPlan.FREE)
        manager = QuotaManager(tracker, allow_overage=False)

        # FREE 플랜 쿼터 소진
        for _ in range(100):
            manager.consume("user1", ResourceType.RAG_QUERY)

        with pytest.raises(QuotaExceededError) as exc_info:
            manager.consume("user1", ResourceType.RAG_QUERY)
        assert exc_info.value.resource == "rag_query"


class TestBillingCalculator:
    """과금 계산 테스트"""

    def test_calculate_overage(self):
        """초과 사용량 계산"""
        tracker = UsageTracker()
        tracker.set_plan("user1", BillingPlan.FREE)
        calculator = BillingCalculator(tracker)

        # FREE 플랜 초과 (100 이상)
        for _ in range(150):
            tracker.record("user1", ResourceType.RAG_QUERY)

        from datetime import datetime
        now = datetime.utcnow()
        overages = calculator.calculate_overage("user1", now.year, now.month)

        # FREE 플랜은 overage_rate가 없으므로 비용 0
        rag_overage = overages.get(ResourceType.RAG_QUERY, (0, 0))
        assert rag_overage[0] == 50  # 150 - 100 = 50 초과

    def test_generate_invoice(self):
        """청구서 생성"""
        tracker = UsageTracker()
        tracker.set_plan("user1", BillingPlan.STARTER)
        calculator = BillingCalculator(tracker)

        from datetime import datetime
        now = datetime.utcnow()
        invoice = calculator.generate_invoice("user1", now.year, now.month)

        assert invoice.user_id == "user1"
        assert invoice.base_amount == 29.0  # STARTER 가격
        assert len(invoice.line_items) >= 1


class TestBillingService:
    """과금 서비스 테스트"""

    def test_full_workflow(self):
        """전체 워크플로우"""
        service = BillingService(allow_overage=True)
        service.set_user_plan("user1", BillingPlan.STARTER)

        # 사용량 추적
        for _ in range(10):
            service.track_usage("user1", ResourceType.RAG_QUERY)

        # 쿼터 확인
        assert service.check_quota("user1", ResourceType.RAG_QUERY)

        # 사용량 요약
        summary = service.get_usage_summary("user1")
        assert summary.usage[ResourceType.RAG_QUERY] == 10

        # 청구서 생성
        invoice = service.generate_invoice("user1")
        assert invoice.total_amount >= 29.0


class TestUsageContext:
    """사용량 컨텍스트 테스트"""

    def test_context_manager(self):
        """컨텍스트 매니저"""
        service = BillingService()
        service.set_user_plan("user1", BillingPlan.STARTER)

        with UsageContext(service, "user1", ResourceType.RAG_QUERY):
            pass  # 작업 수행

        summary = service.get_usage_summary("user1")
        assert summary.usage[ResourceType.RAG_QUERY] == 1


# =============================================================================
# Middleware Tests
# =============================================================================

class TestRequestContext:
    """RequestContext 테스트"""

    def test_duration(self):
        """소요 시간"""
        ctx = RequestContext(
            path="/api/test",
            method="GET",
        )
        time.sleep(0.01)
        assert ctx.duration_ms > 0


class TestAuthMiddleware:
    """인증 미들웨어 테스트"""

    def test_skip_excluded_paths(self):
        """제외 경로 건너뛰기"""
        api_key_auth = APIKeyAuthenticator()
        manager = AuthManager(api_key_auth=api_key_auth)
        middleware = AuthMiddleware(manager, exclude_paths=["/health"])

        ctx = RequestContext(path="/health", method="GET")
        response = middleware.process(None, "/health", ctx)
        assert response.allowed

    def test_missing_auth(self):
        """인증 헤더 누락"""
        api_key_auth = APIKeyAuthenticator()
        manager = AuthManager(api_key_auth=api_key_auth)
        middleware = AuthMiddleware(manager)

        ctx = RequestContext(path="/api/test", method="GET")
        response = middleware.process(None, "/api/test", ctx)
        assert not response.allowed
        assert response.status_code == 401


class TestRateLimitMiddleware:
    """Rate Limit 미들웨어 테스트"""

    def test_allow_request(self):
        """요청 허용"""
        middleware = RateLimitMiddleware()

        ctx = RequestContext(
            path="/api/test",
            method="GET",
            client_ip="127.0.0.1",
        )
        response = middleware.process(ctx)
        assert response.allowed
        assert "X-RateLimit-Limit" in response.headers


class TestQuotaMiddleware:
    """쿼터 미들웨어 테스트"""

    def test_allow_within_quota(self):
        """쿼터 내 허용"""
        service = BillingService()
        service.set_user_plan("user1", BillingPlan.STARTER)
        middleware = QuotaMiddleware(service)

        user = User(id="user1", permissions={Permission.RAG_QUERY})
        ctx = RequestContext(
            path="/api/rag/query",
            method="POST",
        )
        ctx.user = user

        response = middleware.process(ctx)
        assert response.allowed


class TestPermissionMiddleware:
    """권한 미들웨어 테스트"""

    def test_allow_with_permission(self):
        """권한 있으면 허용"""
        middleware = PermissionMiddleware()

        user = User(id="user1", permissions={Permission.RAG_QUERY, Permission.WRITE})
        ctx = RequestContext(
            path="/api/rag/query",
            method="POST",
        )
        ctx.user = user

        response = middleware.process(ctx)
        assert response.allowed

    def test_deny_without_permission(self):
        """권한 없으면 거부"""
        middleware = PermissionMiddleware()

        user = User(id="user1", permissions={Permission.READ})
        ctx = RequestContext(
            path="/api/rag/query",
            method="POST",
        )
        ctx.user = user

        response = middleware.process(ctx)
        assert not response.allowed
        assert response.status_code == 403


class TestAPIMiddlewareChain:
    """미들웨어 체인 테스트"""

    def test_full_chain(self):
        """전체 체인"""
        api_key_auth = APIKeyAuthenticator()
        plain_key, _ = api_key_auth.register_key(
            user_id="user1",
            name="Test Key",
            permissions={Permission.RAG_QUERY, Permission.WRITE},
            rate_limit_plan="basic",
        )

        auth_manager = AuthManager(api_key_auth=api_key_auth)
        billing_service = BillingService()
        billing_service.set_user_plan("user1", BillingPlan.STARTER)

        chain = create_middleware_chain(
            auth_manager=auth_manager,
            billing_service=billing_service,
        )

        response = chain.process(
            authorization=f"ApiKey {plain_key}",
            path="/api/rag/query",
            method="POST",
            client_ip="127.0.0.1",
        )

        assert response.allowed
        assert response.context.user is not None


class TestRequestLogger:
    """요청 로거 테스트"""

    def test_mask_sensitive(self):
        """민감 정보 마스킹"""
        logger_obj = RequestLogger()

        data = {
            "user": "test",
            "authorization": "secret_key",
            "nested": {
                "password": "secret123",
            },
        }

        masked = logger_obj._mask_sensitive(data)
        assert masked["user"] == "test"
        assert masked["authorization"] == "***MASKED***"
        assert masked["nested"]["password"] == "***MASKED***"
