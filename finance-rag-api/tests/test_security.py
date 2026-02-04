# -*- coding: utf-8 -*-
"""
보안 모듈 테스트 (Phase 13)

JWT 인증, API 키 관리, RBAC, 감사 로그 테스트
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


# ============================================================
# Security API 엔드포인트 테스트
# ============================================================

class TestSecurityAPI:
    """보안 API 테스트"""

    def test_security_health(self):
        """보안 모듈 헬스체크"""
        response = client.get("/api/v1/security/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "jwt_available" in data
        assert "features" in data
        assert data["features"]["api_key_auth"] is True
        assert data["features"]["rbac"] is True
        assert data["features"]["audit_logging"] is True

    def test_login_success(self):
        """로그인 성공"""
        response = client.post(
            "/api/v1/security/auth/login",
            json={"username": "admin", "password": "admin"}
        )
        # JWT가 없으면 501, 있으면 200
        assert response.status_code in [200, 501]
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data
            assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self):
        """로그인 실패 - 잘못된 자격증명"""
        response = client.post(
            "/api/v1/security/auth/login",
            json={"username": "admin", "password": "wrong_password"}
        )
        # JWT가 없으면 501, 있으면 401
        assert response.status_code in [401, 501]

    def test_login_unknown_user(self):
        """로그인 실패 - 존재하지 않는 사용자"""
        response = client.post(
            "/api/v1/security/auth/login",
            json={"username": "unknown", "password": "password"}
        )
        assert response.status_code in [401, 501]

    def test_get_me_without_auth(self):
        """인증 없이 사용자 정보 조회"""
        response = client.get("/api/v1/security/auth/me")
        assert response.status_code == 401

    def test_logout_without_auth(self):
        """인증 없이 로그아웃"""
        response = client.post("/api/v1/security/auth/logout")
        assert response.status_code in [401, 501]

    def test_api_keys_list_without_auth(self):
        """인증 없이 API 키 목록 조회"""
        response = client.get("/api/v1/security/api-keys")
        assert response.status_code == 401

    def test_api_keys_create_without_auth(self):
        """인증 없이 API 키 생성"""
        response = client.post(
            "/api/v1/security/api-keys",
            json={"name": "test-key", "role": "user"}
        )
        assert response.status_code in [401, 403]

    def test_audit_logs_without_auth(self):
        """인증 없이 감사 로그 조회"""
        response = client.get("/api/v1/security/audit/logs")
        assert response.status_code in [401, 403]

    def test_audit_stats_without_auth(self):
        """인증 없이 감사 통계 조회"""
        response = client.get("/api/v1/security/audit/stats")
        assert response.status_code in [401, 403]


# ============================================================
# Security 모듈 단위 테스트
# ============================================================

class TestSecurityModule:
    """보안 모듈 단위 테스트"""

    def test_import_security_module(self):
        """보안 모듈 임포트"""
        from src.core.security import (
            Permission,
            Role,
            User,
            SecurityConfig,
            JWTManager,
            APIKey,
            APIKeyManager,
            AuditAction,
            AuditLog,
            AuditLogger,
            JWT_AVAILABLE,
        )
        assert Permission is not None
        assert Role is not None

    def test_permission_enum(self):
        """권한 열거형"""
        from src.core.security import Permission

        assert Permission.READ_DOCUMENTS.value == "read:documents"
        assert Permission.WRITE_DOCUMENTS.value == "write:documents"
        assert Permission.ADMIN.value == "admin:all"

    def test_role_enum(self):
        """역할 열거형"""
        from src.core.security import Role

        assert Role.GUEST.value == "guest"
        assert Role.USER.value == "user"
        assert Role.ANALYST.value == "analyst"
        assert Role.ADMIN.value == "admin"

    def test_role_permissions(self):
        """역할별 권한"""
        from src.core.security import Role, Permission, ROLE_PERMISSIONS

        guest_perms = ROLE_PERMISSIONS[Role.GUEST]
        assert Permission.READ_DOCUMENTS in guest_perms
        assert Permission.WRITE_DOCUMENTS not in guest_perms

        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.MANAGE_USERS in admin_perms
        assert Permission.MANAGE_API_KEYS in admin_perms

    def test_user_creation(self):
        """사용자 생성"""
        from src.core.security import User, Role

        user = User(
            user_id="test-001",
            username="testuser",
            email="test@example.com",
            role=Role.USER,
        )
        assert user.user_id == "test-001"
        assert user.username == "testuser"
        assert user.role == Role.USER

    def test_user_has_permission(self):
        """사용자 권한 확인"""
        from src.core.security import User, Role, Permission

        user = User(
            user_id="user-001",
            username="user",
            email="user@example.com",
            role=Role.USER,
        )
        assert user.has_permission(Permission.READ_DOCUMENTS)
        assert user.has_permission(Permission.WRITE_DOCUMENTS)
        assert not user.has_permission(Permission.ADMIN)

        admin = User(
            user_id="admin-001",
            username="admin",
            email="admin@example.com",
            role=Role.ADMIN,
            permissions={Permission.ADMIN},
        )
        assert admin.has_permission(Permission.ADMIN)

    def test_audit_action_enum(self):
        """감사 액션 열거형"""
        from src.core.security import AuditAction

        assert AuditAction.LOGIN.value == "login"
        assert AuditAction.LOGOUT.value == "logout"
        assert AuditAction.QUERY_EXECUTE.value == "query_execute"


# ============================================================
# API Key Manager 테스트
# ============================================================

class TestAPIKeyManager:
    """API 키 관리자 테스트"""

    def test_api_key_manager_singleton(self):
        """API 키 관리자 싱글톤"""
        from src.core.security import get_api_key_manager

        mgr1 = get_api_key_manager()
        mgr2 = get_api_key_manager()
        assert mgr1 is mgr2

    def test_generate_and_verify_key(self):
        """API 키 생성 및 검증"""
        from src.core.security import APIKeyManager, Role

        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="test-key",
            user_id="user-001",
            role=Role.USER,
        )

        assert raw_key.startswith("frag_")
        assert api_key.name == "test-key"
        assert api_key.role == Role.USER
        assert api_key.is_active is True

        # 키 검증
        verified = manager.verify_key(raw_key)
        assert verified is not None
        assert verified.key_id == api_key.key_id

    def test_verify_invalid_key(self):
        """잘못된 API 키 검증"""
        from src.core.security import APIKeyManager

        manager = APIKeyManager()
        result = manager.verify_key("invalid_key")
        assert result is None

    def test_revoke_key(self):
        """API 키 폐기"""
        from src.core.security import APIKeyManager, Role

        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="to-revoke",
            user_id="user-001",
            role=Role.USER,
        )

        # 폐기
        result = manager.revoke_key(api_key.key_id)
        assert result is True

        # 폐기된 키는 검증 실패
        verified = manager.verify_key(raw_key)
        assert verified is None

    def test_list_keys(self):
        """API 키 목록 조회"""
        from src.core.security import APIKeyManager, Role

        manager = APIKeyManager()
        manager._keys.clear()  # 초기화

        manager.generate_key("key1", "user-001", Role.USER)
        manager.generate_key("key2", "user-001", Role.USER)
        manager.generate_key("key3", "user-002", Role.ANALYST)

        # 전체 목록
        all_keys = manager.list_keys()
        assert len(all_keys) == 3

        # 사용자별 목록
        user1_keys = manager.list_keys("user-001")
        assert len(user1_keys) == 2


# ============================================================
# Audit Logger 테스트
# ============================================================

class TestAuditLogger:
    """감사 로거 테스트"""

    def test_audit_logger_singleton(self):
        """감사 로거 싱글톤"""
        from src.core.security import get_audit_logger

        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2

    def test_log_action(self):
        """액션 로깅"""
        from src.core.security import AuditLogger, AuditAction

        logger = AuditLogger()
        logger._logs.clear()  # 초기화

        log = logger.log(
            action=AuditAction.LOGIN,
            user_id="user-001",
            username="testuser",
            ip_address="127.0.0.1",
        )

        assert log.action == AuditAction.LOGIN
        assert log.user_id == "user-001"
        assert log.success is True

    def test_get_logs_with_filter(self):
        """로그 조회 (필터)"""
        from src.core.security import AuditLogger, AuditAction

        logger = AuditLogger()
        logger._logs.clear()

        logger.log(AuditAction.LOGIN, user_id="user-001")
        logger.log(AuditAction.QUERY_EXECUTE, user_id="user-001")
        logger.log(AuditAction.LOGIN, user_id="user-002")

        # 전체
        all_logs = logger.get_logs()
        assert len(all_logs) == 3

        # 사용자별
        user1_logs = logger.get_logs(user_id="user-001")
        assert len(user1_logs) == 2

        # 액션별
        login_logs = logger.get_logs(action=AuditAction.LOGIN)
        assert len(login_logs) == 2

    def test_get_stats(self):
        """감사 통계 조회"""
        from src.core.security import AuditLogger, AuditAction

        logger = AuditLogger()
        logger._logs.clear()

        logger.log(AuditAction.LOGIN, user_id="user-001")
        logger.log(AuditAction.LOGIN, user_id="user-002")
        logger.log(AuditAction.QUERY_EXECUTE, user_id="user-001")

        stats = logger.get_stats()
        assert stats["total_logs"] == 3
        assert stats["by_action"]["login"] == 2
        assert stats["by_action"]["query_execute"] == 1


# ============================================================
# JWT Manager 테스트 (JWT 사용 가능 시)
# ============================================================

class TestJWTManager:
    """JWT 관리자 테스트"""

    def test_jwt_manager_singleton(self):
        """JWT 관리자 싱글톤"""
        from src.core.security import get_jwt_manager

        mgr1 = get_jwt_manager()
        mgr2 = get_jwt_manager()
        assert mgr1 is mgr2

    def test_jwt_config_defaults(self):
        """JWT 설정 기본값"""
        from src.core.security import SecurityConfig

        config = SecurityConfig()
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_access_token_expire_minutes == 30
        assert config.jwt_refresh_token_expire_days == 7

    def test_create_and_verify_token(self):
        """토큰 생성 및 검증"""
        from src.core.security import JWTManager, User, Role, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")

        manager = JWTManager()
        user = User(
            user_id="test-001",
            username="testuser",
            email="test@example.com",
            role=Role.USER,
        )

        # 액세스 토큰
        access_token = manager.create_access_token(user)
        assert access_token is not None

        # 토큰 검증
        payload = manager.verify_token(access_token)
        assert payload is not None
        assert payload["sub"] == "test-001"
        assert payload["username"] == "testuser"

    def test_revoke_token(self):
        """토큰 폐기"""
        from src.core.security import JWTManager, User, Role, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")

        manager = JWTManager()
        user = User(
            user_id="test-001",
            username="testuser",
            email="test@example.com",
            role=Role.USER,
        )

        token = manager.create_access_token(user)

        # 폐기
        manager.revoke_token(token)

        # 폐기된 토큰 검증 실패
        payload = manager.verify_token(token)
        assert payload is None

    def test_verify_invalid_token(self):
        """잘못된 토큰 검증"""
        from src.core.security import JWTManager, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")

        manager = JWTManager()
        result = manager.verify_token("invalid.token.here")
        assert result is None


# ============================================================
# 라우터 통합 테스트
# ============================================================

class TestSecurityRouterIntegration:
    """보안 라우터 통합 테스트"""

    def test_security_endpoints_registered(self):
        """보안 엔드포인트 등록 확인"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/api/v1/security/auth/login" in paths
        assert "/api/v1/security/auth/me" in paths
        assert "/api/v1/security/api-keys" in paths
        assert "/api/v1/security/audit/logs" in paths
        assert "/api/v1/security/health" in paths

    def test_root_endpoint_includes_security(self):
        """루트 엔드포인트에 보안 정보 포함"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "2.4.0"
        assert "security" in data
        assert "login" in data["security"]
        assert "api_keys" in data["security"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
