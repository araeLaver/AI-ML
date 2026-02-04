# -*- coding: utf-8 -*-
"""
보안 모듈

JWT 인증, API 키 관리, RBAC 권한 관리, 감사 로깅
"""

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None  # type: ignore

from .logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 설정 및 상수
# ============================================================

class Permission(Enum):
    """권한 정의"""
    # 읽기 권한
    READ_DOCUMENTS = "read:documents"
    READ_QUERIES = "read:queries"
    READ_STATS = "read:stats"

    # 쓰기 권한
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"

    # 관리 권한
    MANAGE_USERS = "manage:users"
    MANAGE_API_KEYS = "manage:api_keys"
    MANAGE_SETTINGS = "manage:settings"

    # 시스템 권한
    ADMIN = "admin:all"


class Role(Enum):
    """역할 정의"""
    GUEST = "guest"
    USER = "user"
    ANALYST = "analyst"
    ADMIN = "admin"
    SYSTEM = "system"


# 역할별 권한 매핑
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.GUEST: {
        Permission.READ_DOCUMENTS,
        Permission.READ_QUERIES,
    },
    Role.USER: {
        Permission.READ_DOCUMENTS,
        Permission.READ_QUERIES,
        Permission.READ_STATS,
        Permission.WRITE_DOCUMENTS,
    },
    Role.ANALYST: {
        Permission.READ_DOCUMENTS,
        Permission.READ_QUERIES,
        Permission.READ_STATS,
        Permission.WRITE_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
    },
    Role.ADMIN: {
        Permission.READ_DOCUMENTS,
        Permission.READ_QUERIES,
        Permission.READ_STATS,
        Permission.WRITE_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_API_KEYS,
        Permission.MANAGE_SETTINGS,
    },
    Role.SYSTEM: {
        Permission.ADMIN,
    },
}


@dataclass
class SecurityConfig:
    """보안 설정"""
    # JWT
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # API Key
    api_key_prefix: str = "frag_"
    api_key_length: int = 32

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Security Headers
    enable_cors: bool = True
    enable_csrf: bool = False


# ============================================================
# 사용자 및 토큰 모델
# ============================================================

@dataclass
class User:
    """사용자 정보"""
    user_id: str
    username: str
    email: str
    role: Role = Role.USER
    permissions: set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """권한 확인"""
        # ADMIN 권한은 모든 것 허용
        if Permission.ADMIN in self.permissions:
            return True
        # 역할 기반 권한 확인
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        return permission in self.permissions or permission in role_perms

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


@dataclass
class TokenPayload:
    """JWT 토큰 페이로드"""
    sub: str  # user_id
    username: str
    role: str
    permissions: list[str]
    exp: datetime
    iat: datetime = field(default_factory=datetime.now)
    jti: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_type: str = "access"


@dataclass
class APIKey:
    """API 키"""
    key_id: str
    key_hash: str  # 해시된 키
    name: str
    user_id: str
    role: Role = Role.USER
    permissions: set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "name": self.name,
            "user_id": self.user_id,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
        }


# ============================================================
# JWT 토큰 관리
# ============================================================

class JWTManager:
    """JWT 토큰 관리자"""

    def __init__(self, config: Optional[SecurityConfig] = None):
        if not JWT_AVAILABLE:
            logger.warning("PyJWT not installed, JWT features disabled")

        self.config = config or SecurityConfig()
        self._revoked_tokens: set[str] = set()  # 폐기된 토큰 (실제로는 Redis 사용)

    def create_access_token(self, user: User) -> str:
        """액세스 토큰 생성"""
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT is required for JWT tokens")

        expire = datetime.utcnow() + timedelta(
            minutes=self.config.jwt_access_token_expire_minutes
        )

        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "access",
        }

        return jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
        )

    def create_refresh_token(self, user: User) -> str:
        """리프레시 토큰 생성"""
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT is required for JWT tokens")

        expire = datetime.utcnow() + timedelta(
            days=self.config.jwt_refresh_token_expire_days
        )

        payload = {
            "sub": user.user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "refresh",
        }

        return jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
        )

    def verify_token(self, token: str) -> Optional[dict]:
        """토큰 검증"""
        if not JWT_AVAILABLE:
            return None

        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )

            # 폐기된 토큰 확인
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                logger.warning(f"Revoked token used: {jti}")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """토큰 폐기"""
        if not JWT_AVAILABLE:
            return False

        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": False},  # 만료된 토큰도 폐기 가능
            )
            jti = payload.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                return True
        except Exception:
            pass
        return False

    def refresh_access_token(self, refresh_token: str, user: User) -> Optional[str]:
        """액세스 토큰 갱신"""
        payload = self.verify_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            if payload.get("sub") == user.user_id:
                return self.create_access_token(user)
        return None


# ============================================================
# API 키 관리
# ============================================================

class APIKeyManager:
    """API 키 관리자"""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey (실제로는 DB 사용)

    def generate_key(
        self,
        name: str,
        user_id: str,
        role: Role = Role.USER,
        permissions: Optional[set[Permission]] = None,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """새 API 키 생성

        Returns:
            (raw_key, api_key_object)
        """
        # 랜덤 키 생성
        raw_key = (
            self.config.api_key_prefix +
            secrets.token_urlsafe(self.config.api_key_length)
        )

        # 해시 저장
        key_hash = self._hash_key(raw_key)
        key_id = str(uuid.uuid4())

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            role=role,
            permissions=permissions or set(),
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        logger.info(f"API key created: {name} for user {user_id}")

        return raw_key, api_key

    def verify_key(self, raw_key: str) -> Optional[APIKey]:
        """API 키 검증"""
        key_hash = self._hash_key(raw_key)

        for api_key in self._keys.values():
            if api_key.key_hash == key_hash:
                if not api_key.is_active:
                    logger.warning(f"Inactive API key used: {api_key.key_id}")
                    return None
                if api_key.is_expired():
                    logger.warning(f"Expired API key used: {api_key.key_id}")
                    return None

                # 사용 기록 업데이트
                api_key.last_used = datetime.now()
                api_key.usage_count += 1
                return api_key

        return None

    def revoke_key(self, key_id: str) -> bool:
        """API 키 폐기"""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            logger.info(f"API key revoked: {key_id}")
            return True
        return False

    def list_keys(self, user_id: Optional[str] = None) -> list[APIKey]:
        """API 키 목록"""
        keys = list(self._keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys

    def _hash_key(self, raw_key: str) -> str:
        """키 해시"""
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ============================================================
# 감사 로깅
# ============================================================

class AuditAction(Enum):
    """감사 액션"""
    LOGIN = "login"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    DOCUMENT_CREATE = "document_create"
    DOCUMENT_DELETE = "document_delete"
    QUERY_EXECUTE = "query_execute"
    PERMISSION_CHANGE = "permission_change"
    SETTINGS_CHANGE = "settings_change"


@dataclass
class AuditLog:
    """감사 로그 항목"""
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    action: AuditAction = AuditAction.QUERY_EXECUTE
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
        }


class AuditLogger:
    """감사 로거"""

    def __init__(self, max_logs: int = 10000):
        self._logs: list[AuditLog] = []
        self._max_logs = max_logs

    def log(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditLog:
        """감사 로그 기록"""
        audit_log = AuditLog(
            action=action,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            resource_id=resource_id,
            details=details or {},
            success=success,
            error_message=error_message,
        )

        self._logs.append(audit_log)

        # 로그 크기 제한
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]

        # 로깅
        log_level = "info" if success else "warning"
        log_msg = (
            f"AUDIT: {action.value} | user={username or user_id} | "
            f"resource={resource}:{resource_id} | success={success}"
        )
        getattr(logger, log_level)(log_msg)

        return audit_log

    def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        """감사 로그 조회"""
        logs = self._logs.copy()

        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if action:
            logs = [l for l in logs if l.action == action]
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]

        return logs[-limit:]

    def get_stats(self) -> dict:
        """감사 통계"""
        total = len(self._logs)
        success = sum(1 for l in self._logs if l.success)
        by_action = {}
        for log in self._logs:
            action = log.action.value
            by_action[action] = by_action.get(action, 0) + 1

        return {
            "total_logs": total,
            "successful": success,
            "failed": total - success,
            "by_action": by_action,
        }


# ============================================================
# 글로벌 인스턴스
# ============================================================

_jwt_manager: Optional[JWTManager] = None
_api_key_manager: Optional[APIKeyManager] = None
_audit_logger: Optional[AuditLogger] = None


def get_jwt_manager(config: Optional[SecurityConfig] = None) -> JWTManager:
    """JWT 관리자 싱글톤"""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager(config)
    return _jwt_manager


def get_api_key_manager(config: Optional[SecurityConfig] = None) -> APIKeyManager:
    """API 키 관리자 싱글톤"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager(config)
    return _api_key_manager


def get_audit_logger() -> AuditLogger:
    """감사 로거 싱글톤"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
