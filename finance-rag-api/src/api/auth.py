# -*- coding: utf-8 -*-
"""
인증 모듈

[기능]
- API Key 인증
- JWT 토큰 인증
- OAuth 2.0 지원
- 권한 관리
"""

import base64
import hashlib
import hmac
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """인증 오류"""

    def __init__(self, message: str, code: str = "AUTH_ERROR"):
        super().__init__(message)
        self.code = code


class AuthorizationError(Exception):
    """권한 오류"""

    def __init__(self, message: str, required_permissions: Optional[List[str]] = None):
        super().__init__(message)
        self.required_permissions = required_permissions or []


class Permission(Enum):
    """권한 열거형"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

    # RAG 관련 권한
    RAG_QUERY = "rag:query"
    RAG_INDEX = "rag:index"
    RAG_DELETE = "rag:delete"

    # 관리 권한
    API_KEY_CREATE = "api_key:create"
    API_KEY_REVOKE = "api_key:revoke"
    USAGE_VIEW = "usage:view"
    BILLING_MANAGE = "billing:manage"


@dataclass
class User:
    """사용자"""
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def has_permission(self, permission: Permission) -> bool:
        """권한 확인"""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """권한 중 하나라도 보유"""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """모든 권한 보유"""
        return all(self.has_permission(p) for p in permissions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class APIKey:
    """API 키"""
    id: str
    key_hash: str  # 해시된 키 저장
    key_prefix: str  # 키 접두사 (식별용)
    user_id: str
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    rate_limit_plan: str = "basic"
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """만료 확인"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """유효성 확인"""
        return self.is_active and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "key_prefix": self.key_prefix,
            "user_id": self.user_id,
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "rate_limit_plan": self.rate_limit_plan,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
        }


@dataclass
class JWTPayload:
    """JWT 페이로드"""
    sub: str  # Subject (user_id)
    iat: float  # Issued at
    exp: float  # Expiration
    permissions: List[str] = field(default_factory=list)
    jti: Optional[str] = None  # JWT ID
    iss: Optional[str] = None  # Issuer
    aud: Optional[str] = None  # Audience
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        return time.time() > self.exp

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "sub": self.sub,
            "iat": self.iat,
            "exp": self.exp,
            "permissions": self.permissions,
        }
        if self.jti:
            data["jti"] = self.jti
        if self.iss:
            data["iss"] = self.iss
        if self.aud:
            data["aud"] = self.aud
        if self.metadata:
            data.update(self.metadata)
        return data


class Authenticator(ABC):
    """인증기 기본 클래스"""

    @abstractmethod
    def authenticate(self, credentials: Any) -> User:
        """인증 수행"""
        pass


class APIKeyAuthenticator(Authenticator):
    """API Key 인증기"""

    def __init__(self, key_store: Optional[Dict[str, APIKey]] = None):
        self._keys: Dict[str, APIKey] = key_store or {}
        self._key_hashes: Dict[str, str] = {}  # hash -> key_id

    @staticmethod
    def generate_key(prefix: str = "sk") -> tuple[str, str]:
        """API 키 생성

        Returns:
            (plain_key, key_hash)
        """
        # 32바이트 랜덤 = 256비트 엔트로피
        random_bytes = secrets.token_bytes(32)
        key_suffix = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")
        plain_key = f"{prefix}_{key_suffix}"

        # SHA-256 해시
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        return plain_key, key_hash

    @staticmethod
    def hash_key(plain_key: str) -> str:
        """키 해시"""
        return hashlib.sha256(plain_key.encode()).hexdigest()

    def register_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[Set[Permission]] = None,
        rate_limit_plan: str = "basic",
        expires_in_days: Optional[int] = None,
        prefix: str = "sk",
    ) -> tuple[str, APIKey]:
        """API 키 등록

        Returns:
            (plain_key, api_key_object)
        """
        plain_key, key_hash = self.generate_key(prefix)
        key_prefix = plain_key[:10]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            id=str(uuid.uuid4()),
            key_hash=key_hash,
            key_prefix=key_prefix,
            user_id=user_id,
            name=name,
            permissions=permissions or {Permission.RAG_QUERY},
            rate_limit_plan=rate_limit_plan,
            expires_at=expires_at,
        )

        self._keys[api_key.id] = api_key
        self._key_hashes[key_hash] = api_key.id

        logger.info(f"API key created: {key_prefix}... for user {user_id}")
        return plain_key, api_key

    def revoke_key(self, key_id: str) -> bool:
        """API 키 폐기"""
        if key_id in self._keys:
            api_key = self._keys[key_id]
            api_key.is_active = False
            logger.info(f"API key revoked: {api_key.key_prefix}...")
            return True
        return False

    def authenticate(self, credentials: str) -> User:
        """API 키 인증"""
        key_hash = self.hash_key(credentials)

        if key_hash not in self._key_hashes:
            raise AuthenticationError("Invalid API key", "INVALID_KEY")

        key_id = self._key_hashes[key_hash]
        api_key = self._keys.get(key_id)

        if not api_key:
            raise AuthenticationError("API key not found", "KEY_NOT_FOUND")

        if not api_key.is_active:
            raise AuthenticationError("API key has been revoked", "KEY_REVOKED")

        if api_key.is_expired():
            raise AuthenticationError("API key has expired", "KEY_EXPIRED")

        # 마지막 사용 시간 업데이트
        api_key.last_used_at = datetime.utcnow()

        return User(
            id=api_key.user_id,
            permissions=api_key.permissions,
            metadata={
                "api_key_id": api_key.id,
                "rate_limit_plan": api_key.rate_limit_plan,
            },
        )

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """키 조회"""
        return self._keys.get(key_id)

    def list_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """키 목록"""
        keys = list(self._keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys


class JWTAuthenticator(Authenticator):
    """JWT 인증기"""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        token_ttl_seconds: int = 3600,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.token_ttl = token_ttl_seconds

        # 폐기된 토큰 (jti)
        self._revoked_tokens: Set[str] = set()

    def _base64_encode(self, data: bytes) -> str:
        """Base64 URL-safe 인코딩"""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _base64_decode(self, data: str) -> bytes:
        """Base64 URL-safe 디코딩"""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def _sign(self, message: str) -> str:
        """HMAC 서명"""
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).digest()
        return self._base64_encode(signature)

    def create_token(
        self,
        user_id: str,
        permissions: List[Permission],
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """JWT 토큰 생성"""
        import json

        now = time.time()
        ttl = ttl_seconds or self.token_ttl

        payload = JWTPayload(
            sub=user_id,
            iat=now,
            exp=now + ttl,
            permissions=[p.value for p in permissions],
            jti=str(uuid.uuid4()),
            iss=self.issuer,
            aud=self.audience,
            metadata=metadata or {},
        )

        # 헤더
        header = {"alg": self.algorithm, "typ": "JWT"}
        header_b64 = self._base64_encode(json.dumps(header).encode())

        # 페이로드
        payload_b64 = self._base64_encode(json.dumps(payload.to_dict()).encode())

        # 서명
        message = f"{header_b64}.{payload_b64}"
        signature = self._sign(message)

        return f"{header_b64}.{payload_b64}.{signature}"

    def verify_token(self, token: str) -> JWTPayload:
        """JWT 토큰 검증"""
        import json

        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Invalid token format", "INVALID_TOKEN")

        header_b64, payload_b64, signature = parts

        # 서명 검증
        message = f"{header_b64}.{payload_b64}"
        expected_signature = self._sign(message)

        if not hmac.compare_digest(signature, expected_signature):
            raise AuthenticationError("Invalid token signature", "INVALID_SIGNATURE")

        # 페이로드 파싱
        try:
            payload_json = self._base64_decode(payload_b64)
            payload_data = json.loads(payload_json)
        except Exception:
            raise AuthenticationError("Invalid token payload", "INVALID_PAYLOAD")

        payload = JWTPayload(
            sub=payload_data.get("sub", ""),
            iat=payload_data.get("iat", 0),
            exp=payload_data.get("exp", 0),
            permissions=payload_data.get("permissions", []),
            jti=payload_data.get("jti"),
            iss=payload_data.get("iss"),
            aud=payload_data.get("aud"),
        )

        # 만료 확인
        if payload.is_expired():
            raise AuthenticationError("Token has expired", "TOKEN_EXPIRED")

        # 폐기 확인
        if payload.jti and payload.jti in self._revoked_tokens:
            raise AuthenticationError("Token has been revoked", "TOKEN_REVOKED")

        # Issuer 확인
        if self.issuer and payload.iss != self.issuer:
            raise AuthenticationError("Invalid token issuer", "INVALID_ISSUER")

        # Audience 확인
        if self.audience and payload.aud != self.audience:
            raise AuthenticationError("Invalid token audience", "INVALID_AUDIENCE")

        return payload

    def authenticate(self, credentials: str) -> User:
        """JWT 토큰 인증"""
        payload = self.verify_token(credentials)

        permissions = set()
        for p in payload.permissions:
            try:
                permissions.add(Permission(p))
            except ValueError:
                pass

        return User(
            id=payload.sub,
            permissions=permissions,
            metadata={"jti": payload.jti},
        )

    def revoke_token(self, jti: str) -> None:
        """토큰 폐기"""
        self._revoked_tokens.add(jti)

    def refresh_token(self, token: str, extend_ttl: Optional[int] = None) -> str:
        """토큰 갱신"""
        payload = self.verify_token(token)

        # 기존 토큰 폐기
        if payload.jti:
            self.revoke_token(payload.jti)

        permissions = [Permission(p) for p in payload.permissions]

        return self.create_token(
            user_id=payload.sub,
            permissions=permissions,
            ttl_seconds=extend_ttl,
            metadata=payload.metadata,
        )


@dataclass
class OAuthToken:
    """OAuth 토큰"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }
        if self.refresh_token:
            data["refresh_token"] = self.refresh_token
        if self.scope:
            data["scope"] = self.scope
        return data


class OAuthProvider(ABC):
    """OAuth 제공자 기본 클래스"""

    @abstractmethod
    def get_authorization_url(self, state: str) -> str:
        """인증 URL 생성"""
        pass

    @abstractmethod
    def exchange_code(self, code: str) -> OAuthToken:
        """인증 코드 교환"""
        pass

    @abstractmethod
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """사용자 정보 조회"""
        pass


class MockOAuthProvider(OAuthProvider):
    """테스트용 OAuth 제공자"""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

        self._codes: Dict[str, Dict[str, Any]] = {}
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def get_authorization_url(self, state: str) -> str:
        """인증 URL 생성"""
        return (
            f"https://oauth.example.com/authorize"
            f"?client_id={self.client_id}"
            f"&redirect_uri={self.redirect_uri}"
            f"&state={state}"
            f"&response_type=code"
        )

    def simulate_authorization(self, state: str, user_id: str) -> str:
        """인증 시뮬레이션 (테스트용)"""
        code = secrets.token_urlsafe(32)
        self._codes[code] = {
            "user_id": user_id,
            "state": state,
            "created_at": time.time(),
        }
        return code

    def exchange_code(self, code: str) -> OAuthToken:
        """인증 코드 교환"""
        if code not in self._codes:
            raise AuthenticationError("Invalid authorization code", "INVALID_CODE")

        code_data = self._codes.pop(code)
        user_id = code_data["user_id"]

        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        self._tokens[access_token] = {
            "user_id": user_id,
            "created_at": time.time(),
            "expires_in": 3600,
        }

        return OAuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600,
            scope="read write",
        )

    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """사용자 정보 조회"""
        if access_token not in self._tokens:
            raise AuthenticationError("Invalid access token", "INVALID_TOKEN")

        token_data = self._tokens[access_token]

        return {
            "id": token_data["user_id"],
            "email": f"{token_data['user_id']}@example.com",
            "name": f"User {token_data['user_id']}",
        }


class AuthManager:
    """인증 관리자"""

    def __init__(
        self,
        api_key_auth: Optional[APIKeyAuthenticator] = None,
        jwt_auth: Optional[JWTAuthenticator] = None,
        oauth_providers: Optional[Dict[str, OAuthProvider]] = None,
    ):
        self.api_key_auth = api_key_auth
        self.jwt_auth = jwt_auth
        self.oauth_providers = oauth_providers or {}

    def authenticate(
        self,
        credentials: str,
        auth_type: str = "api_key",
    ) -> User:
        """통합 인증"""
        if auth_type == "api_key":
            if not self.api_key_auth:
                raise AuthenticationError("API key authentication not configured")
            return self.api_key_auth.authenticate(credentials)

        elif auth_type == "jwt":
            if not self.jwt_auth:
                raise AuthenticationError("JWT authentication not configured")
            return self.jwt_auth.authenticate(credentials)

        elif auth_type.startswith("oauth:"):
            provider_name = auth_type.split(":")[1]
            if provider_name not in self.oauth_providers:
                raise AuthenticationError(f"OAuth provider not found: {provider_name}")

            provider = self.oauth_providers[provider_name]
            user_info = provider.get_user_info(credentials)

            return User(
                id=user_info["id"],
                email=user_info.get("email"),
                name=user_info.get("name"),
                permissions={Permission.RAG_QUERY},
            )

        else:
            raise AuthenticationError(f"Unknown auth type: {auth_type}")

    def parse_authorization_header(self, header: str) -> tuple[str, str]:
        """Authorization 헤더 파싱

        Returns:
            (auth_type, credentials)
        """
        parts = header.split(" ", 1)
        if len(parts) != 2:
            raise AuthenticationError("Invalid authorization header")

        scheme, credentials = parts

        if scheme.lower() == "bearer":
            # JWT 또는 OAuth 토큰
            if credentials.count(".") == 2:
                return "jwt", credentials
            return "oauth:default", credentials

        elif scheme.lower() == "apikey":
            return "api_key", credentials

        else:
            raise AuthenticationError(f"Unknown authorization scheme: {scheme}")


def require_permissions(*permissions: Permission):
    """권한 체크 데코레이터"""

    def decorator(func: Callable):
        def wrapper(user: User, *args, **kwargs):
            for perm in permissions:
                if not user.has_permission(perm):
                    raise AuthorizationError(
                        f"Permission required: {perm.value}",
                        required_permissions=[perm.value],
                    )
            return func(user, *args, **kwargs)
        return wrapper
    return decorator
