# -*- coding: utf-8 -*-
"""
보안 API 라우터

인증, API 키 관리, 감사 로그 API 엔드포인트
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.security import (
    User,
    Role,
    Permission,
    AuditAction,
    get_jwt_manager,
    get_api_key_manager,
    get_audit_logger,
    JWT_AVAILABLE,
)

logger = get_logger(__name__)

# 라우터 생성
security_router = APIRouter(tags=["Security"])
bearer_scheme = HTTPBearer(auto_error=False)


# ============================================================
# 스키마 정의
# ============================================================

class LoginRequest(BaseModel):
    """로그인 요청"""
    username: str = Field(..., description="사용자명")
    password: str = Field(..., description="비밀번호")


class LoginResponse(BaseModel):
    """로그인 응답"""
    access_token: str = Field(..., description="액세스 토큰")
    refresh_token: str = Field(..., description="리프레시 토큰")
    token_type: str = Field("bearer", description="토큰 타입")
    expires_in: int = Field(..., description="만료 시간 (초)")


class TokenRefreshRequest(BaseModel):
    """토큰 갱신 요청"""
    refresh_token: str = Field(..., description="리프레시 토큰")


class TokenRefreshResponse(BaseModel):
    """토큰 갱신 응답"""
    access_token: str = Field(..., description="새 액세스 토큰")
    token_type: str = Field("bearer", description="토큰 타입")


class APIKeyCreateRequest(BaseModel):
    """API 키 생성 요청"""
    name: str = Field(..., description="키 이름")
    role: str = Field("user", description="역할")
    expires_in_days: Optional[int] = Field(None, description="만료일 (일)")


class APIKeyCreateResponse(BaseModel):
    """API 키 생성 응답"""
    key: str = Field(..., description="API 키 (한 번만 표시)")
    key_id: str = Field(..., description="키 ID")
    name: str = Field(..., description="키 이름")
    expires_at: Optional[str] = Field(None, description="만료 시간")


class APIKeyResponse(BaseModel):
    """API 키 정보"""
    key_id: str = Field(..., description="키 ID")
    name: str = Field(..., description="키 이름")
    role: str = Field(..., description="역할")
    is_active: bool = Field(..., description="활성 상태")
    created_at: str = Field(..., description="생성 시간")
    expires_at: Optional[str] = Field(None, description="만료 시간")
    last_used: Optional[str] = Field(None, description="마지막 사용")
    usage_count: int = Field(..., description="사용 횟수")


class AuditLogResponse(BaseModel):
    """감사 로그"""
    log_id: str = Field(..., description="로그 ID")
    timestamp: str = Field(..., description="시간")
    action: str = Field(..., description="액션")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    username: Optional[str] = Field(None, description="사용자명")
    resource: Optional[str] = Field(None, description="리소스")
    success: bool = Field(..., description="성공 여부")


class UserInfoResponse(BaseModel):
    """사용자 정보 응답"""
    user_id: str = Field(..., description="사용자 ID")
    username: str = Field(..., description="사용자명")
    email: str = Field(..., description="이메일")
    role: str = Field(..., description="역할")
    permissions: list[str] = Field(..., description="권한")


# ============================================================
# 인증 의존성
# ============================================================

# 데모용 사용자 (실제로는 DB에서 조회)
DEMO_USERS = {
    "admin": User(
        user_id="admin-001",
        username="admin",
        email="admin@example.com",
        role=Role.ADMIN,
        permissions={Permission.ADMIN},
    ),
    "user": User(
        user_id="user-001",
        username="user",
        email="user@example.com",
        role=Role.USER,
    ),
    "analyst": User(
        user_id="analyst-001",
        username="analyst",
        email="analyst@example.com",
        role=Role.ANALYST,
    ),
}


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(None),
) -> Optional[User]:
    """현재 사용자 조회 (JWT 또는 API 키)"""

    # 1. Bearer 토큰 확인
    if credentials:
        jwt_manager = get_jwt_manager()
        payload = jwt_manager.verify_token(credentials.credentials)
        if payload:
            username = payload.get("username")
            if username in DEMO_USERS:
                return DEMO_USERS[username]

    # 2. API 키 확인
    if x_api_key:
        api_key_manager = get_api_key_manager()
        api_key = api_key_manager.verify_key(x_api_key)
        if api_key:
            # API 키의 user_id로 사용자 조회
            for user in DEMO_USERS.values():
                if user.user_id == api_key.user_id:
                    return user
            # 사용자 없으면 임시 사용자 생성
            return User(
                user_id=api_key.user_id,
                username=api_key.name,
                email="",
                role=api_key.role,
                permissions=api_key.permissions,
            )

    return None


async def require_auth(
    user: Optional[User] = Depends(get_current_user),
) -> User:
    """인증 필수"""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_permission(permission: Permission):
    """특정 권한 필수 데코레이터"""
    async def check_permission(user: User = Depends(require_auth)) -> User:
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value}",
            )
        return user
    return check_permission


# ============================================================
# 인증 API
# ============================================================

@security_router.post(
    "/auth/login",
    response_model=LoginResponse,
    summary="로그인",
    description="사용자명과 비밀번호로 로그인합니다.",
)
async def login(request: Request, body: LoginRequest):
    """로그인"""
    if not JWT_AVAILABLE:
        raise HTTPException(status_code=501, detail="JWT not available")

    # 데모: 간단한 인증 (실제로는 DB에서 확인)
    if body.username not in DEMO_USERS:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 데모: 비밀번호는 username과 동일하게 설정
    if body.password != body.username:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = DEMO_USERS[body.username]
    user.last_login = datetime.now()

    jwt_manager = get_jwt_manager()
    access_token = jwt_manager.create_access_token(user)
    refresh_token = jwt_manager.create_refresh_token(user)

    # 감사 로그
    audit_logger = get_audit_logger()
    audit_logger.log(
        action=AuditAction.LOGIN,
        user_id=user.user_id,
        username=user.username,
        ip_address=request.client.host if request.client else None,
        details={"method": "password"},
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=jwt_manager.config.jwt_access_token_expire_minutes * 60,
    )


@security_router.post(
    "/auth/refresh",
    response_model=TokenRefreshResponse,
    summary="토큰 갱신",
    description="리프레시 토큰으로 새 액세스 토큰을 발급합니다.",
)
async def refresh_token(body: TokenRefreshRequest):
    """토큰 갱신"""
    if not JWT_AVAILABLE:
        raise HTTPException(status_code=501, detail="JWT not available")

    jwt_manager = get_jwt_manager()
    payload = jwt_manager.verify_token(body.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # 사용자 조회
    user_id = payload.get("sub")
    user = None
    for u in DEMO_USERS.values():
        if u.user_id == user_id:
            user = u
            break

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    new_access_token = jwt_manager.create_access_token(user)

    # 감사 로그
    audit_logger = get_audit_logger()
    audit_logger.log(
        action=AuditAction.TOKEN_REFRESH,
        user_id=user.user_id,
        username=user.username,
    )

    return TokenRefreshResponse(access_token=new_access_token)


@security_router.post(
    "/auth/logout",
    summary="로그아웃",
    description="토큰을 폐기합니다.",
)
async def logout(
    user: User = Depends(require_auth),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """로그아웃"""
    if not JWT_AVAILABLE:
        raise HTTPException(status_code=501, detail="JWT not available")

    jwt_manager = get_jwt_manager()
    jwt_manager.revoke_token(credentials.credentials)

    # 감사 로그
    audit_logger = get_audit_logger()
    audit_logger.log(
        action=AuditAction.LOGOUT,
        user_id=user.user_id,
        username=user.username,
    )

    return {"message": "Logged out successfully"}


@security_router.get(
    "/auth/me",
    response_model=UserInfoResponse,
    summary="현재 사용자 정보",
    description="현재 인증된 사용자 정보를 조회합니다.",
)
async def get_me(user: User = Depends(require_auth)):
    """현재 사용자 정보"""
    return UserInfoResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        role=user.role.value,
        permissions=[p.value for p in user.permissions],
    )


# ============================================================
# API 키 관리 API
# ============================================================

@security_router.post(
    "/api-keys",
    response_model=APIKeyCreateResponse,
    summary="API 키 생성",
    description="새 API 키를 생성합니다. 키는 한 번만 표시됩니다.",
)
async def create_api_key(
    body: APIKeyCreateRequest,
    user: User = Depends(require_permission(Permission.MANAGE_API_KEYS)),
):
    """API 키 생성"""
    api_key_manager = get_api_key_manager()

    try:
        role = Role(body.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.role}")

    raw_key, api_key = api_key_manager.generate_key(
        name=body.name,
        user_id=user.user_id,
        role=role,
        expires_in_days=body.expires_in_days,
    )

    # 감사 로그
    audit_logger = get_audit_logger()
    audit_logger.log(
        action=AuditAction.API_KEY_CREATE,
        user_id=user.user_id,
        username=user.username,
        resource="api_key",
        resource_id=api_key.key_id,
        details={"name": body.name, "role": body.role},
    )

    return APIKeyCreateResponse(
        key=raw_key,
        key_id=api_key.key_id,
        name=api_key.name,
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )


@security_router.get(
    "/api-keys",
    response_model=list[APIKeyResponse],
    summary="API 키 목록",
    description="사용자의 API 키 목록을 조회합니다.",
)
async def list_api_keys(
    user: User = Depends(require_auth),
):
    """API 키 목록"""
    api_key_manager = get_api_key_manager()

    # 관리자는 모든 키, 일반 사용자는 자신의 키만
    user_id = None if user.has_permission(Permission.MANAGE_API_KEYS) else user.user_id
    keys = api_key_manager.list_keys(user_id)

    return [
        APIKeyResponse(
            key_id=k.key_id,
            name=k.name,
            role=k.role.value,
            is_active=k.is_active,
            created_at=k.created_at.isoformat(),
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            last_used=k.last_used.isoformat() if k.last_used else None,
            usage_count=k.usage_count,
        )
        for k in keys
    ]


@security_router.delete(
    "/api-keys/{key_id}",
    summary="API 키 폐기",
    description="API 키를 폐기합니다.",
)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(require_permission(Permission.MANAGE_API_KEYS)),
):
    """API 키 폐기"""
    api_key_manager = get_api_key_manager()

    if not api_key_manager.revoke_key(key_id):
        raise HTTPException(status_code=404, detail="API key not found")

    # 감사 로그
    audit_logger = get_audit_logger()
    audit_logger.log(
        action=AuditAction.API_KEY_REVOKE,
        user_id=user.user_id,
        username=user.username,
        resource="api_key",
        resource_id=key_id,
    )

    return {"message": "API key revoked"}


# ============================================================
# 감사 로그 API
# ============================================================

@security_router.get(
    "/audit/logs",
    response_model=list[AuditLogResponse],
    summary="감사 로그 조회",
    description="감사 로그를 조회합니다.",
)
async def get_audit_logs(
    user_id: Optional[str] = Query(None, description="사용자 ID 필터"),
    action: Optional[str] = Query(None, description="액션 필터"),
    limit: int = Query(100, description="조회 수", ge=1, le=1000),
    user: User = Depends(require_permission(Permission.READ_STATS)),
):
    """감사 로그 조회"""
    audit_logger = get_audit_logger()

    audit_action = None
    if action:
        try:
            audit_action = AuditAction(action)
        except ValueError:
            pass

    logs = audit_logger.get_logs(
        user_id=user_id,
        action=audit_action,
        limit=limit,
    )

    return [
        AuditLogResponse(
            log_id=l.log_id,
            timestamp=l.timestamp.isoformat(),
            action=l.action.value,
            user_id=l.user_id,
            username=l.username,
            resource=l.resource,
            success=l.success,
        )
        for l in logs
    ]


@security_router.get(
    "/audit/stats",
    summary="감사 통계",
    description="감사 로그 통계를 조회합니다.",
)
async def get_audit_stats(
    user: User = Depends(require_permission(Permission.READ_STATS)),
):
    """감사 통계"""
    audit_logger = get_audit_logger()
    return audit_logger.get_stats()


# ============================================================
# 헬스체크
# ============================================================

@security_router.get(
    "/health",
    summary="보안 모듈 헬스체크",
)
async def security_health():
    """보안 모듈 헬스체크"""
    return {
        "status": "ok",
        "jwt_available": JWT_AVAILABLE,
        "features": {
            "jwt_auth": JWT_AVAILABLE,
            "api_key_auth": True,
            "rbac": True,
            "audit_logging": True,
        },
    }
