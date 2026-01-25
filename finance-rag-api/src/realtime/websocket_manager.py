# -*- coding: utf-8 -*-
"""
WebSocket 실시간 알림 모듈

공시 발생 시 실시간으로 클라이언트에게 알림을 전송합니다.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Set

try:
    from fastapi import WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = Any


class NotificationType(Enum):
    """알림 유형"""
    DISCLOSURE = "disclosure"  # 새 공시
    SYNC_STATUS = "sync_status"  # 동기화 상태
    SYSTEM = "system"  # 시스템 메시지
    HEARTBEAT = "heartbeat"  # 연결 유지


@dataclass
class ConnectionInfo:
    """WebSocket 연결 정보

    Attributes:
        connection_id: 연결 고유 ID
        websocket: WebSocket 객체
        connected_at: 연결 시간
        subscriptions: 구독 목록
        user_id: 사용자 ID (옵션)
    """
    connection_id: str
    websocket: Any  # WebSocket
    connected_at: datetime = field(default_factory=datetime.now)
    subscriptions: Set[str] = field(default_factory=set)
    user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.connection_id)


@dataclass
class Subscription:
    """구독 정보

    Attributes:
        subscription_id: 구독 ID
        subscription_type: 구독 유형 (company, report_type, all)
        filter_value: 필터 값 (회사코드, 공시유형 등)
    """
    subscription_id: str
    subscription_type: str  # "company", "report_type", "all"
    filter_value: Optional[str] = None

    def matches(self, disclosure: dict) -> bool:
        """공시가 구독 조건에 맞는지 확인"""
        if self.subscription_type == "all":
            return True
        elif self.subscription_type == "company":
            return disclosure.get("corp_code") == self.filter_value
        elif self.subscription_type == "report_type":
            report_nm = disclosure.get("report_nm", "")
            return self.filter_value in report_nm if self.filter_value else True
        return False


@dataclass
class Notification:
    """알림 메시지

    Attributes:
        notification_id: 알림 ID
        type: 알림 유형
        data: 알림 데이터
        created_at: 생성 시간
    """
    notification_id: str
    type: NotificationType
    data: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps({
            "id": self.notification_id,
            "type": self.type.value,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
        }, ensure_ascii=False)


class WebSocketManager:
    """WebSocket 연결 관리자

    다수의 클라이언트 연결을 관리하고 실시간 알림을 전송합니다.
    """

    def __init__(self):
        """초기화"""
        self._connections: dict[str, ConnectionInfo] = {}
        self._subscriptions: dict[str, list[Subscription]] = {}  # connection_id -> subscriptions
        self._lock = asyncio.Lock()

        # 통계
        self._total_connections = 0
        self._total_messages_sent = 0
        self._started_at = datetime.now()

    @property
    def active_connections(self) -> int:
        """활성 연결 수"""
        return len(self._connections)

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return {
            "active_connections": self.active_connections,
            "total_connections": self._total_connections,
            "total_messages_sent": self._total_messages_sent,
            "uptime_seconds": (datetime.now() - self._started_at).total_seconds(),
        }

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
    ) -> ConnectionInfo:
        """새 WebSocket 연결 등록

        Args:
            websocket: WebSocket 객체
            user_id: 사용자 ID (옵션)

        Returns:
            연결 정보
        """
        connection_id = str(uuid.uuid4())

        async with self._lock:
            connection = ConnectionInfo(
                connection_id=connection_id,
                websocket=websocket,
                user_id=user_id,
            )
            self._connections[connection_id] = connection
            self._subscriptions[connection_id] = []
            self._total_connections += 1

        # 연결 확인 메시지 전송
        await self._send_to_connection(
            connection,
            Notification(
                notification_id=str(uuid.uuid4()),
                type=NotificationType.SYSTEM,
                data={
                    "message": "Connected",
                    "connection_id": connection_id,
                },
            ),
        )

        return connection

    async def disconnect(self, connection_id: str) -> None:
        """WebSocket 연결 해제

        Args:
            connection_id: 연결 ID
        """
        async with self._lock:
            if connection_id in self._connections:
                del self._connections[connection_id]
            if connection_id in self._subscriptions:
                del self._subscriptions[connection_id]

    async def subscribe(
        self,
        connection_id: str,
        subscription_type: str,
        filter_value: Optional[str] = None,
    ) -> Subscription:
        """구독 추가

        Args:
            connection_id: 연결 ID
            subscription_type: 구독 유형 (company, report_type, all)
            filter_value: 필터 값

        Returns:
            구독 정보
        """
        subscription = Subscription(
            subscription_id=str(uuid.uuid4()),
            subscription_type=subscription_type,
            filter_value=filter_value,
        )

        async with self._lock:
            if connection_id in self._subscriptions:
                self._subscriptions[connection_id].append(subscription)

        # 구독 확인 메시지
        if connection_id in self._connections:
            await self._send_to_connection(
                self._connections[connection_id],
                Notification(
                    notification_id=str(uuid.uuid4()),
                    type=NotificationType.SYSTEM,
                    data={
                        "message": "Subscribed",
                        "subscription_id": subscription.subscription_id,
                        "subscription_type": subscription_type,
                        "filter_value": filter_value,
                    },
                ),
            )

        return subscription

    async def unsubscribe(
        self,
        connection_id: str,
        subscription_id: str,
    ) -> bool:
        """구독 해제

        Args:
            connection_id: 연결 ID
            subscription_id: 구독 ID

        Returns:
            성공 여부
        """
        async with self._lock:
            if connection_id not in self._subscriptions:
                return False

            original_count = len(self._subscriptions[connection_id])
            self._subscriptions[connection_id] = [
                s for s in self._subscriptions[connection_id]
                if s.subscription_id != subscription_id
            ]
            return len(self._subscriptions[connection_id]) < original_count

    async def broadcast_disclosure(self, disclosure: dict[str, Any]) -> int:
        """공시 알림 브로드캐스트

        구독 조건에 맞는 연결에만 알림을 전송합니다.

        Args:
            disclosure: 공시 정보

        Returns:
            전송된 연결 수
        """
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            type=NotificationType.DISCLOSURE,
            data=disclosure,
        )

        sent_count = 0
        dead_connections = []

        for connection_id, connection in list(self._connections.items()):
            subscriptions = self._subscriptions.get(connection_id, [])

            # 구독 조건 확인
            should_send = any(sub.matches(disclosure) for sub in subscriptions)

            # 구독 없으면 전송하지 않음 (all 구독 제외)
            if not should_send and not any(s.subscription_type == "all" for s in subscriptions):
                continue

            try:
                await self._send_to_connection(connection, notification)
                sent_count += 1
            except Exception:
                dead_connections.append(connection_id)

        # 죽은 연결 정리
        for conn_id in dead_connections:
            await self.disconnect(conn_id)

        return sent_count

    async def broadcast_sync_status(self, status: dict[str, Any]) -> int:
        """동기화 상태 브로드캐스트

        Args:
            status: 동기화 상태 정보

        Returns:
            전송된 연결 수
        """
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            type=NotificationType.SYNC_STATUS,
            data=status,
        )

        return await self._broadcast_all(notification)

    async def send_heartbeat(self) -> int:
        """하트비트 전송

        Returns:
            전송된 연결 수
        """
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            type=NotificationType.HEARTBEAT,
            data={"timestamp": datetime.now().isoformat()},
        )

        return await self._broadcast_all(notification)

    async def _broadcast_all(self, notification: Notification) -> int:
        """모든 연결에 브로드캐스트"""
        sent_count = 0
        dead_connections = []

        for connection_id, connection in list(self._connections.items()):
            try:
                await self._send_to_connection(connection, notification)
                sent_count += 1
            except Exception:
                dead_connections.append(connection_id)

        for conn_id in dead_connections:
            await self.disconnect(conn_id)

        return sent_count

    async def _send_to_connection(
        self,
        connection: ConnectionInfo,
        notification: Notification,
    ) -> None:
        """특정 연결에 알림 전송"""
        try:
            if hasattr(connection.websocket, 'send_text'):
                await connection.websocket.send_text(notification.to_json())
            else:
                # 테스트 모의 객체
                pass
            self._total_messages_sent += 1
        except Exception:
            raise

    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """연결 정보 조회"""
        return self._connections.get(connection_id)

    def get_subscriptions(self, connection_id: str) -> list[Subscription]:
        """구독 목록 조회"""
        return self._subscriptions.get(connection_id, [])


# 전역 WebSocket 매니저
_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """전역 WebSocket 매니저 반환"""
    global _manager
    if _manager is None:
        _manager = WebSocketManager()
    return _manager


async def broadcast_disclosure(disclosure: dict[str, Any]) -> int:
    """공시 알림 브로드캐스트 (편의 함수)"""
    manager = get_websocket_manager()
    return await manager.broadcast_disclosure(disclosure)


async def subscribe_to_company(
    connection_id: str,
    corp_code: str,
) -> Subscription:
    """회사별 구독 (편의 함수)"""
    manager = get_websocket_manager()
    return await manager.subscribe(connection_id, "company", corp_code)


async def subscribe_to_type(
    connection_id: str,
    report_type: str,
) -> Subscription:
    """공시 유형별 구독 (편의 함수)"""
    manager = get_websocket_manager()
    return await manager.subscribe(connection_id, "report_type", report_type)


# FastAPI 라우터 헬퍼
if FASTAPI_AVAILABLE:
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket 엔드포인트 핸들러

        Usage:
            @app.websocket("/ws")
            async def ws_endpoint(websocket: WebSocket):
                await websocket_endpoint(websocket)
        """
        await websocket.accept()
        manager = get_websocket_manager()
        connection = await manager.connect(websocket)

        try:
            while True:
                # 클라이언트 메시지 수신
                data = await websocket.receive_text()
                message = json.loads(data)

                # 명령 처리
                action = message.get("action")

                if action == "subscribe":
                    await manager.subscribe(
                        connection.connection_id,
                        message.get("type", "all"),
                        message.get("filter"),
                    )
                elif action == "unsubscribe":
                    await manager.unsubscribe(
                        connection.connection_id,
                        message.get("subscription_id"),
                    )
                elif action == "ping":
                    await manager._send_to_connection(
                        connection,
                        Notification(
                            notification_id=str(uuid.uuid4()),
                            type=NotificationType.HEARTBEAT,
                            data={"pong": True},
                        ),
                    )

        except Exception:
            pass
        finally:
            await manager.disconnect(connection.connection_id)
