# -*- coding: utf-8 -*-
"""Real-time 기능 모듈

Phase 6: 실시간 기능
- DART API 실시간 연동
- WebSocket 실시간 알림
- SSE 스트리밍 응답
"""

from .dart_sync import (
    SyncConfig,
    SyncStatus,
    SyncResult,
    DARTSyncScheduler,
    DARTWebhook,
    start_sync_scheduler,
    stop_sync_scheduler,
    get_sync_status,
    trigger_manual_sync,
)
from .websocket_manager import (
    WebSocketManager,
    ConnectionInfo,
    Subscription,
    NotificationType,
    Notification,
    broadcast_disclosure,
    subscribe_to_company,
    subscribe_to_type,
)
from .streaming import (
    StreamingConfig,
    StreamingRAGService,
    StreamChunk,
    stream_rag_response,
    create_sse_response,
)

__all__ = [
    # DART Sync
    "SyncConfig",
    "SyncStatus",
    "SyncResult",
    "DARTSyncScheduler",
    "DARTWebhook",
    "start_sync_scheduler",
    "stop_sync_scheduler",
    "get_sync_status",
    "trigger_manual_sync",
    # WebSocket
    "WebSocketManager",
    "ConnectionInfo",
    "Subscription",
    "NotificationType",
    "Notification",
    "broadcast_disclosure",
    "subscribe_to_company",
    "subscribe_to_type",
    # Streaming
    "StreamingConfig",
    "StreamingRAGService",
    "StreamChunk",
    "stream_rag_response",
    "create_sse_response",
]
