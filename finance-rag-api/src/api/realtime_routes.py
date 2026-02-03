# -*- coding: utf-8 -*-
"""
실시간 API 라우터

DART 동기화, WebSocket 알림, SSE 스트리밍 엔드포인트를 제공합니다.
"""

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, BackgroundTasks
from fastapi.responses import StreamingResponse

from .schemas import (
    SyncStatusResponse,
    SyncResultResponse,
    SyncHistoryResponse,
)
from ..core.config import get_settings
from ..core.logging import get_logger
from ..realtime import (
    get_sync_status,
    trigger_manual_sync,
    get_websocket_manager,
    NotificationType,
    Notification,
    StreamingRAGService,
    StreamingConfig,
)

logger = get_logger(__name__)

# 라우터 생성
realtime_router = APIRouter(tags=["Realtime"])


# ============================================================
# 동기화 API
# ============================================================

@realtime_router.get(
    "/sync/status",
    response_model=SyncStatusResponse,
    summary="동기화 상태 조회",
    description="DART 동기화 스케줄러의 현재 상태를 조회합니다."
)
async def get_sync_status_endpoint():
    """동기화 상태 조회 API"""
    status = get_sync_status()
    return SyncStatusResponse(**status)


@realtime_router.post(
    "/sync/trigger",
    response_model=SyncResultResponse,
    summary="수동 동기화 실행",
    description="DART 공시 데이터를 수동으로 동기화합니다."
)
async def trigger_sync_endpoint(background_tasks: BackgroundTasks):
    """수동 동기화 실행 API"""
    logger.info("Manual sync triggered")
    result = trigger_manual_sync()

    # WebSocket으로 동기화 상태 브로드캐스트
    manager = get_websocket_manager()
    await manager.broadcast_sync_status(result.to_dict())

    return SyncResultResponse(**result.to_dict())


@realtime_router.get(
    "/sync/history",
    response_model=SyncHistoryResponse,
    summary="동기화 이력 조회",
    description="최근 동기화 이력을 조회합니다."
)
async def get_sync_history_endpoint(
    limit: int = Query(default=10, ge=1, le=100, description="조회할 이력 수")
):
    """동기화 이력 조회 API"""
    from ..realtime.dart_sync import get_scheduler

    scheduler = get_scheduler()
    history = scheduler.get_history(limit)

    return SyncHistoryResponse(
        history=history,
        total_count=len(history)
    )


# ============================================================
# WebSocket API
# ============================================================

@realtime_router.websocket("/ws/notifications")
async def websocket_notifications_endpoint(websocket: WebSocket):
    """
    WebSocket 실시간 알림 엔드포인트

    클라이언트는 다음 메시지를 보낼 수 있습니다:
    - {"action": "subscribe", "type": "all"} - 모든 공시 구독
    - {"action": "subscribe", "type": "company", "filter": "005930"} - 특정 회사 구독
    - {"action": "subscribe", "type": "report_type", "filter": "정기보고서"} - 공시 유형 구독
    - {"action": "unsubscribe", "subscription_id": "..."} - 구독 해제
    - {"action": "ping"} - 연결 확인
    """
    await websocket.accept()

    manager = get_websocket_manager()
    connection = await manager.connect(websocket)

    settings = get_settings()
    heartbeat_interval = settings.ws_heartbeat_interval

    logger.info(f"WebSocket connected: {connection.connection_id}")

    try:
        # 하트비트 태스크 시작
        heartbeat_task = asyncio.create_task(
            _send_heartbeat(manager, connection.connection_id, heartbeat_interval)
        )

        while True:
            try:
                # 클라이언트 메시지 수신 (타임아웃 설정)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=heartbeat_interval * 2
                )

                message = json.loads(data)
                action = message.get("action")

                if action == "subscribe":
                    sub_type = message.get("type", "all")
                    filter_value = message.get("filter")
                    await manager.subscribe(
                        connection.connection_id,
                        sub_type,
                        filter_value
                    )
                    logger.info(f"Subscription added: {sub_type}/{filter_value}")

                elif action == "unsubscribe":
                    subscription_id = message.get("subscription_id")
                    if subscription_id:
                        await manager.unsubscribe(
                            connection.connection_id,
                            subscription_id
                        )
                        logger.info(f"Subscription removed: {subscription_id}")

                elif action == "ping":
                    await manager._send_to_connection(
                        connection,
                        Notification(
                            notification_id=str(uuid.uuid4()),
                            type=NotificationType.HEARTBEAT,
                            data={"pong": True}
                        )
                    )

            except asyncio.TimeoutError:
                # 타임아웃은 정상 - 하트비트가 연결 유지
                continue
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {connection.connection_id}")
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection.connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        await manager.disconnect(connection.connection_id)


async def _send_heartbeat(manager, connection_id: str, interval: int):
    """주기적 하트비트 전송"""
    while True:
        await asyncio.sleep(interval)
        connection = manager.get_connection(connection_id)
        if connection:
            try:
                await manager._send_to_connection(
                    connection,
                    Notification(
                        notification_id=str(uuid.uuid4()),
                        type=NotificationType.HEARTBEAT,
                        data={"heartbeat": True}
                    )
                )
            except Exception:
                break
        else:
            break


# ============================================================
# SSE 스트리밍 API
# ============================================================

@realtime_router.get(
    "/stream/query",
    summary="RAG 스트리밍 응답",
    description="RAG 쿼리 결과를 SSE 스트림으로 반환합니다."
)
async def stream_query_endpoint(
    q: str = Query(..., min_length=2, max_length=500, description="검색 쿼리"),
    top_k: int = Query(default=5, ge=1, le=10, description="검색 결과 수")
):
    """
    RAG 쿼리 스트리밍 엔드포인트

    응답 형식 (Server-Sent Events):
    - event: start - 스트림 시작
    - event: chunk - 텍스트 청크
    - event: source - 소스 문서
    - event: metadata - 메타데이터
    - event: end - 스트림 종료
    """
    logger.info(f"Streaming query: {q[:50]}...")

    # RAG 서비스 가져오기 (lazy loading)
    try:
        from .routes import get_rag_service
        rag_service = get_rag_service()
    except Exception:
        rag_service = None

    streaming_service = StreamingRAGService(
        rag_service=rag_service,
        config=StreamingConfig(
            chunk_size=20,
            chunk_delay_ms=30,
            include_sources=True,
            include_metadata=True
        )
    )

    async def generate():
        async for chunk in streaming_service.stream_query(q, top_k=top_k):
            yield chunk.to_sse()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@realtime_router.get(
    "/stream/health",
    summary="스트리밍 서비스 상태",
    description="스트리밍 서비스의 상태를 확인합니다."
)
async def stream_health_endpoint():
    """스트리밍 서비스 헬스체크"""
    return {
        "status": "ok",
        "service": "streaming",
        "sse_enabled": True,
        "websocket_enabled": True,
    }


# ============================================================
# WebSocket 통계 API
# ============================================================

@realtime_router.get(
    "/ws/stats",
    summary="WebSocket 통계",
    description="WebSocket 연결 통계를 조회합니다."
)
async def websocket_stats_endpoint():
    """WebSocket 연결 통계"""
    manager = get_websocket_manager()
    return manager.stats
