# -*- coding: utf-8 -*-
"""
실시간 기능 통합 테스트

DART 동기화, WebSocket 알림, SSE 스트리밍 API 통합 테스트
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import app


@pytest.fixture
def client():
    """FastAPI 테스트 클라이언트"""
    return TestClient(app)


class TestSyncAPI:
    """동기화 API 테스트"""

    def test_get_sync_status(self, client):
        """동기화 상태 조회 테스트"""
        response = client.get("/api/v1/sync/status")

        assert response.status_code == 200
        data = response.json()

        # 필수 필드 확인
        assert "running" in data
        assert "status" in data
        assert "config" in data

        # 상태 값 확인
        assert data["status"] in ["idle", "running", "success", "failed", "scheduled"]

    def test_trigger_sync(self, client):
        """수동 동기화 트리거 테스트"""
        response = client.post("/api/v1/sync/trigger")

        assert response.status_code == 200
        data = response.json()

        # 필수 필드 확인
        assert "status" in data
        assert "started_at" in data
        assert "new_disclosures" in data
        assert "updated_disclosures" in data

    def test_get_sync_history(self, client):
        """동기화 이력 조회 테스트"""
        response = client.get("/api/v1/sync/history")

        assert response.status_code == 200
        data = response.json()

        assert "history" in data
        assert "total_count" in data
        assert isinstance(data["history"], list)

    def test_get_sync_history_with_limit(self, client):
        """동기화 이력 조회 - limit 파라미터 테스트"""
        response = client.get("/api/v1/sync/history?limit=5")

        assert response.status_code == 200
        data = response.json()

        assert len(data["history"]) <= 5


class TestStreamingAPI:
    """SSE 스트리밍 API 테스트"""

    def test_stream_query(self, client):
        """스트리밍 쿼리 테스트"""
        response = client.get(
            "/api/v1/stream/query",
            params={"q": "ETF란 무엇인가요?", "top_k": 3}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # SSE 데이터 파싱
        content = response.text
        assert "event:" in content
        assert "data:" in content

    def test_stream_query_validation(self, client):
        """스트리밍 쿼리 유효성 검사 테스트"""
        # 너무 짧은 쿼리
        response = client.get(
            "/api/v1/stream/query",
            params={"q": "a"}
        )

        assert response.status_code == 422  # Validation Error

    def test_stream_health(self, client):
        """스트리밍 헬스체크 테스트"""
        response = client.get("/api/v1/stream/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["sse_enabled"] is True
        assert data["websocket_enabled"] is True


class TestWebSocketAPI:
    """WebSocket API 테스트"""

    def test_websocket_connect(self, client):
        """WebSocket 연결 테스트"""
        with client.websocket_connect("/api/v1/ws/notifications") as websocket:
            # 연결 확인 메시지 수신
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "system"
            assert "connection_id" in message["data"]
            assert message["data"]["message"] == "Connected"

    def test_websocket_subscribe_all(self, client):
        """WebSocket 전체 구독 테스트"""
        with client.websocket_connect("/api/v1/ws/notifications") as websocket:
            # 연결 메시지 수신
            websocket.receive_text()

            # 전체 구독 요청
            websocket.send_text(json.dumps({
                "action": "subscribe",
                "type": "all"
            }))

            # 구독 확인 메시지 수신
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "system"
            assert message["data"]["message"] == "Subscribed"
            assert message["data"]["subscription_type"] == "all"

    def test_websocket_subscribe_company(self, client):
        """WebSocket 회사별 구독 테스트"""
        with client.websocket_connect("/api/v1/ws/notifications") as websocket:
            # 연결 메시지 수신
            websocket.receive_text()

            # 특정 회사 구독 요청
            websocket.send_text(json.dumps({
                "action": "subscribe",
                "type": "company",
                "filter": "005930"  # 삼성전자
            }))

            # 구독 확인 메시지 수신
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "system"
            assert message["data"]["subscription_type"] == "company"
            assert message["data"]["filter_value"] == "005930"

    def test_websocket_ping(self, client):
        """WebSocket ping 테스트"""
        with client.websocket_connect("/api/v1/ws/notifications") as websocket:
            # 연결 메시지 수신
            websocket.receive_text()

            # ping 요청
            websocket.send_text(json.dumps({"action": "ping"}))

            # pong 응답 수신
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "heartbeat"
            assert message["data"]["pong"] is True

    def test_websocket_stats(self, client):
        """WebSocket 통계 테스트"""
        response = client.get("/api/v1/ws/stats")

        assert response.status_code == 200
        data = response.json()

        assert "active_connections" in data
        assert "total_connections" in data
        assert "total_messages_sent" in data


class TestE2EFlow:
    """E2E 플로우 테스트"""

    def test_sync_and_check_status(self, client):
        """동기화 후 상태 확인 플로우"""
        # 1. 초기 상태 확인
        status_response = client.get("/api/v1/sync/status")
        assert status_response.status_code == 200

        # 2. 동기화 트리거
        trigger_response = client.post("/api/v1/sync/trigger")
        assert trigger_response.status_code == 200

        # 3. 이력 확인
        history_response = client.get("/api/v1/sync/history")
        assert history_response.status_code == 200

        # 동기화 후 이력에 기록되었는지 확인
        history = history_response.json()["history"]
        if len(history) > 0:
            latest = history[0]
            assert "status" in latest
            assert "started_at" in latest

    def test_websocket_receive_notification_flow(self, client):
        """WebSocket 알림 수신 플로우"""
        with client.websocket_connect("/api/v1/ws/notifications") as websocket:
            # 1. 연결
            connect_msg = json.loads(websocket.receive_text())
            assert connect_msg["type"] == "system"

            # 2. 구독
            websocket.send_text(json.dumps({
                "action": "subscribe",
                "type": "all"
            }))
            subscribe_msg = json.loads(websocket.receive_text())
            assert subscribe_msg["data"]["message"] == "Subscribed"

            # 3. ping/pong 확인
            websocket.send_text(json.dumps({"action": "ping"}))
            pong_msg = json.loads(websocket.receive_text())
            assert pong_msg["data"]["pong"] is True


class TestRealtimeModules:
    """실시간 모듈 단위 테스트"""

    def test_sync_config(self):
        """동기화 설정 테스트"""
        from src.realtime import SyncConfig

        config = SyncConfig(
            interval_hours=2,
            daily_time="10:00",
            max_disclosures_per_sync=50
        )

        assert config.interval_hours == 2
        assert config.daily_time == "10:00"
        assert config.max_disclosures_per_sync == 50

    def test_sync_status_enum(self):
        """동기화 상태 Enum 테스트"""
        from src.realtime import SyncStatus

        assert SyncStatus.IDLE.value == "idle"
        assert SyncStatus.RUNNING.value == "running"
        assert SyncStatus.SUCCESS.value == "success"
        assert SyncStatus.FAILED.value == "failed"

    def test_notification_type(self):
        """알림 유형 테스트"""
        from src.realtime import NotificationType

        assert NotificationType.DISCLOSURE.value == "disclosure"
        assert NotificationType.SYNC_STATUS.value == "sync_status"
        assert NotificationType.HEARTBEAT.value == "heartbeat"

    def test_streaming_config(self):
        """스트리밍 설정 테스트"""
        from src.realtime import StreamingConfig

        config = StreamingConfig(
            chunk_size=20,
            chunk_delay_ms=100
        )

        assert config.chunk_size == 20
        assert config.chunk_delay_ms == 100

    @pytest.mark.asyncio
    async def test_websocket_manager_connect_disconnect(self):
        """WebSocket 매니저 연결/해제 테스트"""
        from src.realtime import WebSocketManager

        manager = WebSocketManager()

        # Mock WebSocket
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()

        # 연결
        connection = await manager.connect(mock_ws)
        assert manager.active_connections == 1
        assert connection.connection_id is not None

        # 해제
        await manager.disconnect(connection.connection_id)
        assert manager.active_connections == 0

    @pytest.mark.asyncio
    async def test_websocket_manager_subscribe(self):
        """WebSocket 매니저 구독 테스트"""
        from src.realtime import WebSocketManager

        manager = WebSocketManager()

        # Mock WebSocket
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()

        # 연결
        connection = await manager.connect(mock_ws)

        # 구독
        subscription = await manager.subscribe(
            connection.connection_id,
            "company",
            "005930"
        )

        assert subscription.subscription_type == "company"
        assert subscription.filter_value == "005930"

        # 구독 확인
        subs = manager.get_subscriptions(connection.connection_id)
        assert len(subs) == 1

        await manager.disconnect(connection.connection_id)


class TestRootEndpoint:
    """루트 엔드포인트 테스트"""

    def test_root_returns_realtime_info(self, client):
        """루트 엔드포인트에 실시간 정보 포함 확인"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["version"] == "2.5.0"
        assert "realtime" in data
        assert "websocket" in data["realtime"]
        assert "sync_status" in data["realtime"]
        assert "stream_query" in data["realtime"]
