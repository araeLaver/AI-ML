# -*- coding: utf-8 -*-
"""Tests for Real-time module (Phase 6)."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.realtime.dart_sync import (
    SyncConfig,
    SyncStatus,
    SyncResult,
    DARTSyncScheduler,
    DARTSyncState,
    DARTWebhook,
    get_sync_status,
    trigger_manual_sync,
)
from src.realtime.websocket_manager import (
    WebSocketManager,
    ConnectionInfo,
    Subscription,
    NotificationType,
    Notification,
    get_websocket_manager,
)
from src.realtime.streaming import (
    StreamingConfig,
    StreamingRAGService,
    StreamChunk,
    StreamEventType,
    stream_rag_response,
    LLMStreamingHelper,
)


# ============================================================
# DART Sync Tests
# ============================================================

class TestSyncConfig:
    """SyncConfig 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = SyncConfig()

        assert config.interval_hours == 1
        assert config.daily_time == "09:00"
        assert config.max_disclosures_per_sync == 100
        assert "A" in config.report_types

    def test_custom_config(self):
        """커스텀 설정 테스트."""
        config = SyncConfig(
            interval_hours=2,
            daily_time="10:00",
            max_disclosures_per_sync=50,
            lookback_days=7,
        )

        assert config.interval_hours == 2
        assert config.daily_time == "10:00"
        assert config.max_disclosures_per_sync == 50
        assert config.lookback_days == 7


class TestSyncResult:
    """SyncResult 테스트."""

    def test_create_result(self):
        """결과 생성 테스트."""
        result = SyncResult(
            status=SyncStatus.SUCCESS,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            new_disclosures=5,
            updated_disclosures=2,
        )

        assert result.status == SyncStatus.SUCCESS
        assert result.new_disclosures == 5
        assert result.updated_disclosures == 2

    def test_duration_calculation(self):
        """소요 시간 계산 테스트."""
        start = datetime.now()
        end = start + timedelta(seconds=10)

        result = SyncResult(
            status=SyncStatus.SUCCESS,
            started_at=start,
            completed_at=end,
        )

        assert result.duration_seconds == pytest.approx(10.0, rel=0.1)

    def test_to_dict(self):
        """딕셔너리 변환 테스트."""
        result = SyncResult(
            status=SyncStatus.SUCCESS,
            started_at=datetime.now(),
            new_disclosures=3,
        )

        data = result.to_dict()

        assert data["status"] == "success"
        assert data["new_disclosures"] == 3
        assert "started_at" in data


class TestDARTSyncState:
    """DARTSyncState 테스트."""

    @pytest.fixture
    def state_file(self, tmp_path):
        """임시 상태 파일."""
        return tmp_path / "sync_state.json"

    def test_init_empty(self, state_file):
        """빈 상태 초기화 테스트."""
        state = DARTSyncState(state_file)

        assert state.last_sync is None
        assert state.last_success is None

    def test_is_known(self, state_file):
        """알려진 공시 확인 테스트."""
        state = DARTSyncState(state_file)

        assert not state.is_known("12345", "abc123")

        state.mark_synced("12345", "abc123")
        assert state.is_known("12345", "abc123")
        assert not state.is_known("12345", "different_hash")

    def test_record_sync(self, state_file):
        """동기화 기록 테스트."""
        state = DARTSyncState(state_file)

        result = SyncResult(
            status=SyncStatus.SUCCESS,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            new_disclosures=5,
        )

        state.record_sync(result)

        assert state.last_sync is not None
        assert len(state.get_history()) == 1

    def test_persistence(self, state_file):
        """상태 영속성 테스트."""
        state1 = DARTSyncState(state_file)
        state1.mark_synced("12345", "abc123")
        state1._save_state()

        state2 = DARTSyncState(state_file)
        assert state2.is_known("12345", "abc123")


class TestDARTSyncScheduler:
    """DARTSyncScheduler 테스트."""

    @pytest.fixture
    def scheduler(self, tmp_path):
        """스케줄러 인스턴스."""
        config = SyncConfig(
            data_dir=str(tmp_path / "sync"),
            interval_hours=0,  # 자동 스케줄 비활성화
        )
        return DARTSyncScheduler(config)

    def test_init(self, scheduler):
        """초기화 테스트."""
        assert scheduler.status == SyncStatus.IDLE
        assert scheduler.last_sync is None

    def test_start_stop(self, scheduler):
        """시작/중지 테스트."""
        scheduler.start()
        assert scheduler._running is True

        scheduler.stop()
        assert scheduler._running is False

    def test_get_status(self, scheduler):
        """상태 조회 테스트."""
        status = scheduler.get_status()

        assert "running" in status
        assert "status" in status
        assert "config" in status

    def test_callbacks(self, scheduler):
        """콜백 테스트."""
        callback_called = []

        def on_complete(result):
            callback_called.append(result)

        scheduler.on_sync_complete(on_complete)

        # 동기화 실행 (collector 없어서 실패하지만 콜백은 호출됨)
        result = scheduler.sync_now()

        assert len(callback_called) == 1
        assert callback_called[0].status in [SyncStatus.SUCCESS, SyncStatus.FAILED]

    def test_get_history(self, scheduler):
        """히스토리 조회 테스트."""
        history = scheduler.get_history()
        assert isinstance(history, list)


class TestDARTWebhook:
    """DARTWebhook 테스트."""

    def test_init_empty(self):
        """빈 초기화 테스트."""
        webhook = DARTWebhook()

        assert not webhook.enabled
        assert len(webhook.urls) == 0

    def test_add_remove_url(self):
        """URL 추가/제거 테스트."""
        webhook = DARTWebhook()

        webhook.add_url("https://example.com/webhook")
        assert len(webhook.urls) == 1
        assert webhook.enabled

        webhook.remove_url("https://example.com/webhook")
        assert len(webhook.urls) == 0

    def test_enable_disable(self):
        """활성화/비활성화 테스트."""
        webhook = DARTWebhook(urls=["https://example.com"])

        assert webhook.enabled

        webhook.disable()
        assert not webhook.enabled

        webhook.enable()
        assert webhook.enabled

    def test_build_payload(self):
        """페이로드 생성 테스트."""
        webhook = DARTWebhook()

        # 딕셔너리 입력
        payload = webhook._build_payload({"rcept_no": "12345"})
        assert payload["event"] == "new_disclosure"
        assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_notify_async_disabled(self):
        """비활성화 시 알림 테스트."""
        webhook = DARTWebhook()
        webhook.disable()

        results = await webhook.notify_async({"test": True})
        assert results == {}


# ============================================================
# WebSocket Manager Tests
# ============================================================

class TestConnectionInfo:
    """ConnectionInfo 테스트."""

    def test_create_connection(self):
        """연결 정보 생성 테스트."""
        ws_mock = MagicMock()
        conn = ConnectionInfo(
            connection_id="test-123",
            websocket=ws_mock,
            user_id="user-456",
        )

        assert conn.connection_id == "test-123"
        assert conn.user_id == "user-456"
        assert len(conn.subscriptions) == 0


class TestSubscription:
    """Subscription 테스트."""

    def test_matches_all(self):
        """전체 구독 매칭 테스트."""
        sub = Subscription(
            subscription_id="sub-1",
            subscription_type="all",
        )

        assert sub.matches({"corp_code": "123", "report_nm": "사업보고서"})

    def test_matches_company(self):
        """회사 구독 매칭 테스트."""
        sub = Subscription(
            subscription_id="sub-1",
            subscription_type="company",
            filter_value="00126380",  # 삼성전자
        )

        assert sub.matches({"corp_code": "00126380"})
        assert not sub.matches({"corp_code": "00164779"})

    def test_matches_report_type(self):
        """공시 유형 구독 매칭 테스트."""
        sub = Subscription(
            subscription_id="sub-1",
            subscription_type="report_type",
            filter_value="사업보고서",
        )

        assert sub.matches({"report_nm": "[정정] 사업보고서"})
        assert not sub.matches({"report_nm": "주요사항보고서"})


class TestNotification:
    """Notification 테스트."""

    def test_create_notification(self):
        """알림 생성 테스트."""
        notification = Notification(
            notification_id="notif-1",
            type=NotificationType.DISCLOSURE,
            data={"corp_name": "삼성전자"},
        )

        assert notification.type == NotificationType.DISCLOSURE
        assert notification.data["corp_name"] == "삼성전자"

    def test_to_json(self):
        """JSON 변환 테스트."""
        notification = Notification(
            notification_id="notif-1",
            type=NotificationType.SYSTEM,
            data={"message": "Connected"},
        )

        json_str = notification.to_json()
        data = json.loads(json_str)

        assert data["type"] == "system"
        assert data["data"]["message"] == "Connected"


class TestWebSocketManager:
    """WebSocketManager 테스트."""

    @pytest.fixture
    def manager(self):
        """매니저 인스턴스."""
        return WebSocketManager()

    @pytest.fixture
    def mock_websocket(self):
        """모의 WebSocket."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        """연결 테스트."""
        conn = await manager.connect(mock_websocket, user_id="user-1")

        assert conn.connection_id is not None
        assert conn.user_id == "user-1"
        assert manager.active_connections == 1

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """연결 해제 테스트."""
        conn = await manager.connect(mock_websocket)
        await manager.disconnect(conn.connection_id)

        assert manager.active_connections == 0

    @pytest.mark.asyncio
    async def test_subscribe(self, manager, mock_websocket):
        """구독 테스트."""
        conn = await manager.connect(mock_websocket)

        sub = await manager.subscribe(
            conn.connection_id,
            "company",
            "00126380",
        )

        assert sub.subscription_type == "company"
        assert sub.filter_value == "00126380"

        subs = manager.get_subscriptions(conn.connection_id)
        assert len(subs) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, manager, mock_websocket):
        """구독 해제 테스트."""
        conn = await manager.connect(mock_websocket)
        sub = await manager.subscribe(conn.connection_id, "all")

        result = await manager.unsubscribe(conn.connection_id, sub.subscription_id)

        assert result is True
        assert len(manager.get_subscriptions(conn.connection_id)) == 0

    @pytest.mark.asyncio
    async def test_broadcast_disclosure(self, manager, mock_websocket):
        """공시 브로드캐스트 테스트."""
        conn = await manager.connect(mock_websocket)
        await manager.subscribe(conn.connection_id, "all")

        disclosure = {
            "rcept_no": "12345",
            "corp_code": "00126380",
            "corp_name": "삼성전자",
            "report_nm": "사업보고서",
        }

        sent_count = await manager.broadcast_disclosure(disclosure)

        assert sent_count == 1

    @pytest.mark.asyncio
    async def test_stats(self, manager, mock_websocket):
        """통계 테스트."""
        await manager.connect(mock_websocket)

        stats = manager.stats
        assert stats["active_connections"] == 1
        assert stats["total_connections"] == 1


# ============================================================
# Streaming Tests
# ============================================================

class TestStreamingConfig:
    """StreamingConfig 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = StreamingConfig()

        assert config.chunk_size == 10
        assert config.chunk_delay_ms == 50
        assert config.include_sources is True

    def test_custom_config(self):
        """커스텀 설정 테스트."""
        config = StreamingConfig(
            chunk_size=20,
            chunk_delay_ms=100,
            include_sources=False,
        )

        assert config.chunk_size == 20
        assert config.include_sources is False


class TestStreamChunk:
    """StreamChunk 테스트."""

    def test_create_chunk(self):
        """청크 생성 테스트."""
        chunk = StreamChunk(
            event_type=StreamEventType.CHUNK,
            data={"text": "Hello"},
        )

        assert chunk.event_type == StreamEventType.CHUNK
        assert chunk.data["text"] == "Hello"

    def test_to_sse(self):
        """SSE 변환 테스트."""
        chunk = StreamChunk(
            event_type=StreamEventType.START,
            data={"stream_id": "abc123"},
        )

        sse = chunk.to_sse()

        assert "event: start" in sse
        assert "data:" in sse
        assert "id:" in sse


class TestStreamingRAGService:
    """StreamingRAGService 테스트."""

    @pytest.fixture
    def streaming_service(self):
        """스트리밍 서비스."""
        config = StreamingConfig(
            chunk_size=5,
            chunk_delay_ms=0,  # 테스트용으로 지연 없음
        )
        return StreamingRAGService(config=config)

    @pytest.mark.asyncio
    async def test_stream_query(self, streaming_service):
        """쿼리 스트리밍 테스트."""
        chunks = []

        async for chunk in streaming_service.stream_query("테스트 쿼리"):
            chunks.append(chunk)

        # 시작, 텍스트 청크들, 메타데이터, 종료
        assert len(chunks) >= 3

        # 시작 이벤트 확인
        assert chunks[0].event_type == StreamEventType.START

        # 종료 이벤트 확인
        assert chunks[-1].event_type == StreamEventType.END

    @pytest.mark.asyncio
    async def test_stream_with_rag_service(self):
        """RAG 서비스와 스트리밍 테스트."""
        # RAGResponse 객체처럼 동작하는 mock
        class MockRAGResponse:
            def __init__(self):
                self.answer = "이것은 테스트 응답입니다."
                self.sources = [{"title": "Test Doc"}]
                self.metadata = {"score": 0.95}

        # query_async를 사용하여 비동기 호출 처리
        mock_rag = MagicMock()
        mock_rag.query_async = AsyncMock(return_value=MockRAGResponse())

        service = StreamingRAGService(
            rag_service=mock_rag,
            config=StreamingConfig(chunk_size=5, chunk_delay_ms=0),
        )

        chunks = []
        async for chunk in service.stream_query("테스트"):
            chunks.append(chunk)

        # 시작/종료 이벤트 확인
        assert chunks[0].event_type == StreamEventType.START
        assert chunks[-1].event_type == StreamEventType.END

        # 텍스트 청크 확인
        text_chunks = [c for c in chunks if c.event_type == StreamEventType.CHUNK]
        assert len(text_chunks) > 0

        # 소스 청크 확인
        source_chunks = [c for c in chunks if c.event_type == StreamEventType.SOURCE]
        assert len(source_chunks) == 1

    @pytest.mark.asyncio
    async def test_stats(self, streaming_service):
        """통계 테스트."""
        async for _ in streaming_service.stream_query("테스트"):
            pass

        stats = streaming_service.stats
        assert stats["total_streams"] == 1
        assert stats["total_chunks_sent"] > 0


class TestStreamRAGResponse:
    """stream_rag_response 함수 테스트."""

    @pytest.mark.asyncio
    async def test_stream_rag_response(self):
        """SSE 스트림 응답 테스트."""
        chunks = []

        async for sse_chunk in stream_rag_response(
            "테스트 쿼리",
            config=StreamingConfig(chunk_delay_ms=0),
        ):
            chunks.append(sse_chunk)

        assert len(chunks) > 0
        assert all("event:" in c for c in chunks)
        assert all("data:" in c for c in chunks)


class TestLLMStreamingHelper:
    """LLMStreamingHelper 테스트."""

    @pytest.mark.asyncio
    async def test_stream_ollama_error_handling(self):
        """Ollama 스트리밍 오류 처리 테스트."""
        chunks = []

        # 존재하지 않는 URL로 테스트 (오류 발생)
        async for chunk in LLMStreamingHelper.stream_ollama(
            base_url="http://localhost:99999",
            prompt="test",
        ):
            chunks.append(chunk)

        # 오류 메시지 반환
        assert len(chunks) == 1
        assert "[Error:" in chunks[0]


# ============================================================
# Integration Tests
# ============================================================

class TestRealtimeIntegration:
    """실시간 모듈 통합 테스트."""

    @pytest.mark.asyncio
    async def test_sync_to_websocket_flow(self, tmp_path):
        """동기화 → WebSocket 알림 플로우 테스트."""
        # 스케줄러 설정
        config = SyncConfig(
            data_dir=str(tmp_path / "sync"),
            interval_hours=0,
        )
        scheduler = DARTSyncScheduler(config)

        # WebSocket 매니저
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()

        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.connection_id, "all")

        # 동기화 완료 시 WebSocket 알림
        async def on_sync_complete(result):
            await manager.broadcast_sync_status(result.to_dict())

        # 콜백은 동기 함수여야 하므로 래핑
        def sync_callback(result):
            asyncio.create_task(on_sync_complete(result))

        scheduler.on_sync_complete(sync_callback)

        # 동기화 실행
        result = scheduler.sync_now()

        # 잠시 대기 (비동기 콜백 처리)
        await asyncio.sleep(0.1)

        assert result.status in [SyncStatus.SUCCESS, SyncStatus.FAILED]

    @pytest.mark.asyncio
    async def test_websocket_to_streaming_integration(self):
        """WebSocket과 스트리밍 통합 테스트."""
        manager = WebSocketManager()
        streaming = StreamingRAGService(
            config=StreamingConfig(chunk_delay_ms=0),
        )

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()

        conn = await manager.connect(mock_ws)

        # 스트리밍 응답 수집
        all_text = ""
        async for chunk in streaming.stream_query("테스트"):
            if chunk.event_type == StreamEventType.CHUNK:
                all_text += chunk.data.get("text", "")

        assert len(all_text) > 0


# ============================================================
# Module Import Tests
# ============================================================

class TestModuleImports:
    """모듈 임포트 테스트."""

    def test_import_dart_sync(self):
        """dart_sync 임포트 테스트."""
        from src.realtime import (
            SyncConfig,
            SyncStatus,
            SyncResult,
            DARTSyncScheduler,
            DARTWebhook,
        )

        assert SyncConfig is not None
        assert DARTSyncScheduler is not None

    def test_import_websocket(self):
        """websocket_manager 임포트 테스트."""
        from src.realtime import (
            WebSocketManager,
            ConnectionInfo,
            Subscription,
            NotificationType,
            Notification,
        )

        assert WebSocketManager is not None
        assert NotificationType is not None

    def test_import_streaming(self):
        """streaming 임포트 테스트."""
        from src.realtime import (
            StreamingConfig,
            StreamingRAGService,
            StreamChunk,
            stream_rag_response,
        )

        assert StreamingRAGService is not None
        assert stream_rag_response is not None
