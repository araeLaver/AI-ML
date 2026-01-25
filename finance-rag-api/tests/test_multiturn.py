# -*- coding: utf-8 -*-
"""
멀티턴 대화 테스트

[테스트 범위]
- ConversationManager: 세션 관리
- ContextResolver: 컨텍스트 해결
- MultiTurnRAGService: 통합 서비스
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.rag.conversation_manager import (
    Message,
    ConversationSession,
    ConversationManager,
    create_session,
    add_message,
    get_history,
)
from src.rag.context_resolver import (
    ContextResolver,
    ResolvedQuery,
    resolve_query,
    extract_entities,
)
from src.rag.multiturn_rag import (
    MultiTurnRAGService,
    MultiTurnResponse,
)


# =============================================================================
# Message 테스트
# =============================================================================

class TestMessage:
    """Message 데이터클래스 테스트"""

    def test_create_message(self):
        """메시지 생성"""
        msg = Message(role="user", content="안녕하세요")

        assert msg.role == "user"
        assert msg.content == "안녕하세요"
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """메타데이터 포함 메시지"""
        msg = Message(
            role="assistant",
            content="도움이 필요하신가요?",
            metadata={"source": "greeting"}
        )

        assert msg.metadata["source"] == "greeting"

    def test_message_to_dict(self):
        """딕셔너리 변환"""
        msg = Message(role="user", content="테스트")
        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "테스트"
        assert "timestamp" in result


# =============================================================================
# ConversationSession 테스트
# =============================================================================

class TestConversationSession:
    """ConversationSession 테스트"""

    def test_create_session(self):
        """세션 생성"""
        session = ConversationSession(session_id="test-123")

        assert session.session_id == "test-123"
        assert len(session.messages) == 0
        assert session.entities == {}

    def test_add_message(self):
        """메시지 추가"""
        session = ConversationSession(session_id="test")
        session.add_message("user", "삼성전자 실적")
        session.add_message("assistant", "삼성전자의 영업이익은...")

        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_get_recent_messages(self):
        """최근 메시지 조회"""
        session = ConversationSession(session_id="test")
        for i in range(10):
            session.add_message("user", f"메시지 {i}")

        recent = session.get_recent_messages(n=3)
        assert len(recent) == 3
        assert recent[-1].content == "메시지 9"

    def test_get_context_window(self):
        """컨텍스트 윈도우"""
        session = ConversationSession(session_id="test")
        # 긴 메시지 추가
        session.add_message("user", "A" * 1000)
        session.add_message("assistant", "B" * 1000)
        session.add_message("user", "C" * 100)

        # 토큰 제한으로 조회
        context = session.get_context_window(max_tokens=500)
        # 가장 최근 메시지부터 포함
        assert len(context) > 0


# =============================================================================
# ConversationManager 테스트
# =============================================================================

class TestConversationManager:
    """ConversationManager 테스트"""

    @pytest.fixture
    def manager(self):
        """테스트용 매니저"""
        return ConversationManager(
            max_sessions=10,
            session_ttl_minutes=5,
            max_messages_per_session=20
        )

    def test_create_session(self, manager):
        """세션 생성"""
        session_id = manager.create_session()

        assert session_id is not None
        assert len(session_id) > 0

        session = manager.get_session(session_id)
        assert session is not None

    def test_create_session_with_metadata(self, manager):
        """메타데이터 포함 세션 생성"""
        session_id = manager.create_session(metadata={"user": "test_user"})
        session = manager.get_session(session_id)

        assert session.metadata["user"] == "test_user"

    def test_add_message(self, manager):
        """메시지 추가"""
        session_id = manager.create_session()
        msg = manager.add_message(session_id, "user", "테스트 메시지")

        assert msg is not None
        assert msg.content == "테스트 메시지"

    def test_add_message_invalid_session(self, manager):
        """존재하지 않는 세션에 메시지 추가"""
        msg = manager.add_message("invalid-session", "user", "테스트")
        assert msg is None

    def test_get_history(self, manager):
        """히스토리 조회"""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "질문 1")
        manager.add_message(session_id, "assistant", "답변 1")
        manager.add_message(session_id, "user", "질문 2")

        history = manager.get_history(session_id)
        assert len(history) == 3

        recent = manager.get_history(session_id, n=2)
        assert len(recent) == 2

    def test_update_entities(self, manager):
        """엔티티 업데이트"""
        session_id = manager.create_session()
        manager.update_entities(session_id, {"company": "삼성전자"})

        entities = manager.get_entities(session_id)
        assert entities["company"] == "삼성전자"

    def test_set_topic(self, manager):
        """주제 설정"""
        session_id = manager.create_session()
        manager.set_topic(session_id, "삼성전자 분석")

        session = manager.get_session(session_id)
        assert session.current_topic == "삼성전자 분석"

    def test_delete_session(self, manager):
        """세션 삭제"""
        session_id = manager.create_session()
        assert manager.delete_session(session_id)
        assert manager.get_session(session_id) is None

    def test_max_sessions_limit(self, manager):
        """최대 세션 수 제한"""
        # 10개 세션 생성
        sessions = [manager.create_session() for _ in range(10)]

        # 11번째 세션 생성 시 가장 오래된 세션 제거
        new_session = manager.create_session()

        assert manager.get_session(sessions[0]) is None  # 첫 번째 세션 제거됨
        assert manager.get_session(new_session) is not None

    def test_max_messages_limit(self, manager):
        """최대 메시지 수 제한"""
        session_id = manager.create_session()

        # 25개 메시지 추가 (제한: 20)
        for i in range(25):
            manager.add_message(session_id, "user", f"메시지 {i}")

        history = manager.get_history(session_id)
        assert len(history) <= 20

    def test_get_stats(self, manager):
        """통계 조회"""
        manager.create_session()
        manager.create_session()

        stats = manager.get_stats()
        assert stats["active_sessions"] == 2
        assert stats["max_sessions"] == 10


# =============================================================================
# ContextResolver 테스트
# =============================================================================

class TestContextResolver:
    """ContextResolver 테스트"""

    @pytest.fixture
    def resolver(self):
        """테스트용 리졸버"""
        return ContextResolver()

    def test_extract_company(self, resolver):
        """회사명 추출"""
        entities = resolver.extract_entities("삼성전자 주가가 올랐어요")
        assert entities.get("company") == "삼성전자"

    def test_extract_company_abbreviation(self, resolver):
        """회사 약어 추출"""
        entities = resolver.extract_entities("삼전 실적이 좋대요")
        assert entities.get("company") == "삼성전자"

    def test_extract_metric(self, resolver):
        """지표 추출"""
        entities = resolver.extract_entities("PER이 낮은 종목 추천해줘")
        assert entities.get("metric") == "PER"

    def test_extract_multiple_entities(self, resolver):
        """복수 엔티티 추출"""
        entities = resolver.extract_entities("SK하이닉스의 ROE가 궁금해")
        assert entities.get("company") == "SK하이닉스"
        assert entities.get("metric") == "ROE"

    def test_detect_pronouns(self, resolver):
        """대명사 탐지"""
        pronouns = resolver.detect_pronouns("그 회사의 실적이 궁금해요")

        assert len(pronouns) > 0
        assert pronouns[0][0] == "그 회사"
        assert pronouns[0][1] == "company"

    def test_resolve_pronoun_company(self, resolver):
        """회사 대명사 해결"""
        resolved = resolver.resolve_query(
            "그 회사의 PER은?",
            entities={"company": "삼성전자"}
        )

        assert "삼성전자" in resolved.resolved_query
        assert len(resolved.references_resolved) > 0

    def test_resolve_pronoun_metric(self, resolver):
        """지표 대명사 해결"""
        resolved = resolver.resolve_query(
            "그 지표는 어떻게 계산해?",
            entities={"metric": "ROE"}
        )

        assert "ROE" in resolved.resolved_query

    def test_resolve_implicit_reference(self, resolver):
        """암시적 참조 해결"""
        resolved = resolver.resolve_query(
            "PER은?",
            entities={"company": "삼성전자"}
        )

        # 회사명이 추가되어야 함
        assert "삼성전자" in resolved.resolved_query

    def test_resolve_no_pronoun(self, resolver):
        """대명사 없는 쿼리"""
        resolved = resolver.resolve_query(
            "카카오 주가",
            entities={}
        )

        # 엔티티 추출됨
        assert resolved.extracted_entities.get("company") == "카카오"
        # 원본 쿼리와 동일 (대명사 없음)
        assert resolved.original_query == "카카오 주가"

    def test_resolve_with_history(self, resolver):
        """히스토리 기반 해결"""
        history = [
            Message(role="user", content="삼성전자 실적 알려줘"),
            Message(role="assistant", content="삼성전자의 영업이익은 6조원입니다."),
        ]

        resolved = resolver.resolve_with_history(
            "그 회사의 PER은?",
            history
        )

        assert "삼성전자" in resolved.resolved_query

    def test_get_search_queries(self, resolver):
        """검색 쿼리 생성"""
        resolved = ResolvedQuery(
            original_query="그 회사의 PER은?",
            resolved_query="삼성전자의 PER은?",
            extracted_entities={"company": "삼성전자"},
            references_resolved=["그 회사 → 삼성전자"],
            confidence=0.9
        )

        queries = resolver.get_search_queries(resolved)
        assert len(queries) == 2
        assert "삼성전자의 PER은?" in queries
        assert "그 회사의 PER은?" in queries

    def test_confidence_decreases_on_unresolved(self, resolver):
        """해결 실패 시 신뢰도 감소"""
        resolved = resolver.resolve_query(
            "그 회사의 실적은?",
            entities={}  # 엔티티 없음
        )

        # 대명사가 있지만 해결 못함
        assert resolved.confidence < 1.0


# =============================================================================
# MultiTurnRAGService 테스트
# =============================================================================

class TestMultiTurnRAGService:
    """MultiTurnRAGService 테스트"""

    @pytest.fixture
    def service(self):
        """테스트용 서비스 (RAG 없음)"""
        return MultiTurnRAGService(rag_service=None)

    def test_create_session(self, service):
        """세션 생성"""
        session_id = service.create_session()
        assert session_id is not None

        info = service.get_session_info(session_id)
        assert info is not None
        assert info["message_count"] == 0

    def test_query_basic(self, service):
        """기본 쿼리"""
        session_id = service.create_session()
        response = service.query(session_id, "삼성전자 실적")

        assert response.session_id == session_id
        assert response.query == "삼성전자 실적"
        assert response.answer is not None

    def test_query_context_resolution(self, service):
        """컨텍스트 해결 쿼리"""
        session_id = service.create_session()

        # 첫 번째 쿼리
        service.query(session_id, "삼성전자 실적 알려줘")

        # 두 번째 쿼리 (대명사 사용)
        response = service.query(session_id, "그 회사의 PER은?")

        # 삼성전자로 해결되어야 함
        assert "삼성전자" in response.resolved_query

    def test_query_entity_persistence(self, service):
        """엔티티 지속성"""
        session_id = service.create_session()

        service.query(session_id, "SK하이닉스 분석해줘")

        info = service.get_session_info(session_id)
        assert info["entities"].get("company") == "SK하이닉스"

    def test_query_topic_update(self, service):
        """주제 업데이트"""
        session_id = service.create_session()

        service.query(session_id, "현대차 전기차 전망")

        info = service.get_session_info(session_id)
        assert info["current_topic"] == "현대자동차"

    def test_get_history(self, service):
        """히스토리 조회"""
        session_id = service.create_session()

        service.query(session_id, "질문 1")
        service.query(session_id, "질문 2")

        history = service.get_history(session_id)
        assert len(history) == 4  # 질문 2개 + 답변 2개

    def test_clear_session(self, service):
        """세션 초기화"""
        session_id = service.create_session()
        service.query(session_id, "테스트")

        assert service.clear_session(session_id)
        assert service.get_session_info(session_id) is None

    def test_get_stats(self):
        """통계 조회"""
        # 새로운 매니저로 독립 테스트
        manager = ConversationManager()
        service = MultiTurnRAGService(conversation_manager=manager)

        service.create_session()
        service.create_session()

        stats = service.get_stats()
        assert stats["active_sessions"] == 2

    def test_query_auto_create_session(self, service):
        """세션 없으면 자동 생성"""
        response = service.query("invalid-session", "테스트")

        # 새 세션이 생성되어야 함
        assert response.session_id != "invalid-session"
        assert service.get_session_info(response.session_id) is not None


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_conversation_flow(self):
        """전체 대화 흐름"""
        service = MultiTurnRAGService()

        # 1. 세션 생성
        session_id = service.create_session()

        # 2. 첫 번째 질문
        r1 = service.query(session_id, "삼성전자 영업이익 알려줘")
        assert "삼성전자" in r1.resolved_query

        # 3. 후속 질문 (대명사)
        r2 = service.query(session_id, "그 회사의 PER은?")
        assert "삼성전자" in r2.resolved_query
        assert len(r2.context_info["references_resolved"]) > 0

        # 4. 또 다른 후속 질문
        r3 = service.query(session_id, "ROE는?")
        assert "삼성전자" in r3.resolved_query

        # 5. 히스토리 확인
        history = service.get_history(session_id)
        assert len(history) == 6  # 3 질문 + 3 답변

    def test_topic_change(self):
        """주제 변경"""
        service = MultiTurnRAGService()
        session_id = service.create_session()

        # 삼성전자 주제
        service.query(session_id, "삼성전자 실적")

        # 주제 변경
        r = service.query(session_id, "SK하이닉스 HBM 현황")

        # 새 주제로 업데이트
        info = service.get_session_info(session_id)
        assert info["current_topic"] == "SK하이닉스"

    def test_multiple_sessions(self):
        """다중 세션"""
        service = MultiTurnRAGService()

        # 세션 1: 삼성전자
        s1 = service.create_session()
        service.query(s1, "삼성전자 분석")

        # 세션 2: SK하이닉스
        s2 = service.create_session()
        service.query(s2, "SK하이닉스 분석")

        # 각 세션 독립적
        info1 = service.get_session_info(s1)
        info2 = service.get_session_info(s2)

        assert info1["entities"]["company"] == "삼성전자"
        assert info2["entities"]["company"] == "SK하이닉스"


# =============================================================================
# 편의 함수 테스트
# =============================================================================

class TestConvenienceFunctions:
    """편의 함수 테스트"""

    def test_create_session_function(self):
        """create_session 함수"""
        session_id = create_session()
        assert session_id is not None

    def test_add_message_function(self):
        """add_message 함수"""
        session_id = create_session()
        msg = add_message(session_id, "user", "테스트")
        assert msg is not None

    def test_get_history_function(self):
        """get_history 함수"""
        session_id = create_session()
        add_message(session_id, "user", "테스트")
        history = get_history(session_id)
        assert len(history) == 1

    def test_resolve_query_function(self):
        """resolve_query 함수"""
        resolved = resolve_query(
            "그 회사의 PER은?",
            entities={"company": "삼성전자"}
        )
        assert "삼성전자" in resolved.resolved_query

    def test_extract_entities_function(self):
        """extract_entities 함수"""
        entities = extract_entities("삼성전자 ROE")
        assert entities.get("company") == "삼성전자"
        assert entities.get("metric") == "ROE"
