# -*- coding: utf-8 -*-
"""
멀티턴 RAG 서비스

[기능]
- 대화 세션 기반 RAG
- 컨텍스트 해결 및 쿼리 재작성
- 대화 히스토리 기반 검색
- 응답 생성 시 히스토리 활용

[사용 예시]
>>> service = MultiTurnRAGService(rag_service)
>>> session_id = service.create_session()
>>> response = service.query(session_id, "삼성전자 실적 알려줘")
>>> response = service.query(session_id, "그 회사의 PER은?")  # 컨텍스트 유지
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .conversation_manager import (
    ConversationManager,
    ConversationSession,
    Message,
    get_conversation_manager,
)
from .context_resolver import (
    ContextResolver,
    ResolvedQuery,
    get_context_resolver,
)
from .rag_service import RAGService, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class MultiTurnResponse:
    """멀티턴 RAG 응답"""
    session_id: str
    query: str
    resolved_query: str
    answer: str
    sources: List[Dict[str, Any]]
    context_info: Dict[str, Any]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "resolved_query": self.resolved_query,
            "answer": self.answer,
            "sources": self.sources,
            "context_info": self.context_info,
            "confidence": self.confidence,
        }


class MultiTurnRAGService:
    """
    멀티턴 대화 지원 RAG 서비스

    [처리 흐름]
    1. 세션에서 대화 히스토리 로드
    2. 컨텍스트 해결 (대명사, 참조)
    3. 해결된 쿼리로 RAG 검색
    4. 응답 생성 (히스토리 컨텍스트 포함)
    5. 대화 저장
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        conversation_manager: Optional[ConversationManager] = None,
        context_resolver: Optional[ContextResolver] = None,
        max_context_messages: int = 5,
        context_window_tokens: int = 2000,
    ):
        self.rag_service = rag_service
        self.conversation_manager = conversation_manager or get_conversation_manager()
        self.context_resolver = context_resolver or get_context_resolver()
        self.max_context_messages = max_context_messages
        self.context_window_tokens = context_window_tokens

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """새 대화 세션 생성"""
        session_id = self.conversation_manager.create_session(metadata)
        logger.info(f"Created multi-turn session: {session_id}")
        return session_id

    def query(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        use_context: bool = True,
    ) -> MultiTurnResponse:
        """
        멀티턴 쿼리 처리

        Args:
            session_id: 세션 ID
            query: 사용자 쿼리
            top_k: 검색 결과 수
            use_context: 컨텍스트 해결 사용 여부

        Returns:
            MultiTurnResponse: 응답
        """
        # 1. 세션 확인
        session = self.conversation_manager.get_session(session_id)
        if session is None:
            # 세션이 없으면 새로 생성
            session_id = self.create_session()
            session = self.conversation_manager.get_session(session_id)

        # 2. 컨텍스트 해결
        resolved: Optional[ResolvedQuery] = None
        resolved_query = query

        if use_context:
            history = self.conversation_manager.get_context(
                session_id,
                max_tokens=self.context_window_tokens
            )
            entities = self.conversation_manager.get_entities(session_id)

            resolved = self.context_resolver.resolve_query(
                query,
                entities=entities,
                history=history,
                topic=session.current_topic if session else None
            )
            resolved_query = resolved.resolved_query

            # 세션 엔티티 업데이트
            if resolved.extracted_entities:
                self.conversation_manager.update_entities(
                    session_id,
                    resolved.extracted_entities
                )

            # 주제 업데이트
            if resolved.extracted_entities.get("company"):
                self.conversation_manager.set_topic(
                    session_id,
                    resolved.extracted_entities["company"]
                )

        # 3. RAG 검색 및 응답 생성
        answer = ""
        sources = []

        if self.rag_service:
            # 히스토리 컨텍스트 생성
            history_context = self._build_history_context(session_id)

            # RAG 쿼리 실행
            rag_response = self.rag_service.query(
                resolved_query,
                top_k=top_k,
                context=history_context
            )
            answer = rag_response.answer
            sources = [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', None),
                }
                for doc in rag_response.source_documents
            ]
        else:
            # RAG 서비스가 없으면 모의 응답
            answer = f"[Mock] 해결된 쿼리: {resolved_query}"

        # 4. 대화 저장
        self.conversation_manager.add_message(
            session_id, "user", query,
            resolved_query=resolved_query
        )
        self.conversation_manager.add_message(
            session_id, "assistant", answer,
            sources=len(sources)
        )

        # 5. 응답 생성
        context_info = {
            "references_resolved": resolved.references_resolved if resolved else [],
            "entities": resolved.extracted_entities if resolved else {},
            "history_messages": len(self.conversation_manager.get_history(session_id)),
        }

        return MultiTurnResponse(
            session_id=session_id,
            query=query,
            resolved_query=resolved_query,
            answer=answer,
            sources=sources,
            context_info=context_info,
            confidence=resolved.confidence if resolved else 1.0,
        )

    def _build_history_context(self, session_id: str) -> str:
        """히스토리 컨텍스트 문자열 생성"""
        history = self.conversation_manager.get_history(
            session_id,
            n=self.max_context_messages
        )

        if not history:
            return ""

        context_parts = ["[이전 대화]"]
        for msg in history:
            role_kr = "사용자" if msg.role == "user" else "시스템"
            # 긴 메시지는 요약
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            context_parts.append(f"{role_kr}: {content}")

        return "\n".join(context_parts)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        session = self.conversation_manager.get_session(session_id)
        if session is None:
            return None

        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "entities": session.entities,
            "current_topic": session.current_topic,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
        }

    def get_history(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """대화 히스토리 조회"""
        messages = self.conversation_manager.get_history(session_id, n)
        return [msg.to_dict() for msg in messages]

    def clear_session(self, session_id: str) -> bool:
        """세션 초기화"""
        return self.conversation_manager.delete_session(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """서비스 통계"""
        return self.conversation_manager.get_stats()


# =============================================================================
# 편의 함수
# =============================================================================

_default_service: Optional[MultiTurnRAGService] = None


def get_multiturn_service(
    rag_service: Optional[RAGService] = None
) -> MultiTurnRAGService:
    """기본 멀티턴 서비스 조회"""
    global _default_service
    if _default_service is None:
        _default_service = MultiTurnRAGService(rag_service)
    return _default_service


def create_chat_session(metadata: Optional[Dict[str, Any]] = None) -> str:
    """채팅 세션 생성 (편의 함수)"""
    return get_multiturn_service().create_session(metadata)


def chat(
    session_id: str,
    message: str,
    top_k: int = 5
) -> MultiTurnResponse:
    """채팅 (편의 함수)"""
    return get_multiturn_service().query(session_id, message, top_k)
