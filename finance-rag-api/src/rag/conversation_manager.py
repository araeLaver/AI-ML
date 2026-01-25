# -*- coding: utf-8 -*-
"""
멀티턴 대화 관리 모듈

[기능]
- 세션 기반 대화 히스토리 관리
- 메시지 저장 및 조회
- 세션 만료 및 정리
- 컨텍스트 윈도우 관리

[사용 예시]
>>> manager = ConversationManager()
>>> session_id = manager.create_session()
>>> manager.add_message(session_id, "user", "삼성전자 실적 알려줘")
>>> manager.add_message(session_id, "assistant", "삼성전자의 영업이익은...")
>>> history = manager.get_history(session_id)
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """대화 메시지"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ConversationSession:
    """대화 세션"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 컨텍스트 정보
    entities: Dict[str, str] = field(default_factory=dict)  # 추출된 엔티티
    current_topic: Optional[str] = None  # 현재 주제

    def add_message(self, role: str, content: str, **metadata) -> Message:
        """메시지 추가"""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """최근 n개 메시지 조회"""
        return self.messages[-n:]

    def get_context_window(self, max_tokens: int = 2000) -> List[Message]:
        """토큰 제한 내 컨텍스트 윈도우"""
        # 간단한 근사: 한글 1자 ≈ 2 토큰
        messages = []
        total_chars = 0
        char_limit = max_tokens // 2

        for msg in reversed(self.messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > char_limit:
                break
            messages.insert(0, msg)
            total_chars += msg_chars

        return messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "entities": self.entities,
            "current_topic": self.current_topic,
            "metadata": self.metadata,
        }


class ConversationManager:
    """
    대화 세션 관리자

    [특징]
    - 인메모리 세션 저장 (프로덕션에서는 Redis 권장)
    - 자동 세션 만료
    - 스레드 안전
    - LRU 기반 세션 관리
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        session_ttl_minutes: int = 60,
        max_messages_per_session: int = 100
    ):
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.max_messages = max_messages_per_session

        # LRU 캐시로 세션 저장
        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()
        self._lock = Lock()

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())

        with self._lock:
            # 최대 세션 수 초과 시 가장 오래된 세션 제거
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)

            session = ConversationSession(
                session_id=session_id,
                metadata=metadata or {}
            )
            self._sessions[session_id] = session

        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """세션 조회"""
        with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                return None

            # 만료 확인
            if datetime.now() - session.last_activity > self.session_ttl:
                self._sessions.pop(session_id, None)
                logger.info(f"Session expired: {session_id}")
                return None

            # LRU 업데이트
            self._sessions.move_to_end(session_id)

            return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **metadata
    ) -> Optional[Message]:
        """메시지 추가"""
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"Session not found: {session_id}")
            return None

        # 최대 메시지 수 초과 시 오래된 메시지 제거
        while len(session.messages) >= self.max_messages:
            session.messages.pop(0)

        message = session.add_message(role, content, **metadata)
        logger.debug(f"Added message to session {session_id}: {role}")
        return message

    def get_history(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[Message]:
        """대화 히스토리 조회"""
        session = self.get_session(session_id)
        if session is None:
            return []

        if n is not None:
            return session.get_recent_messages(n)
        return session.messages.copy()

    def get_context(
        self,
        session_id: str,
        max_tokens: int = 2000
    ) -> List[Message]:
        """컨텍스트 윈도우 조회"""
        session = self.get_session(session_id)
        if session is None:
            return []

        return session.get_context_window(max_tokens)

    def update_entities(
        self,
        session_id: str,
        entities: Dict[str, str]
    ) -> bool:
        """세션 엔티티 업데이트"""
        session = self.get_session(session_id)
        if session is None:
            return False

        session.entities.update(entities)
        return True

    def set_topic(self, session_id: str, topic: str) -> bool:
        """현재 주제 설정"""
        session = self.get_session(session_id)
        if session is None:
            return False

        session.current_topic = topic
        return True

    def get_entities(self, session_id: str) -> Dict[str, str]:
        """세션 엔티티 조회"""
        session = self.get_session(session_id)
        if session is None:
            return {}
        return session.entities.copy()

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
        return False

    def cleanup_expired(self) -> int:
        """만료된 세션 정리"""
        now = datetime.now()
        expired = []

        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_activity > self.session_ttl:
                    expired.append(session_id)

            for session_id in expired:
                del self._sessions[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        with self._lock:
            total_messages = sum(
                len(s.messages) for s in self._sessions.values()
            )
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "total_messages": total_messages,
                "session_ttl_minutes": self.session_ttl.total_seconds() / 60,
            }


# =============================================================================
# 편의 함수
# =============================================================================

# 글로벌 매니저 인스턴스
_default_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """기본 대화 매니저 조회"""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConversationManager()
    return _default_manager


def create_session(metadata: Optional[Dict[str, Any]] = None) -> str:
    """새 세션 생성 (편의 함수)"""
    return get_conversation_manager().create_session(metadata)


def add_message(session_id: str, role: str, content: str) -> Optional[Message]:
    """메시지 추가 (편의 함수)"""
    return get_conversation_manager().add_message(session_id, role, content)


def get_history(session_id: str, n: Optional[int] = None) -> List[Message]:
    """히스토리 조회 (편의 함수)"""
    return get_conversation_manager().get_history(session_id, n)
