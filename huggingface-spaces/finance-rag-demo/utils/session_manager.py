# -*- coding: utf-8 -*-
"""
Session Manager for Chat History

대화 히스토리 관리를 위한 세션 매니저
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class ChatMessage:
    """채팅 메시지"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """딕셔너리에서 생성"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(),
            metadata=data.get("metadata", {})
        )


@dataclass
class ChatSession:
    """채팅 세션 (질문-답변 쌍)"""
    id: str
    question: str
    answer: str
    timestamp: datetime
    sources: List[Dict[str, Any]] = field(default_factory=list)
    search_mode: str = "hybrid"
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources,
            "search_mode": self.search_mode,
            "elapsed_time": self.elapsed_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """딕셔너리에서 생성"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            question=data["question"],
            answer=data["answer"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(),
            sources=data.get("sources", []),
            search_mode=data.get("search_mode", "hybrid"),
            elapsed_time=data.get("elapsed_time", 0.0)
        )


class SessionManager:
    """세션 관리자"""

    SESSION_KEY = "chat_history"
    MAX_HISTORY = 50

    @classmethod
    def init(cls, max_history: int = 50):
        """세션 초기화"""
        cls.MAX_HISTORY = max_history
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = []

    @classmethod
    def add_session(
        cls,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]] = None,
        search_mode: str = "hybrid",
        elapsed_time: float = 0.0
    ) -> ChatSession:
        """새 세션 추가"""
        cls.init()

        session = ChatSession(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            sources=sources or [],
            search_mode=search_mode,
            elapsed_time=elapsed_time
        )

        # 히스토리에 추가
        history = st.session_state[cls.SESSION_KEY]
        history.insert(0, session.to_dict())

        # 최대 개수 유지
        if len(history) > cls.MAX_HISTORY:
            st.session_state[cls.SESSION_KEY] = history[:cls.MAX_HISTORY]

        return session

    @classmethod
    def get_history(cls) -> List[ChatSession]:
        """히스토리 가져오기"""
        cls.init()
        history = st.session_state.get(cls.SESSION_KEY, [])
        return [ChatSession.from_dict(h) for h in history]

    @classmethod
    def get_session_by_id(cls, session_id: str) -> Optional[ChatSession]:
        """ID로 세션 가져오기"""
        cls.init()
        history = st.session_state.get(cls.SESSION_KEY, [])
        for h in history:
            if h.get("id") == session_id:
                return ChatSession.from_dict(h)
        return None

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """세션 삭제"""
        cls.init()
        history = st.session_state.get(cls.SESSION_KEY, [])
        new_history = [h for h in history if h.get("id") != session_id]

        if len(new_history) != len(history):
            st.session_state[cls.SESSION_KEY] = new_history
            return True
        return False

    @classmethod
    def clear_history(cls):
        """히스토리 전체 삭제"""
        st.session_state[cls.SESSION_KEY] = []

    @classmethod
    def get_count(cls) -> int:
        """히스토리 개수"""
        cls.init()
        return len(st.session_state.get(cls.SESSION_KEY, []))

    @classmethod
    def export_to_list(cls) -> List[Dict[str, Any]]:
        """내보내기용 리스트"""
        cls.init()
        return st.session_state.get(cls.SESSION_KEY, [])
