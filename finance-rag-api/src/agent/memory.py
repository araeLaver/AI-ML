# -*- coding: utf-8 -*-
"""
에이전트 메모리 모듈

[기능]
- 대화 메모리
- 작업 메모리
- 장기 메모리
- 컨텍스트 관리
"""

import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """메모리 유형"""
    CONVERSATION = "conversation"  # 대화 기록
    WORKING = "working"  # 작업 메모리
    LONG_TERM = "long_term"  # 장기 메모리
    EPISODIC = "episodic"  # 에피소드 메모리
    SEMANTIC = "semantic"  # 의미 메모리


@dataclass
class MemoryEntry:
    """메모리 항목"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.WORKING
    content: Any = None
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 ~ 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": str(self.content)[:200] if self.content else None,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
        }


class AgentMemory(ABC):
    """에이전트 메모리 인터페이스"""

    @abstractmethod
    def add(self, content: Any, importance: float = 0.5, **metadata) -> MemoryEntry:
        """메모리 추가"""
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """메모리 조회"""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """메모리 검색"""
        pass

    @abstractmethod
    def forget(self, entry_id: str) -> bool:
        """메모리 삭제"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """전체 삭제"""
        pass


class ConversationMemory(AgentMemory):
    """
    대화 메모리

    최근 대화 기록 관리
    """

    def __init__(self, max_turns: int = 100):
        self.max_turns = max_turns
        self._messages: deque = deque(maxlen=max_turns)
        self._entries: Dict[str, MemoryEntry] = {}

    def add(
        self,
        content: Any,
        importance: float = 0.5,
        role: str = "user",
        **metadata,
    ) -> MemoryEntry:
        """대화 추가"""
        entry = MemoryEntry(
            memory_type=MemoryType.CONVERSATION,
            content=content,
            importance=importance,
            metadata={"role": role, **metadata},
        )

        self._messages.append(entry)
        self._entries[entry.id] = entry

        return entry

    def add_user_message(self, content: str) -> MemoryEntry:
        """사용자 메시지 추가"""
        return self.add(content, role="user")

    def add_assistant_message(self, content: str) -> MemoryEntry:
        """어시스턴트 메시지 추가"""
        return self.add(content, role="assistant")

    def add_system_message(self, content: str) -> MemoryEntry:
        """시스템 메시지 추가"""
        return self.add(content, role="system", importance=0.8)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """메모리 조회"""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
        return entry

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """최근 대화"""
        return list(self._messages)[-n:]

    def get_by_role(self, role: str) -> List[MemoryEntry]:
        """역할별 메시지"""
        return [
            e for e in self._messages
            if e.metadata.get("role") == role
        ]

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """대화 검색 (키워드 기반)"""
        query_lower = query.lower()
        results = []

        for entry in self._messages:
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                results.append(entry)

        return results[:limit]

    def forget(self, entry_id: str) -> bool:
        """메모리 삭제"""
        if entry_id in self._entries:
            entry = self._entries.pop(entry_id)
            self._messages = deque(
                [e for e in self._messages if e.id != entry_id],
                maxlen=self.max_turns,
            )
            return True
        return False

    def clear(self) -> None:
        """전체 삭제"""
        self._messages.clear()
        self._entries.clear()

    def get_context_window(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """컨텍스트 윈도우 (LLM용)"""
        messages = []
        total_length = 0

        for entry in reversed(list(self._messages)):
            content_str = str(entry.content)
            content_length = len(content_str)

            if total_length + content_length > max_tokens:
                break

            messages.insert(0, {
                "role": entry.metadata.get("role", "user"),
                "content": content_str,
            })
            total_length += content_length

        return messages

    def summarize(self, llm_func: Optional[Callable] = None) -> str:
        """대화 요약"""
        if not self._messages:
            return ""

        messages = [str(e.content) for e in self._messages]
        full_text = "\n".join(messages)

        if llm_func:
            return llm_func(f"다음 대화를 요약해주세요:\n{full_text}")

        # 간단한 요약 (처음과 마지막)
        return f"대화 시작: {messages[0][:100]}... 마지막: {messages[-1][:100]}"


class WorkingMemory(AgentMemory):
    """
    작업 메모리

    현재 작업에 필요한 임시 정보 저장
    """

    def __init__(self, max_items: int = 50):
        self.max_items = max_items
        self._items: Dict[str, MemoryEntry] = {}
        self._variables: Dict[str, Any] = {}

    def add(
        self,
        content: Any,
        importance: float = 0.5,
        key: Optional[str] = None,
        **metadata,
    ) -> MemoryEntry:
        """작업 항목 추가"""
        entry = MemoryEntry(
            memory_type=MemoryType.WORKING,
            content=content,
            importance=importance,
            metadata=metadata,
        )

        if key:
            entry.metadata["key"] = key

        self._items[entry.id] = entry

        # 최대 크기 초과 시 중요도 낮은 항목 제거
        if len(self._items) > self.max_items:
            self._evict_least_important()

        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """항목 조회"""
        entry = self._items.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
        return entry

    def get_by_key(self, key: str) -> Optional[MemoryEntry]:
        """키로 조회"""
        for entry in self._items.values():
            if entry.metadata.get("key") == key:
                entry.access_count += 1
                entry.last_accessed = time.time()
                return entry
        return None

    def set_variable(self, name: str, value: Any) -> None:
        """변수 설정"""
        self._variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """변수 조회"""
        return self._variables.get(name, default)

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """검색"""
        query_lower = query.lower()
        results = []

        for entry in self._items.values():
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                results.append(entry)

        # 중요도순 정렬
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:limit]

    def forget(self, entry_id: str) -> bool:
        """항목 삭제"""
        if entry_id in self._items:
            del self._items[entry_id]
            return True
        return False

    def clear(self) -> None:
        """전체 삭제"""
        self._items.clear()
        self._variables.clear()

    def _evict_least_important(self) -> None:
        """가장 덜 중요한 항목 제거"""
        if not self._items:
            return

        # 중요도와 접근 시간 고려
        def score(entry: MemoryEntry) -> float:
            recency = time.time() - entry.last_accessed
            return entry.importance - (recency / 3600)  # 시간당 감소

        least_important = min(self._items.values(), key=score)
        del self._items[least_important.id]

    def get_all(self) -> List[MemoryEntry]:
        """모든 항목"""
        return list(self._items.values())

    def get_context(self) -> Dict[str, Any]:
        """현재 컨텍스트"""
        return {
            "items": [e.to_dict() for e in self._items.values()],
            "variables": self._variables,
            "total_items": len(self._items),
        }


class LongTermMemory(AgentMemory):
    """
    장기 메모리

    영구 저장 및 유사도 기반 검색
    """

    def __init__(
        self,
        embedding_func: Optional[Callable] = None,
        max_entries: int = 10000,
    ):
        self.embedding_func = embedding_func
        self.max_entries = max_entries
        self._entries: Dict[str, MemoryEntry] = {}
        self._index: Dict[str, List[str]] = {}  # 키워드 -> entry_ids

    def add(
        self,
        content: Any,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        **metadata,
    ) -> MemoryEntry:
        """장기 메모리 추가"""
        entry = MemoryEntry(
            memory_type=MemoryType.LONG_TERM,
            content=content,
            importance=importance,
            metadata={"tags": tags or [], **metadata},
        )

        # 임베딩 생성
        if self.embedding_func:
            entry.embedding = self.embedding_func(str(content))

        self._entries[entry.id] = entry

        # 인덱스 업데이트
        self._update_index(entry)

        # 최대 크기 관리
        if len(self._entries) > self.max_entries:
            self._consolidate()

        return entry

    def _update_index(self, entry: MemoryEntry) -> None:
        """인덱스 업데이트"""
        # 태그 인덱싱
        for tag in entry.metadata.get("tags", []):
            if tag not in self._index:
                self._index[tag] = []
            self._index[tag].append(entry.id)

        # 키워드 인덱싱
        content_str = str(entry.content).lower()
        words = set(content_str.split())
        for word in words:
            if len(word) > 2:  # 2글자 초과
                if word not in self._index:
                    self._index[word] = []
                self._index[word].append(entry.id)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """메모리 조회"""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
        return entry

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """검색 (키워드 + 유사도)"""
        results = []
        seen = set()

        # 키워드 검색
        query_words = query.lower().split()
        for word in query_words:
            for entry_id in self._index.get(word, []):
                if entry_id not in seen:
                    entry = self._entries.get(entry_id)
                    if entry:
                        results.append(entry)
                        seen.add(entry_id)

        # 유사도 기반 검색 (임베딩)
        if self.embedding_func and len(results) < limit:
            query_embedding = self.embedding_func(query)
            similarities = []

            for entry in self._entries.values():
                if entry.id not in seen and entry.embedding:
                    sim = self._cosine_similarity(query_embedding, entry.embedding)
                    similarities.append((entry, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            for entry, sim in similarities[:limit - len(results)]:
                if sim > 0.5:  # 유사도 임계값
                    results.append(entry)

        # 중요도와 최신성으로 정렬
        def score(entry: MemoryEntry) -> float:
            recency = (time.time() - entry.timestamp) / 86400  # 일 단위
            return entry.importance * (1 / (1 + recency * 0.1))

        results.sort(key=score, reverse=True)
        return results[:limit]

    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """태그로 검색"""
        entry_ids = self._index.get(tag, [])
        return [
            self._entries[eid]
            for eid in entry_ids
            if eid in self._entries
        ]

    def search_similar(
        self,
        content: str,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> List[Tuple[MemoryEntry, float]]:
        """유사한 메모리 검색"""
        if not self.embedding_func:
            return []

        query_embedding = self.embedding_func(content)
        results = []

        for entry in self._entries.values():
            if entry.embedding:
                sim = self._cosine_similarity(query_embedding, entry.embedding)
                if sim >= threshold:
                    results.append((entry, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def forget(self, entry_id: str) -> bool:
        """메모리 삭제"""
        if entry_id in self._entries:
            entry = self._entries.pop(entry_id)

            # 인덱스에서 제거
            for tag in entry.metadata.get("tags", []):
                if tag in self._index and entry_id in self._index[tag]:
                    self._index[tag].remove(entry_id)

            return True
        return False

    def clear(self) -> None:
        """전체 삭제"""
        self._entries.clear()
        self._index.clear()

    def _consolidate(self) -> None:
        """메모리 통합 (오래되고 덜 중요한 항목 정리)"""
        if len(self._entries) <= self.max_entries:
            return

        # 점수 계산
        def score(entry: MemoryEntry) -> float:
            recency = (time.time() - entry.timestamp) / 86400
            access = entry.access_count
            return entry.importance + access * 0.1 - recency * 0.01

        entries_with_score = [
            (eid, score(e))
            for eid, e in self._entries.items()
        ]
        entries_with_score.sort(key=lambda x: x[1])

        # 하위 10% 제거
        remove_count = len(entries_with_score) // 10
        for eid, _ in entries_with_score[:remove_count]:
            self.forget(eid)

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """코사인 유사도"""
        if len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def get_statistics(self) -> Dict[str, Any]:
        """통계"""
        return {
            "total_entries": len(self._entries),
            "index_size": len(self._index),
            "avg_importance": (
                sum(e.importance for e in self._entries.values()) / len(self._entries)
                if self._entries else 0
            ),
            "avg_access_count": (
                sum(e.access_count for e in self._entries.values()) / len(self._entries)
                if self._entries else 0
            ),
        }
