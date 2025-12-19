# -*- coding: utf-8 -*-
"""
대화 히스토리 모듈

[설계 의도]
- 멀티턴 대화 지원
- 컨텍스트 유지로 자연스러운 대화
- 메모리 효율적 관리

[왜 대화 히스토리가 필요한가?]
1. 대명사 해석
   - "그 회사 주가는?" → "삼성전자 주가는?"
   - 이전 대화 참조 필요

2. 후속 질문
   - "더 자세히 알려줘"
   - "비교해줘"

3. 사용자 경험
   - 매번 전체 맥락 반복 불필요
   - 자연스러운 대화 흐름
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json


@dataclass
class Message:
    """대화 메시지"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """대화 턴 (질문-응답 쌍)"""
    user_message: Message
    assistant_message: Message
    contexts_used: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)


class ConversationMemory:
    """
    대화 메모리 관리자

    [메모리 전략]
    1. Window Memory: 최근 N개 턴만 유지
    2. Summary Memory: 이전 대화 요약 (미구현)
    3. Entity Memory: 언급된 엔티티 추적 (미구현)
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 2000
    ):
        """
        Args:
            max_turns: 유지할 최대 대화 턴 수
            max_tokens: 컨텍스트에 포함할 최대 토큰 수 (근사)
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: deque = deque(maxlen=max_turns)
        self.entities: Dict[str, str] = {}  # 추적 엔티티

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        contexts: Optional[List[str]] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """대화 턴 추가"""
        turn = ConversationTurn(
            user_message=Message(role="user", content=user_message),
            assistant_message=Message(role="assistant", content=assistant_message),
            contexts_used=contexts or [],
            sources=sources or []
        )
        self.turns.append(turn)

        # 엔티티 추출 (간단한 버전)
        self._extract_entities(user_message + " " + assistant_message)

    def _extract_entities(self, text: str):
        """간단한 엔티티 추출"""
        # 회사명 패턴
        import re
        company_patterns = [
            r'(삼성전자|SK하이닉스|카카오|네이버|LG에너지솔루션|현대차)',
            r'([A-Z]{2,})',  # 대문자 약어 (NVIDIA, AMD 등)
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    self.entities[match.lower()] = match

    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        LLM에 전달할 대화 히스토리

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        messages = []
        total_chars = 0

        # 최근 턴부터 역순으로 추가 (토큰 제한 내에서)
        for turn in reversed(self.turns):
            user_content = turn.user_message.content
            assistant_content = turn.assistant_message.content

            turn_chars = len(user_content) + len(assistant_content)

            if total_chars + turn_chars > self.max_tokens * 4:  # 대략 4자 = 1토큰
                break

            # 역순으로 추가했으므로 앞에 삽입
            messages.insert(0, {"role": "assistant", "content": assistant_content})
            messages.insert(0, {"role": "user", "content": user_content})
            total_chars += turn_chars

        return messages

    def get_conversation_context(self) -> str:
        """
        대화 컨텍스트를 문자열로 반환

        프롬프트에 직접 포함할 때 사용
        """
        context_parts = []

        for turn in self.turns:
            context_parts.append(f"사용자: {turn.user_message.content}")
            # 긴 응답은 축약
            response = turn.assistant_message.content
            if len(response) > 200:
                response = response[:200] + "..."
            context_parts.append(f"AI: {response}")

        return "\n".join(context_parts)

    def get_recent_entities(self) -> Dict[str, str]:
        """최근 언급된 엔티티"""
        return self.entities.copy()

    def resolve_references(self, query: str) -> str:
        """
        대명사/참조 해결

        "그 회사" → "삼성전자" 등으로 변환
        """
        resolved = query

        # 간단한 대명사 해결
        pronouns = {
            "그 회사": None,
            "이 회사": None,
            "해당 기업": None,
            "거기": None,
        }

        # 가장 최근 언급된 회사로 대체
        if self.entities:
            recent_company = list(self.entities.values())[-1]
            for pronoun in pronouns:
                if pronoun in resolved:
                    resolved = resolved.replace(pronoun, recent_company)

        return resolved

    def clear(self):
        """대화 초기화"""
        self.turns.clear()
        self.entities.clear()

    def to_dict(self) -> Dict[str, Any]:
        """직렬화"""
        return {
            "turns": [
                {
                    "user": turn.user_message.content,
                    "assistant": turn.assistant_message.content,
                    "timestamp": turn.user_message.timestamp.isoformat(),
                    "sources": turn.sources
                }
                for turn in self.turns
            ],
            "entities": self.entities
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """역직렬화"""
        memory = cls()
        for turn_data in data.get("turns", []):
            memory.add_turn(
                user_message=turn_data["user"],
                assistant_message=turn_data["assistant"],
                sources=turn_data.get("sources", [])
            )
        memory.entities = data.get("entities", {})
        return memory

    def get_stats(self) -> Dict[str, Any]:
        """대화 통계"""
        return {
            "total_turns": len(self.turns),
            "total_messages": len(self.turns) * 2,
            "entities_tracked": len(self.entities),
            "entities": list(self.entities.values())
        }


class ConversationalRAG:
    """
    대화형 RAG 래퍼

    기존 RAG 서비스에 대화 기능 추가
    """

    def __init__(
        self,
        rag_service,
        memory: Optional[ConversationMemory] = None
    ):
        """
        Args:
            rag_service: 기본 RAG 서비스
            memory: 대화 메모리 (없으면 새로 생성)
        """
        self.rag = rag_service
        self.memory = memory or ConversationMemory()

    def query(
        self,
        question: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        대화형 질의

        Args:
            question: 사용자 질문
            include_history: 대화 히스토리 포함 여부

        Returns:
            답변 및 메타데이터
        """
        # 1. 참조 해결
        resolved_question = self.memory.resolve_references(question)

        # 2. 대화 컨텍스트 구성
        if include_history and len(self.memory.turns) > 0:
            history_context = self.memory.get_conversation_context()
            enhanced_question = f"""[이전 대화]
{history_context}

[현재 질문]
{resolved_question}"""
        else:
            enhanced_question = resolved_question

        # 3. RAG 질의
        response = self.rag.query(enhanced_question)

        # 4. 메모리에 저장
        self.memory.add_turn(
            user_message=question,
            assistant_message=response.answer,
            sources=response.sources
        )

        return {
            "question": question,
            "resolved_question": resolved_question,
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "conversation_stats": self.memory.get_stats()
        }

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.memory.clear()


def explain_conversation_memory() -> str:
    """대화 메모리 설명 (포트폴리오용)"""
    return """
## 대화 히스토리 관리

### 왜 필요한가?

**단일 턴 RAG의 한계:**
```
사용자: 삼성전자 실적 알려줘
AI: 삼성전자 3분기 영업이익은 9조원입니다.

사용자: 그 회사 주가는?  ← "그 회사"가 뭔지 모름
AI: ??? (문맥 없음)
```

**멀티턴 RAG:**
```
사용자: 삼성전자 실적 알려줘
AI: 삼성전자 3분기 영업이익은 9조원입니다.

사용자: 그 회사 주가는?
AI: [이전 대화에서 "삼성전자" 추출]
    삼성전자 현재 주가는 XX원입니다.
```

### 구현 전략

1. **Window Memory**
   - 최근 N개 대화만 유지
   - 단순하고 효율적
   - 이 프로젝트에서 사용

2. **Summary Memory**
   - 오래된 대화를 요약
   - 장기 컨텍스트 유지
   - LLM 비용 발생

3. **Entity Memory**
   - 언급된 엔티티(회사, 인물) 추적
   - 대명사 해결에 활용

### 참조 해결 (Reference Resolution)

```python
# 입력
"그 회사 주가는?"

# 엔티티 추적
entities = {"삼성전자": "Samsung"}

# 해결된 쿼리
"삼성전자 주가는?"
```
"""
