# -*- coding: utf-8 -*-
"""
대화 컨텍스트 해결 모듈

[기능]
- 대명사 해결 ("그 회사" → "삼성전자")
- 암시적 참조 해결 ("PER은?" → "삼성전자의 PER은?")
- 엔티티 추출 (기업명, 재무지표 등)
- 쿼리 재작성 (컨텍스트 기반)

[사용 예시]
>>> resolver = ContextResolver()
>>> resolved = resolver.resolve_query(
...     "그 회사의 PER은?",
...     entities={"company": "삼성전자"}
... )
>>> print(resolved)  # "삼성전자의 PER은?"
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from .conversation_manager import Message
from .financial_dictionary import FINANCIAL_SYNONYMS

logger = logging.getLogger(__name__)


@dataclass
class ResolvedQuery:
    """해결된 쿼리 결과"""
    original_query: str
    resolved_query: str
    extracted_entities: Dict[str, str]
    references_resolved: List[str]
    confidence: float  # 0.0 ~ 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "resolved_query": self.resolved_query,
            "extracted_entities": self.extracted_entities,
            "references_resolved": self.references_resolved,
            "confidence": self.confidence,
        }


class ContextResolver:
    """
    대화 컨텍스트 해결기

    [처리 순서]
    1. 현재 쿼리에서 엔티티 추출
    2. 대명사/지시어 탐지
    3. 이전 컨텍스트에서 참조 대상 찾기
    4. 쿼리 재작성
    """

    # 대명사 및 지시어 패턴
    PRONOUNS = {
        # 회사/종목 관련
        "그 회사": "company",
        "이 회사": "company",
        "해당 회사": "company",
        "그 기업": "company",
        "이 기업": "company",
        "해당 기업": "company",
        "그 종목": "company",
        "이 종목": "company",
        "해당 종목": "company",
        "거기": "company",
        "그곳": "company",

        # 지표 관련
        "그 지표": "metric",
        "이 지표": "metric",
        "해당 지표": "metric",
        "그것": "metric",
        "이것": "metric",

        # 일반
        "그": "general",
        "이": "general",
        "저": "general",
    }

    # 한국 주요 기업 패턴 (정규식)
    COMPANY_PATTERNS = [
        # 대기업
        r"삼성전자|삼전|SEC",
        r"SK하이닉스|하이닉스|하닉",
        r"현대자동차|현대차|현차",
        r"기아|기아차",
        r"LG전자|엘지전자",
        r"네이버|NAVER",
        r"카카오|KAKAO",
        r"삼성바이오로직스|삼바",
        r"셀트리온",
        r"포스코홀딩스|포스코|POSCO",
        r"현대모비스|모비스",
        r"LG화학|엘지화학",
        r"삼성SDI|삼성에스디아이",
        r"KB금융|KB|국민은행",
        r"신한지주|신한|신한은행",
        r"하나금융|하나|하나은행",

        # 일반 회사 패턴
        r"[가-힣]+(?:전자|자동차|화학|금융|은행|증권|보험|통신|건설|제약|바이오)",
    ]

    # 재무 지표 패턴
    METRIC_PATTERNS = [
        r"PER|P/E|주가수익비율",
        r"PBR|P/B|주가순자산비율",
        r"ROE|자기자본이익률",
        r"ROA|총자산이익률",
        r"EPS|주당순이익",
        r"EBITDA|에비타",
        r"영업이익|영업이익률",
        r"순이익|당기순이익",
        r"매출|매출액",
        r"시가총액|시총",
        r"배당|배당금|배당률|배당수익률",
        r"부채비율",
        r"유동비율",
    ]

    def __init__(self):
        # 정규식 컴파일
        self.company_regex = re.compile(
            "|".join(self.COMPANY_PATTERNS),
            re.IGNORECASE
        )
        self.metric_regex = re.compile(
            "|".join(self.METRIC_PATTERNS),
            re.IGNORECASE
        )
        self.pronoun_regex = re.compile(
            "|".join(re.escape(p) for p in self.PRONOUNS.keys())
        )

    def extract_entities(self, text: str) -> Dict[str, str]:
        """텍스트에서 엔티티 추출"""
        entities = {}

        # 회사명 추출
        company_match = self.company_regex.search(text)
        if company_match:
            entities["company"] = self._normalize_company(company_match.group())

        # 지표 추출
        metric_match = self.metric_regex.search(text)
        if metric_match:
            entities["metric"] = self._normalize_metric(metric_match.group())

        return entities

    def _normalize_company(self, name: str) -> str:
        """회사명 정규화"""
        # 약어를 정식 명칭으로
        mappings = {
            "삼전": "삼성전자",
            "sec": "삼성전자",
            "하닉": "SK하이닉스",
            "하이닉스": "SK하이닉스",
            "현차": "현대자동차",
            "현대차": "현대자동차",
            "기아차": "기아",
            "엘지전자": "LG전자",
            "삼바": "삼성바이오로직스",
            "엘지화학": "LG화학",
            "삼성에스디아이": "삼성SDI",
            "모비스": "현대모비스",
        }
        return mappings.get(name.lower(), name)

    def _normalize_metric(self, metric: str) -> str:
        """지표명 정규화"""
        metric_upper = metric.upper()
        mappings = {
            "P/E": "PER",
            "주가수익비율": "PER",
            "P/B": "PBR",
            "주가순자산비율": "PBR",
            "자기자본이익률": "ROE",
            "총자산이익률": "ROA",
            "주당순이익": "EPS",
            "에비타": "EBITDA",
            "시총": "시가총액",
        }
        return mappings.get(metric_upper, mappings.get(metric, metric))

    def detect_pronouns(self, query: str) -> List[Tuple[str, str, int, int]]:
        """대명사/지시어 탐지

        Returns:
            List of (pronoun, entity_type, start, end)
        """
        pronouns = []

        for match in self.pronoun_regex.finditer(query):
            pronoun = match.group()
            entity_type = self.PRONOUNS.get(pronoun, "general")
            pronouns.append((
                pronoun,
                entity_type,
                match.start(),
                match.end()
            ))

        return pronouns

    def resolve_query(
        self,
        query: str,
        entities: Optional[Dict[str, str]] = None,
        history: Optional[List[Message]] = None,
        topic: Optional[str] = None
    ) -> ResolvedQuery:
        """
        쿼리 해결

        Args:
            query: 현재 쿼리
            entities: 이전 대화에서 추출된 엔티티
            history: 대화 히스토리
            topic: 현재 주제

        Returns:
            ResolvedQuery: 해결된 쿼리
        """
        entities = entities or {}
        references_resolved = []
        confidence = 1.0

        # 1. 현재 쿼리에서 엔티티 추출
        current_entities = self.extract_entities(query)

        # 2. 대명사 탐지
        pronouns = self.detect_pronouns(query)

        # 3. 쿼리 재작성
        resolved = query
        if pronouns:
            # 대명사를 실제 엔티티로 대체
            for pronoun, entity_type, start, end in reversed(pronouns):
                replacement = None

                # 엔티티 타입에 맞는 대체어 찾기
                if entity_type == "company":
                    replacement = entities.get("company") or current_entities.get("company")
                elif entity_type == "metric":
                    replacement = entities.get("metric") or current_entities.get("metric")
                elif entity_type == "general":
                    # 일반 대명사는 회사 → 지표 순으로 시도
                    replacement = (
                        entities.get("company") or
                        entities.get("metric") or
                        current_entities.get("company")
                    )

                if replacement:
                    resolved = resolved[:start] + replacement + resolved[end:]
                    references_resolved.append(f"{pronoun} → {replacement}")
                else:
                    confidence *= 0.8  # 해결 못하면 신뢰도 감소

        # 4. 암시적 참조 해결 (예: "PER은?" → "삼성전자의 PER은?")
        if not current_entities.get("company") and entities.get("company"):
            # 지표만 있고 회사가 없는 경우
            if current_entities.get("metric") or self.metric_regex.search(query):
                company = entities.get("company")
                # 회사명이 쿼리에 없으면 추가
                if company and company not in resolved:
                    resolved = f"{company}의 " + resolved
                    references_resolved.append(f"암시적 참조: {company}")
                    confidence *= 0.9

        # 5. 주제 기반 보완
        if topic and not current_entities and not pronouns:
            # 주제가 있고 새로운 정보가 없으면 주제 컨텍스트 활용
            if topic not in resolved:
                resolved = f"{topic} 관련: " + resolved
                confidence *= 0.85

        # 최종 엔티티 병합
        final_entities = {**entities, **current_entities}

        return ResolvedQuery(
            original_query=query,
            resolved_query=resolved,
            extracted_entities=final_entities,
            references_resolved=references_resolved,
            confidence=confidence,
        )

    def resolve_with_history(
        self,
        query: str,
        history: List[Message],
        max_lookback: int = 5
    ) -> ResolvedQuery:
        """
        히스토리 기반 쿼리 해결

        Args:
            query: 현재 쿼리
            history: 대화 히스토리
            max_lookback: 최대 참조 메시지 수
        """
        # 히스토리에서 엔티티 추출
        entities = {}
        topic = None

        for msg in history[-max_lookback:]:
            msg_entities = self.extract_entities(msg.content)

            # 가장 최근 엔티티 우선
            for key, value in msg_entities.items():
                if key not in entities:
                    entities[key] = value

            # 주제 추출 (사용자 메시지에서)
            if msg.role == "user" and not topic:
                if msg_entities.get("company"):
                    topic = msg_entities["company"]

        return self.resolve_query(query, entities, history, topic)

    def get_search_queries(
        self,
        resolved: ResolvedQuery,
        include_original: bool = True
    ) -> List[str]:
        """
        검색용 쿼리 목록 생성

        해결된 쿼리와 원본 쿼리를 함께 반환하여
        검색 재현율을 높임
        """
        queries = [resolved.resolved_query]

        if include_original and resolved.original_query != resolved.resolved_query:
            queries.append(resolved.original_query)

        return queries


# =============================================================================
# 편의 함수
# =============================================================================

_default_resolver: Optional[ContextResolver] = None


def get_context_resolver() -> ContextResolver:
    """기본 컨텍스트 해결기 조회"""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = ContextResolver()
    return _default_resolver


def resolve_query(
    query: str,
    entities: Optional[Dict[str, str]] = None,
    history: Optional[List[Message]] = None
) -> ResolvedQuery:
    """쿼리 해결 (편의 함수)"""
    resolver = get_context_resolver()
    if history:
        return resolver.resolve_with_history(query, history)
    return resolver.resolve_query(query, entities)


def extract_entities(text: str) -> Dict[str, str]:
    """엔티티 추출 (편의 함수)"""
    return get_context_resolver().extract_entities(text)
