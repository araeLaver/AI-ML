"""
쿼리 확장 모듈

금융 동의어 사전을 활용하여 사용자 쿼리를 확장합니다.
검색 재현율(Recall)을 높이기 위해 동의어, 약어, 영문 표현 등을 추가합니다.

[사용 예시]
- "PER 높은 기업" → ["PER 높은 기업", "주가수익비율 높은 기업"]
- "삼전 실적" → ["삼전 실적", "삼성전자 실적"]
- "HBM 관련주" → ["HBM 관련주", "고대역폭메모리 관련주"]
"""

import re
import logging
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass

from .financial_dictionary import (
    FINANCIAL_SYNONYMS,
    REVERSE_SYNONYMS,
    get_synonyms,
    get_canonical_term,
)

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """쿼리 확장 결과"""
    original_query: str
    expanded_queries: List[str]
    expansions: List[Dict[str, str]]  # [{"term": "PER", "expanded_to": "주가수익비율"}, ...]
    num_expansions: int


class QueryExpander:
    """
    금융 도메인 쿼리 확장기

    [동작 방식]
    1. 쿼리에서 금융 용어 추출
    2. 각 용어의 동의어 검색
    3. 동의어로 대체한 확장 쿼리 생성
    4. 중복 제거 및 정렬

    [확장 전략]
    - 정방향: 약어 → 풀네임 (PER → 주가수익비율)
    - 역방향: 풀네임 → 약어 (주가수익비율 → PER)
    - 표준화: 비표준 → 표준 (삼전 → 삼성전자)
    """

    def __init__(
        self,
        tokenizer=None,
        max_expansions_per_term: int = 2,
        max_total_queries: int = 5,
        include_original: bool = True,
    ):
        """
        Args:
            tokenizer: 형태소 분석기 (KiwiTokenizer 등), None이면 정규식 사용
            max_expansions_per_term: 용어당 최대 확장 수
            max_total_queries: 최대 확장 쿼리 수
            include_original: 원본 쿼리 포함 여부
        """
        self.tokenizer = tokenizer
        self.max_expansions_per_term = max_expansions_per_term
        self.max_total_queries = max_total_queries
        self.include_original = include_original

        # 빠른 검색을 위한 정규식 패턴 생성
        self._build_patterns()

    def _build_patterns(self):
        """동의어 검색용 정규식 패턴 생성"""
        # 모든 표준 용어 (대소문자 구분 없이)
        all_terms = set(FINANCIAL_SYNONYMS.keys())
        for synonyms in FINANCIAL_SYNONYMS.values():
            all_terms.update(synonyms)

        # 긴 것부터 매칭하도록 정렬 (삼성전자가 삼성보다 먼저 매칭되도록)
        sorted_terms = sorted(all_terms, key=len, reverse=True)

        # 정규식 패턴 생성 (특수문자 이스케이프)
        escaped_terms = [re.escape(term) for term in sorted_terms]
        self._term_pattern = re.compile(
            r'\b(' + '|'.join(escaped_terms) + r')\b',
            re.IGNORECASE
        )

    def expand(self, query: str) -> ExpansionResult:
        """
        쿼리 확장 수행

        Args:
            query: 원본 쿼리

        Returns:
            ExpansionResult: 확장 결과
        """
        expanded_queries = []
        expansions = []

        if self.include_original:
            expanded_queries.append(query)

        # 쿼리에서 금융 용어 찾기
        found_terms = self._find_financial_terms(query)

        if not found_terms:
            return ExpansionResult(
                original_query=query,
                expanded_queries=expanded_queries,
                expansions=[],
                num_expansions=0,
            )

        # 각 용어에 대해 확장 쿼리 생성
        for term, start, end in found_terms:
            synonyms = self._get_expansion_synonyms(term)

            for syn in synonyms[:self.max_expansions_per_term]:
                # 원본 용어를 동의어로 대체
                expanded = query[:start] + syn + query[end:]
                if expanded not in expanded_queries:
                    expanded_queries.append(expanded)
                    expansions.append({
                        "term": term,
                        "expanded_to": syn,
                        "position": (start, end),
                    })

                # 최대 쿼리 수 체크
                if len(expanded_queries) >= self.max_total_queries:
                    break

            if len(expanded_queries) >= self.max_total_queries:
                break

        return ExpansionResult(
            original_query=query,
            expanded_queries=expanded_queries,
            expansions=expansions,
            num_expansions=len(expanded_queries) - (1 if self.include_original else 0),
        )

    def expand_simple(self, query: str) -> List[str]:
        """
        간단한 쿼리 확장 (쿼리 리스트만 반환)

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 리스트
        """
        result = self.expand(query)
        return result.expanded_queries

    def _find_financial_terms(self, query: str) -> List[Tuple[str, int, int]]:
        """
        쿼리에서 금융 용어 찾기

        Returns:
            [(용어, 시작위치, 끝위치), ...]
        """
        found = []

        # 토크나이저가 있으면 형태소 분석 사용
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(query)
            for token in tokens:
                if self._is_financial_term(token):
                    # 쿼리에서 토큰 위치 찾기
                    match = re.search(re.escape(token), query, re.IGNORECASE)
                    if match:
                        found.append((token, match.start(), match.end()))
        else:
            # 정규식으로 직접 매칭
            for match in self._term_pattern.finditer(query):
                found.append((match.group(), match.start(), match.end()))

        # 위치 기준 정렬 (뒤에서부터 처리하기 위해 역순)
        found.sort(key=lambda x: x[1], reverse=True)

        return found

    def _is_financial_term(self, term: str) -> bool:
        """금융 용어 여부 확인"""
        term_lower = term.lower()
        term_upper = term.upper()

        return (
            term_upper in FINANCIAL_SYNONYMS or
            term_lower in REVERSE_SYNONYMS
        )

    def _get_expansion_synonyms(self, term: str) -> List[str]:
        """
        확장할 동의어 반환 (원본 제외)

        우선순위:
        1. 표준 용어 (약어 → 풀네임)
        2. 한글 동의어
        3. 영문 동의어
        """
        all_synonyms = get_synonyms(term)
        all_synonyms.discard(term)  # 원본 제외

        # 우선순위에 따라 정렬
        def priority(s):
            # 한글 우선
            if re.match(r'^[가-힣]+$', s):
                return 0
            # 한글+영문 혼합
            elif re.search(r'[가-힣]', s):
                return 1
            # 영문 약어 (대문자)
            elif s.isupper():
                return 2
            # 기타 영문
            else:
                return 3

        sorted_synonyms = sorted(all_synonyms, key=priority)

        return sorted_synonyms

    def get_all_synonyms(self, term: str) -> Set[str]:
        """특정 용어의 모든 동의어 반환 (원본 포함)"""
        return get_synonyms(term)

    def normalize(self, query: str) -> str:
        """
        쿼리를 표준 용어로 정규화

        예: "삼전 1분기 실적" → "삼성전자 1분기 실적"
        """
        found_terms = self._find_financial_terms(query)
        normalized = query

        # 뒤에서부터 치환 (위치가 밀리지 않도록)
        for term, start, end in found_terms:
            canonical = get_canonical_term(term)
            if canonical != term:
                normalized = normalized[:start] + canonical + normalized[end:]

        return normalized


class MultiTermExpander(QueryExpander):
    """
    다중 용어 확장기

    한 쿼리에 여러 금융 용어가 있을 때,
    모든 조합을 생성하는 대신 가장 효과적인 확장만 선택
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_combinations = kwargs.get('max_combinations', 3)

    def expand(self, query: str) -> ExpansionResult:
        """다중 용어 확장"""
        found_terms = self._find_financial_terms(query)

        if len(found_terms) <= 1:
            # 단일 용어는 기본 확장 사용
            return super().expand(query)

        # 각 용어별 최선의 확장만 선택
        expanded_queries = [query] if self.include_original else []
        expansions = []

        for term, start, end in found_terms:
            best_synonym = self._get_best_synonym(term)
            if best_synonym:
                expanded = query[:start] + best_synonym + query[end:]
                if expanded not in expanded_queries:
                    expanded_queries.append(expanded)
                    expansions.append({
                        "term": term,
                        "expanded_to": best_synonym,
                    })

            if len(expanded_queries) >= self.max_total_queries:
                break

        return ExpansionResult(
            original_query=query,
            expanded_queries=expanded_queries,
            expansions=expansions,
            num_expansions=len(expansions),
        )

    def _get_best_synonym(self, term: str) -> Optional[str]:
        """가장 효과적인 동의어 반환"""
        synonyms = self._get_expansion_synonyms(term)
        return synonyms[0] if synonyms else None


# =============================================================================
# 편의 함수
# =============================================================================

# 기본 확장기 인스턴스
_default_expander: Optional[QueryExpander] = None


def get_expander(tokenizer=None) -> QueryExpander:
    """기본 쿼리 확장기 반환"""
    global _default_expander

    if _default_expander is None or tokenizer is not None:
        _default_expander = QueryExpander(tokenizer=tokenizer)

    return _default_expander


def expand_query(query: str, tokenizer=None) -> List[str]:
    """쿼리 확장 (편의 함수)"""
    expander = get_expander(tokenizer)
    return expander.expand_simple(query)


def normalize_query(query: str, tokenizer=None) -> str:
    """쿼리 정규화 (편의 함수)"""
    expander = get_expander(tokenizer)
    return expander.normalize(query)


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("쿼리 확장 테스트")
    print("=" * 60)

    expander = QueryExpander()

    test_queries = [
        "PER 높은 기업",
        "삼전 2024년 1분기 실적",
        "HBM 관련주 추천",
        "SK하이닉스 ROE 분석",
        "2차전지 시장 전망",
        "삼성전자 영업이익",  # 이미 표준 용어
    ]

    for query in test_queries:
        result = expander.expand(query)
        print(f"\n원본: {query}")
        print(f"확장: {result.expanded_queries}")
        if result.expansions:
            for exp in result.expansions:
                print(f"  - {exp['term']} → {exp['expanded_to']}")

    print("\n" + "=" * 60)
    print("정규화 테스트")
    print("=" * 60)

    normalize_tests = [
        "삼전 실적",
        "하닉 HBM 매출",
        "주가수익비율 낮은 종목",
    ]

    for query in normalize_tests:
        normalized = expander.normalize(query)
        print(f"{query} → {normalized}")
