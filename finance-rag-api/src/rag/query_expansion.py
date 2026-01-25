# -*- coding: utf-8 -*-
"""
Query Expansion 모듈

금융 도메인 동의어/유의어 확장을 통해 검색 재현율을 향상시킵니다.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExpansionConfig:
    """Query Expansion 설정

    Attributes:
        enable_synonyms: 동의어 확장 활성화
        enable_abbreviations: 약어 확장 활성화
        enable_related_terms: 관련어 확장 활성화
        max_expansions: 최대 확장 수
        boost_original: 원본 쿼리 가중치
    """
    enable_synonyms: bool = True
    enable_abbreviations: bool = True
    enable_related_terms: bool = True
    max_expansions: int = 5
    boost_original: float = 2.0


class FinancialSynonymDict:
    """금융 도메인 동의어 사전"""

    # 동의어 그룹 (양방향 매핑)
    SYNONYM_GROUPS: list[set[str]] = [
        # 재무 지표
        {"PER", "주가수익비율", "P/E", "주가순이익비율"},
        {"PBR", "주가순자산비율", "P/B"},
        {"ROE", "자기자본이익률", "자기자본수익률"},
        {"ROA", "총자산이익률", "총자산수익률"},
        {"EPS", "주당순이익", "주당이익"},
        {"BPS", "주당순자산"},
        {"영업이익", "영업손익", "operating profit"},
        {"당기순이익", "순이익", "net income", "순손익"},
        {"매출액", "매출", "revenue", "sales"},
        {"시가총액", "시총", "market cap"},

        # 공시 유형
        {"사업보고서", "연간보고서", "연차보고서", "annual report"},
        {"분기보고서", "분기실적", "quarterly report"},
        {"반기보고서", "반기실적"},
        {"주요사항보고서", "주요사항보고"},
        {"임원", "경영진", "이사회"},

        # 산업/기술
        {"반도체", "semiconductor", "칩"},
        {"HBM", "고대역폭메모리", "High Bandwidth Memory"},
        {"OLED", "유기발광다이오드"},
        {"파운드리", "foundry", "위탁생산"},
        {"배터리", "이차전지", "2차전지", "battery"},
        {"전기차", "EV", "전기자동차", "Electric Vehicle"},

        # 기업 관련
        {"배당", "배당금", "dividend"},
        {"주주", "shareholder", "투자자"},
        {"자사주", "자기주식", "treasury stock"},
        {"유상증자", "증자"},
        {"무상증자", "무상배정"},
        {"합병", "M&A", "인수합병"},
        {"상장", "IPO", "기업공개"},
        {"상장폐지", "delisting"},

        # 시장
        {"코스피", "KOSPI", "유가증권시장"},
        {"코스닥", "KOSDAQ"},
        {"나스닥", "NASDAQ"},
        {"다우", "다우존스", "DOW"},
        {"S&P500", "S&P", "에스앤피500"},
    ]

    # 약어 확장 (단방향)
    ABBREVIATIONS: dict[str, list[str]] = {
        "삼전": ["삼성전자"],
        "하닉": ["SK하이닉스"],
        "엘솔": ["LG에너지솔루션"],
        "현차": ["현대차", "현대자동차"],
        "기아": ["기아자동차"],
        "네카라쿠배": ["네이버", "카카오", "라인", "쿠팡", "배달의민족"],
        "빅테크": ["구글", "애플", "메타", "아마존", "마이크로소프트"],
        "FAANG": ["페이스북", "애플", "아마존", "넷플릭스", "구글"],
        "IR": ["기업설명회", "투자설명회", "Investor Relations"],
        "CB": ["전환사채", "Convertible Bond"],
        "BW": ["신주인수권부사채", "Bond with Warrant"],
        "EB": ["교환사채", "Exchangeable Bond"],
    }

    # 관련어 (의미적으로 관련된 용어)
    RELATED_TERMS: dict[str, list[str]] = {
        "실적": ["매출", "영업이익", "순이익", "분기실적"],
        "주가": ["시세", "주식가격", "종가", "시가"],
        "상승": ["급등", "오름", "증가", "상향"],
        "하락": ["급락", "내림", "감소", "하향"],
        "투자": ["매수", "매입", "인수"],
        "매도": ["매각", "처분", "청산"],
        "호황": ["성장", "증가", "확대"],
        "불황": ["침체", "위축", "감소"],
    }

    def __init__(self):
        """동의어 사전 초기화"""
        self._synonym_map: dict[str, set[str]] = {}
        self._build_synonym_map()

    def _build_synonym_map(self) -> None:
        """동의어 맵 구축"""
        for group in self.SYNONYM_GROUPS:
            for term in group:
                term_lower = term.lower()
                if term_lower not in self._synonym_map:
                    self._synonym_map[term_lower] = set()
                # 자기 자신 제외한 모든 동의어 추가
                self._synonym_map[term_lower].update(
                    t for t in group if t.lower() != term_lower
                )

    def get_synonyms(self, term: str) -> set[str]:
        """동의어 조회"""
        return self._synonym_map.get(term.lower(), set())

    def get_abbreviation_expansions(self, term: str) -> list[str]:
        """약어 확장 조회"""
        return self.ABBREVIATIONS.get(term, [])

    def get_related_terms(self, term: str) -> list[str]:
        """관련어 조회"""
        return self.RELATED_TERMS.get(term, [])


@dataclass
class ExpandedQuery:
    """확장된 쿼리

    Attributes:
        original: 원본 쿼리
        expanded_terms: 확장된 용어 목록
        all_terms: 모든 검색어 (원본 + 확장)
        expansion_info: 확장 정보 (어떤 용어가 어떻게 확장되었는지)
    """
    original: str
    expanded_terms: list[str] = field(default_factory=list)
    all_terms: list[str] = field(default_factory=list)
    expansion_info: dict[str, list[str]] = field(default_factory=dict)

    def to_query_string(self, operator: str = "OR") -> str:
        """검색 쿼리 문자열로 변환

        Args:
            operator: 검색 연산자 (OR, AND)

        Returns:
            검색 쿼리 문자열
        """
        if not self.all_terms:
            return self.original

        # 원본 쿼리와 확장 용어 결합
        terms = [self.original] + self.expanded_terms
        unique_terms = list(dict.fromkeys(terms))  # 순서 유지하며 중복 제거

        return f" {operator} ".join(f'"{t}"' for t in unique_terms)


class QueryExpander:
    """쿼리 확장기

    금융 도메인 동의어/유의어를 활용하여 쿼리를 확장합니다.
    """

    def __init__(
        self,
        config: Optional[ExpansionConfig] = None,
        synonym_dict: Optional[FinancialSynonymDict] = None,
    ):
        """
        Args:
            config: 확장 설정
            synonym_dict: 동의어 사전
        """
        self.config = config or ExpansionConfig()
        self.synonym_dict = synonym_dict or FinancialSynonymDict()

        # 통계
        self._total_expansions = 0
        self._expansion_counts: dict[str, int] = {}

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return {
            "total_expansions": self._total_expansions,
            "expansion_counts": self._expansion_counts,
        }

    def expand(self, query: str) -> ExpandedQuery:
        """쿼리 확장

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리
        """
        expanded_terms: list[str] = []
        expansion_info: dict[str, list[str]] = {}

        # 쿼리에서 용어 추출 (간단한 토큰화)
        terms = self._extract_terms(query)

        for term in terms:
            term_expansions: list[str] = []

            # 동의어 확장
            if self.config.enable_synonyms:
                synonyms = self.synonym_dict.get_synonyms(term)
                term_expansions.extend(synonyms)

            # 약어 확장
            if self.config.enable_abbreviations:
                abbreviations = self.synonym_dict.get_abbreviation_expansions(term)
                term_expansions.extend(abbreviations)

            # 관련어 확장
            if self.config.enable_related_terms:
                related = self.synonym_dict.get_related_terms(term)
                term_expansions.extend(related)

            if term_expansions:
                # 최대 확장 수 제한
                term_expansions = term_expansions[:self.config.max_expansions]
                expanded_terms.extend(term_expansions)
                expansion_info[term] = term_expansions

                # 통계 업데이트
                self._expansion_counts[term] = self._expansion_counts.get(term, 0) + 1

        self._total_expansions += 1

        # 모든 용어 (원본 + 확장)
        all_terms = [query] + list(set(expanded_terms))

        return ExpandedQuery(
            original=query,
            expanded_terms=list(set(expanded_terms)),
            all_terms=all_terms,
            expansion_info=expansion_info,
        )

    def _extract_terms(self, query: str) -> list[str]:
        """쿼리에서 용어 추출

        Args:
            query: 쿼리 문자열

        Returns:
            추출된 용어 목록
        """
        # 영어 단어, 한글 단어, 약어 패턴 매칭
        # 알파벳/한글 연속 문자열 + 숫자 포함 용어
        pattern = r'[A-Za-z가-힣][A-Za-z가-힣0-9/&]*'
        terms = re.findall(pattern, query)

        # 중복 제거 및 필터링 (너무 짧은 용어 제외)
        unique_terms = []
        seen = set()
        for term in terms:
            if len(term) >= 2 and term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return unique_terms

    def expand_for_hybrid_search(
        self,
        query: str,
    ) -> dict[str, Any]:
        """하이브리드 검색용 쿼리 확장

        BM25와 Vector Search에 적합한 형태로 확장합니다.

        Args:
            query: 원본 쿼리

        Returns:
            {
                "original": 원본 쿼리,
                "bm25_query": BM25용 확장 쿼리,
                "vector_query": Vector Search용 쿼리,
                "expanded_terms": 확장된 용어 목록,
            }
        """
        expanded = self.expand(query)

        # BM25: 확장된 모든 용어 포함
        bm25_terms = [expanded.original] + expanded.expanded_terms
        bm25_query = " ".join(bm25_terms)

        # Vector Search: 원본 쿼리 + 주요 확장 용어 (너무 많으면 의미 희석)
        vector_terms = [expanded.original]
        if expanded.expanded_terms:
            vector_terms.extend(expanded.expanded_terms[:2])  # 상위 2개만
        vector_query = " ".join(vector_terms)

        return {
            "original": expanded.original,
            "bm25_query": bm25_query,
            "vector_query": vector_query,
            "expanded_terms": expanded.expanded_terms,
            "expansion_info": expanded.expansion_info,
        }


class ContextualQueryExpander(QueryExpander):
    """컨텍스트 기반 쿼리 확장기

    이전 대화 컨텍스트를 고려하여 쿼리를 확장합니다.
    """

    def __init__(
        self,
        config: Optional[ExpansionConfig] = None,
        synonym_dict: Optional[FinancialSynonymDict] = None,
    ):
        super().__init__(config, synonym_dict)
        self._context_terms: list[str] = []

    def set_context(self, context_terms: list[str]) -> None:
        """컨텍스트 용어 설정

        Args:
            context_terms: 컨텍스트에서 추출한 주요 용어
        """
        self._context_terms = context_terms

    def expand_with_context(
        self,
        query: str,
        context: Optional[list[str]] = None,
    ) -> ExpandedQuery:
        """컨텍스트를 고려한 쿼리 확장

        Args:
            query: 원본 쿼리
            context: 컨텍스트 용어 (없으면 저장된 컨텍스트 사용)

        Returns:
            확장된 쿼리
        """
        # 기본 확장 수행
        expanded = self.expand(query)

        # 컨텍스트 용어 추가
        context_terms = context or self._context_terms
        if context_terms:
            # 컨텍스트에서 쿼리와 관련된 용어만 추가
            relevant_context = self._filter_relevant_context(
                query, context_terms
            )
            expanded.expanded_terms.extend(relevant_context)
            expanded.all_terms.extend(relevant_context)

            if relevant_context:
                expanded.expansion_info["_context"] = relevant_context

        return expanded

    def _filter_relevant_context(
        self,
        query: str,
        context_terms: list[str],
    ) -> list[str]:
        """쿼리와 관련된 컨텍스트 용어 필터링"""
        relevant = []
        query_lower = query.lower()

        for term in context_terms:
            # 쿼리에 이미 포함된 용어는 제외
            if term.lower() in query_lower:
                continue

            # 동의어 관계가 있으면 추가
            synonyms = self.synonym_dict.get_synonyms(term)
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    relevant.append(term)
                    break

        return relevant[:3]  # 최대 3개


# 편의 함수
def expand_query(query: str, config: Optional[ExpansionConfig] = None) -> ExpandedQuery:
    """쿼리 확장 (편의 함수)

    Args:
        query: 원본 쿼리
        config: 확장 설정

    Returns:
        확장된 쿼리
    """
    expander = QueryExpander(config)
    return expander.expand(query)


def get_synonyms(term: str) -> set[str]:
    """동의어 조회 (편의 함수)

    Args:
        term: 검색어

    Returns:
        동의어 집합
    """
    synonym_dict = FinancialSynonymDict()
    return synonym_dict.get_synonyms(term)
