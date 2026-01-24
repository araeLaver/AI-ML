"""
쿼리 확장 모듈 테스트

테스트 항목:
1. 금융 동의어 사전 통계 및 무결성
2. 쿼리 확장 기능
3. 쿼리 정규화 기능
4. 엣지 케이스 처리
"""

import pytest
from src.rag.financial_dictionary import (
    FINANCIAL_SYNONYMS,
    REVERSE_SYNONYMS,
    FINANCIAL_METRICS,
    COMPANY_ABBREVIATIONS,
    INDUSTRY_TERMS,
    get_synonyms,
    get_canonical_term,
    get_statistics,
    get_all_terms,
)
from src.rag.query_expander import (
    QueryExpander,
    MultiTermExpander,
    expand_query,
    normalize_query,
    ExpansionResult,
)


# =============================================================================
# 금융 동의어 사전 테스트
# =============================================================================

class TestFinancialDictionary:
    """금융 동의어 사전 테스트"""

    def test_dictionary_not_empty(self):
        """사전이 비어있지 않은지 확인"""
        assert len(FINANCIAL_SYNONYMS) > 0
        assert len(REVERSE_SYNONYMS) > 0

    def test_dictionary_size(self):
        """사전 크기가 최소 요구사항 충족하는지 확인"""
        stats = get_statistics()
        assert stats['total_canonical_terms'] >= 100, "최소 100개 이상의 표준 용어 필요"
        assert stats['total_synonyms'] >= 200, "최소 200개 이상의 동의어 필요"

    def test_category_coverage(self):
        """모든 카테고리가 포함되어 있는지 확인"""
        stats = get_statistics()
        categories = stats['categories']

        assert 'metrics' in categories
        assert 'companies' in categories
        assert 'industry' in categories
        assert 'disclosure' in categories
        assert 'market' in categories
        assert 'time' in categories

        # 각 카테고리에 최소 5개 이상
        for cat, count in categories.items():
            assert count >= 5, f"{cat} 카테고리에 최소 5개 이상 필요"

    def test_financial_metrics_included(self):
        """주요 재무 지표가 포함되어 있는지 확인"""
        required_metrics = ["PER", "PBR", "ROE", "ROA", "EPS", "EBITDA"]

        for metric in required_metrics:
            assert metric in FINANCIAL_METRICS, f"{metric} 지표가 누락됨"
            assert len(FINANCIAL_METRICS[metric]) >= 1, f"{metric}에 동의어가 없음"

    def test_company_abbreviations_included(self):
        """주요 기업 약어가 포함되어 있는지 확인"""
        required_companies = ["삼전", "하닉", "현차"]

        for company in required_companies:
            assert company in COMPANY_ABBREVIATIONS, f"{company} 기업이 누락됨"

    def test_reverse_mapping_consistency(self):
        """역방향 매핑이 일관성 있는지 확인"""
        for canonical, synonyms in FINANCIAL_SYNONYMS.items():
            # 표준 용어 자체도 역방향에 있어야 함
            assert canonical.lower() in REVERSE_SYNONYMS

            # 모든 동의어가 역방향에 있어야 함
            for syn in synonyms:
                assert syn.lower() in REVERSE_SYNONYMS, f"{syn}이 역방향 매핑에 없음"
                assert REVERSE_SYNONYMS[syn.lower()] == canonical


class TestGetSynonyms:
    """get_synonyms 함수 테스트"""

    def test_get_synonyms_forward(self):
        """정방향 검색 (약어 → 풀네임)"""
        synonyms = get_synonyms("PER")
        assert "주가수익비율" in synonyms
        assert "PER" in synonyms

    def test_get_synonyms_reverse(self):
        """역방향 검색 (풀네임 → 약어)"""
        synonyms = get_synonyms("주가수익비율")
        assert "PER" in synonyms
        assert "주가수익비율" in synonyms

    def test_get_synonyms_case_insensitive(self):
        """대소문자 무관 검색 - 핵심 동의어 포함 확인"""
        synonyms_upper = get_synonyms("PER")
        synonyms_lower = get_synonyms("per")

        # 핵심 동의어가 둘 다 포함되어야 함
        assert "주가수익비율" in synonyms_upper
        assert "주가수익비율" in synonyms_lower
        assert "PER" in synonyms_upper
        assert "PER" in synonyms_lower

    def test_get_synonyms_unknown_term(self):
        """알 수 없는 용어"""
        synonyms = get_synonyms("알수없는용어")
        assert synonyms == {"알수없는용어"}

    def test_get_canonical_term(self):
        """표준 용어 반환"""
        assert get_canonical_term("주가수익비율") == "PER"
        assert get_canonical_term("삼전") == "삼전"  # 이미 표준
        assert get_canonical_term("삼성전자") == "삼전"  # 동의어 → 표준


# =============================================================================
# 쿼리 확장 테스트
# =============================================================================

class TestQueryExpander:
    """QueryExpander 클래스 테스트"""

    @pytest.fixture
    def expander(self):
        """기본 쿼리 확장기"""
        return QueryExpander()

    def test_expand_single_term(self, expander):
        """단일 용어 확장"""
        result = expander.expand("PER 높은 기업")

        assert isinstance(result, ExpansionResult)
        assert result.original_query == "PER 높은 기업"
        assert len(result.expanded_queries) >= 2  # 원본 + 확장
        assert "PER 높은 기업" in result.expanded_queries
        # 확장이 실제로 발생했는지 확인
        assert result.num_expansions >= 1
        # expansions에서 PER이 확장되었는지 확인
        assert any(exp.get("term") == "PER" for exp in result.expansions)

    def test_expand_company_abbreviation(self, expander):
        """기업 약어 확장"""
        result = expander.expand("삼전 실적")

        assert any("삼성전자" in q for q in result.expanded_queries)

    def test_expand_industry_term(self, expander):
        """산업 용어 확장"""
        result = expander.expand("HBM 관련주")

        assert any("고대역폭메모리" in q for q in result.expanded_queries)

    def test_expand_no_financial_term(self, expander):
        """금융 용어 없는 쿼리"""
        result = expander.expand("오늘 날씨 어때")

        assert len(result.expanded_queries) == 1
        assert result.expanded_queries[0] == "오늘 날씨 어때"
        assert result.num_expansions == 0

    def test_expand_already_standard(self, expander):
        """이미 표준 용어인 경우"""
        result = expander.expand("삼성전자 영업이익")

        # 표준 용어도 역방향 확장 가능
        assert len(result.expanded_queries) >= 1

    def test_expand_simple(self, expander):
        """간단한 확장 (리스트만 반환)"""
        queries = expander.expand_simple("PER 분석")

        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_max_expansions_limit(self):
        """최대 확장 수 제한"""
        expander = QueryExpander(max_total_queries=3)
        result = expander.expand("삼전 PER ROE 분석")

        assert len(result.expanded_queries) <= 3

    def test_include_original_false(self):
        """원본 쿼리 제외 옵션"""
        expander = QueryExpander(include_original=False)
        result = expander.expand("PER 높은 기업")

        if result.num_expansions > 0:
            assert "PER 높은 기업" not in result.expanded_queries


class TestQueryNormalization:
    """쿼리 정규화 테스트"""

    @pytest.fixture
    def expander(self):
        return QueryExpander()

    def test_normalize_abbreviation(self, expander):
        """약어 정규화"""
        # 삼전 → 삼전 (이미 표준)
        normalized = expander.normalize("삼전 실적")
        assert "삼전" in normalized

    def test_normalize_full_name_to_abbr(self, expander):
        """풀네임 → 약어 정규화"""
        normalized = expander.normalize("주가수익비율 분석")
        assert "PER" in normalized

    def test_normalize_no_change(self, expander):
        """변경 없는 쿼리"""
        query = "오늘 날씨"
        normalized = expander.normalize(query)
        assert normalized == query


class TestMultiTermExpander:
    """다중 용어 확장기 테스트"""

    @pytest.fixture
    def expander(self):
        return MultiTermExpander()

    def test_multi_term_expansion(self, expander):
        """다중 용어 확장"""
        result = expander.expand("삼전 PER ROE 분석")

        assert len(result.expanded_queries) >= 1
        assert len(result.expanded_queries) <= 5  # 기본 최대값


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    def test_expand_query_function(self):
        """expand_query 함수"""
        queries = expand_query("PER 분석")

        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_normalize_query_function(self):
        """normalize_query 함수"""
        normalized = normalize_query("주가수익비율 분석")

        assert isinstance(normalized, str)


# =============================================================================
# 엣지 케이스 테스트
# =============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    @pytest.fixture
    def expander(self):
        return QueryExpander()

    def test_empty_query(self, expander):
        """빈 쿼리"""
        result = expander.expand("")

        assert result.original_query == ""
        assert len(result.expanded_queries) == 1

    def test_special_characters(self, expander):
        """특수문자 포함 쿼리"""
        result = expander.expand("PER(주가수익비율) 분석!")

        assert result.original_query == "PER(주가수익비율) 분석!"

    def test_mixed_case(self, expander):
        """대소문자 혼합"""
        result = expander.expand("per 분석")

        # 소문자 per도 인식해야 함
        assert len(result.expanded_queries) >= 1

    def test_unicode_handling(self, expander):
        """유니코드 처리"""
        result = expander.expand("삼성전자 2024년 실적")

        assert "삼성전자" in result.original_query

    def test_long_query(self, expander):
        """긴 쿼리"""
        long_query = "삼성전자 2024년 1분기 실적 발표에서 영업이익과 순이익이 어떻게 나왔는지 알려주세요"
        result = expander.expand(long_query)

        assert result.original_query == long_query

    def test_duplicate_terms(self, expander):
        """중복 용어"""
        result = expander.expand("PER PER 분석")

        # 중복 확장 없어야 함
        unique_queries = set(result.expanded_queries)
        assert len(unique_queries) == len(result.expanded_queries)


# =============================================================================
# 성능 테스트
# =============================================================================

class TestPerformance:
    """성능 테스트"""

    def test_expansion_speed(self):
        """확장 속도 테스트"""
        import time

        expander = QueryExpander()
        queries = [
            "PER 높은 기업",
            "삼전 실적",
            "HBM 관련주",
            "2차전지 시장 전망",
            "영업이익 증가 기업",
        ] * 100  # 500개 쿼리

        start = time.time()
        for query in queries:
            expander.expand(query)
        elapsed = time.time() - start

        # 500개 쿼리가 1초 이내에 처리되어야 함
        assert elapsed < 1.0, f"확장 속도가 너무 느림: {elapsed:.2f}초"

    def test_dictionary_lookup_speed(self):
        """사전 조회 속도 테스트"""
        import time

        terms = list(FINANCIAL_SYNONYMS.keys()) * 100  # 조회 횟수

        start = time.time()
        for term in terms:
            get_synonyms(term)
        elapsed = time.time() - start

        # 빠른 조회 확인
        assert elapsed < 0.5, f"사전 조회가 너무 느림: {elapsed:.2f}초"
