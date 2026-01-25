# -*- coding: utf-8 -*-
"""
고급 기능 테스트

Query Expansion, A/B Testing, Fine-tuned Embedding, Multi-modal 테스트
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================================
# Query Expansion Tests
# ============================================================================

class TestExpansionConfig:
    """ExpansionConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        from src.rag.query_expansion import ExpansionConfig

        config = ExpansionConfig()
        assert config.enable_synonyms is True
        assert config.enable_abbreviations is True
        assert config.max_expansions == 5

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        from src.rag.query_expansion import ExpansionConfig

        config = ExpansionConfig(
            enable_synonyms=False,
            max_expansions=10,
        )
        assert config.enable_synonyms is False
        assert config.max_expansions == 10


class TestFinancialSynonymDict:
    """FinancialSynonymDict 테스트"""

    def test_get_synonyms(self):
        """동의어 조회 테스트"""
        from src.rag.query_expansion import FinancialSynonymDict

        synonym_dict = FinancialSynonymDict()

        # PER 동의어
        synonyms = synonym_dict.get_synonyms("PER")
        assert "주가수익비율" in synonyms
        assert "P/E" in synonyms

    def test_get_synonyms_case_insensitive(self):
        """대소문자 무관 조회 테스트"""
        from src.rag.query_expansion import FinancialSynonymDict

        synonym_dict = FinancialSynonymDict()

        synonyms1 = synonym_dict.get_synonyms("per")
        synonyms2 = synonym_dict.get_synonyms("PER")
        assert synonyms1 == synonyms2

    def test_get_abbreviation_expansions(self):
        """약어 확장 테스트"""
        from src.rag.query_expansion import FinancialSynonymDict

        synonym_dict = FinancialSynonymDict()

        expansions = synonym_dict.get_abbreviation_expansions("삼전")
        assert "삼성전자" in expansions

    def test_get_related_terms(self):
        """관련어 조회 테스트"""
        from src.rag.query_expansion import FinancialSynonymDict

        synonym_dict = FinancialSynonymDict()

        related = synonym_dict.get_related_terms("실적")
        assert "매출" in related
        assert "영업이익" in related


class TestQueryExpander:
    """QueryExpander 테스트"""

    def test_expand_simple_query(self):
        """단순 쿼리 확장 테스트"""
        from src.rag.query_expansion import QueryExpander

        expander = QueryExpander()
        result = expander.expand("PER 분석")

        assert result.original == "PER 분석"
        assert len(result.expanded_terms) > 0
        assert "주가수익비율" in result.expanded_terms or "P/E" in result.expanded_terms

    def test_expand_abbreviation(self):
        """약어 확장 테스트"""
        from src.rag.query_expansion import QueryExpander

        expander = QueryExpander()
        result = expander.expand("삼전 주가")

        assert "삼성전자" in result.expanded_terms

    def test_expand_stats(self):
        """확장 통계 테스트"""
        from src.rag.query_expansion import QueryExpander

        expander = QueryExpander()
        expander.expand("PER 분석")
        expander.expand("ROE 비교")

        assert expander.stats["total_expansions"] == 2

    def test_expand_for_hybrid_search(self):
        """하이브리드 검색용 확장 테스트"""
        from src.rag.query_expansion import QueryExpander

        expander = QueryExpander()
        result = expander.expand_for_hybrid_search("삼전 영업이익")

        assert "original" in result
        assert "bm25_query" in result
        assert "vector_query" in result
        assert len(result["bm25_query"]) >= len(result["original"])


class TestExpandedQuery:
    """ExpandedQuery 테스트"""

    def test_to_query_string(self):
        """쿼리 문자열 변환 테스트"""
        from src.rag.query_expansion import ExpandedQuery

        query = ExpandedQuery(
            original="PER 분석",
            expanded_terms=["주가수익비율", "P/E"],
            all_terms=["PER 분석", "주가수익비율", "P/E"],
        )

        query_str = query.to_query_string()
        assert "PER 분석" in query_str
        assert "OR" in query_str


class TestContextualQueryExpander:
    """ContextualQueryExpander 테스트"""

    def test_expand_with_context(self):
        """컨텍스트 기반 확장 테스트"""
        from src.rag.query_expansion import ContextualQueryExpander

        expander = ContextualQueryExpander()
        expander.set_context(["삼성전자", "반도체"])

        result = expander.expand_with_context("영업이익은?")
        assert result.original == "영업이익은?"


# ============================================================================
# A/B Testing Tests
# ============================================================================

class TestExperimentConfig:
    """ExperimentConfig 테스트"""

    def test_is_active(self):
        """활성 상태 확인 테스트"""
        from src.rag.ab_testing import ExperimentConfig

        config = ExperimentConfig(
            experiment_id="exp1",
            name="Test Experiment",
        )
        assert config.is_active() is True

    def test_is_active_ended(self):
        """종료된 실험 테스트"""
        from src.rag.ab_testing import ExperimentConfig

        config = ExperimentConfig(
            experiment_id="exp1",
            name="Test Experiment",
            end_time=datetime.now() - timedelta(hours=1),
        )
        assert config.is_active() is False


class TestVariant:
    """Variant 테스트"""

    def test_create_variant(self):
        """변형 생성 테스트"""
        from src.rag.ab_testing import Variant, VariantType

        variant = Variant(
            variant_id="v1",
            variant_type=VariantType.CONTROL,
            config={"reranker": "keyword"},
        )

        assert variant.variant_id == "v1"
        assert variant.variant_type == VariantType.CONTROL


class TestABTestManager:
    """ABTestManager 테스트"""

    def test_create_experiment(self):
        """실험 생성 테스트"""
        from src.rag.ab_testing import ABTestManager

        manager = ABTestManager()
        config = manager.create_experiment(
            name="Reranker Test",
            control_config={"reranker": "keyword"},
            treatment_config={"reranker": "cross-encoder"},
        )

        assert config.name == "Reranker Test"
        assert config.experiment_id is not None

    def test_get_variant_consistency(self):
        """변형 할당 일관성 테스트"""
        from src.rag.ab_testing import ABTestManager

        manager = ABTestManager()
        config = manager.create_experiment(
            name="Test",
            control_config={},
            treatment_config={},
        )

        # 동일 사용자는 항상 같은 변형 받음
        variant1 = manager.get_variant(config.experiment_id, "user123")
        variant2 = manager.get_variant(config.experiment_id, "user123")

        assert variant1.variant_id == variant2.variant_id

    def test_record_result(self):
        """결과 기록 테스트"""
        from src.rag.ab_testing import ABTestManager

        manager = ABTestManager()
        config = manager.create_experiment(
            name="Test",
            control_config={},
            treatment_config={},
        )

        variant = manager.get_variant(config.experiment_id, "user1")

        result = manager.record_result(
            experiment_id=config.experiment_id,
            variant_id=variant.variant_id,
            user_id="user1",
            query="test query",
            response="test response",
            metrics={"latency_ms": 100},
        )

        assert result.experiment_id == config.experiment_id
        assert result.metrics["latency_ms"] == 100

    def test_get_summary(self):
        """요약 통계 테스트"""
        from src.rag.ab_testing import ABTestManager

        manager = ABTestManager()
        config = manager.create_experiment(
            name="Test",
            control_config={},
            treatment_config={},
            traffic_split=0.5,
        )

        # 여러 결과 기록
        for i in range(10):
            variant = manager.get_variant(config.experiment_id, f"user{i}")
            manager.record_result(
                experiment_id=config.experiment_id,
                variant_id=variant.variant_id,
                user_id=f"user{i}",
                query=f"query{i}",
                response=f"response{i}",
                metrics={"latency_ms": 100 + i * 10},
            )

        summary = manager.get_summary(config.experiment_id)
        assert summary.total_requests == 10

    def test_list_experiments(self):
        """실험 목록 조회 테스트"""
        from src.rag.ab_testing import ABTestManager

        manager = ABTestManager()
        manager.create_experiment("Exp1", {}, {})
        manager.create_experiment("Exp2", {}, {})

        experiments = manager.list_experiments()
        assert len(experiments) == 2

    def test_stop_experiment(self):
        """실험 중지 테스트"""
        from src.rag.ab_testing import ABTestManager

        manager = ABTestManager()
        config = manager.create_experiment("Test", {}, {})

        assert config.is_active() is True
        manager.stop_experiment(config.experiment_id)
        assert config.is_active() is False


class TestRAGExperiment:
    """RAGExperiment 테스트"""

    def test_create_from_template(self):
        """템플릿 기반 실험 생성 테스트"""
        from src.rag.ab_testing import RAGExperiment

        rag_exp = RAGExperiment()
        config = rag_exp.create_from_template("reranker_comparison")

        assert config is not None
        assert config.name == "Re-ranker 비교"

    def test_run_with_experiment(self):
        """실험 적용 실행 테스트"""
        from src.rag.ab_testing import RAGExperiment

        rag_exp = RAGExperiment()
        config = rag_exp.create_from_template("query_expansion")

        def control_fn(q):
            return f"control: {q}"

        def treatment_fn(q):
            return f"treatment: {q}"

        response, metadata = rag_exp.run_with_experiment(
            experiment_id=config.experiment_id,
            user_id="test_user",
            query="test query",
            control_fn=control_fn,
            treatment_fn=treatment_fn,
        )

        assert response is not None
        assert "variant" in metadata


# ============================================================================
# Fine-tuned Embedding Tests
# ============================================================================

class TestFineTuneConfig:
    """FineTuneConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        from src.rag.fine_tuned_embedding import FineTuneConfig

        config = FineTuneConfig()
        assert config.epochs == 3
        assert config.batch_size == 16

    def test_finance_models(self):
        """금융 모델 설정 테스트"""
        from src.rag.fine_tuned_embedding import FineTuneConfig

        assert "ko-sroberta" in FineTuneConfig.FINANCE_MODELS
        assert "bge-m3" in FineTuneConfig.FINANCE_MODELS


class TestTrainingDataset:
    """TrainingDataset 테스트"""

    def test_add_pair(self):
        """학습 쌍 추가 테스트"""
        from src.rag.fine_tuned_embedding import TrainingDataset

        dataset = TrainingDataset(name="test")
        dataset.add_pair(
            query="삼성전자 실적",
            positive="삼성전자 영업이익 증가",
        )

        assert len(dataset) == 1
        assert dataset.pairs[0].query == "삼성전자 실적"

    def test_to_json(self):
        """JSON 저장 테스트"""
        from src.rag.fine_tuned_embedding import TrainingDataset

        dataset = TrainingDataset(name="test")
        dataset.add_pair("q1", "p1")
        dataset.add_pair("q2", "p2")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            dataset.to_json(f.name)

            with open(f.name, 'r', encoding='utf-8') as rf:
                data = json.load(rf)

            assert data["name"] == "test"
            assert len(data["pairs"]) == 2

    def test_from_json(self):
        """JSON 로드 테스트"""
        from src.rag.fine_tuned_embedding import TrainingDataset

        # 테스트 데이터 생성
        data = {
            "name": "test",
            "description": "test description",
            "pairs": [
                {"query": "q1", "positive": "p1", "score": 1.0},
            ],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            temp_path = f.name

        try:
            dataset = TrainingDataset.from_json(temp_path)
            assert dataset.name == "test"
            assert len(dataset) == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestFinanceDatasetGenerator:
    """FinanceDatasetGenerator 테스트"""

    def test_generate_dataset(self):
        """데이터셋 생성 테스트"""
        from src.rag.fine_tuned_embedding import FinanceDatasetGenerator

        generator = FinanceDatasetGenerator()
        dataset = generator.generate_dataset(num_pairs=50)

        assert len(dataset) == 50
        assert dataset.pairs[0].query is not None

    def test_generate_with_negatives(self):
        """네거티브 샘플 포함 생성 테스트"""
        from src.rag.fine_tuned_embedding import FinanceDatasetGenerator

        generator = FinanceDatasetGenerator()
        dataset = generator.generate_dataset(
            num_pairs=10,
            include_negatives=True,
        )

        # 일부 샘플에 네거티브 있음
        has_negative = any(p.negative for p in dataset.pairs)
        assert has_negative


class TestFinanceEmbeddingModel:
    """FinanceEmbeddingModel 테스트"""

    def test_encode(self):
        """임베딩 생성 테스트"""
        from src.rag.fine_tuned_embedding import FinanceEmbeddingModel

        model = FinanceEmbeddingModel()
        embeddings = model.encode(["삼성전자 실적", "SK하이닉스 반도체"])

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_similarity(self):
        """유사도 계산 테스트"""
        from src.rag.fine_tuned_embedding import FinanceEmbeddingModel

        model = FinanceEmbeddingModel()
        similarities = model.similarity(
            query="삼성전자 영업이익",
            documents=["삼성전자 실적 발표", "현대차 신차 출시"],
        )

        assert len(similarities) == 2


# ============================================================================
# Multi-modal Tests
# ============================================================================

class TestTableData:
    """TableData 테스트"""

    def test_to_markdown(self):
        """마크다운 변환 테스트"""
        from src.rag.multimodal import TableData

        table = TableData(
            headers=["구분", "금액", "비율"],
            rows=[
                ["매출", "100억", "50%"],
                ["영업이익", "20억", "10%"],
            ],
            caption="재무 요약",
        )

        md = table.to_markdown()
        assert "재무 요약" in md
        assert "| 구분 | 금액 | 비율 |" in md
        assert "매출" in md

    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        from src.rag.multimodal import TableData

        table = TableData(
            headers=["A", "B"],
            rows=[["1", "2"]],
        )

        d = table.to_dict()
        assert d["headers"] == ["A", "B"]
        assert len(d["rows"]) == 1


class TestChartData:
    """ChartData 테스트"""

    def test_to_text(self):
        """텍스트 변환 테스트"""
        from src.rag.multimodal import ChartData

        chart = ChartData(
            chart_type="bar",
            title="매출 추이",
            labels=["Q1", "Q2", "Q3"],
            values=[100, 120, 150],
        )

        text = chart.to_text()
        assert "bar" in text
        assert "매출 추이" in text
        assert "Q1" in text


class TestTableExtractor:
    """TableExtractor 테스트"""

    def test_extract_from_html(self):
        """HTML 표 추출 테스트"""
        from src.rag.multimodal import TableExtractor

        extractor = TableExtractor()

        html = """
        <table>
            <tr><th>구분</th><th>금액</th></tr>
            <tr><td>매출</td><td>100억</td></tr>
            <tr><td>영업이익</td><td>20억</td></tr>
        </table>
        """

        tables = extractor.extract_from_html(html)
        assert len(tables) >= 1
        assert "구분" in tables[0].headers

    def test_extract_from_text_pipe(self):
        """파이프 구분 표 추출 테스트"""
        from src.rag.multimodal import TableExtractor

        extractor = TableExtractor()

        text = """
| 구분 | 금액 | 비율 |
|------|------|------|
| 매출 | 100억 | 50% |
| 영업이익 | 20억 | 10% |
        """

        tables = extractor.extract_from_text(text)
        assert len(tables) >= 1

    def test_is_financial_table(self):
        """재무제표 여부 확인 테스트"""
        from src.rag.multimodal import TableExtractor, TableData

        extractor = TableExtractor()

        financial_table = TableData(
            headers=["구분", "당기", "전기"],
            rows=[
                ["매출액", "100억", "90억"],
                ["영업이익", "20억", "15억"],
            ],
        )

        non_financial_table = TableData(
            headers=["이름", "나이", "직업"],
            rows=[["홍길동", "30", "개발자"]],
        )

        assert extractor.is_financial_table(financial_table) is True
        assert extractor.is_financial_table(non_financial_table) is False


class TestMultiModalProcessor:
    """MultiModalProcessor 테스트"""

    def test_process_document_text(self):
        """텍스트 문서 처리 테스트"""
        from src.rag.multimodal import MultiModalProcessor

        processor = MultiModalProcessor()

        content = """
삼성전자 2024년 1분기 실적 발표

| 구분 | 금액 |
|------|------|
| 매출 | 100조 |
| 영업이익 | 10조 |

반도체 부문이 실적을 견인했습니다.
        """

        extracted = processor.process_document(content, "text")

        # 표와 텍스트 추출
        assert len(extracted) >= 1

    def test_process_document_html(self):
        """HTML 문서 처리 테스트"""
        from src.rag.multimodal import MultiModalProcessor

        processor = MultiModalProcessor()

        html = """
<h1>실적 요약</h1>
<table>
    <tr><th>항목</th><th>금액</th></tr>
    <tr><td>매출</td><td>100억</td></tr>
</table>
<p>좋은 실적입니다.</p>
        """

        extracted = processor.process_document(html, "html")
        assert len(extracted) >= 1

    def test_to_searchable_text(self):
        """검색용 텍스트 변환 테스트"""
        from src.rag.multimodal import MultiModalProcessor, ExtractedContent, ContentType

        processor = MultiModalProcessor()

        contents = [
            ExtractedContent(
                content_type=ContentType.TEXT,
                content="삼성전자 실적 발표",
            ),
            ExtractedContent(
                content_type=ContentType.TABLE,
                content={"headers": ["A", "B"], "rows": [["1", "2"]]},
            ),
        ]

        text = processor.to_searchable_text(contents)
        assert "삼성전자" in text

    def test_stats(self):
        """통계 테스트"""
        from src.rag.multimodal import MultiModalProcessor

        processor = MultiModalProcessor()
        processor.process_document("테스트 | 내용\n------|------\n값 | 1", "text")

        assert processor.stats["processed_count"] == 1


class TestExtractedContent:
    """ExtractedContent 테스트"""

    def test_to_text_table(self):
        """표 텍스트 변환 테스트"""
        from src.rag.multimodal import ExtractedContent, ContentType

        content = ExtractedContent(
            content_type=ContentType.TABLE,
            content=[["A", "B"], ["1", "2"]],
        )

        text = content.to_text()
        assert "A | B" in text

    def test_to_text_chart(self):
        """차트 텍스트 변환 테스트"""
        from src.rag.multimodal import ExtractedContent, ContentType

        content = ExtractedContent(
            content_type=ContentType.CHART,
            content={
                "title": "매출 추이",
                "type": "bar",
            },
        )

        text = content.to_text()
        assert "매출 추이" in text


class TestMultiModalEmbedding:
    """MultiModalEmbedding 테스트"""

    def test_encode_text(self):
        """텍스트 임베딩 테스트"""
        from src.rag.multimodal import MultiModalEmbedding

        embedding = MultiModalEmbedding()
        result = embedding.encode_text(["삼성전자", "SK하이닉스"])

        assert result.shape[0] == 2

    def test_encode_multimodal(self):
        """멀티모달 임베딩 테스트"""
        from src.rag.multimodal import MultiModalEmbedding

        embedding = MultiModalEmbedding()
        result = embedding.encode_multimodal(
            texts=["테스트 텍스트"],
            images=[None],
        )

        assert result.shape[0] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_query_expansion_with_ab_test(self):
        """Query Expansion + A/B 테스트 통합"""
        from src.rag.query_expansion import QueryExpander
        from src.rag.ab_testing import RAGExperiment

        expander = QueryExpander()
        rag_exp = RAGExperiment()

        # A/B 실험 설정
        config = rag_exp.create_from_template("query_expansion")

        def control_fn(query):
            return {"query": query, "expanded": False}

        def treatment_fn(query):
            expanded = expander.expand(query)
            return {"query": expanded.to_query_string(), "expanded": True}

        response, metadata = rag_exp.run_with_experiment(
            experiment_id=config.experiment_id,
            user_id="test_user",
            query="PER 분석",
            control_fn=control_fn,
            treatment_fn=treatment_fn,
        )

        assert response is not None

    def test_multimodal_with_embedding(self):
        """Multi-modal + Embedding 통합"""
        from src.rag.multimodal import MultiModalProcessor
        from src.rag.fine_tuned_embedding import FinanceEmbeddingModel

        processor = MultiModalProcessor()
        model = FinanceEmbeddingModel()

        # 문서 처리
        content = "삼성전자 매출 100조원 달성"
        extracted = processor.process_document(content, "text")

        # 임베딩 생성
        text = processor.to_searchable_text(extracted)
        embeddings = model.encode([text])

        assert embeddings.shape[0] == 1


class TestModuleImports:
    """모듈 임포트 테스트"""

    def test_import_query_expansion(self):
        """Query Expansion 임포트 테스트"""
        from src.rag.query_expansion import (
            ExpansionConfig,
            FinancialSynonymDict,
            QueryExpander,
            ExpandedQuery,
            expand_query,
            get_synonyms,
        )
        assert ExpansionConfig is not None

    def test_import_ab_testing(self):
        """A/B Testing 임포트 테스트"""
        from src.rag.ab_testing import (
            ExperimentConfig,
            Variant,
            ABTestManager,
            RAGExperiment,
            create_experiment,
            get_variant,
        )
        assert ABTestManager is not None

    def test_import_fine_tuned_embedding(self):
        """Fine-tuned Embedding 임포트 테스트"""
        from src.rag.fine_tuned_embedding import (
            FineTuneConfig,
            TrainingDataset,
            FinanceDatasetGenerator,
            FinanceEmbeddingModel,
            load_finance_embedding,
            generate_training_data,
        )
        assert FinanceEmbeddingModel is not None

    def test_import_multimodal(self):
        """Multi-modal 임포트 테스트"""
        from src.rag.multimodal import (
            ContentType,
            ExtractedContent,
            TableData,
            ChartData,
            TableExtractor,
            MultiModalProcessor,
            extract_tables,
            process_document,
        )
        assert MultiModalProcessor is not None
