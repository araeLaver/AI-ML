# -*- coding: utf-8 -*-
"""
임베딩 트레이너 및 평가기 테스트

[테스트 범위]
- TrainingExample: 학습 데이터 구조
- TrainingConfig: 학습 설정
- TrainingDataGenerator: 학습 데이터 생성
- FinancialEmbeddingTrainer: 모델 학습 (모킹)
- EmbeddingEvaluator: 평가 메트릭
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List
import tempfile
import json
from pathlib import Path

# 테스트 대상 모듈
from src.rag.embedding_trainer import (
    TrainingExample,
    TrainingConfig,
    TrainingDataGenerator,
    FinancialEmbeddingTrainer,
)
from src.rag.embedding_evaluator import (
    EvaluationResult,
    EmbeddingEvaluator,
    FinancialEvaluationDataset,
)


# =============================================================================
# TrainingExample 테스트
# =============================================================================

class TestTrainingExample:
    """TrainingExample 데이터클래스 테스트"""

    def test_create_training_example(self):
        """학습 예제 생성"""
        example = TrainingExample(
            query="삼성전자 실적",
            positive="삼성전자의 2024년 영업이익은 6조원입니다.",
            negative="현대차 판매량이 증가했습니다.",
            metadata={"type": "query_document"}
        )

        assert example.query == "삼성전자 실적"
        assert example.positive is not None
        assert example.negative is not None
        assert example.metadata["type"] == "query_document"

    def test_training_example_without_negative(self):
        """네거티브 없는 학습 예제"""
        example = TrainingExample(
            query="PER",
            positive="주가수익비율",
        )

        assert example.negative is None
        assert example.metadata == {}

    def test_training_example_default_metadata(self):
        """기본 메타데이터 확인"""
        example = TrainingExample(
            query="ROE",
            positive="자기자본이익률",
        )

        assert isinstance(example.metadata, dict)


# =============================================================================
# TrainingConfig 테스트
# =============================================================================

class TestTrainingConfig:
    """학습 설정 테스트"""

    def test_default_config(self):
        """기본 설정 확인"""
        config = TrainingConfig()

        assert config.base_model == "intfloat/multilingual-e5-base"
        assert config.epochs == 3
        assert config.batch_size == 16
        assert config.use_lora is True

    def test_custom_config(self):
        """커스텀 설정"""
        config = TrainingConfig(
            base_model="custom-model",
            epochs=5,
            batch_size=32,
            use_lora=False,
            output_dir="custom_output"
        )

        assert config.base_model == "custom-model"
        assert config.epochs == 5
        assert config.batch_size == 32
        assert config.use_lora is False
        assert config.output_dir == "custom_output"

    def test_config_to_dict(self):
        """딕셔너리 변환"""
        config = TrainingConfig()
        config_dict = config.to_dict()

        assert "base_model" in config_dict
        assert "epochs" in config_dict
        assert "batch_size" in config_dict
        assert "use_lora" in config_dict


# =============================================================================
# TrainingDataGenerator 테스트
# =============================================================================

class TestTrainingDataGenerator:
    """학습 데이터 생성기 테스트"""

    @pytest.fixture
    def generator(self, tmp_path):
        """테스트용 생성기"""
        return TrainingDataGenerator(data_dir=str(tmp_path))

    def test_generate_synonym_pairs(self, generator):
        """동의어 쌍 생성"""
        examples = generator.generate_synonym_pairs()

        assert len(examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in examples)
        assert all(ex.metadata.get("type") == "synonym" for ex in examples)

    def test_synonym_pairs_have_query_positive(self, generator):
        """동의어 쌍은 query와 positive 필수"""
        examples = generator.generate_synonym_pairs()

        for ex in examples:
            assert ex.query is not None
            assert len(ex.query) > 0
            assert ex.positive is not None
            assert len(ex.positive) > 0

    def test_generate_financial_context_pairs(self, generator):
        """금융 컨텍스트 쌍 생성"""
        examples = generator.generate_financial_context_pairs()

        assert len(examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in examples)
        assert all(ex.metadata.get("type") == "curated" for ex in examples)

    def test_generate_all_training_data(self, generator):
        """전체 학습 데이터 생성"""
        examples = generator.generate_all(include_hard_negatives=False)

        # 최소 개수 확인
        assert len(examples) >= 10

        # 다양한 타입 포함 확인
        types = set(ex.metadata.get("type") for ex in examples)
        assert "synonym" in types
        assert "curated" in types

    def test_generate_hard_negatives(self, generator):
        """Hard Negative 추가"""
        # 간단한 예시 데이터
        examples = [
            TrainingExample(query="q1", positive="p1"),
            TrainingExample(query="q2", positive="p2"),
            TrainingExample(query="q3", positive="p3"),
        ]

        enhanced = generator.generate_hard_negatives(examples)

        # Hard negative가 추가됨
        assert len(enhanced) == len(examples)
        assert any(ex.negative is not None for ex in enhanced)

    def test_save_and_load_training_data(self, generator):
        """학습 데이터 저장 및 로드"""
        examples = [
            TrainingExample(query="PER", positive="주가수익비율", metadata={"type": "synonym"}),
            TrainingExample(query="ROE", positive="자기자본이익률", metadata={"type": "synonym"}),
        ]

        # 저장
        output_path = generator.save_training_data(examples, "test_data.jsonl")
        assert Path(output_path).exists()

        # 로드
        loaded = generator.load_training_data("test_data.jsonl")
        assert len(loaded) == 2
        assert loaded[0].query == "PER"
        assert loaded[1].query == "ROE"


# =============================================================================
# FinancialEmbeddingTrainer 테스트 (모킹)
# =============================================================================

class TestFinancialEmbeddingTrainer:
    """임베딩 트레이너 테스트"""

    def test_trainer_initialization(self):
        """트레이너 초기화"""
        config = TrainingConfig(use_lora=False)
        trainer = FinancialEmbeddingTrainer(config=config)

        assert trainer is not None
        assert trainer.config.base_model == "intfloat/multilingual-e5-base"

    def test_trainer_with_custom_config(self):
        """커스텀 설정으로 트레이너 생성"""
        config = TrainingConfig(
            base_model="custom-model",
            output_dir="./custom_output",
            use_lora=True,
            lora_r=16
        )
        trainer = FinancialEmbeddingTrainer(config=config)

        assert trainer.config.output_dir == "./custom_output"
        assert trainer.config.use_lora is True
        assert trainer.config.lora_r == 16

    def test_check_dependencies(self):
        """의존성 확인"""
        trainer = FinancialEmbeddingTrainer()
        # sentence_transformers가 설치되어 있으면 True
        result = trainer._check_dependencies()
        assert isinstance(result, bool)

    def test_supported_models(self):
        """지원 모델 목록"""
        assert len(FinancialEmbeddingTrainer.SUPPORTED_MODELS) > 0
        assert "intfloat/multilingual-e5-base" in FinancialEmbeddingTrainer.SUPPORTED_MODELS


# =============================================================================
# EmbeddingEvaluator 테스트
# =============================================================================

class TestEmbeddingEvaluator:
    """임베딩 평가기 테스트"""

    @pytest.fixture
    def evaluator(self):
        """테스트용 평가기"""
        return EmbeddingEvaluator()

    @pytest.fixture
    def mock_model(self):
        """모킹된 모델"""
        model = MagicMock()
        # 5개 텍스트에 대해 768차원 임베딩 반환
        model.encode.return_value = np.random.randn(5, 768)
        return model

    def test_compute_embeddings(self, evaluator, mock_model):
        """임베딩 계산"""
        texts = ["삼성전자", "SK하이닉스", "현대차", "LG전자", "네이버"]
        embeddings = evaluator.compute_embeddings(mock_model, texts)

        assert embeddings.shape == (5, 768)

    def test_cosine_similarity(self, evaluator):
        """코사인 유사도 계산"""
        query_emb = np.array([[1, 0, 0], [0, 1, 0]])
        doc_emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        similarity = evaluator.cosine_similarity(query_emb, doc_emb)

        assert similarity.shape == (2, 3)
        # 동일 벡터는 유사도 1
        assert np.isclose(similarity[0, 0], 1.0)
        assert np.isclose(similarity[1, 1], 1.0)
        # 직교 벡터는 유사도 0
        assert np.isclose(similarity[0, 2], 0.0)

    def test_compute_mrr_perfect_ranking(self, evaluator):
        """MRR 계산 - 완벽한 랭킹"""
        # 쿼리 i의 관련 문서가 항상 가장 유사한 경우
        similarity_matrix = np.array([
            [1.0, 0.5, 0.3],  # 쿼리 0 → 문서 0이 최고
            [0.2, 1.0, 0.4],  # 쿼리 1 → 문서 1이 최고
            [0.1, 0.3, 1.0],  # 쿼리 2 → 문서 2가 최고
        ])
        relevant_indices = [0, 1, 2]

        mrr = evaluator.compute_mrr(similarity_matrix, relevant_indices)
        assert mrr == 1.0

    def test_compute_mrr_imperfect_ranking(self, evaluator):
        """MRR 계산 - 불완전한 랭킹"""
        # 쿼리 0의 관련 문서가 2위
        similarity_matrix = np.array([
            [0.5, 1.0, 0.3],  # 쿼리 0 → 문서 1이 최고, 문서 0은 2위
            [0.2, 1.0, 0.4],  # 쿼리 1 → 문서 1이 최고
        ])
        relevant_indices = [0, 1]

        mrr = evaluator.compute_mrr(similarity_matrix, relevant_indices)
        # (1/2 + 1/1) / 2 = 0.75
        assert np.isclose(mrr, 0.75)

    def test_compute_recall_at_k(self, evaluator):
        """Recall@K 계산"""
        similarity_matrix = np.array([
            [1.0, 0.5, 0.3, 0.1],
            [0.2, 1.0, 0.4, 0.3],
            [0.1, 0.3, 0.2, 1.0],
        ])
        relevant_indices = [0, 1, 3]

        # Recall@1: 모든 관련 문서가 1위
        recall_1 = evaluator.compute_recall_at_k(similarity_matrix, relevant_indices, k=1)
        assert recall_1 == 1.0

        # Recall@3
        recall_3 = evaluator.compute_recall_at_k(similarity_matrix, relevant_indices, k=3)
        assert recall_3 == 1.0

    def test_compute_ndcg_at_k(self, evaluator):
        """NDCG@K 계산"""
        similarity_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.2, 1.0, 0.4],
        ])
        relevant_indices = [0, 1]

        ndcg = evaluator.compute_ndcg_at_k(similarity_matrix, relevant_indices, k=3)
        # 완벽한 랭킹이므로 NDCG = 1.0
        assert ndcg == 1.0

    def test_compute_synonym_similarity(self, evaluator):
        """동의어 유사도 계산"""
        mock_model = MagicMock()
        # 동의어 쌍에 대해 높은 유사도를 반환하도록 설정
        def encode_side_effect(texts, **kwargs):
            # 간단히 동일한 벡터 반환 (유사도 = 1)
            return np.ones((len(texts), 768))

        mock_model.encode.side_effect = encode_side_effect

        synonym_pairs = [("PER", "주가수익비율"), ("ROE", "자기자본이익률")]
        similarity = evaluator.compute_synonym_similarity(mock_model, synonym_pairs)

        # 부동소수점 비교
        assert np.isclose(similarity, 1.0)

    def test_evaluate_model(self, evaluator, mock_model):
        """모델 종합 평가"""
        # 적절한 임베딩 반환 설정
        def encode_side_effect(texts, **kwargs):
            n = len(texts)
            # 각 텍스트에 대해 다른 임베딩 생성
            embeddings = np.eye(n, 768)  # 단위 행렬 기반
            return embeddings

        mock_model.encode.side_effect = encode_side_effect

        queries = ["쿼리1", "쿼리2", "쿼리3"]
        documents = ["문서1", "문서2", "문서3"]

        result = evaluator.evaluate_model(
            mock_model, queries, documents,
            relevant_indices=[0, 1, 2]
        )

        assert isinstance(result, EvaluationResult)
        assert result.num_queries == 3
        assert 0 <= result.mrr <= 1
        assert 0 <= result.recall_at_1 <= 1
        assert 0 <= result.recall_at_5 <= 1

    def test_evaluation_result_to_dict(self):
        """평가 결과 딕셔너리 변환"""
        result = EvaluationResult(
            mrr=0.85,
            recall_at_1=0.75,
            recall_at_5=0.90,
            recall_at_10=0.95,
            ndcg_at_10=0.88,
            synonym_similarity=0.92,
            num_queries=100,
            details={}
        )

        result_dict = result.to_dict()

        assert result_dict["mrr"] == 0.85
        assert result_dict["recall@1"] == 0.75
        assert result_dict["recall@5"] == 0.90
        assert result_dict["recall@10"] == 0.95
        assert result_dict["ndcg@10"] == 0.88
        assert result_dict["synonym_similarity"] == 0.92


# =============================================================================
# FinancialEvaluationDataset 테스트
# =============================================================================

class TestFinancialEvaluationDataset:
    """금융 평가 데이터셋 테스트"""

    def test_get_synonym_test_pairs(self):
        """동의어 테스트 쌍 조회"""
        pairs = FinancialEvaluationDataset.get_synonym_test_pairs()

        assert len(pairs) > 0
        assert all(isinstance(pair, tuple) for pair in pairs)
        assert all(len(pair) == 2 for pair in pairs)

        # 주요 금융 용어 포함 확인
        terms = [pair[0] for pair in pairs]
        assert "PER" in terms
        assert "ROE" in terms

    def test_get_query_document_test_pairs(self):
        """쿼리-문서 테스트 쌍 조회"""
        pairs = FinancialEvaluationDataset.get_query_document_test_pairs()

        assert len(pairs) > 0
        assert all(isinstance(pair, tuple) for pair in pairs)
        assert all(len(pair) == 2 for pair in pairs)

        # 쿼리와 문서 확인
        for query, doc in pairs:
            assert len(query) > 0
            assert len(doc) > 0


# =============================================================================
# 모델 비교 테스트
# =============================================================================

class TestModelComparison:
    """모델 비교 테스트"""

    def test_compare_models(self):
        """베이스라인 vs Fine-tuned 모델 비교"""
        evaluator = EmbeddingEvaluator()

        # 모킹된 모델들
        baseline_model = MagicMock()
        finetuned_model = MagicMock()

        # 베이스라인: 낮은 유사도
        def baseline_encode(texts, **kwargs):
            n = len(texts)
            return np.random.randn(n, 768) * 0.1

        # Fine-tuned: 높은 유사도 (대각 행렬)
        def finetuned_encode(texts, **kwargs):
            n = len(texts)
            return np.eye(n, 768)

        baseline_model.encode.side_effect = baseline_encode
        finetuned_model.encode.side_effect = finetuned_encode

        queries = ["쿼리1", "쿼리2"]
        documents = ["문서1", "문서2"]

        comparison = evaluator.compare_models(
            baseline_model, finetuned_model,
            queries, documents,
            relevant_indices=[0, 1]
        )

        assert "baseline" in comparison
        assert "finetuned" in comparison
        assert "improvement_percent" in comparison


# =============================================================================
# 엣지 케이스 테스트
# =============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_synonym_pairs(self):
        """빈 동의어 쌍"""
        evaluator = EmbeddingEvaluator()
        mock_model = MagicMock()

        similarity = evaluator.compute_synonym_similarity(mock_model, [])
        assert similarity == 0.0

    def test_single_query_evaluation(self):
        """단일 쿼리 평가"""
        evaluator = EmbeddingEvaluator()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1, 0, 0]])

        result = evaluator.evaluate_model(
            mock_model,
            queries=["쿼리"],
            documents=["문서"],
            relevant_indices=[0]
        )

        assert result.num_queries == 1

    def test_training_example_with_negative(self):
        """네거티브 샘플이 있는 학습 예제"""
        example = TrainingExample(
            query="삼성전자",
            positive="삼성전자는 반도체 기업입니다.",
            negative="현대차는 자동차 기업입니다.",
            metadata={"type": "query_document"}
        )

        assert example.negative is not None
        assert "현대차" in example.negative


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_data_generation_pipeline(self, tmp_path):
        """전체 데이터 생성 파이프라인"""
        generator = TrainingDataGenerator(data_dir=str(tmp_path))

        # 전체 데이터 생성
        examples = generator.generate_all(include_hard_negatives=True)
        assert len(examples) > 0

        # 저장
        output_path = generator.save_training_data(examples)
        assert Path(output_path).exists()

        # 로드
        loaded = generator.load_training_data()
        assert len(loaded) == len(examples)

    def test_synonym_generation_coverage(self, tmp_path):
        """동의어 생성 커버리지"""
        generator = TrainingDataGenerator(data_dir=str(tmp_path))
        examples = generator.generate_synonym_pairs()

        # 다양한 방향 포함
        directions = set(ex.metadata.get("direction") for ex in examples)
        assert "forward" in directions
        assert "reverse" in directions
        assert "cross" in directions

    def test_curated_pairs_quality(self, tmp_path):
        """큐레이션된 쌍의 품질"""
        generator = TrainingDataGenerator(data_dir=str(tmp_path))
        examples = generator.generate_financial_context_pairs()

        # 모든 예시가 금융 도메인
        for ex in examples:
            assert ex.metadata.get("domain") == "finance"

        # 쿼리와 positive가 의미적으로 관련됨 (키워드 중복)
        financial_keywords = ["영업이익", "PER", "ROE", "반도체", "HBM", "전기차", "배당"]
        has_relevant = False
        for ex in examples:
            for kw in financial_keywords:
                if kw in ex.query or kw in ex.positive:
                    has_relevant = True
                    break
        assert has_relevant
