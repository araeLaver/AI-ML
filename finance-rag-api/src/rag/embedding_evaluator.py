# -*- coding: utf-8 -*-
"""
임베딩 모델 평가 모듈

[평가 메트릭]
- MRR (Mean Reciprocal Rank)
- Recall@K (K=1, 5, 10)
- NDCG@K (Normalized Discounted Cumulative Gain)
- 동의어 유사도

[사용 예시]
>>> evaluator = EmbeddingEvaluator()
>>> results = evaluator.evaluate_model(model, test_queries, test_docs)
>>> print(results['mrr'], results['recall@5'])
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """평가 결과"""
    mrr: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    synonym_similarity: float
    num_queries: int
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, float]:
        return {
            "mrr": self.mrr,
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "ndcg@10": self.ndcg_at_10,
            "synonym_similarity": self.synonym_similarity,
            "num_queries": self.num_queries,
        }


class EmbeddingEvaluator:
    """
    임베딩 모델 평가기

    [평가 방법]
    1. 쿼리 임베딩과 문서 임베딩 간 코사인 유사도 계산
    2. 유사도 기반 순위 산출
    3. MRR, Recall@K, NDCG@K 계산
    """

    def __init__(self):
        self.results_history = []

    def compute_embeddings(
        self,
        model,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """텍스트 임베딩 계산"""
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    def cosine_similarity(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """코사인 유사도 계산"""
        # 정규화
        query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # 코사인 유사도 행렬 (queries x docs)
        similarity_matrix = np.dot(query_norm, doc_norm.T)
        return similarity_matrix

    def compute_mrr(
        self,
        similarity_matrix: np.ndarray,
        relevant_indices: List[int]
    ) -> float:
        """Mean Reciprocal Rank 계산"""
        reciprocal_ranks = []

        for i, rel_idx in enumerate(relevant_indices):
            # 유사도 기준 순위
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            rank = np.where(sorted_indices == rel_idx)[0]

            if len(rank) > 0:
                reciprocal_ranks.append(1.0 / (rank[0] + 1))
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    def compute_recall_at_k(
        self,
        similarity_matrix: np.ndarray,
        relevant_indices: List[int],
        k: int
    ) -> float:
        """Recall@K 계산"""
        hits = 0

        for i, rel_idx in enumerate(relevant_indices):
            # 상위 K개
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]

            if rel_idx in top_k_indices:
                hits += 1

        return hits / len(relevant_indices)

    def compute_ndcg_at_k(
        self,
        similarity_matrix: np.ndarray,
        relevant_indices: List[int],
        k: int
    ) -> float:
        """NDCG@K 계산"""
        ndcg_scores = []

        for i, rel_idx in enumerate(relevant_indices):
            # 상위 K개
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]

            # DCG 계산
            dcg = 0.0
            for rank, idx in enumerate(top_k_indices):
                if idx == rel_idx:
                    dcg += 1.0 / np.log2(rank + 2)  # rank는 0부터 시작

            # Ideal DCG (관련 문서가 1위일 때)
            idcg = 1.0 / np.log2(2)  # 1위 = 1/log2(2) = 1

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return np.mean(ndcg_scores)

    def compute_synonym_similarity(
        self,
        model,
        synonym_pairs: List[Tuple[str, str]]
    ) -> float:
        """동의어 쌍 유사도 계산"""
        if not synonym_pairs:
            return 0.0

        terms1 = [pair[0] for pair in synonym_pairs]
        terms2 = [pair[1] for pair in synonym_pairs]

        emb1 = self.compute_embeddings(model, terms1)
        emb2 = self.compute_embeddings(model, terms2)

        # 각 쌍의 코사인 유사도
        similarities = []
        for i in range(len(synonym_pairs)):
            sim = np.dot(emb1[i], emb2[i]) / (np.linalg.norm(emb1[i]) * np.linalg.norm(emb2[i]))
            similarities.append(sim)

        return np.mean(similarities)

    def evaluate_model(
        self,
        model,
        queries: List[str],
        documents: List[str],
        relevant_indices: Optional[List[int]] = None,
        synonym_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> EvaluationResult:
        """
        모델 종합 평가

        Args:
            model: SentenceTransformer 모델
            queries: 평가용 쿼리 목록
            documents: 문서 목록
            relevant_indices: 각 쿼리의 관련 문서 인덱스 (None이면 1:1 매핑)
            synonym_pairs: 동의어 쌍 목록 (선택)

        Returns:
            EvaluationResult: 평가 결과
        """
        logger.info(f"Evaluating model with {len(queries)} queries, {len(documents)} documents")

        # 관련 문서 인덱스 기본값 (쿼리 i → 문서 i)
        if relevant_indices is None:
            relevant_indices = list(range(min(len(queries), len(documents))))

        # 임베딩 계산
        query_embeddings = self.compute_embeddings(model, queries)
        doc_embeddings = self.compute_embeddings(model, documents)

        # 유사도 행렬
        similarity_matrix = self.cosine_similarity(query_embeddings, doc_embeddings)

        # 메트릭 계산
        mrr = self.compute_mrr(similarity_matrix, relevant_indices)
        recall_1 = self.compute_recall_at_k(similarity_matrix, relevant_indices, k=1)
        recall_5 = self.compute_recall_at_k(similarity_matrix, relevant_indices, k=5)
        recall_10 = self.compute_recall_at_k(similarity_matrix, relevant_indices, k=10)
        ndcg_10 = self.compute_ndcg_at_k(similarity_matrix, relevant_indices, k=10)

        # 동의어 유사도
        synonym_sim = 0.0
        if synonym_pairs:
            synonym_sim = self.compute_synonym_similarity(model, synonym_pairs)

        result = EvaluationResult(
            mrr=mrr,
            recall_at_1=recall_1,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            ndcg_at_10=ndcg_10,
            synonym_similarity=synonym_sim,
            num_queries=len(queries),
            details={
                "similarity_matrix_shape": similarity_matrix.shape,
                "avg_similarity": float(np.mean(similarity_matrix)),
            }
        )

        self.results_history.append(result)
        logger.info(f"Evaluation complete: MRR={mrr:.4f}, Recall@5={recall_5:.4f}")

        return result

    def compare_models(
        self,
        baseline_model,
        finetuned_model,
        queries: List[str],
        documents: List[str],
        relevant_indices: Optional[List[int]] = None,
        synonym_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        베이스라인 vs Fine-tuned 모델 비교

        Returns:
            비교 결과 딕셔너리
        """
        logger.info("Comparing baseline and fine-tuned models...")

        # 베이스라인 평가
        baseline_result = self.evaluate_model(
            baseline_model, queries, documents, relevant_indices, synonym_pairs
        )

        # Fine-tuned 평가
        finetuned_result = self.evaluate_model(
            finetuned_model, queries, documents, relevant_indices, synonym_pairs
        )

        # 개선율 계산
        improvement = {}
        baseline_dict = baseline_result.to_dict()
        finetuned_dict = finetuned_result.to_dict()

        for key in baseline_dict:
            if key == "num_queries":
                continue
            baseline_val = baseline_dict[key]
            finetuned_val = finetuned_dict[key]

            if baseline_val > 0:
                improvement[key] = ((finetuned_val - baseline_val) / baseline_val) * 100
            else:
                improvement[key] = 0.0

        return {
            "baseline": baseline_dict,
            "finetuned": finetuned_dict,
            "improvement_percent": improvement,
        }


class FinancialEvaluationDataset:
    """금융 도메인 평가 데이터셋"""

    @staticmethod
    def get_synonym_test_pairs() -> List[Tuple[str, str]]:
        """동의어 테스트 쌍"""
        return [
            # 재무 지표
            ("PER", "주가수익비율"),
            ("PBR", "주가순자산비율"),
            ("ROE", "자기자본이익률"),
            ("ROA", "총자산이익률"),
            ("EPS", "주당순이익"),
            ("EBITDA", "감가상각전영업이익"),

            # 기업 약어
            ("삼전", "삼성전자"),
            ("하닉", "SK하이닉스"),
            ("현차", "현대자동차"),

            # 산업 용어
            ("HBM", "고대역폭메모리"),
            ("2차전지", "배터리"),
            ("EV", "전기차"),
            ("AI반도체", "인공지능반도체"),

            # 시장 용어
            ("IPO", "기업공개"),
            ("시총", "시가총액"),
            ("ETF", "상장지수펀드"),
        ]

    @staticmethod
    def get_query_document_test_pairs() -> List[Tuple[str, str]]:
        """쿼리-문서 테스트 쌍"""
        return [
            ("삼성전자 영업이익", "삼성전자의 2024년 1분기 영업이익은 6조 6,000억원을 기록했습니다."),
            ("SK하이닉스 HBM", "SK하이닉스는 HBM3E 양산을 통해 AI 반도체 시장을 선도하고 있습니다."),
            ("현대차 전기차 판매", "현대자동차의 아이오닉 시리즈 판매량이 전년 대비 30% 증가했습니다."),
            ("반도체 업황 전망", "메모리 반도체 가격이 반등하며 업황 개선 기대감이 높아지고 있습니다."),
            ("배당주 추천", "고배당 종목은 안정적인 현금흐름과 함께 배당수익을 제공합니다."),
        ]


# =============================================================================
# 편의 함수
# =============================================================================

def evaluate_embedding_model(
    model_path: str,
    include_synonyms: bool = True
) -> Dict[str, float]:
    """임베딩 모델 평가 (편의 함수)"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    evaluator = EmbeddingEvaluator()

    # 테스트 데이터
    test_pairs = FinancialEvaluationDataset.get_query_document_test_pairs()
    queries = [pair[0] for pair in test_pairs]
    documents = [pair[1] for pair in test_pairs]

    synonym_pairs = None
    if include_synonyms:
        synonym_pairs = FinancialEvaluationDataset.get_synonym_test_pairs()

    result = evaluator.evaluate_model(model, queries, documents, synonym_pairs=synonym_pairs)
    return result.to_dict()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate embedding model")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--compare-with", help="Baseline model to compare with")

    args = parser.parse_args()

    if args.compare_with:
        from sentence_transformers import SentenceTransformer

        baseline = SentenceTransformer(args.compare_with)
        finetuned = SentenceTransformer(args.model)

        evaluator = EmbeddingEvaluator()
        test_pairs = FinancialEvaluationDataset.get_query_document_test_pairs()
        queries = [p[0] for p in test_pairs]
        docs = [p[1] for p in test_pairs]
        synonyms = FinancialEvaluationDataset.get_synonym_test_pairs()

        comparison = evaluator.compare_models(baseline, finetuned, queries, docs, synonym_pairs=synonyms)

        print("\n=== Model Comparison ===")
        print(f"\nBaseline: {args.compare_with}")
        for k, v in comparison["baseline"].items():
            print(f"  {k}: {v:.4f}")

        print(f"\nFine-tuned: {args.model}")
        for k, v in comparison["finetuned"].items():
            print(f"  {k}: {v:.4f}")

        print("\nImprovement (%):")
        for k, v in comparison["improvement_percent"].items():
            sign = "+" if v > 0 else ""
            print(f"  {k}: {sign}{v:.1f}%")
    else:
        result = evaluate_embedding_model(args.model)
        print("\n=== Evaluation Results ===")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
