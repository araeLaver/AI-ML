"""
RAG 벤치마크 비교 실험

다양한 검색/재정렬 전략의 성능을 비교하여 최적 구성을 찾습니다.

[비교 대상]
1. 토크나이저: Simple (2-gram) vs Kiwi (형태소 분석)
2. 검색 방식: Vector Only vs BM25 Only vs Hybrid (RRF)
3. 재정렬: None vs Keyword vs Cross-Encoder
4. 청킹: Fixed vs Sentence vs Recursive vs Semantic

[평가 지표]
- Precision@K: 상위 K개 중 관련 문서 비율
- Recall@K: 전체 관련 문서 중 상위 K개에 포함된 비율
- MRR (Mean Reciprocal Rank): 첫 번째 관련 문서 순위의 역수 평균
- NDCG@K: 순위 가중 관련성 점수
- 응답 시간: 검색 + 재정렬 소요 시간
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

# 상대 임포트
from .hybrid_search import (
    HybridSearcher,
    BM25,
    TokenizerFactory,
    KiwiTokenizer,
    SimpleTokenizer,
)
from .reranker import (
    get_reranker,
    KeywordReranker,
    CrossEncoderReranker,
)


@dataclass
class BenchmarkQuery:
    """벤치마크 쿼리"""
    query: str
    relevant_doc_ids: List[str]  # 정답 문서 ID 리스트
    category: str  # 쿼리 카테고리 (실적, 주가, 공시 등)


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    strategy_name: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    avg_latency_ms: float
    total_queries: int
    config: Dict[str, Any]


class RAGBenchmark:
    """
    RAG 벤치마크 실험 클래스

    다양한 구성의 성능을 비교 측정합니다.
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        queries: List[BenchmarkQuery],
    ):
        """
        Args:
            documents: 문서 리스트 [{"doc_id": ..., "content": ..., "metadata": ...}, ...]
            queries: 벤치마크 쿼리 리스트
        """
        self.documents = documents
        self.queries = queries
        self.results: List[BenchmarkResult] = []

        # 문서 ID → 인덱스 매핑
        self.doc_id_to_idx = {
            doc["doc_id"]: i for i, doc in enumerate(documents)
        }

    def _calculate_precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """Precision@K 계산"""
        if k == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return hits / k

    def _calculate_recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """Recall@K 계산"""
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)

    def _calculate_mrr(
        self,
        retrieved: List[str],
        relevant: List[str],
    ) -> float:
        """MRR (Mean Reciprocal Rank) 계산"""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """NDCG@K 계산"""
        import math

        relevant_set = set(relevant)

        # DCG 계산
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                # 관련 문서면 1, 아니면 0
                rel = 1.0
                dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

        # IDCG 계산 (이상적인 순위)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / math.log2(i + 2)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def run_single_strategy(
        self,
        strategy_name: str,
        search_fn,
        config: Dict[str, Any],
    ) -> BenchmarkResult:
        """
        단일 전략 벤치마크 실행

        Args:
            strategy_name: 전략 이름
            search_fn: 검색 함수 (query: str) -> List[doc_id]
            config: 전략 설정

        Returns:
            벤치마크 결과
        """
        precision_1_list = []
        precision_3_list = []
        precision_5_list = []
        recall_5_list = []
        mrr_list = []
        ndcg_5_list = []
        latencies = []

        for query_obj in self.queries:
            start_time = time.time()

            try:
                retrieved_ids = search_fn(query_obj.query)
            except Exception as e:
                print(f"검색 실패 ({strategy_name}): {e}")
                retrieved_ids = []

            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)

            relevant = query_obj.relevant_doc_ids

            precision_1_list.append(self._calculate_precision_at_k(retrieved_ids, relevant, 1))
            precision_3_list.append(self._calculate_precision_at_k(retrieved_ids, relevant, 3))
            precision_5_list.append(self._calculate_precision_at_k(retrieved_ids, relevant, 5))
            recall_5_list.append(self._calculate_recall_at_k(retrieved_ids, relevant, 5))
            mrr_list.append(self._calculate_mrr(retrieved_ids, relevant))
            ndcg_5_list.append(self._calculate_ndcg_at_k(retrieved_ids, relevant, 5))

        result = BenchmarkResult(
            strategy_name=strategy_name,
            precision_at_1=statistics.mean(precision_1_list) if precision_1_list else 0.0,
            precision_at_3=statistics.mean(precision_3_list) if precision_3_list else 0.0,
            precision_at_5=statistics.mean(precision_5_list) if precision_5_list else 0.0,
            recall_at_5=statistics.mean(recall_5_list) if recall_5_list else 0.0,
            mrr=statistics.mean(mrr_list) if mrr_list else 0.0,
            ndcg_at_5=statistics.mean(ndcg_5_list) if ndcg_5_list else 0.0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            total_queries=len(self.queries),
            config=config,
        )

        self.results.append(result)
        return result

    def run_tokenizer_comparison(self) -> List[BenchmarkResult]:
        """토크나이저 비교 (Simple vs Kiwi)"""
        results = []
        doc_contents = [d["content"] for d in self.documents]
        doc_ids = [d["doc_id"] for d in self.documents]

        # Simple Tokenizer (2-gram)
        TokenizerFactory.reset()
        simple_tokenizer = SimpleTokenizer()
        bm25_simple = BM25(tokenizer=simple_tokenizer)
        bm25_simple.fit(doc_contents, doc_ids)

        def search_simple(query: str) -> List[str]:
            results = bm25_simple.search(query, top_k=10)
            return [r.doc_id for r in results]

        result = self.run_single_strategy(
            "BM25 (Simple 2-gram)",
            search_simple,
            {"tokenizer": "simple", "search": "bm25"}
        )
        results.append(result)

        # Kiwi Tokenizer
        TokenizerFactory.reset()
        try:
            kiwi_tokenizer = KiwiTokenizer()
            bm25_kiwi = BM25(tokenizer=kiwi_tokenizer)
            bm25_kiwi.fit(doc_contents, doc_ids)

            def search_kiwi(query: str) -> List[str]:
                results = bm25_kiwi.search(query, top_k=10)
                return [r.doc_id for r in results]

            result = self.run_single_strategy(
                "BM25 (Kiwi 형태소)",
                search_kiwi,
                {"tokenizer": "kiwi", "search": "bm25"}
            )
            results.append(result)
        except Exception as e:
            print(f"Kiwi 테스트 실패: {e}")

        return results

    def run_search_method_comparison(
        self,
        vector_search_fn=None,
    ) -> List[BenchmarkResult]:
        """
        검색 방식 비교 (Vector vs BM25 vs Hybrid)

        Args:
            vector_search_fn: 벡터 검색 함수 (query) -> List[{"doc_id", "content", "score"}]
        """
        results = []
        doc_contents = [d["content"] for d in self.documents]
        doc_ids = [d["doc_id"] for d in self.documents]
        doc_metadatas = [d.get("metadata", {}) for d in self.documents]

        # BM25 Only
        searcher = HybridSearcher(use_kiwi=True)
        searcher.index_documents(doc_contents, doc_ids, doc_metadatas)

        def search_bm25(query: str) -> List[str]:
            results = searcher.search_keyword_only(query, top_k=10)
            return [r.doc_id for r in results]

        result = self.run_single_strategy(
            "BM25 Only",
            search_bm25,
            {"search": "bm25_only"}
        )
        results.append(result)

        # Vector Only (제공된 경우)
        if vector_search_fn:
            def search_vector(query: str) -> List[str]:
                vec_results = vector_search_fn(query)
                return [r["doc_id"] for r in vec_results[:10]]

            result = self.run_single_strategy(
                "Vector Only",
                search_vector,
                {"search": "vector_only"}
            )
            results.append(result)

            # Hybrid (Vector + BM25 + RRF)
            def search_hybrid(query: str) -> List[str]:
                vec_results = vector_search_fn(query)
                hybrid_results = searcher.search(query, vec_results, top_k=10)
                return [r.doc_id for r in hybrid_results]

            result = self.run_single_strategy(
                "Hybrid (Vector+BM25+RRF)",
                search_hybrid,
                {"search": "hybrid", "fusion": "rrf"}
            )
            results.append(result)

        return results

    def run_reranker_comparison(
        self,
        base_search_fn,
    ) -> List[BenchmarkResult]:
        """
        재정렬 방식 비교 (None vs Keyword vs CrossEncoder)

        Args:
            base_search_fn: 기본 검색 함수 (query) -> List[{"doc_id", "content", "score"}]
        """
        results = []

        # No Reranking
        def search_no_rerank(query: str) -> List[str]:
            base_results = base_search_fn(query)
            return [r["doc_id"] for r in base_results[:10]]

        result = self.run_single_strategy(
            "No Reranking",
            search_no_rerank,
            {"reranker": "none"}
        )
        results.append(result)

        # Keyword Reranker
        keyword_reranker = KeywordReranker()

        def search_keyword_rerank(query: str) -> List[str]:
            base_results = base_search_fn(query)
            reranked = keyword_reranker.rerank(query, base_results, top_k=10)
            return [r.doc_id for r in reranked]

        result = self.run_single_strategy(
            "Keyword Reranker",
            search_keyword_rerank,
            {"reranker": "keyword"}
        )
        results.append(result)

        # Cross-Encoder Reranker
        try:
            cross_reranker = CrossEncoderReranker(model_name="bge-reranker-base")

            def search_cross_rerank(query: str) -> List[str]:
                base_results = base_search_fn(query)
                reranked = cross_reranker.rerank(query, base_results, top_k=10)
                return [r.doc_id for r in reranked]

            result = self.run_single_strategy(
                "Cross-Encoder (bge-reranker)",
                search_cross_rerank,
                {"reranker": "cross_encoder", "model": "bge-reranker-base"}
            )
            results.append(result)
        except Exception as e:
            print(f"Cross-Encoder 테스트 실패: {e}")

        return results

    def run_full_comparison(
        self,
        vector_search_fn=None,
    ) -> List[BenchmarkResult]:
        """전체 비교 실행"""
        print("=" * 60)
        print("RAG 벤치마크 시작")
        print("=" * 60)

        all_results = []

        # 1. 토크나이저 비교
        print("\n[1/3] 토크나이저 비교...")
        all_results.extend(self.run_tokenizer_comparison())

        # 2. 검색 방식 비교
        print("\n[2/3] 검색 방식 비교...")
        all_results.extend(self.run_search_method_comparison(vector_search_fn))

        # 3. 재정렬 비교 (BM25 기반)
        print("\n[3/3] 재정렬 비교...")
        doc_contents = [d["content"] for d in self.documents]
        doc_ids = [d["doc_id"] for d in self.documents]

        bm25 = BM25()
        bm25.fit(doc_contents, doc_ids)

        def base_search(query: str) -> List[Dict]:
            results = bm25.search(query, top_k=20)
            return [
                {"doc_id": r.doc_id, "content": r.content, "score": r.score}
                for r in results
            ]

        all_results.extend(self.run_reranker_comparison(base_search))

        return all_results

    def print_results(self):
        """결과 출력"""
        print("\n" + "=" * 80)
        print("벤치마크 결과")
        print("=" * 80)

        # 테이블 헤더
        header = f"{'전략':<35} {'P@1':>6} {'P@3':>6} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6} {'ms':>8}"
        print(header)
        print("-" * 80)

        # 결과 정렬 (MRR 기준)
        sorted_results = sorted(self.results, key=lambda x: x.mrr, reverse=True)

        for r in sorted_results:
            row = (
                f"{r.strategy_name:<35} "
                f"{r.precision_at_1:>6.3f} "
                f"{r.precision_at_3:>6.3f} "
                f"{r.precision_at_5:>6.3f} "
                f"{r.recall_at_5:>6.3f} "
                f"{r.mrr:>6.3f} "
                f"{r.ndcg_at_5:>6.3f} "
                f"{r.avg_latency_ms:>8.1f}"
            )
            print(row)

        print("-" * 80)

        # 최고 성능 전략
        best = sorted_results[0]
        print(f"\n최고 성능: {best.strategy_name}")
        print(f"  - MRR: {best.mrr:.3f}")
        print(f"  - P@3: {best.precision_at_3:.3f}")
        print(f"  - 평균 응답시간: {best.avg_latency_ms:.1f}ms")

    def save_results(self, output_path: Path) -> Path:
        """결과 저장"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(self.documents),
            "total_queries": len(self.queries),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장: {output_path}")
        return output_path

    def generate_markdown_report(self) -> str:
        """마크다운 리포트 생성"""
        lines = [
            "# RAG 벤치마크 결과",
            "",
            f"- 문서 수: {len(self.documents):,}",
            f"- 쿼리 수: {len(self.queries)}",
            f"- 실행일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 성능 비교",
            "",
            "| 전략 | P@1 | P@3 | P@5 | R@5 | MRR | NDCG@5 | 응답시간(ms) |",
            "|------|-----|-----|-----|-----|-----|--------|-------------|",
        ]

        sorted_results = sorted(self.results, key=lambda x: x.mrr, reverse=True)
        for r in sorted_results:
            lines.append(
                f"| {r.strategy_name} | "
                f"{r.precision_at_1:.3f} | "
                f"{r.precision_at_3:.3f} | "
                f"{r.precision_at_5:.3f} | "
                f"{r.recall_at_5:.3f} | "
                f"{r.mrr:.3f} | "
                f"{r.ndcg_at_5:.3f} | "
                f"{r.avg_latency_ms:.1f} |"
            )

        # 개선률 계산
        if len(sorted_results) >= 2:
            best = sorted_results[0]
            baseline = sorted_results[-1]  # 가장 낮은 성능을 baseline으로

            if baseline.mrr > 0:
                improvement = (best.mrr - baseline.mrr) / baseline.mrr * 100
                lines.extend([
                    "",
                    "## 핵심 인사이트",
                    "",
                    f"- **최고 성능**: {best.strategy_name}",
                    f"- **MRR 개선**: {baseline.strategy_name} 대비 **+{improvement:.1f}%**",
                    f"- **최적 응답시간**: {best.avg_latency_ms:.1f}ms",
                ])

        return "\n".join(lines)


# 샘플 데이터 생성 함수
def create_sample_benchmark_data() -> Tuple[List[Dict], List[BenchmarkQuery]]:
    """테스트용 샘플 데이터 생성"""

    # 샘플 문서
    documents = [
        {
            "doc_id": "doc_1",
            "content": "삼성전자 2024년 3분기 영업이익 9조1834억원 기록. 반도체 부문 HBM 수요 증가로 실적 개선.",
            "metadata": {"source": "실적공시", "corp": "삼성전자"}
        },
        {
            "doc_id": "doc_2",
            "content": "SK하이닉스 HBM3E 양산 시작. AI 반도체 수요 급증으로 2024년 사상 최대 실적 전망.",
            "metadata": {"source": "뉴스", "corp": "SK하이닉스"}
        },
        {
            "doc_id": "doc_3",
            "content": "현대자동차 전기차 아이오닉6 유럽 판매 호조. 2024년 글로벌 전기차 판매 50만대 목표.",
            "metadata": {"source": "실적공시", "corp": "현대자동차"}
        },
        {
            "doc_id": "doc_4",
            "content": "카카오 2024년 2분기 광고 매출 성장. 카카오톡 비즈니스 플랫폼 수익성 개선.",
            "metadata": {"source": "실적공시", "corp": "카카오"}
        },
        {
            "doc_id": "doc_5",
            "content": "네이버 검색광고 매출 증가. AI 서비스 하이퍼클로바X 상용화 본격화.",
            "metadata": {"source": "뉴스", "corp": "네이버"}
        },
        {
            "doc_id": "doc_6",
            "content": "LG에너지솔루션 북미 배터리 공장 증설. 테슬라, GM 납품 계약 확대.",
            "metadata": {"source": "공시", "corp": "LG에너지솔루션"}
        },
        {
            "doc_id": "doc_7",
            "content": "삼성SDI 전고체 배터리 개발 성공. 2027년 양산 목표로 투자 확대.",
            "metadata": {"source": "뉴스", "corp": "삼성SDI"}
        },
        {
            "doc_id": "doc_8",
            "content": "POSCO홀딩스 리튬 사업 본격화. 아르헨티나 리튬 염호 투자 완료.",
            "metadata": {"source": "공시", "corp": "POSCO홀딩스"}
        },
        {
            "doc_id": "doc_9",
            "content": "KB금융 순이익 1조원 돌파. 비은행 부문 성장으로 수익 다각화 성공.",
            "metadata": {"source": "실적공시", "corp": "KB금융"}
        },
        {
            "doc_id": "doc_10",
            "content": "신한지주 디지털 전환 가속. 신한 쏠(SOL) 앱 가입자 2000만 돌파.",
            "metadata": {"source": "뉴스", "corp": "신한지주"}
        },
    ]

    # 샘플 쿼리
    queries = [
        BenchmarkQuery(
            query="삼성전자 3분기 실적",
            relevant_doc_ids=["doc_1"],
            category="실적"
        ),
        BenchmarkQuery(
            query="HBM 반도체 수요",
            relevant_doc_ids=["doc_1", "doc_2"],
            category="산업"
        ),
        BenchmarkQuery(
            query="전기차 배터리 투자",
            relevant_doc_ids=["doc_6", "doc_7"],
            category="투자"
        ),
        BenchmarkQuery(
            query="AI 서비스 상용화",
            relevant_doc_ids=["doc_2", "doc_5"],
            category="기술"
        ),
        BenchmarkQuery(
            query="금융 순이익",
            relevant_doc_ids=["doc_9"],
            category="실적"
        ),
        BenchmarkQuery(
            query="현대차 전기차 판매",
            relevant_doc_ids=["doc_3"],
            category="판매"
        ),
        BenchmarkQuery(
            query="카카오 광고 매출",
            relevant_doc_ids=["doc_4"],
            category="실적"
        ),
        BenchmarkQuery(
            query="리튬 배터리 원재료",
            relevant_doc_ids=["doc_8"],
            category="원자재"
        ),
    ]

    return documents, queries


def main():
    """벤치마크 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 벤치마크")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="결과 저장 경로"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="샘플 데이터로 테스트"
    )

    args = parser.parse_args()

    # 샘플 데이터로 테스트
    if args.sample:
        documents, queries = create_sample_benchmark_data()
        print(f"샘플 데이터: {len(documents)} 문서, {len(queries)} 쿼리")
    else:
        print("--sample 옵션으로 샘플 데이터 테스트를 실행하세요")
        return

    # 벤치마크 실행
    benchmark = RAGBenchmark(documents, queries)
    benchmark.run_full_comparison()

    # 결과 출력
    benchmark.print_results()

    # 결과 저장
    output_path = Path(args.output)
    benchmark.save_results(output_path)

    # 마크다운 리포트
    report = benchmark.generate_markdown_report()
    report_path = output_path.with_suffix(".md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"마크다운 리포트: {report_path}")


if __name__ == "__main__":
    main()
