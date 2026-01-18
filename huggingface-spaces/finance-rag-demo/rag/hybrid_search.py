# -*- coding: utf-8 -*-
"""
Hybrid Search Module

Vector Search + BM25 Keyword Search + RRF Fusion
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .bm25 import BM25, BM25Result
from .vectorstore import VectorStore, VectorSearchResult


@dataclass
class SearchResult:
    """하이브리드 검색 결과"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    search_type: str  # "vector", "keyword", "hybrid"


class HybridSearcher:
    """
    하이브리드 검색기

    벡터 검색 + BM25 키워드 검색 결합
    RRF (Reciprocal Rank Fusion)로 순위 통합

    [왜 하이브리드인가?]
    - 벡터 검색: 의미적 유사성 (동의어, 유사 개념)
    - 키워드 검색: 정확한 용어 매칭 (고유명사, 숫자)
    - 결합: 두 장점 모두 활용

    [RRF 수식]
    score = sum 1/(k + rank)
    - k: 상수 (기본 60), 높을수록 순위 차이 축소
    """

    def __init__(
        self,
        hf_token: str = "",
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Args:
            hf_token: HuggingFace API 토큰 (임베딩용)
            vector_weight: 벡터 검색 가중치 (0~1)
            keyword_weight: 키워드 검색 가중치 (0~1)
            rrf_k: RRF 상수 (높을수록 순위 차이 축소)
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        # 검색 엔진 초기화
        self.vector_store = VectorStore(hf_token=hf_token)
        self.bm25 = BM25()

        # 문서 저장 (중복 방지용)
        self._doc_ids_set: set = set()

    def index_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        문서 색인

        벡터 스토어와 BM25 모두에 색인

        Args:
            documents: 문서 내용 리스트
            doc_ids: 문서 ID 리스트
            metadatas: 메타데이터 리스트 (선택)
        """
        if not documents:
            return

        # 중복 제거
        new_docs = []
        new_ids = []
        new_metas = []

        for i, doc_id in enumerate(doc_ids):
            if doc_id not in self._doc_ids_set:
                new_docs.append(documents[i])
                new_ids.append(doc_id)
                new_metas.append(metadatas[i] if metadatas else {})
                self._doc_ids_set.add(doc_id)

        if not new_docs:
            return

        # 벡터 스토어에 추가
        self.vector_store.add_documents(new_docs, new_ids, new_metas)

        # BM25에 추가 (전체 재색인)
        all_contents = self.vector_store.doc_contents
        all_ids = self.vector_store.doc_ids
        all_metas = self.vector_store.doc_metadatas
        self.bm25.fit(all_contents, all_ids, all_metas)

    def _rrf_score(self, rank: int) -> float:
        """RRF 점수 계산"""
        return 1.0 / (self.rrf_k + rank)

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "hybrid"
    ) -> List[SearchResult]:
        """
        하이브리드 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            search_mode: "hybrid", "vector", "keyword"

        Returns:
            검색 결과 리스트 (점수 내림차순)
        """
        if search_mode == "vector":
            return self._search_vector_only(query, top_k)
        elif search_mode == "keyword":
            return self._search_keyword_only(query, top_k)
        else:
            return self._search_hybrid(query, top_k)

    def _search_hybrid(self, query: str, top_k: int) -> List[SearchResult]:
        """하이브리드 검색 (RRF 결합)"""
        # 1. 벡터 검색
        vector_results = self.vector_store.search(query, top_k=top_k * 2)

        # 2. BM25 키워드 검색
        keyword_results = self.bm25.search(query, top_k=top_k * 2)

        # 3. RRF로 순위 통합
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_contents: Dict[str, str] = {}
        doc_metadatas: Dict[str, Dict[str, Any]] = {}
        doc_types: Dict[str, List[str]] = defaultdict(list)

        # 벡터 검색 결과 처리
        for rank, result in enumerate(vector_results, start=1):
            rrf = self._rrf_score(rank) * self.vector_weight
            doc_scores[result.doc_id] += rrf
            doc_contents[result.doc_id] = result.content
            doc_metadatas[result.doc_id] = result.metadata
            doc_types[result.doc_id].append("vector")

        # 키워드 검색 결과 처리
        for rank, result in enumerate(keyword_results, start=1):
            rrf = self._rrf_score(rank) * self.keyword_weight
            doc_scores[result.doc_id] += rrf
            if result.doc_id not in doc_contents:
                doc_contents[result.doc_id] = result.content
                doc_metadatas[result.doc_id] = result.metadata
            doc_types[result.doc_id].append("keyword")

        # 4. 점수 기준 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 5. 결과 생성
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            search_types = doc_types[doc_id]
            if len(search_types) > 1:
                search_type = "hybrid"
            else:
                search_type = search_types[0]

            results.append(SearchResult(
                doc_id=doc_id,
                content=doc_contents[doc_id],
                score=score,
                metadata=doc_metadatas[doc_id],
                search_type=search_type
            ))

        return results

    def _search_vector_only(self, query: str, top_k: int) -> List[SearchResult]:
        """벡터 검색만"""
        vector_results = self.vector_store.search(query, top_k)
        return [
            SearchResult(
                doc_id=r.doc_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                search_type="vector"
            )
            for r in vector_results
        ]

    def _search_keyword_only(self, query: str, top_k: int) -> List[SearchResult]:
        """키워드 검색만"""
        keyword_results = self.bm25.search(query, top_k)
        return [
            SearchResult(
                doc_id=r.doc_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                search_type="keyword"
            )
            for r in keyword_results
        ]

    def clear(self):
        """검색 엔진 초기화"""
        self.vector_store.clear()
        self.bm25 = BM25()
        self._doc_ids_set = set()

    def get_stats(self) -> Dict[str, Any]:
        """검색 엔진 통계"""
        return {
            "total_documents": len(self._doc_ids_set),
            "vector_store": self.vector_store.get_stats(),
            "bm25": self.bm25.get_stats(),
            "weights": {
                "vector": self.vector_weight,
                "keyword": self.keyword_weight,
            },
            "rrf_k": self.rrf_k,
        }
