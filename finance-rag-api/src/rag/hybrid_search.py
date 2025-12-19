# -*- coding: utf-8 -*-
"""
하이브리드 검색 모듈

[설계 의도]
- 벡터 검색 + 키워드 검색 결합
- 각각의 장단점 보완
- 검색 품질 향상

[왜 하이브리드인가?]
- 벡터 검색: 의미적 유사성 (동의어, 유사 개념)
- 키워드 검색: 정확한 용어 매칭 (고유명사, 숫자)
- 결합: 두 장점 모두 활용

[RRF (Reciprocal Rank Fusion)]
- 여러 검색 결과를 순위 기반으로 통합
- 각 문서의 순위 역수 합산
- score = Σ 1/(k + rank)
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    search_type: str  # "vector", "keyword", "hybrid"


class BM25:
    """
    BM25 키워드 검색 알고리즘

    [BM25 수식]
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))

    파라미터:
    - k1: 용어 빈도 포화도 (기본 1.5)
    - b: 문서 길이 정규화 (기본 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_contents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """
        한국어 + 영어 토큰화

        [한국어 토큰화 전략]
        형태소 분석기 없이 공백 + 2-gram으로 처리
        실무에서는 konlpy, kiwi 등 사용 권장
        """
        # 소문자 변환, 특수문자 제거
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)

        # 공백 기준 분리
        tokens = text.split()

        # 한글 2-gram 추가 (형태소 분석 대체)
        korean_tokens = []
        for token in tokens:
            if re.match(r'^[가-힣]+$', token) and len(token) >= 2:
                # 2-gram 생성
                for i in range(len(token) - 1):
                    korean_tokens.append(token[i:i+2])
                korean_tokens.append(token)  # 원본도 추가
            else:
                korean_tokens.append(token)

        return korean_tokens

    def fit(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """문서 색인"""
        self.doc_contents = documents
        self.doc_ids = doc_ids
        self.doc_metadatas = metadatas or [{} for _ in documents]

        # 문서별 토큰화 및 통계 계산
        doc_freqs: Dict[str, int] = defaultdict(int)

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # 문서 내 용어 빈도
            term_freq: Dict[str, int] = defaultdict(int)
            unique_terms = set()

            for token in tokens:
                term_freq[token] += 1
                self.vocab.add(token)
                unique_terms.add(token)

            self.doc_term_freqs.append(dict(term_freq))

            # 문서 빈도 (해당 용어가 등장하는 문서 수)
            for term in unique_terms:
                doc_freqs[term] += 1

        # 평균 문서 길이
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # IDF 계산
        n_docs = len(documents)
        for term, df in doc_freqs.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """BM25 검색"""
        query_tokens = self._tokenize(query)
        scores: List[Tuple[int, float]] = []

        for doc_idx, term_freqs in enumerate(self.doc_term_freqs):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]

            for token in query_tokens:
                if token not in term_freqs:
                    continue

                tf = term_freqs[token]
                idf = self.idf.get(token, 0)

                # BM25 스코어 계산
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((doc_idx, score))

        # 점수 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 반환
        results = []
        for doc_idx, score in scores[:top_k]:
            results.append(SearchResult(
                doc_id=self.doc_ids[doc_idx],
                content=self.doc_contents[doc_idx],
                score=score,
                metadata=self.doc_metadatas[doc_idx],
                search_type="keyword"
            ))

        return results


class HybridSearcher:
    """
    하이브리드 검색기

    벡터 검색 + BM25 키워드 검색 결합
    RRF (Reciprocal Rank Fusion)로 순위 통합
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Args:
            vector_weight: 벡터 검색 가중치 (0~1)
            keyword_weight: 키워드 검색 가중치 (0~1)
            rrf_k: RRF 상수 (기본 60, 높을수록 순위 차이 축소)
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25()
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

    def index_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """문서 색인 (BM25용)"""
        self._documents = documents
        self._doc_ids = doc_ids
        self._metadatas = metadatas or [{} for _ in documents]
        self.bm25.fit(documents, doc_ids, metadatas)

    def _rrf_score(self, rank: int) -> float:
        """RRF 점수 계산"""
        return 1.0 / (self.rrf_k + rank)

    def search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        하이브리드 검색

        Args:
            query: 검색 쿼리
            vector_results: 벡터 검색 결과 (외부에서 제공)
                [{"doc_id": ..., "content": ..., "score": ..., "metadata": ...}, ...]
            top_k: 반환할 결과 수

        Returns:
            통합 순위 결과
        """
        # 1. BM25 키워드 검색
        keyword_results = self.bm25.search(query, top_k=top_k * 2)

        # 2. RRF로 순위 통합
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_contents: Dict[str, str] = {}
        doc_metadatas: Dict[str, Dict[str, Any]] = {}
        doc_types: Dict[str, List[str]] = defaultdict(list)

        # 벡터 검색 결과 처리
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get("doc_id", f"vec_{rank}")
            rrf = self._rrf_score(rank) * self.vector_weight
            doc_scores[doc_id] += rrf
            doc_contents[doc_id] = result.get("content", "")
            doc_metadatas[doc_id] = result.get("metadata", {})
            doc_types[doc_id].append("vector")

        # 키워드 검색 결과 처리
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result.doc_id
            rrf = self._rrf_score(rank) * self.keyword_weight
            doc_scores[doc_id] += rrf
            if doc_id not in doc_contents:
                doc_contents[doc_id] = result.content
                doc_metadatas[doc_id] = result.metadata
            doc_types[doc_id].append("keyword")

        # 3. 점수 기준 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 결과 생성
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

    def search_keyword_only(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """키워드 검색만 수행"""
        return self.bm25.search(query, top_k)


def explain_hybrid_search() -> str:
    """하이브리드 검색 설명 (포트폴리오용)"""
    return """
## 하이브리드 검색이란?

### 왜 필요한가?

**벡터 검색의 한계:**
- "삼성전자 주가" 검색 시 "삼전 가격"도 찾음 (장점)
- 하지만 "HBM3E"라는 정확한 용어는 놓칠 수 있음 (단점)

**키워드 검색의 한계:**
- "HBM3E" 정확히 매칭 (장점)
- "고대역폭 메모리"로 검색하면 못 찾음 (단점)

### 해결책: 하이브리드

```
최종 점수 = (벡터 점수 × 0.5) + (키워드 점수 × 0.5)
```

### RRF (Reciprocal Rank Fusion)

순위 기반 통합 알고리즘:
- 각 검색 결과의 순위 역수를 합산
- 점수 스케일이 다른 두 검색 결과를 공정하게 통합

```
RRF_score = Σ 1/(k + rank)
```

### 구현 선택 이유

1. **BM25 선택**: TF-IDF보다 문서 길이 정규화 우수
2. **RRF 선택**: 단순 점수 합산보다 안정적
3. **가중치 0.5:0.5**: 실험 결과 균형이 최적
"""
