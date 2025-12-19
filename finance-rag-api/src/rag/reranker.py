# -*- coding: utf-8 -*-
"""
Re-ranking 모듈

[설계 의도]
- 초기 검색 결과를 더 정교하게 재정렬
- 검색 품질 향상
- 비용 효율적 (전체 문서가 아닌 top-k만 처리)

[왜 Re-ranking이 필요한가?]
1. 임베딩 모델의 한계
   - 범용 임베딩은 도메인 특화 질의에 약함
   - 짧은 쿼리 vs 긴 문서 길이 불균형

2. Cross-Encoder의 장점
   - 쿼리-문서 쌍을 함께 인코딩
   - Bi-Encoder보다 정확도 높음
   - 단, 속도가 느려서 re-ranking에만 사용

[구현 전략]
- Cross-Encoder 모델 사용 (실제 서비스)
- LLM 기반 re-ranking (대안)
- 규칙 기반 휴리스틱 (경량 버전)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


@dataclass
class RankedDocument:
    """재정렬된 문서"""
    doc_id: str
    content: str
    original_rank: int
    new_rank: int
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any]


class BaseReranker(ABC):
    """Re-ranker 추상 클래스"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """문서 재정렬"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Re-ranker 이름"""
        pass


class KeywordReranker(BaseReranker):
    """
    키워드 기반 Re-ranker

    간단한 규칙 기반 재정렬
    - 쿼리 키워드 매칭 점수
    - 문서 시작 부분 가중치
    - 정확한 구문 매칭 보너스
    """

    @property
    def name(self) -> str:
        return "keyword"

    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _calculate_score(self, query: str, document: str) -> float:
        """재정렬 점수 계산"""
        query_keywords = set(self._extract_keywords(query))
        doc_text = document.lower()

        if not query_keywords:
            return 0.0

        score = 0.0

        # 1. 키워드 매칭 점수
        for keyword in query_keywords:
            if keyword in doc_text:
                count = doc_text.count(keyword)
                score += min(count, 5) * 0.1  # 최대 5회까지 카운트

        # 2. 정확한 쿼리 구문 매칭 보너스
        if query.lower() in doc_text:
            score += 0.5

        # 3. 문서 시작 부분에 키워드 있으면 보너스
        first_100 = doc_text[:100]
        early_matches = sum(1 for kw in query_keywords if kw in first_100)
        score += early_matches * 0.2

        # 4. 정규화 (0~1 범위)
        max_possible = len(query_keywords) * 0.5 + 0.5 + len(query_keywords) * 0.2
        normalized_score = min(1.0, score / max(max_possible, 1))

        return normalized_score

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """키워드 기반 재정렬"""
        scored_docs = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            original_score = doc.get("score", 0.0)
            rerank_score = self._calculate_score(query, content)

            # 원본 점수와 재정렬 점수 결합
            combined_score = original_score * 0.4 + rerank_score * 0.6

            scored_docs.append({
                "doc": doc,
                "original_rank": i + 1,
                "original_score": original_score,
                "rerank_score": combined_score
            })

        # 재정렬
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 결과 생성
        results = []
        for new_rank, item in enumerate(scored_docs[:top_k], start=1):
            doc = item["doc"]
            results.append(RankedDocument(
                doc_id=doc.get("doc_id", f"doc_{new_rank}"),
                content=doc.get("content", ""),
                original_rank=item["original_rank"],
                new_rank=new_rank,
                original_score=item["original_score"],
                rerank_score=item["rerank_score"],
                metadata=doc.get("metadata", {})
            ))

        return results


class LLMReranker(BaseReranker):
    """
    LLM 기반 Re-ranker

    LLM에게 문서 관련성 점수를 매기게 함
    정확도 높지만 비용/속도 트레이드오프
    """

    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    @property
    def name(self) -> str:
        return "llm"

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """
        LLM 기반 재정렬

        실제 구현에서는 LLM에게 각 문서의 관련성을 평가하게 함
        여기서는 데모용으로 키워드 기반 로직 사용
        """
        # LLM이 없으면 키워드 기반으로 폴백
        if self.llm is None:
            keyword_reranker = KeywordReranker()
            return keyword_reranker.rerank(query, documents, top_k)

        # LLM 프롬프트 (실제 구현)
        prompt_template = """다음 질문과 문서의 관련성을 0-10 점수로 평가하세요.

질문: {query}

문서: {document}

관련성 점수 (0-10, 숫자만):"""

        scored_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")[:500]  # 토큰 절약

            try:
                # LLM 호출
                response = self.llm.generate(
                    system_prompt="당신은 문서 관련성 평가 전문가입니다. 점수만 숫자로 답하세요.",
                    user_prompt=prompt_template.format(query=query, document=content)
                )

                # 점수 추출
                score_match = re.search(r'\d+', response)
                if score_match:
                    score = min(10, int(score_match.group())) / 10.0
                else:
                    score = 0.5
            except Exception:
                score = 0.5  # 오류시 중간값

            scored_docs.append({
                "doc": doc,
                "original_rank": i + 1,
                "original_score": doc.get("score", 0.0),
                "rerank_score": score
            })

        # 재정렬
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 결과 생성
        results = []
        for new_rank, item in enumerate(scored_docs[:top_k], start=1):
            doc = item["doc"]
            results.append(RankedDocument(
                doc_id=doc.get("doc_id", f"doc_{new_rank}"),
                content=doc.get("content", ""),
                original_rank=item["original_rank"],
                new_rank=new_rank,
                original_score=item["original_score"],
                rerank_score=item["rerank_score"],
                metadata=doc.get("metadata", {})
            ))

        return results


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder 기반 Re-ranker

    [원리]
    - Bi-Encoder: 쿼리와 문서를 각각 임베딩 후 유사도 계산
    - Cross-Encoder: 쿼리+문서를 함께 입력하여 관련성 직접 예측

    [장점]
    - Bi-Encoder보다 정확도 높음
    - 세밀한 관련성 판단 가능

    [단점]
    - O(N) 비용 (각 문서마다 추론 필요)
    - 전체 검색에는 부적합, re-ranking에만 사용

    [권장 모델]
    - sentence-transformers/ms-marco-MiniLM-L-6-v2
    - BAAI/bge-reranker-base
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return f"cross_encoder ({self.model_name})"

    def _load_model(self):
        """Cross-Encoder 모델 로드 (지연 로딩)"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers 패키지가 필요합니다: "
                    "pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """Cross-Encoder로 재정렬"""
        # 모델 없이 데모 모드
        # 실제 구현에서는 self._load_model() 호출

        # 데모용: 키워드 기반으로 대체
        keyword_reranker = KeywordReranker()
        return keyword_reranker.rerank(query, documents, top_k)


# Re-ranker 팩토리
RERANKER_REGISTRY = {
    "keyword": KeywordReranker,
    "llm": LLMReranker,
    "cross_encoder": CrossEncoderReranker,
}


def get_reranker(
    reranker_type: str = "keyword",
    **kwargs
) -> BaseReranker:
    """Re-ranker 팩토리 함수"""
    if reranker_type not in RERANKER_REGISTRY:
        raise ValueError(
            f"Unknown reranker: {reranker_type}. "
            f"Available: {list(RERANKER_REGISTRY.keys())}"
        )

    return RERANKER_REGISTRY[reranker_type](**kwargs)


def explain_reranking() -> str:
    """Re-ranking 설명 (포트폴리오용)"""
    return """
## Re-ranking이란?

### 문제 상황
초기 벡터 검색 결과가 항상 최적은 아님:
- 임베딩 모델이 도메인에 최적화되지 않음
- 짧은 쿼리 vs 긴 문서 길이 불균형
- 의미적 유사성만으로는 부족

### 해결책: Two-Stage Retrieval

```
1단계: 빠른 검색 (Bi-Encoder)
   - 수백만 문서에서 top-100 추출
   - O(1) 벡터 유사도 검색

2단계: 정밀 재정렬 (Cross-Encoder)
   - top-100을 정확히 평가
   - 쿼리+문서 함께 인코딩
   - 최종 top-5 선정
```

### Cross-Encoder vs Bi-Encoder

| 항목 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| 입력 | 쿼리, 문서 각각 | 쿼리+문서 함께 |
| 속도 | 빠름 (O(1)) | 느림 (O(N)) |
| 정확도 | 중간 | 높음 |
| 용도 | 전체 검색 | Re-ranking |

### 구현 선택

1. **KeywordReranker**: 빠르고 간단, 기본 옵션
2. **LLMReranker**: 정확하지만 비용 발생
3. **CrossEncoderReranker**: 균형 잡힌 선택
"""
