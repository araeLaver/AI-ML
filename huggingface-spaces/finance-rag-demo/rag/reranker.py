# -*- coding: utf-8 -*-
"""
Re-ranking Module

KeywordReranker (기본, 경량)
CrossEncoderReranker (선택, 330MB 추가)
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
    키워드 기반 Re-ranker (경량)

    간단한 규칙 기반 재정렬
    - 쿼리 키워드 매칭 점수
    - 문서 시작 부분 가중치
    - 정확한 구문 매칭 보너스

    [장점]
    - 추가 모델 불필요 (메모리 절약)
    - 빠른 처리 속도
    - HuggingFace Spaces 2GB 제한 대응
    """

    # 금융 도메인 중요 키워드 (가중치 부여)
    IMPORTANT_KEYWORDS = {
        "실적", "매출", "영업이익", "순이익", "전망", "투자",
        "HBM", "반도체", "AI", "배당", "금리", "환율",
        "삼성전자", "SK하이닉스", "네이버", "카카오",
    }

    @property
    def name(self) -> str:
        return "keyword"

    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 한글, 영문, 숫자 추출
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
                # 중요 키워드면 가중치 부여
                weight = 1.5 if keyword in self.IMPORTANT_KEYWORDS else 1.0
                score += min(count, 5) * 0.1 * weight

        # 2. 정확한 쿼리 구문 매칭 보너스
        if query.lower() in doc_text:
            score += 0.5

        # 3. 문서 시작 부분에 키워드 있으면 보너스
        first_100 = doc_text[:100]
        early_matches = sum(1 for kw in query_keywords if kw in first_100)
        score += early_matches * 0.2

        # 4. 정규화 (0~1 범위)
        max_possible = len(query_keywords) * 0.75 + 0.5 + len(query_keywords) * 0.2
        normalized_score = min(1.0, score / max(max_possible, 1))

        return normalized_score

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """키워드 기반 재정렬"""
        if not documents:
            return []

        scored_docs = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            original_score = doc.get("score", 0.0)
            rerank_score = self._calculate_score(query, content)

            # 원본 점수와 재정렬 점수 결합 (4:6 비율)
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


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder 기반 Re-ranker (선택적)

    [원리]
    - 쿼리+문서를 함께 입력하여 관련성 직접 예측
    - Bi-Encoder보다 정확도 높음

    [주의]
    - 추가 330MB 모델 다운로드 필요
    - HuggingFace Spaces 메모리 제한 확인 필요

    [지원 모델]
    - BAAI/bge-reranker-base (다국어)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (영어)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._device = device

    @property
    def name(self) -> str:
        return f"cross_encoder ({self.model_name})"

    def _load_model(self):
        """Cross-Encoder 모델 로드 (지연 로딩)"""
        if self._initialized:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self._device,
            )
            self._initialized = True
            return self._model

        except ImportError:
            print("sentence-transformers 패키지가 필요합니다. KeywordReranker로 fallback")
            self._initialized = True
            self._model = None
            return None
        except Exception as e:
            print(f"Cross-Encoder 모델 로드 실패: {e}. KeywordReranker로 fallback")
            self._initialized = True
            self._model = None
            return None

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """Cross-Encoder로 재정렬"""
        if not documents:
            return []

        # 모델 로드
        model = self._load_model()

        # 모델 로드 실패 시 KeywordReranker로 fallback
        if model is None:
            return KeywordReranker().rerank(query, documents, top_k)

        try:
            import numpy as np

            # 쿼리-문서 쌍 생성
            pairs = []
            for doc in documents:
                content = doc.get("content", "")
                # 문서가 너무 길면 앞부분만 사용
                if len(content) > 2000:
                    content = content[:2000]
                pairs.append([query, content])

            # Cross-Encoder로 점수 계산
            scores = model.predict(pairs, show_progress_bar=False)

            # sigmoid로 정규화
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            normalized_scores = sigmoid(np.array(scores))

            # 문서에 점수 추가
            scored_docs = []
            for i, (doc, score) in enumerate(zip(documents, normalized_scores)):
                scored_docs.append({
                    "doc": doc,
                    "original_rank": i + 1,
                    "original_score": doc.get("score", 0.0),
                    "rerank_score": float(score),
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

        except Exception as e:
            print(f"Cross-Encoder 추론 실패: {e}. KeywordReranker로 fallback")
            return KeywordReranker().rerank(query, documents, top_k)


def get_reranker(reranker_type: str = "keyword", **kwargs) -> BaseReranker:
    """
    Re-ranker 팩토리

    Args:
        reranker_type: "keyword" 또는 "cross_encoder"
        **kwargs: reranker별 추가 인자

    Returns:
        BaseReranker 인스턴스
    """
    if reranker_type == "keyword":
        return KeywordReranker()
    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker: {reranker_type}. Available: keyword, cross_encoder")
