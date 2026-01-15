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
    - Bi-Encoder보다 정확도 높음 (~15% 향상)
    - 세밀한 관련성 판단 가능

    [단점]
    - O(N) 비용 (각 문서마다 추론 필요)
    - 전체 검색에는 부적합, re-ranking에만 사용

    [지원 모델]
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (영어, 빠름)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (영어, 정확)
    - BAAI/bge-reranker-base (다국어)
    - BAAI/bge-reranker-v2-m3 (다국어, 최신)
    - bongsoo/kpf-cross-encoder-v1 (한국어 특화)

    [성능 벤치마크]
    - MS MARCO MRR@10: 0.39 (MiniLM-L-6) → 0.41 (MiniLM-L-12)
    - 한국어 금융 문서: bge-reranker가 더 효과적
    """

    # 모델별 설정
    MODEL_CONFIGS = {
        "ms-marco-MiniLM-L-6-v2": {
            "full_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_length": 512,
            "language": "en",
            "speed": "fast",
        },
        "ms-marco-MiniLM-L-12-v2": {
            "full_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "max_length": 512,
            "language": "en",
            "speed": "medium",
        },
        "bge-reranker-base": {
            "full_name": "BAAI/bge-reranker-base",
            "max_length": 512,
            "language": "multilingual",
            "speed": "medium",
        },
        "bge-reranker-v2-m3": {
            "full_name": "BAAI/bge-reranker-v2-m3",
            "max_length": 8192,
            "language": "multilingual",
            "speed": "slow",
        },
        "kpf-cross-encoder-v1": {
            "full_name": "bongsoo/kpf-cross-encoder-v1",
            "max_length": 512,
            "language": "ko",
            "speed": "medium",
        },
    }

    # 한국어 추천 모델 (우선순위)
    KOREAN_MODELS = [
        "bge-reranker-v2-m3",
        "bge-reranker-base",
        "kpf-cross-encoder-v1",
    ]

    def __init__(
        self,
        model_name: str = "bge-reranker-base",
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: 모델 이름 (짧은 이름 또는 전체 경로)
            max_length: 최대 토큰 길이 (None이면 모델 기본값)
            device: 디바이스 ("cuda", "cpu", None=자동)
            batch_size: 배치 크기
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._initialized = False

        # 모델 설정 로드
        if model_name in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model_name]
            self._full_model_name = config["full_name"]
            self._max_length = max_length or config["max_length"]
        else:
            # 사용자 지정 모델
            self._full_model_name = model_name
            self._max_length = max_length or 512

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
            import logging

            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

            print(f"Cross-Encoder 모델 로딩 중: {self._full_model_name}")

            self._model = CrossEncoder(
                self._full_model_name,
                max_length=self._max_length,
                device=self._device,
            )

            self._initialized = True
            print(f"Cross-Encoder 모델 로드 완료: {self._full_model_name}")

            return self._model

        except ImportError:
            raise ImportError(
                "sentence-transformers 패키지가 필요합니다:\n"
                "pip install sentence-transformers"
            )
        except Exception as e:
            print(f"Cross-Encoder 모델 로드 실패: {e}")
            print("KeywordReranker로 fallback")
            self._initialized = True
            self._model = None
            return None

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """
        Cross-Encoder로 재정렬

        Args:
            query: 검색 쿼리
            documents: 문서 리스트 [{"doc_id": ..., "content": ..., "score": ...}, ...]
            top_k: 반환할 문서 수

        Returns:
            재정렬된 문서 리스트
        """
        if not documents:
            return []

        # 모델 로드
        model = self._load_model()

        # 모델 로드 실패 시 KeywordReranker로 fallback
        if model is None:
            keyword_reranker = KeywordReranker()
            return keyword_reranker.rerank(query, documents, top_k)

        # 쿼리-문서 쌍 생성
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            # 문서가 너무 길면 앞부분만 사용
            if len(content) > self._max_length * 4:  # 대략적인 토큰 추정
                content = content[:self._max_length * 4]
            pairs.append([query, content])

        # Cross-Encoder로 점수 계산
        try:
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            print(f"Cross-Encoder 추론 실패: {e}")
            keyword_reranker = KeywordReranker()
            return keyword_reranker.rerank(query, documents, top_k)

        # 점수 정규화 (sigmoid로 0~1 범위)
        import numpy as np

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

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        config = self.MODEL_CONFIGS.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "full_name": self._full_model_name,
            "max_length": self._max_length,
            "language": config.get("language", "unknown"),
            "speed": config.get("speed", "unknown"),
            "initialized": self._initialized,
        }


class EnsembleReranker(BaseReranker):
    """
    앙상블 Re-ranker

    여러 Re-ranker 결과를 결합하여 더 안정적인 순위 생성

    [전략]
    - 가중치 기반 점수 결합
    - RRF (Reciprocal Rank Fusion)
    """

    def __init__(
        self,
        rerankers: List[Tuple[BaseReranker, float]],
        fusion_method: str = "weighted"
    ):
        """
        Args:
            rerankers: [(reranker, weight), ...] 리스트
            fusion_method: "weighted" 또는 "rrf"
        """
        self.rerankers = rerankers
        self.fusion_method = fusion_method

    @property
    def name(self) -> str:
        names = [r[0].name for r in self.rerankers]
        return f"ensemble ({', '.join(names)})"

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """앙상블 재정렬"""
        if not documents:
            return []

        # 각 reranker 결과 수집
        all_results = []
        for reranker, weight in self.rerankers:
            try:
                results = reranker.rerank(query, documents, top_k=len(documents))
                all_results.append((results, weight))
            except Exception as e:
                print(f"{reranker.name} 실패: {e}")
                continue

        if not all_results:
            # 모든 reranker 실패 시 원본 순서 유지
            return [
                RankedDocument(
                    doc_id=doc.get("doc_id", f"doc_{i}"),
                    content=doc.get("content", ""),
                    original_rank=i + 1,
                    new_rank=i + 1,
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        # 점수 결합
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict] = {}

        for results, weight in all_results:
            if self.fusion_method == "rrf":
                # RRF 방식
                for result in results:
                    rrf_score = 1.0 / (60 + result.new_rank)
                    doc_scores[result.doc_id] = doc_scores.get(result.doc_id, 0.0) + rrf_score * weight
                    doc_data[result.doc_id] = {
                        "content": result.content,
                        "original_score": result.original_score,
                        "metadata": result.metadata,
                    }
            else:
                # 가중치 평균 방식
                for result in results:
                    doc_scores[result.doc_id] = doc_scores.get(result.doc_id, 0.0) + result.rerank_score * weight
                    doc_data[result.doc_id] = {
                        "content": result.content,
                        "original_score": result.original_score,
                        "metadata": result.metadata,
                    }

        # 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 결과 생성
        results = []
        for new_rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            data = doc_data[doc_id]
            results.append(RankedDocument(
                doc_id=doc_id,
                content=data["content"],
                original_rank=0,  # 앙상블에서는 의미 없음
                new_rank=new_rank,
                original_score=data["original_score"],
                rerank_score=score,
                metadata=data["metadata"]
            ))

        return results


# Re-ranker 팩토리
RERANKER_REGISTRY = {
    "keyword": KeywordReranker,
    "llm": LLMReranker,
    "cross_encoder": CrossEncoderReranker,
    "ensemble": EnsembleReranker,
}


def get_reranker(
    reranker_type: str = "keyword",
    **kwargs
) -> BaseReranker:
    """
    Re-ranker 팩토리 함수

    Args:
        reranker_type: "keyword", "llm", "cross_encoder", "ensemble"
        **kwargs: reranker별 추가 인자

    Returns:
        BaseReranker 인스턴스

    Examples:
        # 키워드 기반 (기본, 빠름)
        reranker = get_reranker("keyword")

        # Cross-Encoder (정확, 느림)
        reranker = get_reranker("cross_encoder", model_name="bge-reranker-base")

        # 앙상블 (Keyword + CrossEncoder)
        keyword = get_reranker("keyword")
        cross = get_reranker("cross_encoder")
        ensemble = get_reranker("ensemble", rerankers=[(keyword, 0.3), (cross, 0.7)])
    """
    if reranker_type not in RERANKER_REGISTRY:
        raise ValueError(
            f"Unknown reranker: {reranker_type}. "
            f"Available: {list(RERANKER_REGISTRY.keys())}"
        )

    return RERANKER_REGISTRY[reranker_type](**kwargs)


def get_best_reranker_for_korean() -> BaseReranker:
    """
    한국어 문서에 최적화된 Re-ranker 반환

    CrossEncoder(bge-reranker-base)를 우선 시도,
    실패 시 KeywordReranker로 fallback
    """
    try:
        reranker = CrossEncoderReranker(model_name="bge-reranker-base")
        # 테스트 추론
        test_docs = [{"doc_id": "test", "content": "테스트 문서", "score": 1.0}]
        reranker.rerank("테스트", test_docs, top_k=1)
        return reranker
    except Exception:
        return KeywordReranker()


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
