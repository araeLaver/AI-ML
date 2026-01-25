# -*- coding: utf-8 -*-
"""
다국어 임베딩 모듈

[기능]
- 다국어 텍스트 임베딩
- Cross-lingual 임베딩
- 언어 정렬 임베딩
"""

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detection import Language, LanguageDetector, HybridDetector

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    text: str
    embedding: List[float]
    language: Language
    dimension: int
    model_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:100] if self.text else None,
            "dimension": self.dimension,
            "language": self.language.value,
            "model_name": self.model_name,
        }


class BaseEmbedding(ABC):
    """임베딩 기본 클래스"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 차원"""
        pass

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """텍스트 임베딩"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 임베딩"""
        pass


class SimpleHashEmbedding(BaseEmbedding):
    """
    해시 기반 간단 임베딩 (시뮬레이션용)

    실제 환경에서는 sentence-transformers 등 사용
    """

    def __init__(self, dim: int = 384, seed: Optional[int] = None):
        self._dim = dim
        self._rng = random.Random(seed)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        """해시 기반 임베딩"""
        # 텍스트의 해시를 시드로 사용
        self._rng.seed(hash(text))

        embedding = [self._rng.gauss(0, 1) for _ in range(self._dim)]

        # 정규화
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            language=Language.UNKNOWN,
            dimension=self._dim,
            model_name="hash_embedding",
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 임베딩"""
        return [self.embed(text) for text in texts]


class MultilingualEmbedding(BaseEmbedding):
    """
    다국어 임베딩

    다양한 언어를 동일 벡터 공간에 임베딩
    """

    def __init__(
        self,
        model_func: Optional[Callable[[str], List[float]]] = None,
        dim: int = 768,
        detector: Optional[LanguageDetector] = None,
    ):
        self._dim = dim
        self.model_func = model_func
        self.detector = detector or HybridDetector()

        # 시뮬레이션용 폴백
        self._fallback = SimpleHashEmbedding(dim=dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        """다국어 임베딩"""
        # 언어 감지
        detection = self.detector.detect(text)

        if self.model_func:
            embedding = self.model_func(text)
        else:
            result = self._fallback.embed(text)
            embedding = result.embedding

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            language=detection.language,
            dimension=self._dim,
            model_name="multilingual_embedding",
            metadata={"confidence": detection.confidence},
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 임베딩"""
        return [self.embed(text) for text in texts]


class CrossLingualEmbedding(BaseEmbedding):
    """
    Cross-Lingual 임베딩

    다른 언어 간 의미적 유사성 보존
    """

    def __init__(
        self,
        base_embedding: Optional[BaseEmbedding] = None,
        language_vectors: Optional[Dict[Language, List[float]]] = None,
    ):
        self.base_embedding = base_embedding or SimpleHashEmbedding(dim=768)
        self._dim = self.base_embedding.dimension

        # 언어별 오프셋 벡터 (언어 간 정렬용)
        self.language_vectors = language_vectors or self._init_language_vectors()

        self.detector = HybridDetector()

    def _init_language_vectors(self) -> Dict[Language, List[float]]:
        """언어별 벡터 초기화"""
        rng = random.Random(42)
        vectors = {}

        for lang in Language:
            if lang != Language.UNKNOWN:
                vec = [rng.gauss(0, 0.1) for _ in range(self._dim)]
                vectors[lang] = vec

        return vectors

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        """Cross-lingual 임베딩"""
        # 기본 임베딩
        base_result = self.base_embedding.embed(text)

        # 언어 감지
        detection = self.detector.detect(text)
        lang = detection.language

        # 언어 벡터 적용
        embedding = base_result.embedding.copy()
        if lang in self.language_vectors:
            lang_vec = self.language_vectors[lang]
            embedding = [e + l for e, l in zip(embedding, lang_vec)]

            # 재정규화
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            language=lang,
            dimension=self._dim,
            model_name="cross_lingual_embedding",
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 임베딩"""
        return [self.embed(text) for text in texts]

    def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Cross-lingual 유사도 계산"""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        # 코사인 유사도
        dot_product = sum(a * b for a, b in zip(emb1.embedding, emb2.embedding))
        return dot_product


class LanguageAlignedEmbedding(BaseEmbedding):
    """
    언어 정렬 임베딩

    특정 기준 언어(예: 영어)에 다른 언어를 정렬
    """

    def __init__(
        self,
        base_embedding: Optional[BaseEmbedding] = None,
        pivot_language: Language = Language.ENGLISH,
        alignment_matrix: Optional[Dict[Language, List[List[float]]]] = None,
    ):
        self.base_embedding = base_embedding or SimpleHashEmbedding(dim=384)
        self._dim = self.base_embedding.dimension
        self.pivot_language = pivot_language

        # 정렬 행렬 (각 언어 -> pivot 언어)
        self.alignment_matrix = alignment_matrix or {}
        self.detector = HybridDetector()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        """정렬된 임베딩"""
        base_result = self.base_embedding.embed(text)
        detection = self.detector.detect(text)
        lang = detection.language

        embedding = base_result.embedding

        # pivot 언어가 아닌 경우 정렬 변환 적용
        if lang != self.pivot_language and lang in self.alignment_matrix:
            matrix = self.alignment_matrix[lang]
            embedding = self._apply_matrix(embedding, matrix)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            language=lang,
            dimension=self._dim,
            model_name="language_aligned_embedding",
            metadata={"aligned_to": self.pivot_language.value},
        )

    def _apply_matrix(
        self,
        vector: List[float],
        matrix: List[List[float]],
    ) -> List[float]:
        """행렬 변환 적용"""
        if not matrix:
            return vector

        result = []
        for row in matrix:
            val = sum(v * m for v, m in zip(vector, row))
            result.append(val)
        return result

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 임베딩"""
        return [self.embed(text) for text in texts]

    def learn_alignment(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_lang: Language,
    ) -> None:
        """
        정렬 행렬 학습 (간소화)

        실제로는 Procrustes 분석 등 사용
        """
        # 간단한 평균 차이 벡터 계산
        source_embs = [self.base_embedding.embed(t).embedding for t in source_texts]
        target_embs = [self.base_embedding.embed(t).embedding for t in target_texts]

        # 평균 계산
        n = len(source_embs)
        if n == 0:
            return

        # 간소화된 정렬: 평균 오프셋
        offset = []
        for i in range(self._dim):
            src_mean = sum(e[i] for e in source_embs) / n
            tgt_mean = sum(e[i] for e in target_embs) / n
            offset.append(tgt_mean - src_mean)

        # 단위 행렬 + 오프셋으로 간소화
        identity_with_offset = [
            [1.0 if i == j else 0.0 for j in range(self._dim)]
            for i in range(self._dim)
        ]

        self.alignment_matrix[source_lang] = identity_with_offset


class EmbeddingRegistry:
    """
    임베딩 레지스트리

    다양한 임베딩 모델 관리
    """

    def __init__(self):
        self._embeddings: Dict[str, BaseEmbedding] = {}
        self._default_name: Optional[str] = None

        # 기본 임베딩 등록
        self.register("simple_hash", SimpleHashEmbedding())
        self.register("multilingual", MultilingualEmbedding())
        self.set_default("multilingual")

    def register(self, name: str, embedding: BaseEmbedding) -> None:
        """임베딩 등록"""
        self._embeddings[name] = embedding
        logger.info(f"Registered embedding: {name}")

    def get(self, name: str) -> Optional[BaseEmbedding]:
        """임베딩 조회"""
        return self._embeddings.get(name)

    def set_default(self, name: str) -> None:
        """기본 임베딩 설정"""
        if name in self._embeddings:
            self._default_name = name

    def get_default(self) -> Optional[BaseEmbedding]:
        """기본 임베딩 반환"""
        if self._default_name:
            return self._embeddings.get(self._default_name)
        return None

    def embed(self, text: str, model_name: Optional[str] = None) -> EmbeddingResult:
        """텍스트 임베딩"""
        embedding = (
            self._embeddings.get(model_name) if model_name
            else self.get_default()
        )

        if embedding is None:
            raise ValueError(f"Embedding model not found: {model_name}")

        return embedding.embed(text)

    def embed_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
    ) -> List[EmbeddingResult]:
        """배치 임베딩"""
        embedding = (
            self._embeddings.get(model_name) if model_name
            else self.get_default()
        )

        if embedding is None:
            raise ValueError(f"Embedding model not found: {model_name}")

        return embedding.embed_batch(texts)

    def list_models(self) -> List[str]:
        """등록된 모델 목록"""
        return list(self._embeddings.keys())

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """모델 정보"""
        embedding = self._embeddings.get(name)
        if embedding:
            return {
                "name": name,
                "dimension": embedding.dimension,
                "type": type(embedding).__name__,
            }
        return None
