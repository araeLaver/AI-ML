# -*- coding: utf-8 -*-
"""
임베딩 서비스 모듈

[백엔드 개발자 관점]
- 외부 API 클라이언트와 동일한 패턴
- 의존성 주입 가능한 구조
- 인터페이스 분리로 교체 용이 (Ollama → OpenAI 전환 가능)
"""

import ollama
from abc import ABC, abstractmethod
from typing import List
from chromadb.utils import embedding_functions


class EmbeddingService(ABC):
    """임베딩 서비스 인터페이스 (Strategy 패턴)"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 텍스트 임베딩"""
        pass


class OllamaEmbedding(EmbeddingService):
    """
    Ollama 기반 임베딩 서비스

    장점:
    - 무료, 로컬 실행
    - API 키 불필요
    - 프라이버시 보장
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def embed(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        result = ollama.embed(model=self.model, input=text)
        return result.embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 텍스트 임베딩 생성"""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings


class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    ChromaDB용 임베딩 함수 어댑터

    ChromaDB가 요구하는 인터페이스에 맞춤
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            result = ollama.embed(model=self.model, input=text)
            embeddings.append(result.embeddings[0])
        return embeddings
