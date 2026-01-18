# -*- coding: utf-8 -*-
"""
Vector Store Module

ChromaDB In-Memory + HuggingFace Embedding API
"""

import hashlib
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """벡터 검색 결과"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """
    In-Memory Vector Store

    HuggingFace Inference API로 임베딩 생성
    NumPy 기반 코사인 유사도 검색

    [특징]
    - ChromaDB 대신 순수 NumPy (메모리 절약)
    - HuggingFace 임베딩 API 사용
    - Fallback: 간단한 문자 임베딩
    """

    HF_API_URL = "https://api-inference.huggingface.co/models/"
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 차원

    def __init__(
        self,
        hf_token: str = "",
        model: str = DEFAULT_MODEL,
        timeout: int = 30
    ):
        """
        Args:
            hf_token: HuggingFace API 토큰
            model: 임베딩 모델 이름
            timeout: API 타임아웃 (초)
        """
        self.hf_token = hf_token
        self.model = model
        self.timeout = timeout

        # 저장소
        self.doc_ids: List[str] = []
        self.doc_contents: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def _get_embeddings_from_api(self, texts: List[str]) -> Optional[List[List[float]]]:
        """HuggingFace API로 임베딩 생성"""
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        try:
            response = requests.post(
                self.HF_API_URL + self.model,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except Exception:
            return None

    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        간단한 문자 기반 임베딩 (Fallback)

        API 실패 시 사용하는 경량 임베딩
        """
        text = text.lower()
        vec = [0.0] * dim

        # 문자 기반 해싱
        for i, char in enumerate(text):
            idx = (ord(char) * (i + 1)) % dim
            vec[idx] += 1.0

        # 2-gram 추가
        for i in range(len(text) - 1):
            gram = text[i:i+2]
            h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16) % dim
            vec[h] += 0.5

        # L2 정규화
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """임베딩 생성 (API 우선, Fallback 지원)"""
        # API 시도
        api_embeddings = self._get_embeddings_from_api(texts)
        if api_embeddings is not None:
            return np.array(api_embeddings)

        # Fallback: 간단한 임베딩
        return np.array([self._simple_embedding(t) for t in texts])

    def add_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        문서 추가

        Args:
            documents: 문서 내용 리스트
            doc_ids: 문서 ID 리스트
            metadatas: 메타데이터 리스트 (선택)
        """
        if not documents:
            return

        # 새 문서 임베딩
        new_embeddings = self._get_embeddings(documents)

        # 기존 데이터에 추가
        self.doc_ids.extend(doc_ids)
        self.doc_contents.extend(documents)
        self.doc_metadatas.extend(metadatas or [{} for _ in documents])

        # 임베딩 배열 업데이트
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> List[VectorSearchResult]:
        """
        벡터 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트 (유사도 내림차순)
        """
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []

        # 쿼리 임베딩
        query_embedding = self._get_embeddings([query])[0]

        # 코사인 유사도 계산
        # cos_sim = (A . B) / (||A|| * ||B||)
        # 임베딩이 정규화되어 있으면 dot product만으로 충분
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(doc_norms, query_norm)

        # 상위 k개 인덱스
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 결과 생성
        results = []
        for idx in top_indices:
            results.append(VectorSearchResult(
                doc_id=self.doc_ids[idx],
                content=self.doc_contents[idx],
                score=float(similarities[idx]),
                metadata=self.doc_metadatas[idx]
            ))

        return results

    def clear(self):
        """저장소 초기화"""
        self.doc_ids = []
        self.doc_contents = []
        self.doc_metadatas = []
        self.embeddings = None

    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        return {
            "num_documents": len(self.doc_ids),
            "embedding_dim": self.EMBEDDING_DIM if self.embeddings is not None else 0,
            "model": self.model,
            "memory_mb": round(self.embeddings.nbytes / 1024 / 1024, 2) if self.embeddings is not None else 0,
        }
