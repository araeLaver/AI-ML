# -*- coding: utf-8 -*-
"""
BM25 Keyword Search Module

BM25 알고리즘 기반 키워드 검색
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tokenizer import BaseTokenizer, get_tokenizer


@dataclass
class BM25Result:
    """BM25 검색 결과"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class BM25:
    """
    BM25 키워드 검색 알고리즘

    [BM25 수식]
    score(D,Q) = sum IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))

    파라미터:
    - k1: 용어 빈도 포화도 (기본 1.5)
    - b: 문서 길이 정규화 (기본 0.75)

    [특징]
    - TF-IDF보다 문서 길이 정규화 우수
    - 한국어 2-gram 토크나이저 사용
    - 메모리 효율적 (ChromaDB 대신 순수 Python)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[BaseTokenizer] = None
    ):
        """
        Args:
            k1: 용어 빈도 포화도 (높을수록 TF 영향 증가)
            b: 문서 길이 정규화 (0=무시, 1=완전 정규화)
            tokenizer: 토크나이저 (None이면 기본 사용)
        """
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or get_tokenizer()

        # 인덱스 데이터
        self.doc_ids: List[str] = []
        self.doc_contents: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.avg_doc_length: float = 0.0
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    def fit(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        문서 색인

        Args:
            documents: 문서 내용 리스트
            doc_ids: 문서 ID 리스트
            metadatas: 메타데이터 리스트 (선택)
        """
        self.doc_contents = documents
        self.doc_ids = doc_ids
        self.doc_metadatas = metadatas or [{} for _ in documents]

        # 초기화
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.vocab = set()
        doc_freqs: Dict[str, int] = defaultdict(int)

        # 문서별 토큰화 및 통계 계산
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
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
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0.0

        # IDF 계산
        n_docs = len(documents)
        for term, df in doc_freqs.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        """
        BM25 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트 (점수 내림차순)
        """
        if not self.doc_contents:
            return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

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
                # score += IDF * (TF * (k1+1)) / (TF + k1 * (1 - b + b * |D|/avgdl))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1))
                score += idf * numerator / denominator

            if score > 0:
                scores.append((doc_idx, score))

        # 점수 기준 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 반환
        results = []
        for doc_idx, score in scores[:top_k]:
            results.append(BM25Result(
                doc_id=self.doc_ids[doc_idx],
                content=self.doc_contents[doc_idx],
                score=score,
                metadata=self.doc_metadatas[doc_idx]
            ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """색인 통계"""
        return {
            "num_documents": len(self.doc_contents),
            "vocab_size": len(self.vocab),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "tokenizer": self.tokenizer.get_name(),
        }
