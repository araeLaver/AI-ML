# -*- coding: utf-8 -*-
"""
벡터 스토어 서비스 모듈

[백엔드 개발자 관점]
- Repository 패턴과 유사
- 데이터 접근 추상화
- 영속성 관리 (인메모리 vs 디스크)
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from pathlib import Path

from .embedding import OllamaEmbeddingFunction


class VectorStoreService:
    """
    벡터 스토어 서비스 (ChromaDB)

    SQL DB와 비교:
    - collection = table
    - document = row
    - embedding = indexed column (but vector similarity)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "finance_docs"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_fn = OllamaEmbeddingFunction()

    def _get_client(self) -> chromadb.Client:
        """ChromaDB 클라이언트 지연 초기화"""
        if self._client is None:
            if self.persist_dir:
                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self._client = chromadb.Client()
        return self._client

    def _get_collection(self):
        """컬렉션 지연 초기화"""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"description": "금융 문서 RAG 시스템"}
            )
        return self._collection

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        문서 추가

        SQL 비유: INSERT INTO finance_docs (id, content, metadata) VALUES ...
        """
        collection = self._get_collection()
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        유사 문서 검색

        SQL 비유: SELECT * FROM finance_docs
                  ORDER BY similarity(embedding, query_embedding) DESC
                  LIMIT top_k
        """
        collection = self._get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def get_document_count(self) -> int:
        """저장된 문서 수 조회"""
        collection = self._get_collection()
        return collection.count()

    def delete_collection(self) -> None:
        """컬렉션 삭제 (테스트/리셋용)"""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
        except ValueError:
            pass

    def list_all_documents(self, limit: int = 100) -> Dict[str, Any]:
        """모든 문서 조회 (디버그용)"""
        collection = self._get_collection()
        results = collection.get(limit=limit, include=["documents", "metadatas"])
        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }
