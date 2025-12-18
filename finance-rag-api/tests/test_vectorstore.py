# -*- coding: utf-8 -*-
"""
벡터 스토어 테스트

[테스트 범위]
- 문서 추가
- 유사도 검색
- 메타데이터 필터링
"""

import pytest
from src.rag.vectorstore import VectorStoreService


class TestVectorStoreService:
    """VectorStoreService 테스트"""

    def test_add_documents(self, vectorstore, sample_texts):
        """문서 추가 테스트"""
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadatas = [{"source": "test"} for _ in sample_texts]

        vectorstore.add_documents(
            documents=sample_texts,
            ids=ids,
            metadatas=metadatas
        )

        count = vectorstore.get_document_count()
        assert count == len(sample_texts)

    def test_search_returns_relevant(self, vectorstore, sample_texts):
        """검색 결과 관련성 테스트"""
        # 문서 추가
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadatas = [{"source": "test"} for _ in sample_texts]
        vectorstore.add_documents(sample_texts, ids, metadatas)

        # ETF 관련 검색
        results = vectorstore.search("ETF가 뭔가요?", top_k=2)

        assert len(results["documents"]) > 0
        # ETF 관련 문서가 상위에 있어야 함
        assert any("ETF" in doc for doc in results["documents"])

    def test_search_with_top_k(self, vectorstore, sample_texts):
        """top_k 파라미터 테스트"""
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadatas = [{"source": "test"} for _ in sample_texts]
        vectorstore.add_documents(sample_texts, ids, metadatas)

        results = vectorstore.search("투자", top_k=2)
        assert len(results["documents"]) == 2

        results = vectorstore.search("투자", top_k=5)
        assert len(results["documents"]) == 5

    def test_search_returns_distances(self, vectorstore, sample_texts):
        """검색 결과에 거리(유사도) 포함"""
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        vectorstore.add_documents(sample_texts, ids)

        results = vectorstore.search("ETF", top_k=3)

        assert "distances" in results
        assert len(results["distances"]) == len(results["documents"])

    def test_search_empty_db(self, vectorstore):
        """빈 DB 검색"""
        results = vectorstore.search("아무거나", top_k=3)
        assert results["documents"] == []

    def test_list_all_documents(self, vectorstore, sample_texts):
        """전체 문서 조회"""
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadatas = [{"source": f"source_{i}"} for i in range(len(sample_texts))]
        vectorstore.add_documents(sample_texts, ids, metadatas)

        result = vectorstore.list_all_documents()

        assert len(result["ids"]) == len(sample_texts)
        assert len(result["documents"]) == len(sample_texts)

    def test_delete_collection(self, vectorstore, sample_texts):
        """컬렉션 삭제"""
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        vectorstore.add_documents(sample_texts, ids)

        assert vectorstore.get_document_count() > 0

        vectorstore.delete_collection()

        # 삭제 후 새 컬렉션은 비어있어야 함
        assert vectorstore.get_document_count() == 0

    def test_metadata_filter(self, vectorstore):
        """메타데이터 필터 검색"""
        # 서로 다른 출처의 문서 추가
        docs = [
            "ETF는 상장지수펀드입니다.",
            "채권은 안정적인 투자입니다.",
            "주식 시장이 상승했습니다."
        ]
        ids = ["doc_0", "doc_1", "doc_2"]
        metadatas = [
            {"source": "가이드", "category": "상품"},
            {"source": "가이드", "category": "상품"},
            {"source": "뉴스", "category": "시황"}
        ]
        vectorstore.add_documents(docs, ids, metadatas)

        # source 필터
        results = vectorstore.search(
            "투자",
            top_k=3,
            filter_metadata={"source": "가이드"}
        )

        # 가이드 출처만 반환되어야 함
        assert len(results["documents"]) == 2
