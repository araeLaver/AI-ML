# -*- coding: utf-8 -*-
"""
문서 로더 테스트

[테스트 범위]
- 텍스트 분할 (청킹)
- Document 객체 생성
- 메타데이터 관리
"""

import pytest
from src.rag.document_loader import (
    Document,
    ChunkingConfig,
    RecursiveTextSplitter,
    TextLoader,
    DocumentLoaderFactory
)


class TestDocument:
    """Document 클래스 테스트"""

    def test_document_creation(self):
        """Document 객체 생성 테스트"""
        doc = Document(
            content="테스트 내용",
            metadata={"source": "test"}
        )
        assert doc.content == "테스트 내용"
        assert doc.metadata["source"] == "test"

    def test_document_id_generation(self):
        """Document ID 자동 생성 테스트"""
        doc = Document(
            content="테스트 내용",
            metadata={"source": "test", "chunk_index": 0}
        )
        doc_id = doc.id
        assert doc_id is not None
        assert "test" in doc_id
        assert len(doc_id) > 0

    def test_same_content_same_hash(self):
        """동일 콘텐츠는 동일 해시 생성"""
        doc1 = Document(content="동일한 내용", metadata={"source": "a", "chunk_index": 0})
        doc2 = Document(content="동일한 내용", metadata={"source": "a", "chunk_index": 0})
        # 해시 부분이 동일해야 함
        assert doc1.id.split("_")[-1] == doc2.id.split("_")[-1]


class TestChunkingConfig:
    """청킹 설정 테스트"""

    def test_default_config(self):
        """기본 설정값 테스트"""
        config = ChunkingConfig()
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 100

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = ChunkingConfig(chunk_size=300, chunk_overlap=50)
        assert config.chunk_size == 300
        assert config.chunk_overlap == 50


class TestRecursiveTextSplitter:
    """재귀적 텍스트 분할기 테스트"""

    def test_short_text_no_split(self, text_splitter):
        """짧은 텍스트는 분할하지 않음"""
        short_text = "짧은 텍스트입니다."
        chunks = text_splitter.split(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_long_text_splits(self, sample_long_text, text_splitter):
        """긴 텍스트는 여러 청크로 분할"""
        chunks = text_splitter.split(sample_long_text)
        assert len(chunks) > 1
        # 각 청크가 최대 크기 이하
        for chunk in chunks:
            assert len(chunk) <= text_splitter.config.chunk_size + 100  # 약간의 여유

    def test_empty_text(self, text_splitter):
        """빈 텍스트 처리"""
        chunks = text_splitter.split("")
        assert chunks == []

    def test_whitespace_only(self, text_splitter):
        """공백만 있는 텍스트"""
        chunks = text_splitter.split("   \n\n   ")
        assert chunks == []

    def test_preserves_content(self, sample_long_text, text_splitter):
        """분할 후에도 핵심 내용 보존"""
        chunks = text_splitter.split(sample_long_text)
        combined = " ".join(chunks)
        # 원본의 핵심 키워드가 보존되어야 함
        assert "분산 투자" in combined
        assert "장기 투자" in combined


class TestTextLoader:
    """텍스트 로더 테스트"""

    def test_load_from_string(self, sample_long_text, chunking_config):
        """문자열에서 로드"""
        loader = TextLoader(chunking_config=chunking_config)
        documents = loader.load_from_string(sample_long_text, "테스트 문서")

        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.metadata["source"] == "테스트 문서" for doc in documents)

    def test_metadata_includes_chunk_info(self, sample_long_text, chunking_config):
        """메타데이터에 청크 정보 포함"""
        loader = TextLoader(chunking_config=chunking_config)
        documents = loader.load_from_string(sample_long_text, "테스트")

        for i, doc in enumerate(documents):
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert doc.metadata["file_type"] == "text"


class TestDocumentLoaderFactory:
    """문서 로더 팩토리 테스트"""

    def test_get_txt_loader(self):
        """TXT 파일 로더 반환"""
        loader = DocumentLoaderFactory.get_loader("test.txt")
        assert isinstance(loader, TextLoader)

    def test_get_md_loader(self):
        """MD 파일 로더 반환"""
        loader = DocumentLoaderFactory.get_loader("test.md")
        assert isinstance(loader, TextLoader)

    def test_unsupported_format(self):
        """지원하지 않는 형식 예외"""
        with pytest.raises(ValueError) as exc_info:
            DocumentLoaderFactory.get_loader("test.xlsx")
        assert "지원하지 않는" in str(exc_info.value)

    def test_supported_extensions(self):
        """지원 확장자 목록"""
        extensions = DocumentLoaderFactory.get_supported_extensions()
        assert ".pdf" in extensions
        assert ".txt" in extensions
        assert ".md" in extensions
