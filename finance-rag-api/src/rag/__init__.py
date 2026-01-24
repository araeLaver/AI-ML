# -*- coding: utf-8 -*-
"""RAG 모듈"""
from .embedding import OllamaEmbedding, OllamaEmbeddingFunction, EmbeddingService
from .vectorstore import VectorStoreService
from .rag_service import RAGService, RAGResponse
from .document_loader import (
    Document,
    ChunkingConfig,
    PDFLoader,
    TextLoader,
    RecursiveTextSplitter,
    DocumentLoaderFactory
)
from .financial_dictionary import (
    FINANCIAL_SYNONYMS,
    get_synonyms,
    get_canonical_term,
    get_statistics as get_dictionary_statistics,
)
from .query_expander import (
    QueryExpander,
    expand_query,
    normalize_query,
)

__all__ = [
    "OllamaEmbedding",
    "OllamaEmbeddingFunction",
    "EmbeddingService",
    "VectorStoreService",
    "RAGService",
    "RAGResponse",
    "Document",
    "ChunkingConfig",
    "PDFLoader",
    "TextLoader",
    "RecursiveTextSplitter",
    "DocumentLoaderFactory",
    # Query Expansion
    "FINANCIAL_SYNONYMS",
    "get_synonyms",
    "get_canonical_term",
    "get_dictionary_statistics",
    "QueryExpander",
    "expand_query",
    "normalize_query",
]
