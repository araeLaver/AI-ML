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
from .embedding_trainer import (
    TrainingExample,
    TrainingConfig,
    TrainingDataGenerator,
    FinancialEmbeddingTrainer,
    prepare_training_data,
    train_financial_embedding,
)
from .embedding_evaluator import (
    EvaluationResult,
    EmbeddingEvaluator,
    FinancialEvaluationDataset,
    evaluate_embedding_model,
)
from .conversation_manager import (
    Message,
    ConversationSession,
    ConversationManager,
    create_session,
    add_message,
    get_history,
)
from .context_resolver import (
    ContextResolver,
    ResolvedQuery,
    resolve_query,
    extract_entities,
)
from .multiturn_rag import (
    MultiTurnRAGService,
    MultiTurnResponse,
    create_chat_session,
    chat,
)
from .cache_service import (
    CacheEntry,
    CacheStats,
    CacheBackend,
    InMemoryCache,
    RedisCache,
    CacheService,
    CachedRAGService,
    get_cache_service,
    cache_result,
    get_cached,
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
    # Embedding Training
    "TrainingExample",
    "TrainingConfig",
    "TrainingDataGenerator",
    "FinancialEmbeddingTrainer",
    "prepare_training_data",
    "train_financial_embedding",
    # Embedding Evaluation
    "EvaluationResult",
    "EmbeddingEvaluator",
    "FinancialEvaluationDataset",
    "evaluate_embedding_model",
    # Multi-turn Conversation
    "Message",
    "ConversationSession",
    "ConversationManager",
    "create_session",
    "add_message",
    "get_history",
    "ContextResolver",
    "ResolvedQuery",
    "resolve_query",
    "extract_entities",
    "MultiTurnRAGService",
    "MultiTurnResponse",
    "create_chat_session",
    "chat",
    # Caching
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "InMemoryCache",
    "RedisCache",
    "CacheService",
    "CachedRAGService",
    "get_cache_service",
    "cache_result",
    "get_cached",
]
