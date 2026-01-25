# -*- coding: utf-8 -*-
"""
Finance RAG - 성능 최적화 모듈

[포함 기능]
- 배치 처리 (Batch Processing)
- 연결 풀링 (Connection Pooling)
- 비동기 처리 (Async Processing)
- 인덱스 최적화 (Index Optimization)
- 멀티레벨 캐싱 (Multi-level Caching)
"""

from .batch_processor import (
    BatchProcessor,
    EmbeddingBatchProcessor,
    QueryBatchProcessor,
    BatchConfig,
)
from .connection_pool import (
    ConnectionPool,
    RedisPool,
    ChromaPool,
    OllamaPool,
)
from .async_executor import (
    AsyncExecutor,
    ParallelQueryExecutor,
    BackgroundTaskManager,
)
from .index_optimizer import (
    IndexOptimizer,
    ChromaIndexOptimizer,
    IndexStats,
)
from .multilevel_cache import (
    MultiLevelCache,
    L1MemoryCache,
    L2RedisCache,
    SemanticCache,
)

__all__ = [
    # Batch Processing
    "BatchProcessor",
    "EmbeddingBatchProcessor",
    "QueryBatchProcessor",
    "BatchConfig",
    # Connection Pooling
    "ConnectionPool",
    "RedisPool",
    "ChromaPool",
    "OllamaPool",
    # Async Processing
    "AsyncExecutor",
    "ParallelQueryExecutor",
    "BackgroundTaskManager",
    # Index Optimization
    "IndexOptimizer",
    "ChromaIndexOptimizer",
    "IndexStats",
    # Multi-level Cache
    "MultiLevelCache",
    "L1MemoryCache",
    "L2RedisCache",
    "SemanticCache",
]
