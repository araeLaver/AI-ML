# -*- coding: utf-8 -*-
"""
배치 처리 모듈

[기능]
- 대량 임베딩 배치 처리
- 쿼리 배치 처리
- 적응형 배치 크기 조절
- 진행 상황 추적
- 재시도 로직

[사용 예시]
>>> processor = EmbeddingBatchProcessor(batch_size=100)
>>> embeddings = processor.process(documents)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """배치 처리 설정"""
    batch_size: int = 100
    max_workers: int = 4
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    adaptive_batch_size: bool = True
    min_batch_size: int = 10
    max_batch_size: int = 500
    target_latency_ms: float = 1000.0


@dataclass
class BatchResult(Generic[R]):
    """배치 처리 결과"""
    results: List[R] = field(default_factory=list)
    failed_indices: List[int] = field(default_factory=list)
    total_time: float = 0.0
    batch_count: int = 0
    avg_batch_time: float = 0.0
    items_per_second: float = 0.0


@dataclass
class BatchStats:
    """배치 처리 통계"""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    total_batches: int = 0
    avg_batch_time_ms: float = 0.0
    throughput_per_sec: float = 0.0
    current_batch_size: int = 0


class BatchProcessor(ABC, Generic[T, R]):
    """
    배치 처리기 베이스 클래스

    [특징]
    - 제네릭 타입 지원
    - 적응형 배치 크기
    - 재시도 로직
    - 진행 상황 콜백
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._current_batch_size = self.config.batch_size
        self._batch_times: List[float] = []
        self._stats = BatchStats()

    @abstractmethod
    def _process_batch(self, batch: List[T]) -> List[R]:
        """단일 배치 처리 (하위 클래스에서 구현)"""
        pass

    def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """
        전체 항목 배치 처리

        Args:
            items: 처리할 항목 리스트
            progress_callback: 진행 상황 콜백 (processed, total)

        Returns:
            BatchResult: 처리 결과
        """
        if not items:
            return BatchResult()

        start_time = time.time()
        results: List[R] = []
        failed_indices: List[int] = []
        batch_count = 0

        self._stats.total_items = len(items)
        self._stats.processed_items = 0

        for batch_start in range(0, len(items), self._current_batch_size):
            batch_end = min(batch_start + self._current_batch_size, len(items))
            batch = items[batch_start:batch_end]
            batch_count += 1

            batch_start_time = time.time()

            try:
                batch_results = self._process_batch_with_retry(batch)
                results.extend(batch_results)
                self._stats.processed_items += len(batch)
            except Exception as e:
                logger.error(f"Batch {batch_count} failed: {e}")
                # 실패한 항목 인덱스 기록
                failed_indices.extend(range(batch_start, batch_end))
                self._stats.failed_items += len(batch)
                # None으로 채워서 인덱스 유지
                results.extend([None] * len(batch))

            batch_time = time.time() - batch_start_time
            self._batch_times.append(batch_time)

            # 적응형 배치 크기 조절
            if self.config.adaptive_batch_size:
                self._adjust_batch_size(batch_time * 1000, len(batch))

            # 진행 상황 콜백
            if progress_callback:
                progress_callback(len(results), len(items))

        total_time = time.time() - start_time

        return BatchResult(
            results=results,
            failed_indices=failed_indices,
            total_time=total_time,
            batch_count=batch_count,
            avg_batch_time=total_time / batch_count if batch_count > 0 else 0,
            items_per_second=len(items) / total_time if total_time > 0 else 0,
        )

    def _process_batch_with_retry(self, batch: List[T]) -> List[R]:
        """재시도 로직이 포함된 배치 처리"""
        last_exception = None

        for attempt in range(self.config.retry_count):
            try:
                return self._process_batch(batch)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Batch processing attempt {attempt + 1}/{self.config.retry_count} failed: {e}"
                )
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))

        raise last_exception

    def _adjust_batch_size(self, batch_time_ms: float, batch_size: int):
        """적응형 배치 크기 조절"""
        target = self.config.target_latency_ms

        if batch_time_ms > target * 1.5:
            # 너무 느림 - 배치 크기 감소
            new_size = max(
                self.config.min_batch_size,
                int(self._current_batch_size * 0.7)
            )
        elif batch_time_ms < target * 0.5:
            # 너무 빠름 - 배치 크기 증가
            new_size = min(
                self.config.max_batch_size,
                int(self._current_batch_size * 1.3)
            )
        else:
            new_size = self._current_batch_size

        if new_size != self._current_batch_size:
            logger.debug(
                f"Adjusting batch size: {self._current_batch_size} -> {new_size} "
                f"(batch_time={batch_time_ms:.1f}ms)"
            )
            self._current_batch_size = new_size
            self._stats.current_batch_size = new_size

    def get_stats(self) -> BatchStats:
        """통계 조회"""
        if self._batch_times:
            self._stats.avg_batch_time_ms = np.mean(self._batch_times) * 1000
            self._stats.throughput_per_sec = (
                self._stats.processed_items / sum(self._batch_times)
                if sum(self._batch_times) > 0 else 0
            )
        self._stats.current_batch_size = self._current_batch_size
        return self._stats


class EmbeddingBatchProcessor(BatchProcessor[str, List[float]]):
    """
    임베딩 배치 처리기

    대량의 텍스트를 효율적으로 임베딩
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        config: Optional[BatchConfig] = None,
    ):
        super().__init__(config)
        self._embedding_fn = embedding_fn

    def set_embedding_function(
        self, fn: Callable[[List[str]], List[List[float]]]
    ):
        """임베딩 함수 설정"""
        self._embedding_fn = fn

    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """배치 임베딩 처리"""
        if self._embedding_fn is None:
            raise ValueError("Embedding function not set")

        return self._embedding_fn(batch)


class QueryBatchProcessor(BatchProcessor[str, Dict[str, Any]]):
    """
    쿼리 배치 처리기

    대량의 쿼리를 병렬로 처리
    """

    def __init__(
        self,
        query_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        config: Optional[BatchConfig] = None,
    ):
        super().__init__(config)
        self._query_fn = query_fn
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers
        )

    def set_query_function(self, fn: Callable[[str], Dict[str, Any]]):
        """쿼리 함수 설정"""
        self._query_fn = fn

    def _process_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """배치 쿼리 처리 (병렬)"""
        if self._query_fn is None:
            raise ValueError("Query function not set")

        futures = [
            self._executor.submit(self._query_fn, query)
            for query in batch
        ]

        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.config.timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Query failed: {e}")
                results.append({"error": str(e)})

        return results

    def __del__(self):
        self._executor.shutdown(wait=False)


class ParallelBatchProcessor(Generic[T, R]):
    """
    병렬 배치 처리기

    여러 배치 처리기를 병렬로 실행
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_parallel(
        self,
        processor: BatchProcessor[T, R],
        item_groups: List[List[T]],
    ) -> List[BatchResult[R]]:
        """여러 그룹을 병렬로 배치 처리"""
        futures = [
            self._executor.submit(processor.process, items)
            for items in item_groups
        ]

        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Parallel batch failed: {e}")
                results.append(BatchResult(failed_indices=list(range(len(item_groups)))))

        return results

    def __del__(self):
        self._executor.shutdown(wait=False)


class StreamingBatchProcessor(Generic[T, R]):
    """
    스트리밍 배치 처리기

    메모리 효율적인 스트리밍 처리
    """

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        batch_size: int = 100,
    ):
        self.process_fn = process_fn
        self.batch_size = batch_size

    def process_stream(
        self, items: Iterator[T]
    ) -> Iterator[R]:
        """스트리밍 배치 처리"""
        batch: List[T] = []

        for item in items:
            batch.append(item)

            if len(batch) >= self.batch_size:
                yield from self.process_fn(batch)
                batch = []

        # 남은 배치 처리
        if batch:
            yield from self.process_fn(batch)

    async def process_stream_async(
        self, items: Iterator[T]
    ):
        """비동기 스트리밍 배치 처리"""
        batch: List[T] = []

        for item in items:
            batch.append(item)

            if len(batch) >= self.batch_size:
                results = await asyncio.to_thread(self.process_fn, batch)
                for result in results:
                    yield result
                batch = []

        if batch:
            results = await asyncio.to_thread(self.process_fn, batch)
            for result in results:
                yield result
