# -*- coding: utf-8 -*-
"""
배치 처리 모듈

문서 일괄 처리, 벌크 임베딩, 비동기 작업 관리
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, TypeVar

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class JobStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchConfig:
    """배치 처리 설정

    Attributes:
        batch_size: 배치 크기
        max_concurrent: 최대 동시 실행 수
        retry_count: 재시도 횟수
        retry_delay: 재시도 대기 시간 (초)
        timeout: 작업 타임아웃 (초)
    """
    batch_size: int = 100
    max_concurrent: int = 5
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 300.0  # 5분


@dataclass
class JobResult:
    """작업 결과"""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: float = 0.0
    retries: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": round(self.duration, 3),
            "retries": self.retries,
        }


@dataclass
class BatchResult:
    """배치 처리 결과"""
    batch_id: str
    total_items: int
    successful: int = 0
    failed: int = 0
    results: list[JobResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_items if self.total_items > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "total_items": self.total_items,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": round(self.duration, 3),
        }


class BatchProcessor:
    """배치 처리기

    대량 데이터를 효율적으로 처리하기 위한 배치 프로세서
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._jobs: dict[str, JobResult] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        return self._semaphore

    async def _execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> tuple[T, int]:
        """재시도 로직이 포함된 실행"""
        last_error = None
        retries = 0

        for attempt in range(self.config.retry_count + 1):
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout,
                )
                return result, retries
            except asyncio.TimeoutError:
                last_error = "Timeout"
                retries = attempt
            except Exception as e:
                last_error = str(e)
                retries = attempt

            if attempt < self.config.retry_count:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise Exception(last_error)

    async def process_item(
        self,
        item: Any,
        processor: Callable[[Any], Coroutine[Any, Any, T]],
        job_id: Optional[str] = None,
    ) -> JobResult:
        """단일 아이템 처리"""
        job_id = job_id or str(uuid.uuid4())
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
        )
        self._jobs[job_id] = job_result

        semaphore = self._get_semaphore()

        async with semaphore:
            job_result.status = JobStatus.RUNNING
            job_result.started_at = datetime.now()

            try:
                result, retries = await self._execute_with_retry(processor, item)
                job_result.result = result
                job_result.status = JobStatus.COMPLETED
                job_result.retries = retries

            except Exception as e:
                job_result.error = str(e)
                job_result.status = JobStatus.FAILED
                job_result.retries = self.config.retry_count

            finally:
                job_result.completed_at = datetime.now()
                job_result.duration = (
                    job_result.completed_at - job_result.started_at
                ).total_seconds()

        return job_result

    async def process_batch(
        self,
        items: list[Any],
        processor: Callable[[Any], Coroutine[Any, Any, T]],
        batch_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """배치 처리"""
        batch_id = batch_id or str(uuid.uuid4())
        batch_result = BatchResult(
            batch_id=batch_id,
            total_items=len(items),
            started_at=datetime.now(),
        )

        logger.info(f"Starting batch {batch_id} with {len(items)} items")

        # 비동기 작업 생성
        tasks = [
            self.process_item(item, processor, f"{batch_id}:{i}")
            for i, item in enumerate(items)
        ]

        # 병렬 실행
        completed = 0
        for coro in asyncio.as_completed(tasks):
            job_result = await coro
            batch_result.results.append(job_result)

            if job_result.status == JobStatus.COMPLETED:
                batch_result.successful += 1
            else:
                batch_result.failed += 1

            completed += 1
            if progress_callback:
                progress_callback(completed, len(items))

        batch_result.completed_at = datetime.now()
        batch_result.duration = (
            batch_result.completed_at - batch_result.started_at
        ).total_seconds()

        logger.info(
            f"Batch {batch_id} completed: "
            f"{batch_result.successful}/{batch_result.total_items} successful, "
            f"duration: {batch_result.duration:.2f}s"
        )

        return batch_result

    async def process_batches(
        self,
        items: list[Any],
        processor: Callable[[Any], Coroutine[Any, Any, T]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[BatchResult]:
        """대량 데이터를 배치로 나누어 처리"""
        results = []
        total_items = len(items)

        for i in range(0, total_items, self.config.batch_size):
            batch_items = items[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1

            batch_result = await self.process_batch(
                batch_items,
                processor,
                batch_id=f"batch_{batch_num}",
                progress_callback=progress_callback,
            )
            results.append(batch_result)

        return results

    def get_job(self, job_id: str) -> Optional[JobResult]:
        """작업 결과 조회"""
        return self._jobs.get(job_id)

    def get_stats(self) -> dict:
        """처리 통계"""
        total = len(self._jobs)
        completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
        running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

        return {
            "total_jobs": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": total - completed - failed - running,
        }


# ============================================================
# 문서 처리 특화 배치
# ============================================================

@dataclass
class DocumentBatch:
    """문서 배치"""
    documents: list[str]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentBatchProcessor:
    """문서 배치 처리기

    대량 문서 임베딩 및 인덱싱
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        embedding_func: Optional[Callable[[str], Coroutine[Any, Any, list[float]]]] = None,
        index_func: Optional[Callable[[str, list[float], dict], Coroutine[Any, Any, str]]] = None,
    ):
        self.config = config or BatchConfig(batch_size=50, max_concurrent=10)
        self.processor = BatchProcessor(self.config)
        self.embedding_func = embedding_func
        self.index_func = index_func

    async def process_documents(
        self,
        documents: list[str],
        source: str = "unknown",
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """문서 일괄 처리 (임베딩 + 인덱싱)"""

        async def process_single_doc(doc: str) -> dict:
            result = {"document": doc[:100], "source": source}

            # 임베딩
            if self.embedding_func:
                embedding = await self.embedding_func(doc)
                result["embedding_dim"] = len(embedding)

                # 인덱싱
                if self.index_func:
                    doc_id = await self.index_func(doc, embedding, metadata or {})
                    result["doc_id"] = doc_id

            return result

        return await self.processor.process_batch(
            documents,
            process_single_doc,
            progress_callback=progress_callback,
        )


# ============================================================
# 작업 큐
# ============================================================

class AsyncJobQueue:
    """비동기 작업 큐"""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._queue: asyncio.Queue[tuple[str, Callable, tuple, dict]] = asyncio.Queue()
        self._results: dict[str, JobResult] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def _worker(self, worker_id: int):
        """워커 루프"""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                job_id, func, args, kwargs = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )

                job_result = JobResult(
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    started_at=datetime.now(),
                )
                self._results[job_id] = job_result

                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    job_result.result = result
                    job_result.status = JobStatus.COMPLETED

                except Exception as e:
                    job_result.error = str(e)
                    job_result.status = JobStatus.FAILED
                    logger.error(f"Job {job_id} failed: {e}")

                finally:
                    job_result.completed_at = datetime.now()
                    job_result.duration = (
                        job_result.completed_at - job_result.started_at
                    ).total_seconds()
                    self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

        logger.debug(f"Worker {worker_id} stopped")

    async def start(self):
        """큐 시작"""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        logger.info(f"Job queue started with {self.max_workers} workers")

    async def stop(self):
        """큐 종료"""
        self._running = False

        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Job queue stopped")

    async def submit(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """작업 제출"""
        job_id = job_id or str(uuid.uuid4())
        await self._queue.put((job_id, func, args, kwargs))

        self._results[job_id] = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
        )

        return job_id

    def get_result(self, job_id: str) -> Optional[JobResult]:
        """작업 결과 조회"""
        return self._results.get(job_id)

    async def wait_for(self, job_id: str, timeout: float = 60.0) -> JobResult:
        """작업 완료 대기"""
        start = time.time()

        while time.time() - start < timeout:
            result = self._results.get(job_id)
            if result and result.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return result
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        total = len(self._results)
        completed = sum(1 for r in self._results.values() if r.status == JobStatus.COMPLETED)
        failed = sum(1 for r in self._results.values() if r.status == JobStatus.FAILED)

        return {
            "total_jobs": total,
            "completed": completed,
            "failed": failed,
            "pending": self.pending_count,
            "workers": len(self._workers),
        }
