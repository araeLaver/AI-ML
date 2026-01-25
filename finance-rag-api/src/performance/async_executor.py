# -*- coding: utf-8 -*-
"""
비동기 실행 모듈

[기능]
- 비동기 쿼리 실행
- 병렬 쿼리 처리
- 백그라운드 태스크 관리
- 작업 큐 관리

[사용 예시]
>>> executor = AsyncExecutor()
>>> results = await executor.execute_parallel([
...     lambda: rag.query("삼성전자"),
...     lambda: rag.query("SK하이닉스"),
... ])
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskStatus(Enum):
    """태스크 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """태스크 결과"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class TaskInfo:
    """태스크 정보"""
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncExecutor:
    """
    비동기 실행기

    [특징]
    - 동기 함수를 비동기로 실행
    - 병렬 실행 지원
    - 타임아웃 관리
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> T:
        """
        동기 함수를 비동기로 실행

        Args:
            func: 실행할 함수
            *args: 위치 인자
            timeout: 타임아웃 (초)
            **kwargs: 키워드 인자

        Returns:
            함수 실행 결과
        """
        loop = asyncio.get_event_loop()

        try:
            if timeout:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda: func(*args, **kwargs)
                    ),
                    timeout=timeout
                )
            else:
                return await loop.run_in_executor(
                    self._executor,
                    lambda: func(*args, **kwargs)
                )
        except asyncio.TimeoutError:
            logger.warning(f"Task timed out after {timeout}s")
            raise

    async def execute_parallel(
        self,
        tasks: List[Callable[[], T]],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[T]:
        """
        여러 작업을 병렬로 실행

        Args:
            tasks: 실행할 함수 리스트
            timeout: 전체 타임아웃
            return_exceptions: 예외를 결과로 반환할지 여부

        Returns:
            결과 리스트
        """
        loop = asyncio.get_event_loop()

        futures = [
            loop.run_in_executor(self._executor, task)
            for task in tasks
        ]

        if timeout:
            done, pending = await asyncio.wait(
                futures,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # 미완료 작업 취소
            for task in pending:
                task.cancel()

            results = []
            for future in done:
                try:
                    results.append(future.result())
                except Exception as e:
                    if return_exceptions:
                        results.append(e)
                    else:
                        raise

            return results
        else:
            return await asyncio.gather(
                *futures,
                return_exceptions=return_exceptions
            )

    def shutdown(self):
        """실행기 종료"""
        self._executor.shutdown(wait=True)


class ParallelQueryExecutor:
    """
    병렬 쿼리 실행기

    여러 RAG 쿼리를 동시에 실행
    """

    def __init__(
        self,
        query_fn: Callable[[str], Dict[str, Any]],
        max_concurrent: int = 5,
    ):
        self.query_fn = query_fn
        self.max_concurrent = max_concurrent
        self._executor = AsyncExecutor(max_workers=max_concurrent)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_queries(
        self,
        queries: List[str],
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        여러 쿼리를 병렬 실행

        Args:
            queries: 쿼리 리스트
            timeout: 개별 쿼리 타임아웃

        Returns:
            결과 리스트
        """
        async def query_with_semaphore(query: str) -> Dict[str, Any]:
            async with self._semaphore:
                try:
                    return await self._executor.execute(
                        self.query_fn,
                        query,
                        timeout=timeout
                    )
                except Exception as e:
                    logger.error(f"Query failed: {query[:50]}... - {e}")
                    return {"error": str(e), "query": query}

        return await asyncio.gather(
            *[query_with_semaphore(q) for q in queries]
        )

    async def execute_with_progress(
        self,
        queries: List[str],
        progress_callback: Callable[[int, int], None],
    ) -> List[Dict[str, Any]]:
        """
        진행 상황 콜백과 함께 쿼리 실행

        Args:
            queries: 쿼리 리스트
            progress_callback: 진행 상황 콜백

        Returns:
            결과 리스트
        """
        results = []
        completed = 0

        for query in queries:
            async with self._semaphore:
                result = await self._executor.execute(
                    self.query_fn, query
                )
                results.append(result)
                completed += 1
                progress_callback(completed, len(queries))

        return results


class BackgroundTaskManager:
    """
    백그라운드 태스크 관리자

    [특징]
    - 태스크 큐 관리
    - 상태 추적
    - 취소 지원
    - 결과 저장
    """

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._tasks: Dict[str, TaskInfo] = {}
        self._futures: Dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = asyncio.Lock()

    def submit(
        self,
        name: str,
        func: Callable[..., T],
        *args,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        백그라운드 태스크 제출

        Args:
            name: 태스크 이름
            func: 실행할 함수
            *args: 위치 인자
            metadata: 메타데이터
            **kwargs: 키워드 인자

        Returns:
            태스크 ID
        """
        task_id = str(uuid.uuid4())

        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task_info

        def wrapped_func():
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = time.time()

            try:
                result = func(*args, **kwargs)
                task_info.result = result
                task_info.status = TaskStatus.COMPLETED
            except Exception as e:
                task_info.error = str(e)
                task_info.status = TaskStatus.FAILED
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                task_info.completed_at = time.time()

            return task_info

        future = self._executor.submit(wrapped_func)
        self._futures[task_id] = future

        logger.info(f"Submitted task: {name} ({task_id})")
        return task_id

    async def submit_async(
        self,
        name: str,
        coro: Awaitable[T],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        비동기 태스크 제출

        Args:
            name: 태스크 이름
            coro: 코루틴
            metadata: 메타데이터

        Returns:
            태스크 ID
        """
        task_id = str(uuid.uuid4())

        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task_info

        async def wrapped_coro():
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = time.time()

            try:
                result = await coro
                task_info.result = result
                task_info.status = TaskStatus.COMPLETED
            except Exception as e:
                task_info.error = str(e)
                task_info.status = TaskStatus.FAILED
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                task_info.completed_at = time.time()

        asyncio.create_task(wrapped_coro())
        logger.info(f"Submitted async task: {name} ({task_id})")
        return task_id

    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """태스크 상태 조회"""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """
        태스크 결과 조회 (블로킹)

        Args:
            task_id: 태스크 ID
            timeout: 대기 타임아웃

        Returns:
            태스크 결과
        """
        future = self._futures.get(task_id)
        task_info = self._tasks.get(task_id)

        if future is None or task_info is None:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="Task not found"
            )

        try:
            future.result(timeout=timeout)
        except Exception as e:
            pass

        return TaskResult(
            task_id=task_id,
            status=task_info.status,
            result=task_info.result,
            error=task_info.error,
            started_at=task_info.started_at,
            completed_at=task_info.completed_at,
        )

    async def wait_for(self, task_id: str, timeout: float = None) -> TaskResult:
        """
        태스크 완료 대기 (비동기)

        Args:
            task_id: 태스크 ID
            timeout: 대기 타임아웃

        Returns:
            태스크 결과
        """
        task_info = self._tasks.get(task_id)
        if task_info is None:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="Task not found"
            )

        start = time.time()
        while task_info.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start) > timeout:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error="Timeout waiting for task"
                )
            await asyncio.sleep(0.1)

        return TaskResult(
            task_id=task_id,
            status=task_info.status,
            result=task_info.result,
            error=task_info.error,
            started_at=task_info.started_at,
            completed_at=task_info.completed_at,
        )

    def cancel(self, task_id: str) -> bool:
        """태스크 취소"""
        future = self._futures.get(task_id)
        task_info = self._tasks.get(task_id)

        if future is None:
            return False

        if future.cancel():
            if task_info:
                task_info.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task: {task_id}")
            return True

        return False

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
    ) -> List[TaskInfo]:
        """
        태스크 목록 조회

        Args:
            status: 상태 필터 (None이면 전체)

        Returns:
            태스크 정보 리스트
        """
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def cleanup_completed(self, max_age: float = 3600.0) -> int:
        """
        완료된 태스크 정리

        Args:
            max_age: 최대 유지 시간 (초)

        Returns:
            정리된 태스크 수
        """
        now = time.time()
        to_remove = []

        for task_id, task_info in self._tasks.items():
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task_info.completed_at and (now - task_info.completed_at) > max_age:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]
            if task_id in self._futures:
                del self._futures[task_id]

        logger.info(f"Cleaned up {len(to_remove)} completed tasks")
        return len(to_remove)

    def shutdown(self):
        """관리자 종료"""
        self._executor.shutdown(wait=False)
