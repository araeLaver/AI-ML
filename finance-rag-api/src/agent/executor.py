# -*- coding: utf-8 -*-
"""
도구 실행기 모듈

[기능]
- 도구 실행 및 결과 처리
- 실행 컨텍스트 관리
- 병렬 실행 지원
- 오류 처리 및 재시도
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tools import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """실행 컨텍스트"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: float = 30.0
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_variable(self, name: str, value: Any) -> None:
        """변수 설정"""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """변수 조회"""
        return self.variables.get(name, default)


@dataclass
class ExecutionResult:
    """실행 결과"""
    tool_name: str
    status: ExecutionStatus
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    retries: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    execution_id: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "retries": self.retries,
            "duration_ms": round(self.duration_ms, 2),
            "execution_id": self.execution_id,
        }


class ToolExecutor:
    """
    도구 실행기

    도구 실행 및 관리
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_workers: int = 4,
        default_timeout: float = 30.0,
    ):
        self.registry = registry
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._execution_history: List[ExecutionResult] = []
        self._hooks: Dict[str, List[Callable]] = {
            "before_execute": [],
            "after_execute": [],
            "on_error": [],
        }

    def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        """동기 실행"""
        context = context or ExecutionContext()
        result = ExecutionResult(
            tool_name=tool_name,
            status=ExecutionStatus.PENDING,
            execution_id=context.execution_id,
        )

        # 도구 조회
        tool = self.registry.get(tool_name)
        if not tool:
            result.status = ExecutionStatus.FAILED
            result.error = f"Tool not found: {tool_name}"
            return result

        # Before hooks
        self._run_hooks("before_execute", tool_name, params, context)

        # 실행
        result.start_time = time.time()
        result.status = ExecutionStatus.RUNNING

        try:
            # 파라미터 검증
            valid, error = tool.validate_params(params)
            if not valid:
                result.status = ExecutionStatus.FAILED
                result.error = error
                return result

            # 변수 치환
            params = self._substitute_variables(params, context)

            # 실행 with timeout
            future = self._executor.submit(tool.execute, **params)
            tool_result = future.result(timeout=context.timeout_seconds)

            result.result = tool_result
            result.status = ExecutionStatus.COMPLETED if tool_result.success else ExecutionStatus.FAILED

            if not tool_result.success:
                result.error = tool_result.error

        except FuturesTimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error = f"Execution timeout ({context.timeout_seconds}s)"
            self._run_hooks("on_error", tool_name, "timeout", context)

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            logger.error(f"Tool execution error: {e}")
            self._run_hooks("on_error", tool_name, str(e), context)

        finally:
            result.end_time = time.time()

        # After hooks
        self._run_hooks("after_execute", tool_name, result, context)

        # 히스토리 저장
        self._execution_history.append(result)
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]

        return result

    async def execute_async(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        """비동기 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute(tool_name, params, context),
        )

    def execute_many(
        self,
        calls: List[Tuple[str, Dict[str, Any]]],
        context: Optional[ExecutionContext] = None,
        parallel: bool = True,
    ) -> List[ExecutionResult]:
        """다중 도구 실행"""
        if not parallel:
            return [
                self.execute(tool_name, params, context)
                for tool_name, params in calls
            ]

        # 병렬 실행
        futures = [
            self._executor.submit(self.execute, tool_name, params, context)
            for tool_name, params in calls
        ]

        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=self.default_timeout * 2))
            except Exception as e:
                results.append(ExecutionResult(
                    tool_name="unknown",
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                ))

        return results

    async def execute_many_async(
        self,
        calls: List[Tuple[str, Dict[str, Any]]],
        context: Optional[ExecutionContext] = None,
    ) -> List[ExecutionResult]:
        """비동기 다중 실행"""
        tasks = [
            self.execute_async(tool_name, params, context)
            for tool_name, params in calls
        ]
        return await asyncio.gather(*tasks)

    def execute_with_retry(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
        max_retries: Optional[int] = None,
    ) -> ExecutionResult:
        """재시도 포함 실행"""
        context = context or ExecutionContext()
        max_retries = max_retries or context.max_retries

        result = None
        for attempt in range(max_retries + 1):
            result = self.execute(tool_name, params, context)
            result.retries = attempt

            if result.status == ExecutionStatus.COMPLETED:
                return result

            # 재시도 전 대기
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))

        return result

    def _substitute_variables(
        self,
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """변수 치환"""
        result = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                result[key] = context.get_variable(var_name, value)
            else:
                result[key] = value
        return result

    def add_hook(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """후크 추가"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _run_hooks(self, event: str, *args) -> None:
        """후크 실행"""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Hook error: {e}")

    def get_history(
        self,
        limit: int = 100,
        tool_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
    ) -> List[ExecutionResult]:
        """실행 히스토리"""
        results = self._execution_history

        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]

        if status:
            results = [r for r in results if r.status == status]

        return results[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """실행 통계"""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
            }

        total = len(self._execution_history)
        successful = sum(
            1 for r in self._execution_history
            if r.status == ExecutionStatus.COMPLETED
        )
        durations = [r.duration_ms for r in self._execution_history]

        # 도구별 통계
        by_tool: Dict[str, Dict[str, Any]] = {}
        for r in self._execution_history:
            if r.tool_name not in by_tool:
                by_tool[r.tool_name] = {"count": 0, "success": 0, "durations": []}
            by_tool[r.tool_name]["count"] += 1
            if r.status == ExecutionStatus.COMPLETED:
                by_tool[r.tool_name]["success"] += 1
            by_tool[r.tool_name]["durations"].append(r.duration_ms)

        tool_stats = {
            name: {
                "count": data["count"],
                "success_rate": data["success"] / data["count"] if data["count"] > 0 else 0,
                "avg_duration_ms": sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0,
            }
            for name, data in by_tool.items()
        }

        return {
            "total_executions": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "by_tool": tool_stats,
        }

    def shutdown(self) -> None:
        """종료"""
        self._executor.shutdown(wait=True)


class ChainedExecutor:
    """
    체인 실행기

    도구들을 순차적으로 실행하고 결과를 전달
    """

    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self._chain: List[Tuple[str, Dict[str, Any], Optional[str]]] = []

    def add(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result_var: Optional[str] = None,
    ) -> "ChainedExecutor":
        """체인에 도구 추가"""
        self._chain.append((tool_name, params, result_var))
        return self

    def execute(
        self,
        context: Optional[ExecutionContext] = None,
    ) -> List[ExecutionResult]:
        """체인 실행"""
        context = context or ExecutionContext()
        results = []

        for tool_name, params, result_var in self._chain:
            result = self.executor.execute(tool_name, params, context)
            results.append(result)

            # 실패 시 중단
            if result.status != ExecutionStatus.COMPLETED:
                break

            # 결과를 변수에 저장
            if result_var and result.result:
                context.set_variable(result_var, result.result.data)

        return results

    def clear(self) -> None:
        """체인 초기화"""
        self._chain.clear()
