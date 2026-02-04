# -*- coding: utf-8 -*-
"""
데이터 파이프라인 모듈

ETL 워크플로우, 데이터 변환, 파이프라인 오케스트레이션
"""

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ============================================================
# 파이프라인 상태 및 설정
# ============================================================

class PipelineStatus(Enum):
    """파이프라인 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """스텝 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    name: str = "default"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: Optional[float] = None
    parallel_steps: bool = False
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "data/pipeline/checkpoints"

    # 알림 설정
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_channels: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    """스텝 실행 결과"""
    step_id: str
    step_name: str
    status: StepStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    input_data: Any = None
    output_data: Any = None
    error: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps: list[StepResult] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0.0
        success = sum(1 for s in self.steps if s.status == StepStatus.SUCCESS)
        return success / len(self.steps)

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "success_rate": self.success_rate,
            "total_steps": len(self.steps),
            "error": self.error,
            "metadata": self.metadata,
        }


# ============================================================
# 파이프라인 스텝
# ============================================================

class PipelineStep(ABC, Generic[T, R]):
    """파이프라인 스텝 기본 클래스"""

    def __init__(
        self,
        name: str,
        description: str = "",
        retries: int = 0,
        timeout: Optional[float] = None,
    ):
        self.step_id = str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.retries = retries
        self.timeout = timeout

    @abstractmethod
    async def execute(self, data: T, context: dict) -> R:
        """스텝 실행"""
        pass

    async def run(self, data: T, context: dict) -> StepResult:
        """스텝 실행 (재시도 포함)"""
        result = StepResult(
            step_id=self.step_id,
            step_name=self.name,
            status=StepStatus.RUNNING,
            started_at=datetime.now(),
            input_data=data,
        )

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                if self.timeout:
                    output = await asyncio.wait_for(
                        self.execute(data, context),
                        timeout=self.timeout
                    )
                else:
                    output = await self.execute(data, context)

                result.status = StepStatus.SUCCESS
                result.output_data = output
                result.completed_at = datetime.now()
                result.metrics["attempts"] = attempt + 1

                logger.info(
                    f"Step '{self.name}' completed in {result.duration_ms:.2f}ms"
                )
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(f"Step '{self.name}' timeout (attempt {attempt + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Step '{self.name}' failed (attempt {attempt + 1}): {e}"
                )

            if attempt < self.retries:
                await asyncio.sleep(1.0 * (attempt + 1))  # 점진적 딜레이

        result.status = StepStatus.FAILED
        result.error = last_error
        result.completed_at = datetime.now()
        result.metrics["attempts"] = self.retries + 1
        return result


class FunctionStep(PipelineStep[T, R]):
    """함수 기반 스텝"""

    def __init__(
        self,
        name: str,
        func: Callable[[T, dict], R],
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.func = func

    async def execute(self, data: T, context: dict) -> R:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(data, context)
        return self.func(data, context)


# ============================================================
# 파이프라인 빌더
# ============================================================

class Pipeline:
    """데이터 파이프라인"""

    def __init__(self, name: str, config: Optional[PipelineConfig] = None):
        self.pipeline_id = str(uuid.uuid4())
        self.name = name
        self.config = config or PipelineConfig(name=name)
        self.steps: list[PipelineStep] = []
        self._context: dict = {}
        self._cancelled = False

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """스텝 추가"""
        self.steps.append(step)
        return self

    def add_function(
        self,
        name: str,
        func: Callable,
        retries: int = 0,
        timeout: Optional[float] = None,
    ) -> "Pipeline":
        """함수 스텝 추가"""
        step = FunctionStep(name, func, retries=retries, timeout=timeout)
        return self.add_step(step)

    def set_context(self, key: str, value: Any) -> "Pipeline":
        """컨텍스트 설정"""
        self._context[key] = value
        return self

    def cancel(self):
        """파이프라인 취소"""
        self._cancelled = True

    async def run(self, initial_data: Any = None) -> PipelineResult:
        """파이프라인 실행"""
        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        logger.info(f"Pipeline '{self.name}' started with {len(self.steps)} steps")

        current_data = initial_data
        context = {**self._context, "pipeline_id": self.pipeline_id}

        try:
            for step in self.steps:
                if self._cancelled:
                    result.status = PipelineStatus.CANCELLED
                    result.error = "Pipeline cancelled"
                    break

                step_result = await step.run(current_data, context)
                result.steps.append(step_result)

                if step_result.status == StepStatus.FAILED:
                    result.status = PipelineStatus.FAILED
                    result.error = f"Step '{step.name}' failed: {step_result.error}"
                    break

                current_data = step_result.output_data
                context["last_output"] = current_data

            else:
                result.status = PipelineStatus.SUCCESS

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            logger.error(f"Pipeline '{self.name}' error: {e}")

        result.completed_at = datetime.now()
        result.metadata["final_data"] = current_data

        logger.info(
            f"Pipeline '{self.name}' completed: {result.status.value} "
            f"({result.duration_ms:.2f}ms, {result.success_rate:.0%} success)"
        )

        return result


# ============================================================
# ETL 스텝 구현
# ============================================================

class ExtractStep(PipelineStep[dict, list]):
    """데이터 추출 스텝"""

    def __init__(
        self,
        name: str = "extract",
        source_type: str = "file",
        source_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.source_type = source_type
        self.source_path = source_path

    async def execute(self, data: dict, context: dict) -> list:
        source_path = data.get("source_path") or self.source_path

        if self.source_type == "file":
            return await self._extract_from_file(source_path)
        elif self.source_type == "api":
            return await self._extract_from_api(data.get("api_url"))
        elif self.source_type == "database":
            return await self._extract_from_db(data.get("query"))
        else:
            return data.get("data", [])

    async def _extract_from_file(self, path: str) -> list:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    async def _extract_from_api(self, url: str) -> list:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
        except ImportError:
            import requests
            response = requests.get(url)
            return response.json()

    async def _extract_from_db(self, query: str) -> list:
        # 플레이스홀더 - 실제 구현은 DB 커넥터에 따라 달라짐
        raise NotImplementedError("Database extraction not implemented")


class TransformStep(PipelineStep[list, list]):
    """데이터 변환 스텝"""

    def __init__(
        self,
        name: str = "transform",
        transformations: Optional[list[Callable]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.transformations = transformations or []

    def add_transformation(self, func: Callable) -> "TransformStep":
        self.transformations.append(func)
        return self

    async def execute(self, data: list, context: dict) -> list:
        result = data

        for transform in self.transformations:
            if asyncio.iscoroutinefunction(transform):
                result = await transform(result, context)
            else:
                result = transform(result, context)

        return result


class LoadStep(PipelineStep[list, dict]):
    """데이터 적재 스텝"""

    def __init__(
        self,
        name: str = "load",
        target_type: str = "file",
        target_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.target_type = target_type
        self.target_path = target_path

    async def execute(self, data: list, context: dict) -> dict:
        target_path = context.get("target_path") or self.target_path

        if self.target_type == "file":
            return await self._load_to_file(data, target_path)
        elif self.target_type == "vectorstore":
            return await self._load_to_vectorstore(data, context)
        elif self.target_type == "database":
            return await self._load_to_db(data, context)
        else:
            return {"loaded_count": len(data), "data": data}

    async def _load_to_file(self, data: list, path: str) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {
            "loaded_count": len(data),
            "target_path": str(path),
            "file_size": path.stat().st_size,
        }

    async def _load_to_vectorstore(self, data: list, context: dict) -> dict:
        # RAG 서비스를 통해 벡터 스토어에 적재
        rag_service = context.get("rag_service")
        if not rag_service:
            raise ValueError("RAG service not provided in context")

        documents = []
        for item in data:
            if isinstance(item, str):
                documents.append(item)
            elif isinstance(item, dict):
                documents.append(item.get("content", str(item)))

        count = rag_service.add_documents(
            documents=documents,
            source=context.get("source", "pipeline"),
        )

        return {"loaded_count": count, "target": "vectorstore"}

    async def _load_to_db(self, data: list, context: dict) -> dict:
        raise NotImplementedError("Database loading not implemented")


# ============================================================
# 데이터 변환 유틸리티
# ============================================================

class DataTransformers:
    """공통 데이터 변환 함수"""

    @staticmethod
    def filter_by(field: str, value: Any) -> Callable:
        """필드 값으로 필터링"""
        def transform(data: list, ctx: dict) -> list:
            return [item for item in data if item.get(field) == value]
        return transform

    @staticmethod
    def filter_not_null(field: str) -> Callable:
        """null이 아닌 항목만 필터링"""
        def transform(data: list, ctx: dict) -> list:
            return [item for item in data if item.get(field) is not None]
        return transform

    @staticmethod
    def select_fields(*fields: str) -> Callable:
        """특정 필드만 선택"""
        def transform(data: list, ctx: dict) -> list:
            return [{k: item.get(k) for k in fields} for item in data]
        return transform

    @staticmethod
    def rename_fields(mapping: dict[str, str]) -> Callable:
        """필드명 변경"""
        def transform(data: list, ctx: dict) -> list:
            return [
                {mapping.get(k, k): v for k, v in item.items()}
                for item in data
            ]
        return transform

    @staticmethod
    def add_field(field: str, value: Any) -> Callable:
        """필드 추가"""
        def transform(data: list, ctx: dict) -> list:
            for item in data:
                item[field] = value() if callable(value) else value
            return data
        return transform

    @staticmethod
    def add_timestamp(field: str = "processed_at") -> Callable:
        """타임스탬프 추가"""
        def transform(data: list, ctx: dict) -> list:
            now = datetime.now().isoformat()
            for item in data:
                item[field] = now
            return data
        return transform

    @staticmethod
    def deduplicate(key_field: str) -> Callable:
        """중복 제거"""
        def transform(data: list, ctx: dict) -> list:
            seen = set()
            result = []
            for item in data:
                key = item.get(key_field)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result
        return transform

    @staticmethod
    def sort_by(field: str, reverse: bool = False) -> Callable:
        """정렬"""
        def transform(data: list, ctx: dict) -> list:
            return sorted(data, key=lambda x: x.get(field, ""), reverse=reverse)
        return transform

    @staticmethod
    def limit(count: int) -> Callable:
        """개수 제한"""
        def transform(data: list, ctx: dict) -> list:
            return data[:count]
        return transform

    @staticmethod
    def map_field(field: str, func: Callable) -> Callable:
        """필드 값 변환"""
        def transform(data: list, ctx: dict) -> list:
            for item in data:
                if field in item:
                    item[field] = func(item[field])
            return data
        return transform


# ============================================================
# 파이프라인 레지스트리
# ============================================================

class PipelineRegistry:
    """파이프라인 레지스트리"""

    _pipelines: Dict[str, "Pipeline"] = {}
    _results: List["PipelineResult"] = []

    @classmethod
    def register(cls, pipeline: "Pipeline") -> None:
        """파이프라인 등록"""
        cls._pipelines[pipeline.name] = pipeline

    @classmethod
    def get(cls, name: str) -> Optional["Pipeline"]:
        """파이프라인 조회"""
        return cls._pipelines.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """등록된 파이프라인 목록"""
        return list(cls._pipelines.keys())

    @classmethod
    def add_result(cls, result: "PipelineResult") -> None:
        """실행 결과 저장"""
        cls._results.append(result)
        # 최근 100개만 유지
        if len(cls._results) > 100:
            cls._results = cls._results[-100:]

    @classmethod
    def get_results(
        cls,
        pipeline_name: Optional[str] = None,
        limit: int = 10,
    ) -> List["PipelineResult"]:
        """실행 결과 조회"""
        results = cls._results
        if pipeline_name:
            results = [r for r in results if r.pipeline_name == pipeline_name]
        return results[-limit:][::-1]

    @classmethod
    def get_stats(cls) -> dict:
        """파이프라인 통계"""
        if not cls._results:
            return {
                "total_runs": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "by_status": {s.value: 0 for s in PipelineStatus},
            }

        success = sum(1 for r in cls._results if r.status == PipelineStatus.SUCCESS)
        total_duration = sum(r.duration_ms for r in cls._results)

        return {
            "total_runs": len(cls._results),
            "success_rate": success / len(cls._results),
            "avg_duration_ms": total_duration / len(cls._results),
            "by_status": {
                status.value: sum(1 for r in cls._results if r.status == status)
                for status in PipelineStatus
            },
        }


# ============================================================
# 파이프라인 빌더 헬퍼
# ============================================================

def create_etl_pipeline(
    name: str,
    source_type: str = "file",
    source_path: Optional[str] = None,
    target_type: str = "file",
    target_path: Optional[str] = None,
    transformations: Optional[list[Callable]] = None,
) -> Pipeline:
    """ETL 파이프라인 생성 헬퍼"""
    pipeline = Pipeline(name)

    # Extract
    extract = ExtractStep(
        name=f"{name}_extract",
        source_type=source_type,
        source_path=source_path,
    )
    pipeline.add_step(extract)

    # Transform
    transform = TransformStep(
        name=f"{name}_transform",
        transformations=transformations or [],
    )
    pipeline.add_step(transform)

    # Load
    load = LoadStep(
        name=f"{name}_load",
        target_type=target_type,
        target_path=target_path,
    )
    pipeline.add_step(load)

    PipelineRegistry.register(pipeline)
    return pipeline


async def run_pipeline(
    pipeline: Pipeline,
    initial_data: Any = None,
) -> PipelineResult:
    """파이프라인 실행 및 결과 저장"""
    result = await pipeline.run(initial_data)
    PipelineRegistry.add_result(result)
    return result
