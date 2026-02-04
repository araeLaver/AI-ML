# -*- coding: utf-8 -*-
"""
파이프라인 API 라우터

ETL 파이프라인 실행, 모니터링, 관리 엔드포인트
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineRegistry,
    PipelineResult,
    PipelineStatus,
    DataTransformers,
    ExtractStep,
    TransformStep,
    LoadStep,
    create_etl_pipeline,
    run_pipeline,
)

logger = get_logger(__name__)

# 라우터 생성
pipeline_router = APIRouter(tags=["Pipeline"])


# ============================================================
# 스키마 정의
# ============================================================

class PipelineCreateRequest(BaseModel):
    """파이프라인 생성 요청"""
    name: str = Field(..., description="파이프라인 이름")
    source_type: str = Field("file", description="소스 타입 (file, api)")
    source_path: Optional[str] = Field(None, description="소스 경로")
    target_type: str = Field("file", description="타겟 타입 (file, vectorstore)")
    target_path: Optional[str] = Field(None, description="타겟 경로")
    transformations: list[str] = Field(
        default_factory=list,
        description="변환 목록 (deduplicate, add_timestamp, limit:100)"
    )


class PipelineRunRequest(BaseModel):
    """파이프라인 실행 요청"""
    pipeline_name: str = Field(..., description="실행할 파이프라인 이름")
    initial_data: Optional[dict] = Field(None, description="초기 데이터")
    async_mode: bool = Field(False, description="비동기 실행 여부")


class PipelineResultResponse(BaseModel):
    """파이프라인 실행 결과"""
    pipeline_id: str
    pipeline_name: str
    status: str
    started_at: str
    completed_at: Optional[str]
    duration_ms: float
    success_rate: float
    total_steps: int
    error: Optional[str]


class PipelineStatsResponse(BaseModel):
    """파이프라인 통계"""
    total_runs: int
    success_rate: float
    avg_duration_ms: float
    by_status: dict[str, int]


# ============================================================
# 변환 파싱 헬퍼
# ============================================================

def parse_transformation(transform_str: str):
    """변환 문자열 파싱"""
    if ":" in transform_str:
        name, arg = transform_str.split(":", 1)
    else:
        name = transform_str
        arg = None

    transformers = {
        "add_timestamp": lambda: DataTransformers.add_timestamp(),
        "deduplicate": lambda: DataTransformers.deduplicate(arg or "id"),
        "limit": lambda: DataTransformers.limit(int(arg or 100)),
        "sort_by": lambda: DataTransformers.sort_by(arg or "id"),
        "sort_by_desc": lambda: DataTransformers.sort_by(arg or "id", reverse=True),
    }

    if name in transformers:
        return transformers[name]()

    return None


# ============================================================
# 파이프라인 관리 엔드포인트
# ============================================================

@pipeline_router.post(
    "/create",
    response_model=dict,
    summary="파이프라인 생성",
    description="새로운 ETL 파이프라인을 생성합니다.",
)
async def create_pipeline(request: PipelineCreateRequest):
    """파이프라인 생성"""
    # 변환 함수 파싱
    transformations = []
    for t in request.transformations:
        transform = parse_transformation(t)
        if transform:
            transformations.append(transform)

    # 파이프라인 생성
    pipeline = create_etl_pipeline(
        name=request.name,
        source_type=request.source_type,
        source_path=request.source_path,
        target_type=request.target_type,
        target_path=request.target_path,
        transformations=transformations,
    )

    return {
        "success": True,
        "pipeline_id": pipeline.pipeline_id,
        "pipeline_name": pipeline.name,
        "steps": [s.name for s in pipeline.steps],
    }


@pipeline_router.get(
    "/list",
    response_model=list[str],
    summary="파이프라인 목록",
    description="등록된 파이프라인 목록을 조회합니다.",
)
async def list_pipelines():
    """파이프라인 목록"""
    return PipelineRegistry.list()


@pipeline_router.post(
    "/run",
    response_model=PipelineResultResponse,
    summary="파이프라인 실행",
    description="등록된 파이프라인을 실행합니다.",
)
async def run_pipeline_endpoint(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """파이프라인 실행"""
    pipeline = PipelineRegistry.get(request.pipeline_name)
    if not pipeline:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline not found: {request.pipeline_name}"
        )

    if request.async_mode:
        # 비동기 실행
        async def run_async():
            await run_pipeline(pipeline, request.initial_data)

        background_tasks.add_task(asyncio.create_task, run_async())

        return PipelineResultResponse(
            pipeline_id=pipeline.pipeline_id,
            pipeline_name=pipeline.name,
            status="pending",
            started_at="",
            completed_at=None,
            duration_ms=0,
            success_rate=0,
            total_steps=len(pipeline.steps),
            error=None,
        )

    # 동기 실행
    result = await run_pipeline(pipeline, request.initial_data)

    return PipelineResultResponse(
        pipeline_id=result.pipeline_id,
        pipeline_name=result.pipeline_name,
        status=result.status.value,
        started_at=result.started_at.isoformat(),
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
        duration_ms=result.duration_ms,
        success_rate=result.success_rate,
        total_steps=len(result.steps),
        error=result.error,
    )


@pipeline_router.post(
    "/run/adhoc",
    response_model=PipelineResultResponse,
    summary="임시 파이프라인 실행",
    description="등록하지 않고 즉시 파이프라인을 실행합니다.",
)
async def run_adhoc_pipeline(request: PipelineCreateRequest):
    """임시 파이프라인 실행"""
    # 변환 함수 파싱
    transformations = []
    for t in request.transformations:
        transform = parse_transformation(t)
        if transform:
            transformations.append(transform)

    # 파이프라인 생성 (등록하지 않음)
    pipeline = Pipeline(request.name)

    # Extract
    extract = ExtractStep(
        name=f"{request.name}_extract",
        source_type=request.source_type,
        source_path=request.source_path,
    )
    pipeline.add_step(extract)

    # Transform
    transform = TransformStep(
        name=f"{request.name}_transform",
        transformations=transformations,
    )
    pipeline.add_step(transform)

    # Load
    load = LoadStep(
        name=f"{request.name}_load",
        target_type=request.target_type,
        target_path=request.target_path,
    )
    pipeline.add_step(load)

    # 실행
    result = await pipeline.run({
        "source_path": request.source_path,
    })
    PipelineRegistry.add_result(result)

    return PipelineResultResponse(
        pipeline_id=result.pipeline_id,
        pipeline_name=result.pipeline_name,
        status=result.status.value,
        started_at=result.started_at.isoformat(),
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
        duration_ms=result.duration_ms,
        success_rate=result.success_rate,
        total_steps=len(result.steps),
        error=result.error,
    )


# ============================================================
# 모니터링 엔드포인트
# ============================================================

@pipeline_router.get(
    "/results",
    response_model=list[PipelineResultResponse],
    summary="실행 결과 목록",
    description="파이프라인 실행 결과 목록을 조회합니다.",
)
async def get_results(
    pipeline_name: Optional[str] = Query(None, description="파이프라인 이름 필터"),
    limit: int = Query(10, ge=1, le=100, description="조회 개수"),
):
    """실행 결과 목록"""
    results = PipelineRegistry.get_results(pipeline_name, limit)

    return [
        PipelineResultResponse(
            pipeline_id=r.pipeline_id,
            pipeline_name=r.pipeline_name,
            status=r.status.value,
            started_at=r.started_at.isoformat(),
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            duration_ms=r.duration_ms,
            success_rate=r.success_rate,
            total_steps=len(r.steps),
            error=r.error,
        )
        for r in results
    ]


@pipeline_router.get(
    "/stats",
    response_model=PipelineStatsResponse,
    summary="파이프라인 통계",
    description="파이프라인 실행 통계를 조회합니다.",
)
async def get_stats():
    """파이프라인 통계"""
    stats = PipelineRegistry.get_stats()
    return PipelineStatsResponse(**stats)


@pipeline_router.get(
    "/health",
    summary="파이프라인 헬스체크",
)
async def pipeline_health():
    """파이프라인 헬스체크"""
    stats = PipelineRegistry.get_stats()

    return {
        "status": "ok",
        "registered_pipelines": len(PipelineRegistry.list()),
        "total_runs": stats["total_runs"],
        "recent_success_rate": stats["success_rate"],
        "features": {
            "etl": True,
            "async_execution": True,
            "transformations": True,
        },
    }


# ============================================================
# 변환 유틸리티 엔드포인트
# ============================================================

@pipeline_router.get(
    "/transformations",
    summary="사용 가능한 변환 목록",
    description="파이프라인에서 사용 가능한 변환 함수 목록입니다.",
)
async def list_transformations():
    """사용 가능한 변환 목록"""
    return {
        "transformations": [
            {
                "name": "add_timestamp",
                "description": "타임스탬프 필드 추가",
                "usage": "add_timestamp",
                "args": "없음",
            },
            {
                "name": "deduplicate",
                "description": "중복 제거",
                "usage": "deduplicate:field_name",
                "args": "키 필드명 (기본: id)",
            },
            {
                "name": "limit",
                "description": "개수 제한",
                "usage": "limit:100",
                "args": "최대 개수",
            },
            {
                "name": "sort_by",
                "description": "정렬 (오름차순)",
                "usage": "sort_by:field_name",
                "args": "정렬 필드명",
            },
            {
                "name": "sort_by_desc",
                "description": "정렬 (내림차순)",
                "usage": "sort_by_desc:field_name",
                "args": "정렬 필드명",
            },
        ]
    }
