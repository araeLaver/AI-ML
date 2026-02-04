# -*- coding: utf-8 -*-
"""
파이프라인 모듈 테스트 (Phase 15)

ETL 파이프라인, 데이터 변환, 파이프라인 API 테스트
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


# ============================================================
# Pipeline API 엔드포인트 테스트
# ============================================================

class TestPipelineAPI:
    """파이프라인 API 테스트"""

    def test_pipeline_health(self):
        """파이프라인 헬스체크"""
        response = client.get("/api/v1/pipeline/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "registered_pipelines" in data
        assert "features" in data
        assert data["features"]["etl"] is True

    def test_list_pipelines(self):
        """파이프라인 목록 조회"""
        response = client.get("/api/v1/pipeline/list")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_transformations(self):
        """변환 목록 조회"""
        response = client.get("/api/v1/pipeline/transformations")
        assert response.status_code == 200
        data = response.json()
        assert "transformations" in data
        assert len(data["transformations"]) > 0

        # 기본 변환들 확인
        names = [t["name"] for t in data["transformations"]]
        assert "add_timestamp" in names
        assert "deduplicate" in names
        assert "limit" in names

    def test_create_pipeline(self):
        """파이프라인 생성"""
        response = client.post(
            "/api/v1/pipeline/create",
            json={
                "name": "test_pipeline",
                "source_type": "file",
                "target_type": "file",
                "transformations": ["add_timestamp", "limit:10"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["pipeline_name"] == "test_pipeline"
        assert len(data["steps"]) == 3  # Extract, Transform, Load

    def test_pipeline_stats(self):
        """파이프라인 통계"""
        response = client.get("/api/v1/pipeline/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_runs" in data
        assert "success_rate" in data
        assert "avg_duration_ms" in data

    def test_pipeline_results(self):
        """파이프라인 결과 조회"""
        response = client.get("/api/v1/pipeline/results")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# ============================================================
# Pipeline 모듈 단위 테스트
# ============================================================

class TestPipelineModule:
    """파이프라인 모듈 단위 테스트"""

    def test_import_pipeline_module(self):
        """파이프라인 모듈 임포트"""
        from src.core.pipeline import (
            Pipeline,
            PipelineConfig,
            PipelineStatus,
            PipelineResult,
            PipelineStep,
            FunctionStep,
            ExtractStep,
            TransformStep,
            LoadStep,
            DataTransformers,
            PipelineRegistry,
        )
        assert Pipeline is not None
        assert PipelineConfig is not None

    def test_pipeline_status_enum(self):
        """파이프라인 상태 열거형"""
        from src.core.pipeline import PipelineStatus

        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.SUCCESS.value == "success"
        assert PipelineStatus.FAILED.value == "failed"

    def test_pipeline_config_defaults(self):
        """파이프라인 설정 기본값"""
        from src.core.pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.name == "default"
        assert config.max_retries == 3
        assert config.parallel_steps is False

    def test_step_result_duration(self):
        """스텝 결과 소요시간 계산"""
        from src.core.pipeline import StepResult, StepStatus
        from datetime import datetime, timedelta

        now = datetime.now()
        result = StepResult(
            step_id="test",
            step_name="test_step",
            status=StepStatus.SUCCESS,
            started_at=now,
            completed_at=now + timedelta(milliseconds=100),
        )

        assert result.duration_ms >= 100

    def test_pipeline_result_success_rate(self):
        """파이프라인 결과 성공률"""
        from src.core.pipeline import PipelineResult, StepResult, StepStatus, PipelineStatus
        from datetime import datetime

        now = datetime.now()
        result = PipelineResult(
            pipeline_id="test",
            pipeline_name="test_pipeline",
            status=PipelineStatus.SUCCESS,
            started_at=now,
            steps=[
                StepResult("1", "step1", StepStatus.SUCCESS, now),
                StepResult("2", "step2", StepStatus.SUCCESS, now),
                StepResult("3", "step3", StepStatus.FAILED, now),
            ]
        )

        assert result.success_rate == pytest.approx(2/3, 0.01)


# ============================================================
# 데이터 변환 테스트
# ============================================================

class TestDataTransformers:
    """데이터 변환 테스트"""

    def test_filter_by(self):
        """필드 값 필터링"""
        from src.core.pipeline import DataTransformers

        data = [
            {"name": "A", "active": True},
            {"name": "B", "active": False},
            {"name": "C", "active": True},
        ]

        transform = DataTransformers.filter_by("active", True)
        result = transform(data, {})

        assert len(result) == 2
        assert all(item["active"] for item in result)

    def test_select_fields(self):
        """필드 선택"""
        from src.core.pipeline import DataTransformers

        data = [
            {"id": 1, "name": "A", "extra": "x"},
            {"id": 2, "name": "B", "extra": "y"},
        ]

        transform = DataTransformers.select_fields("id", "name")
        result = transform(data, {})

        assert len(result) == 2
        assert "extra" not in result[0]
        assert result[0] == {"id": 1, "name": "A"}

    def test_rename_fields(self):
        """필드명 변경"""
        from src.core.pipeline import DataTransformers

        data = [{"old_name": "value"}]

        transform = DataTransformers.rename_fields({"old_name": "new_name"})
        result = transform(data, {})

        assert "new_name" in result[0]
        assert "old_name" not in result[0]

    def test_add_timestamp(self):
        """타임스탬프 추가"""
        from src.core.pipeline import DataTransformers

        data = [{"id": 1}]

        transform = DataTransformers.add_timestamp("created_at")
        result = transform(data, {})

        assert "created_at" in result[0]

    def test_deduplicate(self):
        """중복 제거"""
        from src.core.pipeline import DataTransformers

        data = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
            {"id": 1, "name": "A duplicate"},
        ]

        transform = DataTransformers.deduplicate("id")
        result = transform(data, {})

        assert len(result) == 2

    def test_sort_by(self):
        """정렬"""
        from src.core.pipeline import DataTransformers

        data = [
            {"id": 3},
            {"id": 1},
            {"id": 2},
        ]

        transform = DataTransformers.sort_by("id")
        result = transform(data, {})

        assert result[0]["id"] == 1
        assert result[2]["id"] == 3

    def test_limit(self):
        """개수 제한"""
        from src.core.pipeline import DataTransformers

        data = [{"id": i} for i in range(100)]

        transform = DataTransformers.limit(10)
        result = transform(data, {})

        assert len(result) == 10


# ============================================================
# 파이프라인 실행 테스트
# ============================================================

class TestPipelineExecution:
    """파이프라인 실행 테스트"""

    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        """간단한 파이프라인 실행"""
        from src.core.pipeline import Pipeline, FunctionStep

        async def double(data, ctx):
            return data * 2

        pipeline = Pipeline("test_simple")
        pipeline.add_step(FunctionStep("double", double))

        result = await pipeline.run(5)

        assert result.status.value == "success"
        assert result.metadata["final_data"] == 10

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self):
        """멀티 스텝 파이프라인"""
        from src.core.pipeline import Pipeline, FunctionStep

        async def step1(data, ctx):
            return data + 1

        async def step2(data, ctx):
            return data * 2

        async def step3(data, ctx):
            return {"result": data}

        pipeline = Pipeline("test_multi")
        pipeline.add_step(FunctionStep("add", step1))
        pipeline.add_step(FunctionStep("multiply", step2))
        pipeline.add_step(FunctionStep("wrap", step3))

        result = await pipeline.run(5)

        assert result.status.value == "success"
        assert len(result.steps) == 3
        assert result.metadata["final_data"]["result"] == 12  # (5+1)*2

    @pytest.mark.asyncio
    async def test_pipeline_with_context(self):
        """컨텍스트 사용 파이프라인"""
        from src.core.pipeline import Pipeline, FunctionStep

        async def use_context(data, ctx):
            return data + ctx.get("offset", 0)

        pipeline = Pipeline("test_context")
        pipeline.set_context("offset", 100)
        pipeline.add_step(FunctionStep("add_offset", use_context))

        result = await pipeline.run(5)

        assert result.metadata["final_data"] == 105

    @pytest.mark.asyncio
    async def test_pipeline_failure(self):
        """파이프라인 실패"""
        from src.core.pipeline import Pipeline, FunctionStep

        async def fail(data, ctx):
            raise ValueError("Test error")

        pipeline = Pipeline("test_fail")
        pipeline.add_step(FunctionStep("fail_step", fail))

        result = await pipeline.run(None)

        assert result.status.value == "failed"
        assert "Test error" in result.error


# ============================================================
# 파이프라인 레지스트리 테스트
# ============================================================

class TestPipelineRegistry:
    """파이프라인 레지스트리 테스트"""

    def test_register_pipeline(self):
        """파이프라인 등록"""
        from src.core.pipeline import Pipeline, PipelineRegistry

        pipeline = Pipeline("registry_test")
        PipelineRegistry.register(pipeline)

        assert "registry_test" in PipelineRegistry.list()

    def test_get_pipeline(self):
        """파이프라인 조회"""
        from src.core.pipeline import Pipeline, PipelineRegistry

        pipeline = Pipeline("get_test")
        PipelineRegistry.register(pipeline)

        retrieved = PipelineRegistry.get("get_test")
        assert retrieved is not None
        assert retrieved.name == "get_test"

    def test_get_nonexistent_pipeline(self):
        """존재하지 않는 파이프라인 조회"""
        from src.core.pipeline import PipelineRegistry

        result = PipelineRegistry.get("nonexistent")
        assert result is None


# ============================================================
# 라우터 통합 테스트
# ============================================================

class TestPipelineRouterIntegration:
    """파이프라인 라우터 통합 테스트"""

    def test_pipeline_endpoints_registered(self):
        """파이프라인 엔드포인트 등록 확인"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/api/v1/pipeline/create" in paths
        assert "/api/v1/pipeline/list" in paths
        assert "/api/v1/pipeline/run" in paths
        assert "/api/v1/pipeline/stats" in paths
        assert "/api/v1/pipeline/health" in paths

    def test_root_endpoint_includes_pipeline(self):
        """루트 엔드포인트에 파이프라인 정보 포함"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "2.5.0"
        assert "pipeline" in data
        assert "create" in data["pipeline"]
        assert "run" in data["pipeline"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
