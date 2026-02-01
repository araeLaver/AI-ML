"""CI/CD Pipeline 테스트"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from src.cicd.pipeline import (
    CICDPipeline,
    ModelValidationGate,
    DeploymentManager,
    DeploymentStrategy,
    PipelineStatus,
    PipelineStage,
    PipelineRun,
)


# --- ModelValidationGate Tests ---


class TestModelValidationGate:
    def setup_method(self):
        self.gate = ModelValidationGate(
            min_accuracy=0.8, min_f1=0.75, max_model_size_mb=100, max_latency_ms=50
        )

    def test_validate_performance_pass(self):
        metrics = {"accuracy": 0.9, "f1_score": 0.85}
        result = self.gate.validate_model_performance(metrics)
        assert result["passed"] is True
        assert len(result["checks"]) == 2

    def test_validate_performance_fail_accuracy(self):
        metrics = {"accuracy": 0.7, "f1_score": 0.85}
        result = self.gate.validate_model_performance(metrics)
        assert result["passed"] is False

    def test_validate_performance_fail_f1(self):
        metrics = {"accuracy": 0.9, "f1_score": 0.5}
        result = self.gate.validate_model_performance(metrics)
        assert result["passed"] is False

    def test_validate_model_size_pass(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"x" * 1024)  # 1KB
            f.flush()
            result = self.gate.validate_model_size(f.name)
        os.unlink(f.name)
        assert result["passed"] is True
        assert result["size_mb"] < 1

    def test_validate_model_size_fail(self):
        self.gate.max_model_size_mb = 0.0001  # very small limit
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"x" * 1024 * 1024)  # 1MB
            f.flush()
            result = self.gate.validate_model_size(f.name)
        os.unlink(f.name)
        assert result["passed"] is False

    def test_validate_model_size_missing_file(self):
        result = self.gate.validate_model_size("/nonexistent/model.pkl")
        assert result["passed"] is False
        assert "error" in result

    def test_validate_latency_pass(self):
        fast_fn = lambda x: x * 2
        result = self.gate.validate_model_latency(fast_fn, 42, n_iterations=10)
        assert result["passed"] is True
        assert result["avg_latency_ms"] < 50

    def test_validate_latency_metrics(self):
        fn = lambda x: x
        result = self.gate.validate_model_latency(fn, 1, n_iterations=20)
        assert "p95_latency_ms" in result
        assert "p99_latency_ms" in result
        assert result["n_iterations"] == 20

    def test_compare_with_champion_better(self):
        candidate = {"accuracy": 0.95, "f1_score": 0.90}
        champion = {"accuracy": 0.90, "f1_score": 0.85}
        result = self.gate.compare_with_champion(candidate, champion)
        assert result["passed"] is True
        assert result["recommendation"] == "promote"

    def test_compare_with_champion_worse(self):
        candidate = {"accuracy": 0.80, "f1_score": 0.70}
        champion = {"accuracy": 0.90, "f1_score": 0.85}
        result = self.gate.compare_with_champion(candidate, champion)
        assert result["passed"] is False
        assert result["recommendation"] == "reject"

    def test_run_all_gates_pass(self):
        metrics = {"accuracy": 0.9, "f1_score": 0.85}
        result = self.gate.run_all_gates(metrics=metrics)
        assert result["overall_passed"] is True
        assert "performance" in result["gates"]

    def test_run_all_gates_with_model_path(self):
        metrics = {"accuracy": 0.9, "f1_score": 0.85}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"x" * 100)
            f.flush()
            result = self.gate.run_all_gates(metrics=metrics, model_path=f.name)
        os.unlink(f.name)
        assert result["overall_passed"] is True
        assert "size" in result["gates"]

    def test_run_all_gates_fail(self):
        metrics = {"accuracy": 0.5, "f1_score": 0.3}
        result = self.gate.run_all_gates(metrics=metrics)
        assert result["overall_passed"] is False


# --- DeploymentManager Tests ---


class TestDeploymentManager:
    def setup_method(self):
        self.dm = DeploymentManager(namespace="test")

    def test_canary_deploy_success(self):
        result = self.dm.canary_deploy("myimage:latest", steps=[10, 50, 100])
        assert result["status"] == "completed"
        assert result["current_traffic_pct"] == 100

    def test_canary_deploy_with_health_check_pass(self):
        result = self.dm.canary_deploy(
            "img:v1", steps=[50, 100], health_check_fn=lambda: True
        )
        assert result["status"] == "completed"

    def test_canary_deploy_with_health_check_fail(self):
        result = self.dm.canary_deploy(
            "img:v1", steps=[50, 100], health_check_fn=lambda: False
        )
        assert result["status"] == "rolled_back"
        assert "Health check failed" in result["rollback_reason"]

    def test_canary_deploy_health_check_exception(self):
        def bad_check():
            raise ConnectionError("timeout")

        result = self.dm.canary_deploy("img:v1", steps=[50], health_check_fn=bad_check)
        assert result["status"] == "rolled_back"

    def test_blue_green_deploy(self):
        result = self.dm.blue_green_deploy("img:v2", target_env="green")
        assert result["status"] == "completed"
        assert result["target_env"] == "green"

    def test_rollback_active(self):
        self.dm.canary_deploy("img:v1")
        result = self.dm.rollback()
        assert result["success"] is True

    def test_rollback_no_deployment(self):
        result = self.dm.rollback()
        assert result["success"] is False

    def test_get_deployment_status(self):
        self.dm.canary_deploy("img:v1")
        status = self.dm.get_deployment_status()
        assert status["active_deployment"] is not None
        assert status["total_deployments"] >= 1


# --- PipelineRun / PipelineStage Tests ---


class TestPipelineDataclasses:
    def test_pipeline_stage_defaults(self):
        stage = PipelineStage(name="test")
        assert stage.status == PipelineStatus.PENDING
        assert stage.duration == 0.0
        assert stage.logs == []

    def test_pipeline_run_defaults(self):
        run = PipelineRun()
        assert run.status == PipelineStatus.PENDING
        assert len(run.id) == 8
        assert run.trigger == "manual"


# --- CICDPipeline Tests ---


class TestCICDPipeline:
    def setup_method(self):
        self.pipeline = CICDPipeline(project_dir=".")

    def test_validate_model_pass(self):
        metrics = {"accuracy": 0.9, "f1_score": 0.85}
        stage = self.pipeline.validate_model(metrics)
        assert stage.status == PipelineStatus.SUCCESS

    def test_validate_model_fail(self):
        metrics = {"accuracy": 0.3, "f1_score": 0.2}
        with pytest.raises(RuntimeError):
            self.pipeline.validate_model(metrics)

    def test_deploy_canary(self):
        stage = self.pipeline.deploy("img:v1", DeploymentStrategy.CANARY, "staging")
        assert stage.status == PipelineStatus.SUCCESS

    def test_deploy_blue_green(self):
        stage = self.pipeline.deploy("img:v1", DeploymentStrategy.BLUE_GREEN, "green")
        assert stage.status == PipelineStatus.SUCCESS

    def test_promote_model_no_active(self):
        result = self.pipeline.promote_model()
        assert result["success"] is False

    def test_promote_model_success(self):
        self.pipeline.deployment_manager.canary_deploy("img:v1")
        result = self.pipeline.promote_model("staging", "production")
        assert result["success"] is True
        assert result["to_env"] == "production"

    def test_get_pipeline_status_empty(self):
        status = self.pipeline.get_pipeline_status()
        assert status["current_run"] is None
        assert status["total_runs"] == 0

    def test_get_run_history_empty(self):
        history = self.pipeline.get_run_history()
        assert history == []

    def test_build_image_simulated(self):
        stage = self.pipeline.build_image("test:latest")
        assert stage.status == PipelineStatus.SUCCESS
        assert "image_tag" in stage.artifacts
