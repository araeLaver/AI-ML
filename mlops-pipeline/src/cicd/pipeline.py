"""
CI/CD Pipeline Manager
- 모델 검증 게이트 (성능, 크기, 지연시간)
- 배포 전략 (Canary, Blue/Green, Rolling)
- 파이프라인 오케스트레이션
"""

import os
import time
import uuid
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import docker

    HAS_DOCKER = True
except ImportError:
    docker = None
    HAS_DOCKER = False

try:
    from kubernetes import client as k8s_client, config as k8s_config

    HAS_K8S = True
except ImportError:
    k8s_client = None
    k8s_config = None
    HAS_K8S = False


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStrategy(str, Enum):
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"


@dataclass
class PipelineStage:
    name: str
    status: PipelineStatus = PipelineStatus.PENDING
    duration: float = 0.0
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineRun:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stages: List[PipelineStage] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    git_sha: str = ""
    trigger: str = "manual"
    created_at: datetime = field(default_factory=datetime.now)


class ModelValidationGate:
    """모델 품질 게이트 - 배포 전 모델 검증"""

    def __init__(
        self,
        min_accuracy: float = 0.8,
        min_f1: float = 0.75,
        max_model_size_mb: float = 500.0,
        max_latency_ms: float = 100.0,
    ):
        self.min_accuracy = min_accuracy
        self.min_f1 = min_f1
        self.max_model_size_mb = max_model_size_mb
        self.max_latency_ms = max_latency_ms

    def validate_model_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """정확도/F1 등 임계값 검증"""
        results = {"passed": True, "checks": []}

        accuracy = metrics.get("accuracy", 0.0)
        check_acc = {
            "metric": "accuracy",
            "value": accuracy,
            "threshold": self.min_accuracy,
            "passed": accuracy >= self.min_accuracy,
        }
        results["checks"].append(check_acc)

        f1 = metrics.get("f1_score", 0.0)
        check_f1 = {
            "metric": "f1_score",
            "value": f1,
            "threshold": self.min_f1,
            "passed": f1 >= self.min_f1,
        }
        results["checks"].append(check_f1)

        results["passed"] = all(c["passed"] for c in results["checks"])
        return results

    def validate_model_size(self, model_path: str) -> Dict[str, Any]:
        """모델 파일 크기 제한 검증"""
        path = Path(model_path)
        if not path.exists():
            return {
                "passed": False,
                "error": f"Model file not found: {model_path}",
                "size_mb": 0,
                "max_size_mb": self.max_model_size_mb,
            }

        size_mb = path.stat().st_size / (1024 * 1024)
        passed = size_mb <= self.max_model_size_mb
        return {
            "passed": passed,
            "size_mb": round(size_mb, 2),
            "max_size_mb": self.max_model_size_mb,
        }

    def validate_model_latency(
        self,
        predict_fn: Callable,
        sample_input: Any,
        n_iterations: int = 100,
    ) -> Dict[str, Any]:
        """추론 속도 벤치마크"""
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            predict_fn(sample_input)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        return {
            "passed": avg_latency <= self.max_latency_ms,
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "max_latency_ms": self.max_latency_ms,
            "n_iterations": n_iterations,
        }

    def compare_with_champion(
        self, candidate: Dict[str, float], champion: Dict[str, float]
    ) -> Dict[str, Any]:
        """챔피언 모델 대비 비교"""
        comparisons = []
        overall_better = True

        for metric in ["accuracy", "f1_score", "precision", "recall"]:
            if metric in candidate and metric in champion:
                diff = candidate[metric] - champion[metric]
                is_better = diff >= 0
                comparisons.append(
                    {
                        "metric": metric,
                        "candidate": candidate[metric],
                        "champion": champion[metric],
                        "diff": round(diff, 4),
                        "improved": is_better,
                    }
                )
                if not is_better:
                    overall_better = False

        return {
            "passed": overall_better,
            "comparisons": comparisons,
            "recommendation": "promote" if overall_better else "reject",
        }

    def run_all_gates(
        self,
        metrics: Dict[str, float],
        model_path: Optional[str] = None,
        predict_fn: Optional[Callable] = None,
        sample_input: Any = None,
        champion_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """전체 품질 게이트 실행"""
        results = {"overall_passed": True, "gates": {}}

        # Performance gate
        perf = self.validate_model_performance(metrics)
        results["gates"]["performance"] = perf
        if not perf["passed"]:
            results["overall_passed"] = False

        # Size gate
        if model_path:
            size = self.validate_model_size(model_path)
            results["gates"]["size"] = size
            if not size["passed"]:
                results["overall_passed"] = False

        # Latency gate
        if predict_fn and sample_input is not None:
            latency = self.validate_model_latency(predict_fn, sample_input)
            results["gates"]["latency"] = latency
            if not latency["passed"]:
                results["overall_passed"] = False

        # Champion comparison
        if champion_metrics:
            comparison = self.compare_with_champion(metrics, champion_metrics)
            results["gates"]["champion_comparison"] = comparison
            if not comparison["passed"]:
                results["overall_passed"] = False

        return results


class DeploymentManager:
    """배포 관리자 - Canary, Blue/Green, Rolling 배포 전략"""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.deployments: List[Dict[str, Any]] = []
        self._active_deployment: Optional[Dict[str, Any]] = None

    def canary_deploy(
        self,
        image: str,
        steps: Optional[List[int]] = None,
        health_check_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """카나리 배포 - 점진적 트래픽 이동"""
        if steps is None:
            steps = [1, 5, 25, 100]

        deployment = {
            "id": str(uuid.uuid4())[:8],
            "strategy": DeploymentStrategy.CANARY.value,
            "image": image,
            "status": "in_progress",
            "steps_completed": [],
            "current_traffic_pct": 0,
            "started_at": datetime.now().isoformat(),
        }

        for pct in steps:
            logger.info(f"Canary: routing {pct}% traffic to {image}")
            deployment["current_traffic_pct"] = pct
            deployment["steps_completed"].append(pct)

            if health_check_fn:
                try:
                    healthy = health_check_fn()
                    if not healthy:
                        deployment["status"] = "rolled_back"
                        deployment["rollback_reason"] = f"Health check failed at {pct}%"
                        self.deployments.append(deployment)
                        return deployment
                except Exception as e:
                    deployment["status"] = "rolled_back"
                    deployment["rollback_reason"] = str(e)
                    self.deployments.append(deployment)
                    return deployment

        deployment["status"] = "completed"
        self._active_deployment = deployment
        self.deployments.append(deployment)
        return deployment

    def blue_green_deploy(
        self, image: str, target_env: str = "green"
    ) -> Dict[str, Any]:
        """블루/그린 배포 - 환경 전환"""
        deployment = {
            "id": str(uuid.uuid4())[:8],
            "strategy": DeploymentStrategy.BLUE_GREEN.value,
            "image": image,
            "target_env": target_env,
            "status": "completed",
            "started_at": datetime.now().isoformat(),
        }

        logger.info(f"Blue/Green: deploying {image} to {target_env}")
        logger.info(f"Blue/Green: switching traffic to {target_env}")

        self._active_deployment = deployment
        self.deployments.append(deployment)
        return deployment

    def rollback(self, deployment_id: Optional[str] = None) -> Dict[str, Any]:
        """자동 롤백"""
        if deployment_id:
            target = next(
                (d for d in self.deployments if d["id"] == deployment_id), None
            )
        else:
            target = self._active_deployment

        if not target:
            return {"success": False, "error": "No deployment found to rollback"}

        target["status"] = "rolled_back"
        rollback_record = {
            "id": str(uuid.uuid4())[:8],
            "type": "rollback",
            "original_deployment_id": target["id"],
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        self.deployments.append(rollback_record)
        self._active_deployment = None
        return {"success": True, "rollback": rollback_record}

    def get_deployment_status(self) -> Dict[str, Any]:
        """배포 상태 조회"""
        return {
            "active_deployment": self._active_deployment,
            "total_deployments": len(self.deployments),
            "history": self.deployments[-10:],
        }


class CICDPipeline:
    """CI/CD 파이프라인 오케스트레이터"""

    def __init__(
        self,
        project_dir: str = ".",
        registry: str = "ghcr.io",
        image_name: str = "fraud-detector",
    ):
        self.project_dir = project_dir
        self.registry = registry
        self.image_name = image_name
        self.validation_gate = ModelValidationGate()
        self.deployment_manager = DeploymentManager()
        self._runs: List[PipelineRun] = []
        self._current_run: Optional[PipelineRun] = None

    def _execute_stage(self, stage: PipelineStage, fn: Callable) -> PipelineStage:
        """단일 스테이지 실행"""
        stage.status = PipelineStatus.RUNNING
        start = time.time()
        try:
            result = fn()
            stage.status = PipelineStatus.SUCCESS
            stage.logs.append(f"Stage '{stage.name}' completed successfully")
            if isinstance(result, dict):
                stage.artifacts.update(result)
        except Exception as e:
            stage.status = PipelineStatus.FAILED
            stage.logs.append(f"Stage '{stage.name}' failed: {str(e)}")
            raise
        finally:
            stage.duration = round(time.time() - start, 2)
        return stage

    def run_tests(self) -> PipelineStage:
        """테스트 실행"""
        stage = PipelineStage(name="tests")

        def _run():
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.project_dir,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Tests failed:\n{result.stdout}\n{result.stderr}")
            return {"test_output": result.stdout}

        return self._execute_stage(stage, _run)

    def run_linting(self) -> PipelineStage:
        """린트 실행"""
        stage = PipelineStage(name="linting")

        def _run():
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "src/", "tests/"],
                capture_output=True,
                text=True,
                cwd=self.project_dir,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Linting failed:\n{result.stdout}\n{result.stderr}")
            return {"lint_output": result.stdout}

        return self._execute_stage(stage, _run)

    def build_image(self, tag: Optional[str] = None) -> PipelineStage:
        """Docker 이미지 빌드"""
        stage = PipelineStage(name="build")
        tag = tag or f"{self.registry}/{self.image_name}:latest"

        def _run():
            if HAS_DOCKER:
                try:
                    client = docker.from_env()
                    image, build_logs = client.images.build(
                        path=self.project_dir, tag=tag
                    )
                    return {"image_tag": tag, "image_id": image.id}
                except Exception:
                    logger.warning("Docker daemon not available, simulating build")
                    return {"image_tag": tag, "image_id": "simulated", "simulated": True}
            else:
                logger.warning("Docker not available, simulating build")
                return {"image_tag": tag, "image_id": "simulated", "simulated": True}

        return self._execute_stage(stage, _run)

    def validate_model(self, metrics: Dict[str, float], model_path: Optional[str] = None) -> PipelineStage:
        """모델 품질 게이트 실행"""
        stage = PipelineStage(name="model_validation")

        def _run():
            result = self.validation_gate.run_all_gates(
                metrics=metrics, model_path=model_path
            )
            if not result["overall_passed"]:
                raise RuntimeError(f"Model validation failed: {result}")
            return {"validation_result": str(result)}

        return self._execute_stage(stage, _run)

    def deploy(
        self,
        image: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        target_env: str = "staging",
    ) -> PipelineStage:
        """배포 실행"""
        stage = PipelineStage(name=f"deploy_{target_env}")

        def _run():
            if strategy == DeploymentStrategy.CANARY:
                result = self.deployment_manager.canary_deploy(image)
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                result = self.deployment_manager.blue_green_deploy(image, target_env)
            else:
                result = self.deployment_manager.canary_deploy(image, steps=[100])

            if result.get("status") == "rolled_back":
                raise RuntimeError(f"Deployment rolled back: {result.get('rollback_reason')}")
            return {"deployment_id": result["id"], "strategy": strategy.value}

        return self._execute_stage(stage, _run)

    def promote_model(self, from_env: str = "staging", to_env: str = "production") -> Dict[str, Any]:
        """staging → production 승격"""
        status = self.deployment_manager.get_deployment_status()
        active = status.get("active_deployment")
        if not active:
            return {"success": False, "error": "No active deployment to promote"}

        image = active.get("image", "")
        logger.info(f"Promoting {image} from {from_env} to {to_env}")
        deploy_result = self.deployment_manager.blue_green_deploy(image, to_env)
        return {
            "success": True,
            "from_env": from_env,
            "to_env": to_env,
            "deployment": deploy_result,
        }

    def run_full_pipeline(
        self,
        metrics: Dict[str, float],
        model_path: Optional[str] = None,
        deploy_strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        skip_tests: bool = False,
    ) -> PipelineRun:
        """전체 파이프라인 오케스트레이션"""
        run = PipelineRun(
            git_sha=self._get_git_sha(),
            trigger="pipeline",
        )
        self._current_run = run
        run.status = PipelineStatus.RUNNING

        try:
            # Linting
            lint_stage = self.run_linting()
            run.stages.append(lint_stage)

            # Tests
            if not skip_tests:
                test_stage = self.run_tests()
                run.stages.append(test_stage)

            # Model validation
            val_stage = self.validate_model(metrics, model_path)
            run.stages.append(val_stage)

            # Build
            build_stage = self.build_image()
            run.stages.append(build_stage)

            # Deploy
            image_tag = build_stage.artifacts.get("image_tag", f"{self.registry}/{self.image_name}:latest")
            deploy_stage = self.deploy(image_tag, deploy_strategy)
            run.stages.append(deploy_stage)

            run.status = PipelineStatus.SUCCESS
        except Exception as e:
            run.status = PipelineStatus.FAILED
            logger.error(f"Pipeline failed: {e}")

        self._runs.append(run)
        return run

    def get_pipeline_status(self) -> Dict[str, Any]:
        """현재 파이프라인 상태 조회"""
        current = None
        if self._current_run:
            current = {
                "id": self._current_run.id,
                "status": self._current_run.status.value,
                "stages": [
                    {"name": s.name, "status": s.status.value, "duration": s.duration}
                    for s in self._current_run.stages
                ],
            }
        return {"current_run": current, "total_runs": len(self._runs)}

    def get_run_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """파이프라인 실행 이력"""
        return [
            {
                "id": r.id,
                "status": r.status.value,
                "git_sha": r.git_sha,
                "trigger": r.trigger,
                "created_at": r.created_at.isoformat(),
                "stages": len(r.stages),
            }
            for r in self._runs[-limit:]
        ]

    def _get_git_sha(self) -> str:
        """현재 Git SHA 가져오기"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_dir,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
