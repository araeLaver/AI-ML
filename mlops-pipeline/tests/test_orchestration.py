"""
Orchestration (Airflow DAGs) Tests
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.orchestration import (
    AirflowDAGBuilder,
    DAGConfig,
    TaskDefinition,
    RetryPolicy,
)
from src.orchestration.airflow_dags import TaskType, DAGRunResult


class TestRetryPolicy:
    """RetryPolicy 테스트"""

    def test_default_values(self):
        """기본값 확인"""
        policy = RetryPolicy()

        assert policy.retries == 3
        assert policy.retry_delay == timedelta(minutes=5)
        assert policy.retry_exponential_backoff is True

    def test_custom_values(self):
        """커스텀 값 확인"""
        policy = RetryPolicy(
            retries=5,
            retry_delay=timedelta(seconds=30),
            retry_exponential_backoff=False,
        )

        assert policy.retries == 5
        assert policy.retry_delay == timedelta(seconds=30)
        assert policy.retry_exponential_backoff is False

    def test_to_dict(self):
        """직렬화 확인"""
        policy = RetryPolicy(retries=2, retry_delay=timedelta(minutes=10))

        data = policy.to_dict()

        assert data["retries"] == 2
        assert data["retry_delay_seconds"] == 600


class TestTaskDefinition:
    """TaskDefinition 테스트"""

    def test_python_task(self):
        """Python 태스크 정의"""
        task = TaskDefinition(
            task_id="my_task",
            task_type=TaskType.PYTHON,
            description="Test task",
        )

        assert task.task_id == "my_task"
        assert task.task_type == TaskType.PYTHON
        assert task.dependencies == []

    def test_task_with_dependencies(self):
        """의존성 포함 태스크"""
        task = TaskDefinition(
            task_id="downstream",
            task_type=TaskType.PYTHON,
            dependencies=["upstream1", "upstream2"],
        )

        assert "upstream1" in task.dependencies
        assert "upstream2" in task.dependencies

    def test_bash_task(self):
        """Bash 태스크 정의"""
        task = TaskDefinition(
            task_id="bash_task",
            task_type=TaskType.BASH,
            bash_command="echo 'Hello World'",
        )

        assert task.task_type == TaskType.BASH
        assert task.bash_command == "echo 'Hello World'"

    def test_to_dict(self):
        """직렬화 확인"""
        task = TaskDefinition(
            task_id="test",
            task_type=TaskType.PYTHON,
            dependencies=["dep1"],
            params={"key": "value"},
        )

        data = task.to_dict()

        assert data["task_id"] == "test"
        assert data["task_type"] == "python"
        assert "dep1" in data["dependencies"]


class TestDAGConfig:
    """DAGConfig 테스트"""

    def test_default_config(self):
        """기본 설정"""
        config = DAGConfig(dag_id="test_dag")

        assert config.dag_id == "test_dag"
        assert config.catchup is False
        assert config.max_active_runs == 1

    def test_custom_config(self):
        """커스텀 설정"""
        config = DAGConfig(
            dag_id="my_dag",
            description="My test DAG",
            schedule_interval="0 * * * *",
            catchup=True,
            max_active_runs=3,
            tags=["test", "ml"],
        )

        assert config.dag_id == "my_dag"
        assert config.schedule_interval == "0 * * * *"
        assert config.catchup is True
        assert "test" in config.tags

    def test_to_dict(self):
        """직렬화 확인"""
        config = DAGConfig(
            dag_id="serialized",
            description="Test",
            tags=["tag1"],
        )

        data = config.to_dict()

        assert data["dag_id"] == "serialized"
        assert "tag1" in data["tags"]


class TestAirflowDAGBuilder:
    """AirflowDAGBuilder 테스트"""

    @pytest.fixture
    def temp_dags_path(self):
        """임시 DAGs 경로"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def builder(self, temp_dags_path):
        """DAG 빌더 인스턴스"""
        return AirflowDAGBuilder(dags_path=temp_dags_path, use_airflow=False)

    def test_init_creates_directory(self, temp_dags_path):
        """초기화 시 디렉토리 생성"""
        builder = AirflowDAGBuilder(dags_path=temp_dags_path, use_airflow=False)
        assert Path(temp_dags_path).exists()

    def test_create_dag(self, builder):
        """DAG 생성"""
        config = DAGConfig(
            dag_id="new_dag",
            description="New DAG",
            tags=["test"],
        )

        result = builder.create_dag(config)

        assert result.dag_id == "new_dag"
        assert builder.get_dag("new_dag") is not None

    def test_add_task(self, builder):
        """태스크 추가"""
        builder.create_dag(DAGConfig(dag_id="test_dag"))

        task = TaskDefinition(
            task_id="my_task",
            task_type=TaskType.PYTHON,
        )

        result = builder.add_task("test_dag", task)

        assert result.task_id == "my_task"

    def test_add_task_to_nonexistent_dag(self, builder):
        """존재하지 않는 DAG에 태스크 추가"""
        task = TaskDefinition(task_id="task", task_type=TaskType.PYTHON)

        with pytest.raises(ValueError, match="not found"):
            builder.add_task("nonexistent", task)

    def test_list_dags(self, builder):
        """DAG 목록"""
        builder.create_dag(DAGConfig(dag_id="dag1"))
        builder.create_dag(DAGConfig(dag_id="dag2"))

        dags = builder.list_dags()

        assert len(dags) == 2
        dag_ids = [d["dag_id"] for d in dags]
        assert "dag1" in dag_ids
        assert "dag2" in dag_ids

    def test_delete_dag(self, builder):
        """DAG 삭제"""
        builder.create_dag(DAGConfig(dag_id="to_delete"))

        deleted = builder.delete_dag("to_delete")

        assert deleted is True
        assert builder.get_dag("to_delete") is None

    def test_delete_nonexistent_dag(self, builder):
        """존재하지 않는 DAG 삭제"""
        deleted = builder.delete_dag("nonexistent")
        assert deleted is False

    def test_create_fraud_detection_pipeline(self, builder):
        """사기 탐지 파이프라인 생성"""
        dag = builder.create_fraud_detection_pipeline()

        assert dag.dag_id == "fraud_detection_pipeline"
        assert "ml" in dag.tags
        assert "fraud" in dag.tags

        # 태스크 확인
        deps = builder.get_task_dependencies(dag.dag_id)
        assert "ingest_data" in deps
        assert "validate_data" in deps
        assert "train_model" in deps
        assert "deploy_model" in deps

        # 의존성 확인
        assert "ingest_data" in deps["validate_data"]
        assert "validate_data" in deps["engineer_features"]

    def test_create_auto_retrain_dag(self, builder):
        """자동 재학습 DAG 생성"""
        dag = builder.create_auto_retrain_dag(
            model_name="fraud_model",
            schedule="0 4 * * 1",
            performance_threshold=0.85,
        )

        assert dag.dag_id == "auto_retrain_fraud_model"
        assert "auto-retrain" in dag.tags
        assert dag.params["performance_threshold"] == 0.85

        # 태스크 확인
        deps = builder.get_task_dependencies(dag.dag_id)
        assert "check_drift" in deps
        assert "retrain" in deps
        assert "conditional_deploy" in deps

    def test_simulate_run_success(self, builder):
        """DAG 실행 시뮬레이션 성공"""
        builder.create_dag(DAGConfig(dag_id="simple_dag"))
        builder.add_task("simple_dag", TaskDefinition(
            task_id="task1",
            task_type=TaskType.PYTHON,
        ))
        builder.add_task("simple_dag", TaskDefinition(
            task_id="task2",
            task_type=TaskType.PYTHON,
            dependencies=["task1"],
        ))

        result = builder.simulate_run("simple_dag")

        assert result.success is True
        assert result.dag_id == "simple_dag"
        assert "task1" in result.task_results
        assert "task2" in result.task_results

    def test_simulate_run_with_functions(self, builder):
        """함수 포함 DAG 실행 시뮬레이션"""
        builder.create_dag(DAGConfig(dag_id="func_dag"))
        builder.add_task("func_dag", TaskDefinition(
            task_id="compute",
            task_type=TaskType.PYTHON,
        ))

        execution_count = [0]

        def my_function():
            execution_count[0] += 1
            return {"value": 42}

        result = builder.simulate_run("func_dag", {"compute": my_function})

        assert result.success is True
        assert execution_count[0] == 1
        assert result.task_results["compute"]["result"]["value"] == 42

    def test_simulate_run_with_failure(self, builder):
        """실패하는 DAG 실행 시뮬레이션"""
        builder.create_dag(DAGConfig(dag_id="fail_dag"))
        builder.add_task("fail_dag", TaskDefinition(
            task_id="failing_task",
            task_type=TaskType.PYTHON,
        ))

        def failing_function():
            raise ValueError("Intentional failure")

        result = builder.simulate_run("fail_dag", {"failing_task": failing_function})

        assert result.success is False
        assert len(result.errors) > 0
        assert "Intentional failure" in result.errors[0]

    def test_simulate_run_nonexistent_dag(self, builder):
        """존재하지 않는 DAG 실행 시뮬레이션"""
        with pytest.raises(ValueError, match="not found"):
            builder.simulate_run("nonexistent")

    def test_get_task_dependencies(self, builder):
        """태스크 의존성 그래프"""
        builder.create_dag(DAGConfig(dag_id="dep_dag"))
        builder.add_task("dep_dag", TaskDefinition(
            task_id="a",
            task_type=TaskType.PYTHON,
        ))
        builder.add_task("dep_dag", TaskDefinition(
            task_id="b",
            task_type=TaskType.PYTHON,
            dependencies=["a"],
        ))
        builder.add_task("dep_dag", TaskDefinition(
            task_id="c",
            task_type=TaskType.PYTHON,
            dependencies=["a", "b"],
        ))

        deps = builder.get_task_dependencies("dep_dag")

        assert deps["a"] == []
        assert deps["b"] == ["a"]
        assert set(deps["c"]) == {"a", "b"}

    def test_export_dag_definition(self, builder):
        """DAG 정의 내보내기"""
        builder.create_dag(DAGConfig(
            dag_id="export_test",
            description="Test export",
            schedule_interval="0 0 * * *",
        ))
        builder.add_task("export_test", TaskDefinition(
            task_id="task1",
            task_type=TaskType.PYTHON,
        ))
        builder.add_task("export_test", TaskDefinition(
            task_id="task2",
            task_type=TaskType.PYTHON,
            dependencies=["task1"],
        ))

        code = builder.export_dag_definition("export_test")

        assert code is not None
        assert "from airflow import DAG" in code
        assert "export_test" in code
        assert "task1" in code
        assert "task2" in code
        assert "task1 >> task2" in code

    def test_export_nonexistent_dag(self, builder):
        """존재하지 않는 DAG 내보내기"""
        code = builder.export_dag_definition("nonexistent")
        assert code is None

    def test_dag_persistence(self, temp_dags_path):
        """DAG 영속성"""
        # 첫 번째 인스턴스에서 생성
        builder1 = AirflowDAGBuilder(temp_dags_path, use_airflow=False)
        builder1.create_dag(DAGConfig(dag_id="persistent_dag"))

        # 두 번째 인스턴스에서 로드
        builder2 = AirflowDAGBuilder(temp_dags_path, use_airflow=False)

        assert builder2.get_dag("persistent_dag") is not None


class TestDAGRunResult:
    """DAGRunResult 테스트"""

    def test_create_result(self):
        """결과 생성"""
        result = DAGRunResult(
            dag_id="test",
            run_id="test_123",
            success=True,
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            task_results={"task1": {"success": True}},
        )

        assert result.dag_id == "test"
        assert result.success is True

    def test_to_dict(self):
        """직렬화"""
        result = DAGRunResult(
            dag_id="test",
            run_id="run_1",
            success=True,
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            task_results={},
        )

        data = result.to_dict()

        assert data["dag_id"] == "test"
        assert data["duration_seconds"] == 300  # 5 minutes
        assert "start_time" in data
        assert "end_time" in data
