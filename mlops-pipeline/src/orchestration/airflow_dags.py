"""Airflow DAG definitions for ML pipeline orchestration.

This module provides configurable DAG builders for automated ML workflows,
with a local simulation mode when Airflow is not available.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from enum import Enum

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.sensors.filesystem import FileSensor
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


class TaskType(Enum):
    """Task type enumeration."""
    PYTHON = "python"
    BASH = "bash"
    SENSOR = "sensor"


@dataclass
class RetryPolicy:
    """Retry policy for tasks."""
    retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    retry_exponential_backoff: bool = True
    max_retry_delay: timedelta = field(default_factory=lambda: timedelta(hours=1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "retries": self.retries,
            "retry_delay_seconds": self.retry_delay.total_seconds(),
            "retry_exponential_backoff": self.retry_exponential_backoff,
            "max_retry_delay_seconds": self.max_retry_delay.total_seconds(),
        }


@dataclass
class TaskDefinition:
    """Definition of a single DAG task."""
    task_id: str
    task_type: TaskType
    description: str = ""
    python_callable: Callable | None = None
    bash_command: str | None = None
    file_path: str | None = None  # For file sensors
    dependencies: list[str] = field(default_factory=list)
    retry_policy: RetryPolicy | None = None
    timeout: timedelta | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "dependencies": self.dependencies,
            "retry_policy": self.retry_policy.to_dict() if self.retry_policy else None,
            "timeout_seconds": self.timeout.total_seconds() if self.timeout else None,
            "params": self.params,
        }


@dataclass
class DAGConfig:
    """Configuration for a DAG."""
    dag_id: str
    description: str = ""
    schedule_interval: str | timedelta | None = None
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    catchup: bool = False
    max_active_runs: int = 1
    default_retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    tags: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "description": self.description,
            "schedule_interval": str(self.schedule_interval),
            "start_date": self.start_date.isoformat(),
            "catchup": self.catchup,
            "max_active_runs": self.max_active_runs,
            "default_retry_policy": self.default_retry_policy.to_dict(),
            "tags": self.tags,
            "params": self.params,
        }


@dataclass
class DAGRunResult:
    """Result of a simulated DAG run."""
    dag_id: str
    run_id: str
    success: bool
    start_time: datetime
    end_time: datetime
    task_results: dict[str, dict[str, Any]]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "run_id": self.run_id,
            "success": self.success,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "task_results": self.task_results,
            "errors": self.errors,
        }


class AirflowDAGBuilder:
    """Builder for Airflow DAGs with local simulation support.

    This class provides a high-level interface for building DAGs
    that works with Airflow when available, and provides a local
    simulation mode for testing and development.

    Attributes:
        use_airflow: Whether to use real Airflow (if available)
        dags_path: Path to store DAG definitions
    """

    def __init__(
        self,
        dags_path: str | Path = "dags",
        use_airflow: bool = True,
    ):
        """Initialize the DAG builder.

        Args:
            dags_path: Path to store DAG definitions and metadata
            use_airflow: Whether to use Airflow if available
        """
        self.dags_path = Path(dags_path)
        self.dags_path.mkdir(parents=True, exist_ok=True)

        self.use_airflow = use_airflow and AIRFLOW_AVAILABLE
        self._dags: dict[str, DAGConfig] = {}
        self._tasks: dict[str, dict[str, TaskDefinition]] = {}

        # Load existing definitions
        self._load_definitions()

    def _load_definitions(self) -> None:
        """Load DAG definitions from disk."""
        definitions_file = self.dags_path / "dag_definitions.json"
        if definitions_file.exists():
            with open(definitions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for dag_data in data.get("dags", []):
                    # Reconstruct DAGConfig (simplified)
                    dag_id = dag_data["dag_id"]
                    self._dags[dag_id] = DAGConfig(
                        dag_id=dag_id,
                        description=dag_data.get("description", ""),
                        tags=dag_data.get("tags", []),
                    )

    def _save_definitions(self) -> None:
        """Save DAG definitions to disk."""
        definitions_file = self.dags_path / "dag_definitions.json"
        data = {
            "dags": [dag.to_dict() for dag in self._dags.values()],
            "updated_at": datetime.now().isoformat(),
        }
        with open(definitions_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create_dag(self, config: DAGConfig) -> DAGConfig:
        """Create a new DAG.

        Args:
            config: DAG configuration

        Returns:
            Created DAGConfig
        """
        self._dags[config.dag_id] = config
        self._tasks[config.dag_id] = {}
        self._save_definitions()
        return config

    def add_task(
        self,
        dag_id: str,
        task: TaskDefinition,
    ) -> TaskDefinition:
        """Add a task to a DAG.

        Args:
            dag_id: ID of the DAG
            task: Task definition

        Returns:
            Added TaskDefinition

        Raises:
            ValueError: If DAG not found
        """
        if dag_id not in self._dags:
            raise ValueError(f"DAG '{dag_id}' not found")

        self._tasks[dag_id][task.task_id] = task
        return task

    def get_dag(self, dag_id: str) -> DAGConfig | None:
        """Get a DAG configuration."""
        return self._dags.get(dag_id)

    def list_dags(self) -> list[dict[str, Any]]:
        """List all DAGs."""
        return [
            {
                "dag_id": dag.dag_id,
                "description": dag.description,
                "tags": dag.tags,
                "task_count": len(self._tasks.get(dag.dag_id, {})),
            }
            for dag in self._dags.values()
        ]

    def delete_dag(self, dag_id: str) -> bool:
        """Delete a DAG."""
        if dag_id not in self._dags:
            return False

        del self._dags[dag_id]
        if dag_id in self._tasks:
            del self._tasks[dag_id]
        self._save_definitions()
        return True

    def create_fraud_detection_pipeline(self) -> DAGConfig:
        """Create a predefined fraud detection ML pipeline DAG.

        Returns:
            Created DAGConfig
        """
        config = DAGConfig(
            dag_id="fraud_detection_pipeline",
            description="End-to-end fraud detection ML pipeline",
            schedule_interval="0 2 * * *",  # Daily at 2 AM
            tags=["ml", "fraud", "production"],
            params={"model_name": "fraud_detector_v1"},
        )

        dag = self.create_dag(config)

        # Data ingestion task
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="ingest_data",
                task_type=TaskType.PYTHON,
                description="Ingest new transaction data",
                params={"source": "data/transactions"},
            ),
        )

        # Data validation task
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="validate_data",
                task_type=TaskType.PYTHON,
                description="Validate data quality with Great Expectations",
                dependencies=["ingest_data"],
                params={"suite": "fraud_detection"},
            ),
        )

        # Feature engineering task
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="engineer_features",
                task_type=TaskType.PYTHON,
                description="Generate features using Feast",
                dependencies=["validate_data"],
            ),
        )

        # Model training task
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="train_model",
                task_type=TaskType.PYTHON,
                description="Train fraud detection model",
                dependencies=["engineer_features"],
                retry_policy=RetryPolicy(retries=2),
                timeout=timedelta(hours=2),
            ),
        )

        # Model evaluation task
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="evaluate_model",
                task_type=TaskType.PYTHON,
                description="Evaluate model performance",
                dependencies=["train_model"],
            ),
        )

        # Model deployment task
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="deploy_model",
                task_type=TaskType.PYTHON,
                description="Deploy model if performance threshold met",
                dependencies=["evaluate_model"],
                params={"threshold": 0.85},
            ),
        )

        return dag

    def create_auto_retrain_dag(
        self,
        model_name: str,
        schedule: str = "0 3 * * 0",  # Weekly at 3 AM on Sunday
        performance_threshold: float = 0.80,
    ) -> DAGConfig:
        """Create an auto-retraining DAG.

        Args:
            model_name: Name of the model to retrain
            schedule: Cron schedule expression
            performance_threshold: Minimum performance to deploy

        Returns:
            Created DAGConfig
        """
        dag_id = f"auto_retrain_{model_name}"

        config = DAGConfig(
            dag_id=dag_id,
            description=f"Automated retraining for {model_name}",
            schedule_interval=schedule,
            tags=["ml", "auto-retrain", model_name],
            params={
                "model_name": model_name,
                "performance_threshold": performance_threshold,
            },
        )

        dag = self.create_dag(config)

        # Check for data drift
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="check_drift",
                task_type=TaskType.PYTHON,
                description="Check for data/concept drift",
            ),
        )

        # Prepare training data
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="prepare_data",
                task_type=TaskType.PYTHON,
                description="Prepare training data from feature store",
                dependencies=["check_drift"],
            ),
        )

        # Retrain model
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="retrain",
                task_type=TaskType.PYTHON,
                description="Retrain model with new data",
                dependencies=["prepare_data"],
                timeout=timedelta(hours=4),
            ),
        )

        # Evaluate new model
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="evaluate",
                task_type=TaskType.PYTHON,
                description="Evaluate retrained model",
                dependencies=["retrain"],
            ),
        )

        # Compare with production model
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="compare_models",
                task_type=TaskType.PYTHON,
                description="Compare new model with production",
                dependencies=["evaluate"],
            ),
        )

        # Conditional deployment
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="conditional_deploy",
                task_type=TaskType.PYTHON,
                description="Deploy if better than threshold",
                dependencies=["compare_models"],
                params={"threshold": performance_threshold},
            ),
        )

        # Notify stakeholders
        self.add_task(
            dag.dag_id,
            TaskDefinition(
                task_id="notify",
                task_type=TaskType.PYTHON,
                description="Send notification about retraining result",
                dependencies=["conditional_deploy"],
            ),
        )

        return dag

    def simulate_run(
        self,
        dag_id: str,
        task_functions: dict[str, Callable] | None = None,
    ) -> DAGRunResult:
        """Simulate a DAG run locally (for testing).

        Args:
            dag_id: ID of the DAG to run
            task_functions: Optional mapping of task_id to callable functions

        Returns:
            DAGRunResult with execution details
        """
        if dag_id not in self._dags:
            raise ValueError(f"DAG '{dag_id}' not found")

        task_functions = task_functions or {}
        start_time = datetime.now()
        run_id = f"{dag_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        task_results: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        success = True

        tasks = self._tasks.get(dag_id, {})

        # Simple topological sort for execution order
        executed = set()
        execution_order = []

        def can_execute(task: TaskDefinition) -> bool:
            return all(dep in executed for dep in task.dependencies)

        remaining = list(tasks.values())
        while remaining:
            runnable = [t for t in remaining if can_execute(t)]
            if not runnable:
                errors.append("Circular dependency detected")
                success = False
                break

            for task in runnable:
                task_start = datetime.now()

                try:
                    if task.task_id in task_functions:
                        result = task_functions[task.task_id]()
                    else:
                        # Simulate task execution
                        result = {"status": "simulated", "task_id": task.task_id}

                    task_results[task.task_id] = {
                        "success": True,
                        "start_time": task_start.isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "result": result,
                    }
                    executed.add(task.task_id)
                except Exception as e:
                    task_results[task.task_id] = {
                        "success": False,
                        "start_time": task_start.isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "error": str(e),
                    }
                    errors.append(f"Task {task.task_id} failed: {str(e)}")
                    success = False

                remaining.remove(task)
                execution_order.append(task.task_id)

        end_time = datetime.now()

        return DAGRunResult(
            dag_id=dag_id,
            run_id=run_id,
            success=success,
            start_time=start_time,
            end_time=end_time,
            task_results=task_results,
            errors=errors,
        )

    def get_task_dependencies(self, dag_id: str) -> dict[str, list[str]]:
        """Get task dependency graph for a DAG.

        Args:
            dag_id: ID of the DAG

        Returns:
            Dictionary mapping task_id to list of dependencies
        """
        if dag_id not in self._tasks:
            return {}

        return {
            task.task_id: task.dependencies
            for task in self._tasks[dag_id].values()
        }

    def export_dag_definition(self, dag_id: str) -> str | None:
        """Export DAG as Airflow Python code.

        Args:
            dag_id: ID of the DAG

        Returns:
            Python code string for Airflow DAG
        """
        if dag_id not in self._dags:
            return None

        dag = self._dags[dag_id]
        tasks = self._tasks.get(dag_id, {})

        lines = [
            '"""Auto-generated Airflow DAG."""',
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "from datetime import datetime, timedelta",
            "",
            f'# DAG: {dag.dag_id}',
            f'# Description: {dag.description}',
            "",
            "default_args = {",
            f"    'owner': 'airflow',",
            f"    'retries': {dag.default_retry_policy.retries},",
            f"    'retry_delay': timedelta(seconds={dag.default_retry_policy.retry_delay.total_seconds()}),",
            "}",
            "",
            f"with DAG(",
            f"    dag_id='{dag.dag_id}',",
            f"    description='{dag.description}',",
            f"    schedule_interval='{dag.schedule_interval}',",
            f"    start_date=datetime({dag.start_date.year}, {dag.start_date.month}, {dag.start_date.day}),",
            f"    catchup={dag.catchup},",
            f"    max_active_runs={dag.max_active_runs},",
            f"    tags={dag.tags},",
            "    default_args=default_args,",
            ") as dag:",
            "",
        ]

        # Add task definitions
        for task in tasks.values():
            if task.task_type == TaskType.PYTHON:
                lines.extend([
                    f"    {task.task_id} = PythonOperator(",
                    f"        task_id='{task.task_id}',",
                    f"        python_callable=lambda: None,  # TODO: Implement",
                    "    )",
                    "",
                ])
            elif task.task_type == TaskType.BASH:
                lines.extend([
                    f"    {task.task_id} = BashOperator(",
                    f"        task_id='{task.task_id}',",
                    f"        bash_command='{task.bash_command or 'echo done'}',",
                    "    )",
                    "",
                ])

        # Add dependencies
        lines.append("    # Dependencies")
        for task in tasks.values():
            for dep in task.dependencies:
                lines.append(f"    {dep} >> {task.task_id}")

        return "\n".join(lines)
