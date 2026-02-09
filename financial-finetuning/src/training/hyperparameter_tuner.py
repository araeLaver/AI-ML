# Hyperparameter Tuner with Ray Tune + Optuna
"""
Ray Tune과 Optuna를 활용한 하이퍼파라미터 자동 최적화 모듈

Features:
- ASHA/HyperBand 스케줄러를 통한 조기 종료
- Optuna 기반 베이지안 탐색
- FinancialLoRATrainer와 완전 통합
- 최적 설정 자동 저장 및 재학습 지원
"""

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Ray는 선택적 의존성
RAY_AVAILABLE = False
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

    RAY_AVAILABLE = True
except ImportError:
    pass

OPTUNA_AVAILABLE = False
try:
    from optuna.samplers import TPESampler  # noqa: F401

    OPTUNA_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TuningConfig:
    """하이퍼파라미터 탐색 설정"""

    search_space: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "learning_rate": {"type": "loguniform", "lower": 1e-5, "upper": 1e-3},
        "num_train_epochs": {"type": "choice", "values": [1, 2, 3, 5]},
        "per_device_train_batch_size": {"type": "choice", "values": [2, 4, 8]},
        "gradient_accumulation_steps": {"type": "choice", "values": [2, 4, 8, 16]},
        "warmup_ratio": {"type": "uniform", "lower": 0.0, "upper": 0.1},
        "weight_decay": {"type": "loguniform", "lower": 1e-4, "upper": 0.1},
        "lora_r": {"type": "choice", "values": [8, 16, 32, 64]},
        "lora_alpha": {"type": "choice", "values": [16, 32, 64, 128]},
        "lora_dropout": {"type": "uniform", "lower": 0.0, "upper": 0.2},
    })

    # Scheduler
    scheduler_type: str = "asha"
    scheduler_max_t: int = 5
    scheduler_grace_period: int = 1
    scheduler_reduction_factor: int = 3

    # Search algorithm
    search_algorithm: str = "optuna"

    # Metric
    metric: str = "eval_loss"
    mode: str = "min"

    # Resources
    num_samples: int = 10
    gpus_per_trial: float = 1
    cpus_per_trial: int = 4
    max_concurrent_trials: int = 1

    # Output
    output_dir: str = "./outputs/tuning_results"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_space": self.search_space,
            "scheduler_type": self.scheduler_type,
            "scheduler_max_t": self.scheduler_max_t,
            "scheduler_grace_period": self.scheduler_grace_period,
            "scheduler_reduction_factor": self.scheduler_reduction_factor,
            "search_algorithm": self.search_algorithm,
            "metric": self.metric,
            "mode": self.mode,
            "num_samples": self.num_samples,
            "gpus_per_trial": self.gpus_per_trial,
            "cpus_per_trial": self.cpus_per_trial,
            "max_concurrent_trials": self.max_concurrent_trials,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "TuningConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        scheduler_cfg = raw.get("scheduler", {})
        resources_cfg = raw.get("resources", {})

        return cls(
            search_space=raw.get("search_space", cls.search_space),
            scheduler_type=scheduler_cfg.get("type", "asha"),
            scheduler_max_t=scheduler_cfg.get("max_t", 5),
            scheduler_grace_period=scheduler_cfg.get("grace_period", 1),
            scheduler_reduction_factor=scheduler_cfg.get("reduction_factor", 3),
            search_algorithm=raw.get("search_algorithm", "optuna"),
            metric=raw.get("metric", "eval_loss"),
            mode=raw.get("mode", "min"),
            num_samples=resources_cfg.get("num_samples", 10),
            gpus_per_trial=resources_cfg.get("gpus_per_trial", 1),
            cpus_per_trial=resources_cfg.get("cpus_per_trial", 4),
            max_concurrent_trials=resources_cfg.get("max_concurrent_trials", 1),
            output_dir=raw.get("output_dir", "./outputs/tuning_results"),
        )


@dataclass
class TuningResult:
    """하이퍼파라미터 탐색 결과"""

    best_config: Dict[str, Any]
    best_metric: float
    all_trials: List[Dict[str, Any]]
    num_trials: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config,
            "best_metric": self.best_metric,
            "all_trials": self.all_trials,
            "num_trials": self.num_trials,
            "duration_seconds": self.duration_seconds,
        }

    def summary(self) -> str:
        lines = [
            "=== Hyperparameter Tuning Results ===",
            f"Total trials: {self.num_trials}",
            f"Best {list(self.best_config.keys())[0] if self.best_config else 'metric'}: {self.best_metric:.6f}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Best config:",
        ]
        for k, v in self.best_config.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core tuner
# ---------------------------------------------------------------------------

class HyperparameterTuner:
    """Ray Tune 기반 하이퍼파라미터 최적화"""

    # Flat parameter -> nested config path mapping
    _PARAM_MAP = {
        "learning_rate": ("training", "learning_rate"),
        "num_train_epochs": ("training", "num_train_epochs"),
        "per_device_train_batch_size": ("training", "per_device_train_batch_size"),
        "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
        "warmup_ratio": ("training", "warmup_ratio"),
        "weight_decay": ("training", "weight_decay"),
        "lora_r": ("lora", "r"),
        "lora_alpha": ("lora", "lora_alpha"),
        "lora_dropout": ("lora", "lora_dropout"),
    }

    def __init__(
        self,
        base_config: Dict[str, Any],
        tuning_config: TuningConfig,
        data_path: Optional[str] = None,
    ):
        if not RAY_AVAILABLE:
            raise ImportError(
                "ray[tune] is required for hyperparameter tuning. "
                "Install with: pip install 'ray[tune]>=2.9.0'"
            )

        self.base_config = base_config
        self.tuning_config = tuning_config
        self.data_path = data_path
        self._results = None

    def _create_search_space(self) -> Dict[str, Any]:
        """탐색 공간을 Ray Tune 형식으로 변환"""
        space = {}
        for name, spec in self.tuning_config.search_space.items():
            space[name] = self._spec_to_ray(name, spec)
        return space

    @staticmethod
    def _spec_to_ray(name: str, spec: Dict[str, Any]):
        """개별 파라미터 스펙 → Ray Tune 탐색 공간 객체"""
        t = spec["type"]
        if t == "loguniform":
            return tune.loguniform(spec["lower"], spec["upper"])
        elif t == "uniform":
            return tune.uniform(spec["lower"], spec["upper"])
        elif t == "choice":
            return tune.choice(spec["values"])
        elif t == "randint":
            return tune.randint(spec["lower"], spec["upper"])
        elif t == "quniform":
            return tune.quniform(spec["lower"], spec["upper"], spec["q"])
        else:
            raise ValueError(f"Unknown search space type '{t}' for parameter '{name}'")

    @staticmethod
    def _apply_config_overrides(
        base: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """flat 파라미터 → nested config 구조에 적용 (원본 불변)"""
        config = copy.deepcopy(base)
        for key, value in overrides.items():
            path = HyperparameterTuner._PARAM_MAP.get(key)
            if path is None:
                logger.warning(f"Unknown tuning parameter: {key}")
                continue
            section, param = path
            if section not in config:
                config[section] = {}
            config[section][param] = value
        return config

    def _create_scheduler(self):
        """스케줄러 생성"""
        cfg = self.tuning_config
        if cfg.scheduler_type == "asha":
            return ASHAScheduler(
                metric=cfg.metric,
                mode=cfg.mode,
                max_t=cfg.scheduler_max_t,
                grace_period=cfg.scheduler_grace_period,
                reduction_factor=cfg.scheduler_reduction_factor,
            )
        elif cfg.scheduler_type == "hyperband":
            return HyperBandScheduler(
                metric=cfg.metric,
                mode=cfg.mode,
                max_t=cfg.scheduler_max_t,
                reduction_factor=cfg.scheduler_reduction_factor,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {cfg.scheduler_type}")

    def _create_search_algorithm(self):
        """탐색 알고리즘 생성"""
        if self.tuning_config.search_algorithm == "optuna":
            if OPTUNA_AVAILABLE:
                from ray.tune.search.optuna import OptunaSearch
                return OptunaSearch(metric=self.tuning_config.metric, mode=self.tuning_config.mode)
            else:
                logger.warning("optuna not installed, falling back to random search")
                return None
        # random search: None (Ray default)
        return None

    def _create_trainable(self):
        """Ray Tune trainable 함수 생성 (클로저)"""
        base_config = self.base_config
        data_path = self.data_path

        def trainable(config: Dict[str, Any]):
            """Ray worker에서 실행되는 학습 함수"""
            from transformers import TrainerCallback, TrainerState, TrainerControl
            import ray.train

            class RayTuneReportCallback(TrainerCallback):
                """평가 시 Ray Tune에 메트릭 리포트"""

                def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
                    if metrics:
                        ray.train.report(metrics)

            # flat 파라미터 → nested config 적용
            merged_config = HyperparameterTuner._apply_config_overrides(base_config, config)

            # Ray Tune 환경에서는 HuggingFace 리포터 비활성화
            if "training" not in merged_config:
                merged_config["training"] = {}
            merged_config["training"]["report_to"] = "none"

            from src.training.train_lora import FinancialLoRATrainer

            trainer = FinancialLoRATrainer(
                config=merged_config,
                extra_callbacks=[RayTuneReportCallback()],
            )
            trainer.setup(data_path=data_path)
            trainer.train()

        return trainable

    def tune(self) -> TuningResult:
        """하이퍼파라미터 탐색 실행"""
        cfg = self.tuning_config
        start_time = time.time()

        search_space = self._create_search_space()
        scheduler = self._create_scheduler()
        search_alg = self._create_search_algorithm()

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tuner = tune.Tuner(
            tune.with_resources(
                self._create_trainable(),
                resources={"cpu": cfg.cpus_per_trial, "gpu": cfg.gpus_per_trial},
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=cfg.num_samples,
                max_concurrent_trials=cfg.max_concurrent_trials,
                scheduler=scheduler,
                search_alg=search_alg,
            ),
            run_config=ray.train.RunConfig(
                name="financial_hp_tuning",
                storage_path=str(output_dir),
            ),
        )

        results = tuner.fit()
        duration = time.time() - start_time

        best_result = results.get_best_result(metric=cfg.metric, mode=cfg.mode)

        all_trials = []
        for r in results:
            trial_info = {"config": r.config}
            if r.metrics:
                trial_info["metrics"] = {k: v for k, v in r.metrics.items() if isinstance(v, (int, float))}
            all_trials.append(trial_info)

        self._results = TuningResult(
            best_config=best_result.config,
            best_metric=best_result.metrics.get(cfg.metric, float("inf")),
            all_trials=all_trials,
            num_trials=len(all_trials),
            duration_seconds=duration,
        )

        return self._results

    def get_best_config(self) -> Dict[str, Any]:
        """최적 설정을 full training config로 반환"""
        if self._results is None:
            raise RuntimeError("Call tune() first")
        return self._apply_config_overrides(self.base_config, self._results.best_config)

    def get_results_summary(self) -> str:
        """결과 요약 문자열"""
        if self._results is None:
            raise RuntimeError("Call tune() first")
        return self._results.summary()

    def save_results(self, path: Optional[str] = None) -> str:
        """결과를 JSON + best config YAML로 저장"""
        if self._results is None:
            raise RuntimeError("Call tune() first")

        output_dir = Path(path or self.tuning_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 전체 결과 JSON
        results_path = output_dir / "tuning_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self._results.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # 최적 training config YAML (바로 재학습 가능)
        best_config_path = output_dir / "best_training_config.yaml"
        best_full_config = self.get_best_config()
        with open(best_config_path, "w", encoding="utf-8") as f:
            yaml.dump(best_full_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"  - Full results: {results_path}")
        logger.info(f"  - Best config: {best_config_path}")

        return str(output_dir)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_hyperparameter_search(
    tuning_config_path: str = "configs/tuning_config.yaml",
    base_config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> TuningResult:
    """
    YAML 설정 → 탐색 실행 → 결과 저장까지 원스텝

    Args:
        tuning_config_path: 탐색 설정 YAML 경로
        base_config_path: 기본 학습 설정 경로 (None이면 tuning config 내 경로 사용)
        data_path: 학습 데이터 경로
        output_dir: 결과 저장 경로

    Returns:
        TuningResult
    """
    # 탐색 설정 로드
    tuning_config = TuningConfig.from_yaml(tuning_config_path)

    if output_dir:
        tuning_config.output_dir = output_dir

    # 기본 학습 설정 로드
    if base_config_path is None:
        with open(tuning_config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        base_config_path = raw.get("base_config_path", "configs/training_config.yaml")

    from .train_lora import load_training_config
    base_config = load_training_config(base_config_path)

    # 탐색 실행
    tuner = HyperparameterTuner(
        base_config=base_config,
        tuning_config=tuning_config,
        data_path=data_path,
    )

    result = tuner.tune()
    tuner.save_results()

    print(result.summary())
    return result
