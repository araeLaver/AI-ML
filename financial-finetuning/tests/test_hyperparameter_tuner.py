"""Tests for hyperparameter_tuner module"""

import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.training.hyperparameter_tuner import (
    RAY_AVAILABLE,
    TuningConfig,
    TuningResult,
    HyperparameterTuner,
)


# ===================================================================
# TestTuningConfig
# ===================================================================

class TestTuningConfig:
    """TuningConfig dataclass 테스트"""

    def test_default_values(self):
        cfg = TuningConfig()
        assert cfg.scheduler_type == "asha"
        assert cfg.search_algorithm == "optuna"
        assert cfg.metric == "eval_loss"
        assert cfg.mode == "min"
        assert cfg.num_samples == 10
        assert cfg.gpus_per_trial == 1
        assert cfg.max_concurrent_trials == 1

    def test_custom_values(self):
        cfg = TuningConfig(
            scheduler_type="hyperband",
            num_samples=20,
            gpus_per_trial=2,
            metric="eval_accuracy",
            mode="max",
        )
        assert cfg.scheduler_type == "hyperband"
        assert cfg.num_samples == 20
        assert cfg.gpus_per_trial == 2
        assert cfg.metric == "eval_accuracy"
        assert cfg.mode == "max"

    def test_default_search_space(self):
        cfg = TuningConfig()
        space = cfg.search_space
        assert "learning_rate" in space
        assert "lora_r" in space
        assert "lora_dropout" in space
        assert space["learning_rate"]["type"] == "loguniform"
        assert space["lora_r"]["type"] == "choice"
        assert len(space) == 9

    def test_custom_search_space(self):
        custom_space = {
            "learning_rate": {"type": "uniform", "lower": 1e-4, "upper": 5e-4},
        }
        cfg = TuningConfig(search_space=custom_space)
        assert len(cfg.search_space) == 1
        assert cfg.search_space["learning_rate"]["type"] == "uniform"

    def test_to_dict(self):
        cfg = TuningConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "search_space" in d
        assert "scheduler_type" in d
        assert "num_samples" in d
        assert "output_dir" in d
        assert d["scheduler_type"] == "asha"

    def test_to_dict_roundtrip(self):
        cfg = TuningConfig(num_samples=42, scheduler_type="hyperband")
        d = cfg.to_dict()
        # JSON 직렬화 가능 확인
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["num_samples"] == 42
        assert restored["scheduler_type"] == "hyperband"

    def test_from_yaml(self):
        yaml_content = {
            "search_space": {
                "learning_rate": {"type": "loguniform", "lower": 1e-5, "upper": 1e-3},
                "lora_r": {"type": "choice", "values": [8, 16]},
            },
            "scheduler": {
                "type": "hyperband",
                "max_t": 10,
                "grace_period": 2,
                "reduction_factor": 4,
            },
            "search_algorithm": "random",
            "metric": "eval_f1",
            "mode": "max",
            "resources": {
                "num_samples": 5,
                "gpus_per_trial": 0.5,
                "cpus_per_trial": 2,
                "max_concurrent_trials": 2,
            },
            "output_dir": "/tmp/test_tuning",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name

        try:
            cfg = TuningConfig.from_yaml(tmp_path)
            assert cfg.scheduler_type == "hyperband"
            assert cfg.scheduler_max_t == 10
            assert cfg.scheduler_grace_period == 2
            assert cfg.search_algorithm == "random"
            assert cfg.metric == "eval_f1"
            assert cfg.mode == "max"
            assert cfg.num_samples == 5
            assert cfg.gpus_per_trial == 0.5
            assert cfg.max_concurrent_trials == 2
            assert len(cfg.search_space) == 2
        finally:
            os.unlink(tmp_path)


# ===================================================================
# TestTuningResult
# ===================================================================

class TestTuningResult:
    """TuningResult dataclass 테스트"""

    def _make_result(self) -> TuningResult:
        return TuningResult(
            best_config={"learning_rate": 1e-4, "lora_r": 16},
            best_metric=0.45,
            all_trials=[
                {"config": {"learning_rate": 1e-4}, "metrics": {"eval_loss": 0.45}},
                {"config": {"learning_rate": 5e-4}, "metrics": {"eval_loss": 0.52}},
            ],
            num_trials=2,
            duration_seconds=120.5,
        )

    def test_creation(self):
        result = self._make_result()
        assert result.best_metric == 0.45
        assert result.num_trials == 2
        assert result.duration_seconds == 120.5
        assert result.best_config["lora_r"] == 16

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["best_metric"] == 0.45
        assert d["num_trials"] == 2
        assert len(d["all_trials"]) == 2
        assert "best_config" in d
        assert "duration_seconds" in d

    def test_summary(self):
        result = self._make_result()
        s = result.summary()
        assert "Tuning Results" in s
        assert "Total trials: 2" in s
        assert "120.5s" in s
        assert "learning_rate" in s
        assert "lora_r" in s

    def test_json_serializable(self):
        result = self._make_result()
        d = result.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["best_metric"] == 0.45
        assert restored["num_trials"] == 2


# ===================================================================
# TestConfigApplyOverrides
# ===================================================================

class TestConfigApplyOverrides:
    """_apply_config_overrides 정적 메서드 테스트"""

    def _base_config(self) -> dict:
        return {
            "model": {"name": "test-model"},
            "training": {
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "weight_decay": 0.01,
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            },
        }

    def test_training_params(self):
        base = self._base_config()
        overrides = {"learning_rate": 1e-5, "num_train_epochs": 5}
        result = HyperparameterTuner._apply_config_overrides(base, overrides)
        assert result["training"]["learning_rate"] == 1e-5
        assert result["training"]["num_train_epochs"] == 5
        # 나머지 값은 유지
        assert result["training"]["per_device_train_batch_size"] == 4

    def test_lora_params(self):
        base = self._base_config()
        overrides = {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.1}
        result = HyperparameterTuner._apply_config_overrides(base, overrides)
        assert result["lora"]["r"] == 64
        assert result["lora"]["lora_alpha"] == 128
        assert result["lora"]["lora_dropout"] == 0.1

    def test_original_unchanged(self):
        base = self._base_config()
        original_lr = base["training"]["learning_rate"]
        original_r = base["lora"]["r"]
        overrides = {"learning_rate": 1e-5, "lora_r": 64}

        HyperparameterTuner._apply_config_overrides(base, overrides)

        # 원본이 변경되지 않음을 확인
        assert base["training"]["learning_rate"] == original_lr
        assert base["lora"]["r"] == original_r

    def test_mixed_params(self):
        base = self._base_config()
        overrides = {
            "learning_rate": 5e-5,
            "lora_r": 32,
            "warmup_ratio": 0.05,
            "gradient_accumulation_steps": 8,
        }
        result = HyperparameterTuner._apply_config_overrides(base, overrides)
        assert result["training"]["learning_rate"] == 5e-5
        assert result["training"]["warmup_ratio"] == 0.05
        assert result["training"]["gradient_accumulation_steps"] == 8
        assert result["lora"]["r"] == 32

    def test_empty_overrides(self):
        base = self._base_config()
        result = HyperparameterTuner._apply_config_overrides(base, {})
        assert result == base
        assert result is not base  # deep copy

    def test_missing_section_created(self):
        base = {"model": {"name": "test"}}
        overrides = {"learning_rate": 1e-4}
        result = HyperparameterTuner._apply_config_overrides(base, overrides)
        assert result["training"]["learning_rate"] == 1e-4


# ===================================================================
# TestHyperparameterTuner (requires Ray)
# ===================================================================

@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray[tune] not installed")
class TestHyperparameterTuner:
    """HyperparameterTuner 클래스 테스트 (Ray 필요)"""

    def _make_tuner(self) -> HyperparameterTuner:
        base_config = {
            "model": {"name": "test-model"},
            "training": {"learning_rate": 2e-4, "num_train_epochs": 3},
            "lora": {"r": 16, "lora_alpha": 32},
        }
        tuning_config = TuningConfig(num_samples=2)
        return HyperparameterTuner(base_config, tuning_config)

    def test_init(self):
        tuner = self._make_tuner()
        assert tuner.base_config["model"]["name"] == "test-model"
        assert tuner.tuning_config.num_samples == 2
        assert tuner._results is None

    def test_create_search_space(self):
        tuner = self._make_tuner()
        space = tuner._create_search_space()
        assert "learning_rate" in space
        assert "lora_r" in space
        assert len(space) == 9

    def test_create_scheduler_asha(self):
        tuner = self._make_tuner()
        scheduler = tuner._create_scheduler()
        assert scheduler is not None

    def test_create_scheduler_hyperband(self):
        tuner = self._make_tuner()
        tuner.tuning_config.scheduler_type = "hyperband"
        scheduler = tuner._create_scheduler()
        assert scheduler is not None

    def test_create_scheduler_invalid(self):
        tuner = self._make_tuner()
        tuner.tuning_config.scheduler_type = "invalid"
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            tuner._create_scheduler()

    def test_get_best_config_before_tune(self):
        tuner = self._make_tuner()
        with pytest.raises(RuntimeError, match="Call tune"):
            tuner.get_best_config()

    def test_save_results_before_tune(self):
        tuner = self._make_tuner()
        with pytest.raises(RuntimeError, match="Call tune"):
            tuner.save_results()


# ===================================================================
# TestHyperparameterTunerWithoutRay
# ===================================================================

class TestHyperparameterTunerWithoutRay:
    """Ray 없이 동작하는 부분 테스트"""

    def test_import_error_without_ray(self):
        """RAY_AVAILABLE=False일 때 ImportError 발생 확인"""
        with patch("src.training.hyperparameter_tuner.RAY_AVAILABLE", False):
            with pytest.raises(ImportError, match="ray"):
                HyperparameterTuner(
                    base_config={},
                    tuning_config=TuningConfig(),
                )

    def test_spec_to_ray_invalid_type(self):
        """잘못된 탐색 공간 타입 에러"""
        if not RAY_AVAILABLE:
            pytest.skip("ray needed for _spec_to_ray")
        with pytest.raises(ValueError, match="Unknown search space type"):
            HyperparameterTuner._spec_to_ray("test", {"type": "invalid"})
