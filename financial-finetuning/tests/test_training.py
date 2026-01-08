# Tests for Training Module
"""
학습 모듈 테스트 (모델 로드 없이)
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# peft 호환성 문제로 인해 직접 임포트 사용
try:
    from src.training.train_lora import (
        FinancialLoRATrainer,
        load_training_config,
    )
    from peft import LoraConfig
    from transformers import TrainingArguments
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="peft/transformers 호환성 문제")
class TestTrainingConfig:
    """학습 설정 테스트"""

    def test_load_config(self):
        """설정 파일 로드 테스트"""
        config_path = project_root / "configs" / "training_config.yaml"
        config = load_training_config(str(config_path))

        assert config is not None
        assert "model" in config
        assert "lora" in config
        assert "training" in config

    def test_model_config(self):
        """모델 설정 검증"""
        config_path = project_root / "configs" / "training_config.yaml"
        config = load_training_config(str(config_path))

        model_config = config.get("model", {})
        assert "name" in model_config
        assert "max_seq_length" in model_config
        assert model_config["max_seq_length"] > 0

    def test_lora_config(self):
        """LoRA 설정 검증"""
        config_path = project_root / "configs" / "training_config.yaml"
        config = load_training_config(str(config_path))

        lora_config = config.get("lora", {})
        assert "r" in lora_config
        assert "lora_alpha" in lora_config
        assert "target_modules" in lora_config
        assert lora_config["r"] > 0
        assert len(lora_config["target_modules"]) > 0

    def test_training_config(self):
        """학습 설정 검증"""
        config_path = project_root / "configs" / "training_config.yaml"
        config = load_training_config(str(config_path))

        train_config = config.get("training", {})
        assert "num_train_epochs" in train_config
        assert "learning_rate" in train_config
        assert train_config["num_train_epochs"] > 0
        assert train_config["learning_rate"] > 0


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="peft/transformers 호환성 문제")
class TestFinancialLoRATrainer:
    """FinancialLoRATrainer 클래스 테스트 (모델 로드 없이)"""

    def test_trainer_init(self):
        """트레이너 초기화 테스트"""
        config_path = project_root / "configs" / "training_config.yaml"
        trainer = FinancialLoRATrainer(config_path=str(config_path))

        assert trainer is not None
        assert trainer.config is not None
        assert trainer.model is None  # 아직 로드 안됨

    def test_trainer_with_custom_config(self):
        """커스텀 설정으로 트레이너 초기화"""
        custom_config = {
            "model": {"name": "test-model", "max_seq_length": 1024},
            "lora": {"r": 8, "lora_alpha": 16},
            "training": {"num_train_epochs": 1},
        }

        trainer = FinancialLoRATrainer(config=custom_config)

        assert trainer.config["model"]["name"] == "test-model"
        assert trainer.config["lora"]["r"] == 8

    def test_setup_dataset_only(self):
        """데이터셋만 설정 테스트"""
        config_path = project_root / "configs" / "training_config.yaml"
        trainer = FinancialLoRATrainer(config_path=str(config_path))
        trainer.setup_dataset()

        assert trainer.dataset is not None
        stats = trainer.dataset.get_statistics()
        assert stats["total_samples"] > 0


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="peft/transformers 호환성 문제")
class TestLoRAConfigGeneration:
    """LoRA 설정 생성 테스트"""

    def test_get_lora_config(self):
        """LoRA Config 객체 생성 테스트"""
        config_path = project_root / "configs" / "training_config.yaml"
        trainer = FinancialLoRATrainer(config_path=str(config_path))
        lora_config = trainer._get_lora_config()

        assert isinstance(lora_config, LoraConfig)
        assert lora_config.r == trainer.config["lora"]["r"]
        assert lora_config.lora_alpha == trainer.config["lora"]["lora_alpha"]

    def test_get_training_arguments(self):
        """TrainingArguments 생성 테스트"""
        config_path = project_root / "configs" / "training_config.yaml"
        trainer = FinancialLoRATrainer(config_path=str(config_path))
        training_args = trainer._get_training_arguments()

        assert isinstance(training_args, TrainingArguments)
        assert training_args.num_train_epochs == trainer.config["training"]["num_train_epochs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
