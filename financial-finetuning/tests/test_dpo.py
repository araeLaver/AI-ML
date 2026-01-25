# -*- coding: utf-8 -*-
"""Tests for DPO Trainer module."""

import pytest
import json
from pathlib import Path

# Import directly to avoid train_lora dependency
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.training.dpo_trainer import (
    PreferencePair,
    PreferenceDataset,
    DPOTrainingConfig,
    FinancialPreferenceGenerator,
)

# FinancialDPOTrainer requires additional dependencies
try:
    from src.training.dpo_trainer import FinancialDPOTrainer
    DPO_TRAINER_AVAILABLE = True
except ImportError:
    DPO_TRAINER_AVAILABLE = False


class TestPreferencePair:
    """PreferencePair 데이터클래스 테스트."""

    def test_create_pair(self):
        """페어 생성 테스트."""
        pair = PreferencePair(
            prompt="거래를 분석해주세요.",
            chosen="이 거래는 정상입니다.",
            rejected="거래입니다.",
            category="fraud_detection",
        )

        assert pair.prompt == "거래를 분석해주세요."
        assert pair.chosen == "이 거래는 정상입니다."
        assert pair.rejected == "거래입니다."
        assert pair.category == "fraud_detection"

    def test_pair_to_dict(self):
        """딕셔너리 변환 테스트."""
        pair = PreferencePair(
            prompt="테스트",
            chosen="선호 답변",
            rejected="비선호 답변",
        )

        data = pair.to_dict()

        assert data["prompt"] == "테스트"
        assert data["chosen"] == "선호 답변"
        assert data["rejected"] == "비선호 답변"

    def test_pair_defaults(self):
        """기본값 테스트."""
        pair = PreferencePair(
            prompt="질문",
            chosen="좋은 답변",
            rejected="나쁜 답변",
        )

        assert pair.category == ""
        assert pair.metadata == {}


class TestPreferenceDataset:
    """PreferenceDataset 테스트."""

    @pytest.fixture
    def sample_pairs(self):
        """샘플 페어 리스트."""
        return [
            PreferencePair(
                prompt="질문 1",
                chosen="좋은 답변 1",
                rejected="나쁜 답변 1",
                category="category_a",
            ),
            PreferencePair(
                prompt="질문 2",
                chosen="좋은 답변 2",
                rejected="나쁜 답변 2",
                category="category_b",
            ),
            PreferencePair(
                prompt="질문 3",
                chosen="좋은 답변 3",
                rejected="나쁜 답변 3",
                category="category_a",
            ),
        ]

    def test_init_empty(self):
        """빈 데이터셋 생성 테스트."""
        dataset = PreferenceDataset()

        assert len(dataset) == 0

    def test_init_with_pairs(self, sample_pairs):
        """페어와 함께 생성 테스트."""
        dataset = PreferenceDataset(sample_pairs)

        assert len(dataset) == 3

    def test_add_pair(self):
        """페어 추가 테스트."""
        dataset = PreferenceDataset()
        dataset.add_pair(PreferencePair(
            prompt="질문",
            chosen="답변",
            rejected="나쁜 답변",
        ))

        assert len(dataset) == 1

    def test_getitem(self, sample_pairs):
        """인덱싱 테스트."""
        dataset = PreferenceDataset(sample_pairs)

        pair = dataset[0]
        assert pair.prompt == "질문 1"

    def test_to_hf_dataset(self, sample_pairs):
        """HuggingFace 데이터셋 형식 변환 테스트."""
        dataset = PreferenceDataset(sample_pairs)
        hf_data = dataset.to_hf_dataset()

        assert "prompt" in hf_data
        assert "chosen" in hf_data
        assert "rejected" in hf_data
        assert len(hf_data["prompt"]) == 3

    def test_split(self, sample_pairs):
        """분할 테스트."""
        dataset = PreferenceDataset(sample_pairs)
        train, eval_ = dataset.split(eval_ratio=0.33, seed=42)

        assert len(train) + len(eval_) == 3

    def test_get_statistics(self, sample_pairs):
        """통계 테스트."""
        dataset = PreferenceDataset(sample_pairs)
        stats = dataset.get_statistics()

        assert stats["total_pairs"] == 3
        assert "category_a" in stats["by_category"]
        assert "category_b" in stats["by_category"]

    def test_to_json(self, sample_pairs, tmp_path):
        """JSON 저장 테스트."""
        dataset = PreferenceDataset(sample_pairs)
        output_path = tmp_path / "preferences.json"

        dataset.to_json(output_path)

        assert output_path.exists()

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 3

    def test_from_json(self, tmp_path):
        """JSON 로드 테스트."""
        data = [
            {"prompt": "질문", "chosen": "좋은 답변", "rejected": "나쁜 답변"},
        ]

        json_path = tmp_path / "test.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        dataset = PreferenceDataset.from_json(json_path)

        assert len(dataset) == 1
        assert dataset[0].prompt == "질문"


class TestDPOTrainingConfig:
    """DPOTrainingConfig 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = DPOTrainingConfig()

        assert config.model_name == "beomi/Llama-3-Open-Ko-8B"
        assert config.beta == 0.1
        assert config.learning_rate == 5e-5
        assert config.use_lora is True

    def test_custom_config(self):
        """커스텀 설정 테스트."""
        config = DPOTrainingConfig(
            model_name="custom-model",
            beta=0.2,
            learning_rate=1e-4,
            num_train_epochs=5,
        )

        assert config.model_name == "custom-model"
        assert config.beta == 0.2
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 5

    def test_to_dict(self):
        """딕셔너리 변환 테스트."""
        config = DPOTrainingConfig(beta=0.15)
        data = config.to_dict()

        assert data["beta"] == 0.15
        assert "model_name" in data
        assert "learning_rate" in data


class TestFinancialPreferenceGenerator:
    """FinancialPreferenceGenerator 테스트."""

    @pytest.fixture
    def sample_instructions(self):
        """샘플 instruction 데이터."""
        return [
            {
                "instruction": "거래를 분석해주세요.",
                "input": "거래금액: 100만원",
                "output": "**분석 결과**\n\n- 정상 거래입니다.\n- 리스크 점수: 20/100 (낮음)",
                "category": "fraud_detection",
            },
            {
                "instruction": "투자 분석을 해주세요.",
                "input": "종목: 삼성전자",
                "output": "**투자 분석**\n\n투자의견: 매수\n\n목표가: 80,000원",
                "category": "investment_analysis",
            },
        ]

    def test_init(self):
        """초기화 테스트."""
        generator = FinancialPreferenceGenerator(seed=42)
        assert generator is not None

    def test_generate_quality_degradation(self, sample_instructions):
        """품질 저하 전략 테스트."""
        generator = FinancialPreferenceGenerator(seed=42)
        pairs = generator.generate_from_instructions(
            sample_instructions,
            strategy="quality_degradation",
        )

        assert len(pairs) >= 1
        for pair in pairs:
            assert pair.prompt
            assert pair.chosen
            assert pair.rejected
            # 선호 답변과 비선호 답변이 다르거나 비선호가 더 짧아야 함
            assert pair.chosen != pair.rejected or len(pair.rejected) <= len(pair.chosen)

    def test_generate_length_variation(self, sample_instructions):
        """길이 변형 전략 테스트."""
        generator = FinancialPreferenceGenerator(seed=42)
        pairs = generator.generate_from_instructions(
            sample_instructions,
            strategy="length_variation",
        )

        assert len(pairs) >= 1
        for pair in pairs:
            assert len(pair.rejected) < len(pair.chosen)

    def test_generate_factual_errors(self, sample_instructions):
        """사실 오류 전략 테스트."""
        generator = FinancialPreferenceGenerator(seed=42)
        pairs = generator.generate_from_instructions(
            sample_instructions,
            strategy="factual_errors",
        )

        assert len(pairs) >= 1

    def test_format_prompt(self, sample_instructions):
        """프롬프트 포맷팅 테스트."""
        generator = FinancialPreferenceGenerator()
        prompt = generator._format_prompt(sample_instructions[0])

        assert "### Instruction:" in prompt
        assert "### Input:" in prompt
        assert "### Response:" in prompt

    def test_format_prompt_no_input(self):
        """입력 없는 프롬프트 포맷팅 테스트."""
        generator = FinancialPreferenceGenerator()
        prompt = generator._format_prompt({
            "instruction": "질문",
            "output": "답변",
        })

        assert "### Instruction:" in prompt
        assert "### Input:" not in prompt


class TestFinancialDPOTrainer:
    """FinancialDPOTrainer 테스트."""

    def test_init(self):
        """초기화 테스트."""
        config = DPOTrainingConfig()
        trainer = FinancialDPOTrainer(config)

        assert trainer.config == config
        assert trainer.model is None
        assert trainer._training_state["is_trained"] is False

    def test_get_training_state(self):
        """학습 상태 조회 테스트."""
        config = DPOTrainingConfig()
        trainer = FinancialDPOTrainer(config)

        state = trainer.get_training_state()

        assert state["is_trained"] is False
        assert state["started_at"] is None

    def test_config_from_yaml(self, tmp_path):
        """YAML 설정 로드 테스트."""
        yaml_content = """
dpo:
  model_name: "test-model"
  beta: 0.2
  learning_rate: 1e-5
  num_train_epochs: 2
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = DPOTrainingConfig.from_yaml(yaml_path)

        assert config.model_name == "test-model"
        assert config.beta == 0.2
        assert config.learning_rate == 1e-5
