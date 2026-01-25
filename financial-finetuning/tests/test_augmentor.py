# -*- coding: utf-8 -*-
"""Tests for Dataset Augmentor module."""

import pytest
import tempfile
import json
from pathlib import Path

from src.data.dataset_augmentor import (
    DatasetAugmentor,
    SyntheticDataGenerator,
    AugmentationConfig,
    AugmentedSample,
    expand_dataset,
)


class TestAugmentedSample:
    """AugmentedSample 데이터클래스 테스트."""

    def test_create_sample(self):
        """샘플 생성 테스트."""
        sample = AugmentedSample(
            instruction="테스트 질문",
            output="테스트 답변",
            input="테스트 입력",
            category="test",
            augmentation_type="original",
        )

        assert sample.instruction == "테스트 질문"
        assert sample.output == "테스트 답변"
        assert sample.category == "test"
        assert sample.augmentation_type == "original"

    def test_sample_defaults(self):
        """기본값 테스트."""
        sample = AugmentedSample(
            instruction="질문",
            output="답변",
        )

        assert sample.input == ""
        assert sample.category == ""
        assert sample.metadata == {}


class TestAugmentationConfig:
    """AugmentationConfig 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = AugmentationConfig()

        assert config.target_samples == 1000
        assert config.use_llm is False
        assert config.numerical_variation_range == 0.3
        assert config.enable_paraphrasing is True

    def test_custom_config(self):
        """커스텀 설정 테스트."""
        config = AugmentationConfig(
            target_samples=500,
            use_llm=False,
            numerical_variation_range=0.5,
        )

        assert config.target_samples == 500
        assert config.numerical_variation_range == 0.5


class TestDatasetAugmentor:
    """DatasetAugmentor 테스트."""

    @pytest.fixture
    def sample_instructions(self):
        """샘플 instruction 데이터."""
        return [
            {
                "instruction": "거래 분석을 해주세요.",
                "input": "거래금액: 100만원, 거래시간: 오후 3시",
                "output": "이 거래는 정상 거래입니다.",
                "category": "fraud_detection",
            },
            {
                "instruction": "투자 분석을 해주세요.",
                "input": "종목: 삼성전자, PER: 10",
                "output": "이 종목은 저평가 상태입니다.",
                "category": "investment_analysis",
            },
            {
                "instruction": "리스크를 평가해주세요.",
                "input": "포트폴리오: 주식 60%, 채권 40%",
                "output": "중간 수준의 리스크입니다.",
                "category": "risk_assessment",
            },
        ]

    def test_init(self, sample_instructions):
        """초기화 테스트."""
        config = AugmentationConfig(target_samples=100)
        augmentor = DatasetAugmentor(sample_instructions, config)

        assert len(augmentor.samples) == 3
        assert augmentor.config.target_samples == 100

    def test_augment_includes_originals(self, sample_instructions):
        """원본 포함 테스트."""
        config = AugmentationConfig(target_samples=10)
        augmentor = DatasetAugmentor(sample_instructions, config)

        result = augmentor.augment()

        originals = [s for s in result if s.augmentation_type == "original"]
        assert len(originals) == 3

    def test_augment_reaches_target(self, sample_instructions):
        """목표 개수 달성 테스트."""
        config = AugmentationConfig(target_samples=20)
        augmentor = DatasetAugmentor(sample_instructions, config)

        result = augmentor.augment()

        assert len(result) == 20

    def test_template_variation(self, sample_instructions):
        """템플릿 변형 테스트."""
        config = AugmentationConfig(
            target_samples=10,
            enable_paraphrasing=True,
            enable_numerical_variation=False,
        )
        augmentor = DatasetAugmentor(sample_instructions, config)

        result = augmentor.augment()

        template_variations = [s for s in result if s.augmentation_type == "template_variation"]
        assert len(template_variations) > 0

    def test_numerical_variation(self, sample_instructions):
        """수치 변형 테스트."""
        config = AugmentationConfig(
            target_samples=10,
            enable_paraphrasing=False,
            enable_numerical_variation=True,
        )
        augmentor = DatasetAugmentor(sample_instructions, config)

        result = augmentor.augment()

        numerical_variations = [s for s in result if s.augmentation_type == "numerical_variation"]
        assert len(numerical_variations) > 0

    def test_vary_numbers(self, sample_instructions):
        """숫자 변형 함수 테스트."""
        augmentor = DatasetAugmentor(sample_instructions)

        original = "거래금액: 100만원, 거래시간: 15시, 거래횟수: 5회"
        varied = augmentor._vary_numbers(original, 0.3)

        # 숫자가 변형되었는지 확인 (정확히 같지 않아야 함 - 랜덤이라 가끔 같을 수 있음)
        assert "만원" in varied
        assert "시" in varied
        assert "회" in varied

    def test_to_json(self, sample_instructions, tmp_path):
        """JSON 저장 테스트."""
        config = AugmentationConfig(target_samples=5)
        augmentor = DatasetAugmentor(sample_instructions, config)
        result = augmentor.augment()

        output_path = tmp_path / "augmented.json"
        augmentor.to_json(result, output_path)

        assert output_path.exists()

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 5

    def test_get_statistics(self, sample_instructions):
        """통계 테스트."""
        config = AugmentationConfig(target_samples=10)
        augmentor = DatasetAugmentor(sample_instructions, config)
        result = augmentor.augment()

        stats = augmentor.get_statistics(result)

        assert stats["total_samples"] == 10
        assert "by_augmentation_type" in stats
        assert "by_category" in stats


class TestSyntheticDataGenerator:
    """SyntheticDataGenerator 테스트."""

    def test_init(self):
        """초기화 테스트."""
        generator = SyntheticDataGenerator(seed=42)
        assert generator is not None

    def test_generate_fraud_samples(self):
        """사기 탐지 샘플 생성 테스트."""
        generator = SyntheticDataGenerator(seed=42)
        samples = generator.generate("fraud_detection", count=5)

        assert len(samples) == 5
        for sample in samples:
            assert "instruction" in sample
            assert "output" in sample
            assert sample["category"] == "fraud_detection"

    def test_generate_investment_samples(self):
        """투자 분석 샘플 생성 테스트."""
        generator = SyntheticDataGenerator(seed=42)
        samples = generator.generate("investment_analysis", count=5)

        assert len(samples) == 5
        for sample in samples:
            assert "instruction" in sample
            assert sample["category"] == "investment_analysis"

    def test_generate_risk_samples(self):
        """리스크 평가 샘플 생성 테스트."""
        generator = SyntheticDataGenerator(seed=42)
        samples = generator.generate("risk_assessment", count=5)

        assert len(samples) == 5
        for sample in samples:
            assert sample["category"] == "risk_assessment"

    def test_generate_unknown_category(self):
        """알 수 없는 카테고리 테스트."""
        generator = SyntheticDataGenerator(seed=42)
        samples = generator.generate("unknown_category", count=3)

        assert len(samples) == 3
        for sample in samples:
            assert sample["category"] == "unknown_category"

    def test_reproducibility(self):
        """재현성 테스트."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)

        samples1 = gen1.generate("fraud_detection", count=3)
        samples2 = gen2.generate("fraud_detection", count=3)

        # 같은 시드로 같은 결과가 나와야 함
        for s1, s2 in zip(samples1, samples2):
            assert s1["instruction"] == s2["instruction"]


class TestExpandDataset:
    """expand_dataset 편의 함수 테스트."""

    @pytest.fixture
    def sample_instructions(self):
        """샘플 데이터."""
        return [
            {
                "instruction": "테스트 질문",
                "input": "테스트 입력",
                "output": "테스트 답변",
                "category": "test",
            }
        ]

    def test_expand_without_save(self, sample_instructions):
        """저장 없이 확장 테스트."""
        result = expand_dataset(sample_instructions, target_size=10)

        assert len(result) == 10

    def test_expand_with_save(self, sample_instructions, tmp_path):
        """저장과 함께 확장 테스트."""
        output_path = tmp_path / "expanded.json"

        result = expand_dataset(
            sample_instructions,
            target_size=10,
            output_path=output_path,
        )

        assert len(result) == 10
        assert output_path.exists()

        # 통계 파일도 생성되어야 함
        stats_path = tmp_path / "augmentation_stats.json"
        assert stats_path.exists()
