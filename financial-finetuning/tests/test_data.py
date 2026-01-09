# Tests for Data Module
"""
데이터 모듈 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFinancialInstructions:
    """금융 Instruction 데이터 테스트"""

    def test_import_instructions(self):
        """Instruction 데이터 임포트 테스트"""
        from src.data.financial_instructions import FINANCIAL_INSTRUCTIONS

        assert FINANCIAL_INSTRUCTIONS is not None
        assert len(FINANCIAL_INSTRUCTIONS) >= 100, "최소 100개 이상의 샘플 필요"
        print(f"Total instructions: {len(FINANCIAL_INSTRUCTIONS)}")

    def test_instruction_structure(self):
        """Instruction 데이터 구조 테스트"""
        from src.data.financial_instructions import FINANCIAL_INSTRUCTIONS

        for item in FINANCIAL_INSTRUCTIONS:
            assert "instruction" in item, "instruction 필드 필수"
            assert "output" in item, "output 필드 필수"
            assert len(item["instruction"]) > 0
            assert len(item["output"]) > 0

    def test_instruction_categories(self):
        """카테고리 존재 테스트"""
        from src.data.financial_instructions import FINANCIAL_INSTRUCTIONS

        categories = set(item.get("category") for item in FINANCIAL_INSTRUCTIONS)
        categories.discard(None)

        expected_categories = {
            "fraud_detection",
            "investment_analysis",
            "product_explanation",
            "risk_assessment",
            "market_analysis",
            "term_explanation",
        }

        assert len(categories) >= 6, "6개 카테고리 필요"
        assert categories == expected_categories, f"카테고리 불일치: {categories}"
        print(f"Found categories: {categories}")

    def test_category_distribution(self):
        """카테고리별 최소 샘플 수 테스트"""
        from src.data.financial_instructions import get_dataset_stats

        stats = get_dataset_stats()

        # 각 카테고리별 최소 10개 이상
        for category, count in stats.items():
            if category != "total":
                assert count >= 10, f"{category} 카테고리에 최소 10개 샘플 필요 (현재: {count})"

        print(f"Category distribution: {stats}")


class TestFormatInstruction:
    """format_instruction 함수 테스트"""

    def test_format_with_input(self):
        """입력이 있는 경우 포맷팅 테스트"""
        from src.data.prepare_dataset import format_instruction

        result = format_instruction(
            instruction="테스트 지시사항",
            input_text="테스트 입력",
            output="테스트 출력",
        )

        assert "### 지시사항:" in result
        assert "테스트 지시사항" in result
        assert "### 입력:" in result
        assert "테스트 입력" in result
        assert "### 응답:" in result
        assert "테스트 출력" in result

    def test_format_without_input(self):
        """입력이 없는 경우 포맷팅 테스트"""
        from src.data.prepare_dataset import format_instruction

        result = format_instruction(
            instruction="테스트 지시사항",
            input_text="",
            output="테스트 출력",
        )

        assert "### 지시사항:" in result
        assert "### 입력:" not in result
        assert "### 응답:" in result

    def test_custom_template(self):
        """커스텀 템플릿 테스트"""
        from src.data.prepare_dataset import format_instruction

        custom_template = "[INST] {instruction} [/INST] {output}"
        custom_template_no_input = "[INST] {instruction} [/INST] {output}"

        result = format_instruction(
            instruction="테스트",
            input_text="",
            output="결과",
            prompt_template=custom_template,
            prompt_template_no_input=custom_template_no_input,
        )

        assert "[INST]" in result
        assert "[/INST]" in result


class TestFinancialInstructionDataset:
    """FinancialInstructionDataset 클래스 테스트"""

    def test_dataset_creation(self):
        """데이터셋 생성 테스트"""
        from src.data.prepare_dataset import FinancialInstructionDataset

        dataset = FinancialInstructionDataset(test_size=0.2)
        assert dataset is not None

    def test_get_statistics(self):
        """통계 조회 테스트"""
        from src.data.prepare_dataset import FinancialInstructionDataset

        dataset = FinancialInstructionDataset()
        stats = dataset.get_statistics()

        assert "total_samples" in stats
        assert "train_samples" in stats
        assert "eval_samples" in stats
        assert "categories" in stats
        assert stats["total_samples"] > 0


class TestCreateFinancialDataset:
    """create_financial_dataset 함수 테스트"""

    def test_create_dataset(self):
        """데이터셋 생성 함수 테스트"""
        from src.data.prepare_dataset import create_financial_dataset

        dataset_dict = create_financial_dataset(test_size=0.2)

        assert "train" in dataset_dict
        assert "test" in dataset_dict
        assert len(dataset_dict["train"]) > 0

    def test_dataset_columns(self):
        """데이터셋 컬럼 테스트"""
        from src.data.prepare_dataset import create_financial_dataset

        dataset_dict = create_financial_dataset()
        train_ds = dataset_dict["train"]

        # 필수 컬럼 확인
        assert "text" in train_ds.column_names
        assert "instruction" in train_ds.column_names
        assert "output" in train_ds.column_names


class TestDataAugmentation:
    """데이터 증강 테스트"""

    def test_augment_dataset(self):
        """데이터 증강 함수 테스트"""
        from src.data.prepare_dataset import augment_dataset
        from src.data.financial_instructions import FINANCIAL_INSTRUCTIONS

        original_count = len(FINANCIAL_INSTRUCTIONS)
        augmented = augment_dataset(FINANCIAL_INSTRUCTIONS, augmentation_factor=2)

        assert len(augmented) >= original_count
        print(f"Original: {original_count}, Augmented: {len(augmented)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
