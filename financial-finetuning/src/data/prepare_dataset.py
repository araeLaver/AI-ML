# Dataset Preparation Module
"""
금융 도메인 Fine-tuning을 위한 데이터셋 준비 모듈
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from datasets import Dataset, DatasetDict
from .financial_instructions import FINANCIAL_INSTRUCTIONS


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    prompt_template: Optional[str] = None,
    prompt_template_no_input: Optional[str] = None,
) -> str:
    """
    Instruction을 학습용 프롬프트 형식으로 변환

    Args:
        instruction: 지시사항
        input_text: 입력 텍스트 (선택)
        output: 출력 텍스트
        prompt_template: 입력이 있는 경우의 템플릿
        prompt_template_no_input: 입력이 없는 경우의 템플릿

    Returns:
        포맷된 프롬프트 문자열
    """
    # 기본 템플릿
    if prompt_template is None:
        prompt_template = """### 지시사항:
{instruction}

### 입력:
{input}

### 응답:
{output}"""

    if prompt_template_no_input is None:
        prompt_template_no_input = """### 지시사항:
{instruction}

### 응답:
{output}"""

    if input_text and input_text.strip():
        return prompt_template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return prompt_template_no_input.format(
            instruction=instruction,
            output=output
        )


def create_financial_dataset(
    instructions: Optional[List[Dict[str, Any]]] = None,
    test_size: float = 0.1,
    seed: int = 42,
    prompt_template: Optional[str] = None,
    prompt_template_no_input: Optional[str] = None,
) -> DatasetDict:
    """
    금융 도메인 학습 데이터셋 생성

    Args:
        instructions: 커스텀 instruction 리스트 (없으면 기본값 사용)
        test_size: 테스트 셋 비율
        seed: 랜덤 시드
        prompt_template: 프롬프트 템플릿
        prompt_template_no_input: 입력 없는 프롬프트 템플릿

    Returns:
        DatasetDict with train and test splits
    """
    if instructions is None:
        instructions = FINANCIAL_INSTRUCTIONS

    # 데이터 포맷팅
    formatted_data = []
    for item in instructions:
        formatted_text = format_instruction(
            instruction=item["instruction"],
            input_text=item.get("input", ""),
            output=item["output"],
            prompt_template=prompt_template,
            prompt_template_no_input=prompt_template_no_input,
        )

        formatted_data.append({
            "text": formatted_text,
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"],
            "category": item.get("category", "general"),
        })

    # Dataset 생성
    dataset = Dataset.from_list(formatted_data)

    # Train/Test 분할
    dataset_dict = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    return dataset_dict


class FinancialInstructionDataset:
    """
    금융 도메인 Instruction 데이터셋 클래스

    Usage:
        dataset = FinancialInstructionDataset()
        train_data = dataset.get_train_dataset()
        eval_data = dataset.get_eval_dataset()
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        test_size: float = 0.1,
        seed: int = 42,
        prompt_template: Optional[str] = None,
        prompt_template_no_input: Optional[str] = None,
    ):
        """
        Args:
            data_path: 커스텀 데이터 JSON 파일 경로
            test_size: 테스트 셋 비율
            seed: 랜덤 시드
            prompt_template: 프롬프트 템플릿
            prompt_template_no_input: 입력 없는 프롬프트 템플릿
        """
        self.test_size = test_size
        self.seed = seed
        self.prompt_template = prompt_template
        self.prompt_template_no_input = prompt_template_no_input

        # 데이터 로드
        if data_path and Path(data_path).exists():
            self.instructions = self._load_from_file(data_path)
        else:
            self.instructions = FINANCIAL_INSTRUCTIONS

        # 데이터셋 생성
        self._dataset = None

    def _load_from_file(self, path: str) -> List[Dict[str, Any]]:
        """JSON 파일에서 데이터 로드"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def dataset(self) -> DatasetDict:
        """Lazy loading of dataset"""
        if self._dataset is None:
            self._dataset = create_financial_dataset(
                instructions=self.instructions,
                test_size=self.test_size,
                seed=self.seed,
                prompt_template=self.prompt_template,
                prompt_template_no_input=self.prompt_template_no_input,
            )
        return self._dataset

    def get_train_dataset(self) -> Dataset:
        """학습 데이터셋 반환"""
        return self.dataset["train"]

    def get_eval_dataset(self) -> Dataset:
        """평가 데이터셋 반환"""
        return self.dataset["test"]

    def save_to_disk(self, output_dir: str):
        """데이터셋을 디스크에 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON 형식으로 저장
        train_data = [item for item in self.get_train_dataset()]
        eval_data = [item for item in self.get_eval_dataset()]

        with open(output_path / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(output_path / "eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(train_data)} training samples to {output_path / 'train.json'}")
        print(f"Saved {len(eval_data)} evaluation samples to {output_path / 'eval.json'}")

    def get_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        train_ds = self.get_train_dataset()
        eval_ds = self.get_eval_dataset()

        # 카테고리별 통계
        category_counts = {}
        for item in self.instructions:
            cat = item.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # 텍스트 길이 통계
        train_lengths = [len(item["text"]) for item in train_ds]

        return {
            "total_samples": len(self.instructions),
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "categories": category_counts,
            "avg_text_length": sum(train_lengths) / len(train_lengths) if train_lengths else 0,
            "max_text_length": max(train_lengths) if train_lengths else 0,
            "min_text_length": min(train_lengths) if train_lengths else 0,
        }


def augment_dataset(
    instructions: List[Dict[str, Any]],
    augmentation_factor: int = 2,
) -> List[Dict[str, Any]]:
    """
    데이터 증강 (간단한 패러프레이징)

    실제 프로덕션에서는 LLM을 활용한 고급 증강 사용 권장
    """
    augmented = list(instructions)  # 원본 복사

    # 간단한 동의어 대체 규칙
    replacements = [
        ("분석해주세요", "분석해 주십시오"),
        ("설명해주세요", "설명해 주십시오"),
        ("알려주세요", "알려 주십시오"),
        ("계산해주세요", "계산해 주십시오"),
        ("평가해주세요", "평가해 주십시오"),
    ]

    for _ in range(augmentation_factor - 1):
        for item in instructions:
            new_instruction = item["instruction"]
            for old, new in replacements:
                new_instruction = new_instruction.replace(old, new)

            if new_instruction != item["instruction"]:
                augmented.append({
                    **item,
                    "instruction": new_instruction,
                })

    return augmented


if __name__ == "__main__":
    # 테스트 실행
    print("=" * 50)
    print("Financial Instruction Dataset")
    print("=" * 50)

    # 데이터셋 생성
    dataset = FinancialInstructionDataset()

    # 통계 출력
    stats = dataset.get_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Train samples: {stats['train_samples']}")
    print(f"  Eval samples: {stats['eval_samples']}")
    print(f"\n📁 Categories:")
    for cat, count in stats['categories'].items():
        print(f"  - {cat}: {count}")
    print(f"\n📏 Text Length:")
    print(f"  Average: {stats['avg_text_length']:.0f} chars")
    print(f"  Min: {stats['min_text_length']} chars")
    print(f"  Max: {stats['max_text_length']} chars")

    # 샘플 출력
    print(f"\n📝 Sample formatted text:")
    print("-" * 50)
    train_ds = dataset.get_train_dataset()
    if len(train_ds) > 0:
        print(train_ds[0]["text"][:500] + "...")
