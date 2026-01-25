# -*- coding: utf-8 -*-
"""
LLM-based Dataset Augmentation for Financial Instructions

This module provides automated dataset expansion through:
1. Template-based augmentation (variations of existing samples)
2. LLM-based paraphrasing and generation
3. Cross-category combination
4. Numerical variation for financial examples
"""

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class AugmentationConfig:
    """Configuration for dataset augmentation."""
    target_samples: int = 1000
    use_llm: bool = False
    llm_provider: str = "openai"  # "openai" or "anthropic"
    temperature: float = 0.7
    max_tokens: int = 1024
    numerical_variation_range: float = 0.3  # +/- 30% for numerical values
    enable_paraphrasing: bool = True
    enable_cross_category: bool = True
    enable_numerical_variation: bool = True
    seed: int = 42


@dataclass
class AugmentedSample:
    """An augmented sample with provenance tracking."""
    instruction: str
    output: str
    input: str = ""
    category: str = ""
    augmentation_type: str = ""
    source_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAugmentor:
    """LLM-based dataset augmentation for expanding financial instruction datasets.

    This class provides multiple augmentation strategies:
    1. Template variations - Different phrasings of the same question
    2. Numerical variations - Changing amounts, percentages, dates
    3. Cross-category combinations - Combining elements from different categories
    4. LLM paraphrasing - Using LLM to generate natural variations

    Attributes:
        config: Augmentation configuration
        samples: Original samples to augment
    """

    # Korean financial instruction templates for variation
    INSTRUCTION_TEMPLATES = {
        "fraud_detection": [
            "다음 거래가 이상거래인지 분석해주세요.",
            "이 거래 패턴이 사기일 가능성이 있는지 평가해주세요.",
            "거래 내역을 검토하고 의심스러운 점이 있는지 알려주세요.",
            "이상거래 여부를 판단하고 그 근거를 설명해주세요.",
            "다음 거래의 사기 위험도를 평가해주세요.",
            "거래 패턴을 분석하여 정상/비정상 여부를 판단해주세요.",
        ],
        "investment_analysis": [
            "다음 종목에 대한 투자 분석을 해주세요.",
            "이 주식의 투자 가치를 평가해주세요.",
            "투자 관점에서 이 종목을 분석해주세요.",
            "이 종목의 매수/매도 의견을 알려주세요.",
            "투자 포트폴리오에 포함할 만한지 분석해주세요.",
        ],
        "product_explanation": [
            "다음 금융 상품에 대해 설명해주세요.",
            "이 금융 상품의 장단점을 알려주세요.",
            "이 상품이 어떤 투자자에게 적합한지 설명해주세요.",
            "금융 상품의 특징과 위험요소를 분석해주세요.",
        ],
        "risk_assessment": [
            "다음 포트폴리오의 리스크를 평가해주세요.",
            "이 투자의 위험도를 분석해주세요.",
            "리스크 관점에서 이 자산 배분을 평가해주세요.",
            "투자 위험 요소를 분석하고 개선안을 제시해주세요.",
        ],
        "market_analysis": [
            "현재 시장 상황을 분석해주세요.",
            "시장 전망에 대해 의견을 주세요.",
            "경제 지표를 해석하고 투자 시사점을 알려주세요.",
            "시장 트렌드를 분석하고 투자 전략을 제안해주세요.",
        ],
        "term_explanation": [
            "다음 금융 용어를 설명해주세요.",
            "이 금융 개념에 대해 쉽게 설명해주세요.",
            "초보 투자자도 이해할 수 있게 용어를 설명해주세요.",
            "이 금융 용어의 의미와 실제 활용 사례를 알려주세요.",
        ],
    }

    # Numerical patterns for variation
    KOREAN_NUMBER_PATTERNS = [
        (r"(\d+)만원", "만원"),
        (r"(\d+)억원", "억원"),
        (r"(\d+)%", "%"),
        (r"(\d+)시", "시"),
        (r"(\d+)회", "회"),
        (r"(\d+)일", "일"),
        (r"(\d+)개월", "개월"),
        (r"(\d+)년", "년"),
    ]

    def __init__(
        self,
        samples: list[dict[str, Any]],
        config: AugmentationConfig | None = None,
    ):
        """Initialize the augmentor.

        Args:
            samples: Original samples to augment
            config: Augmentation configuration
        """
        self.samples = samples
        self.config = config or AugmentationConfig()

        random.seed(self.config.seed)

        self._llm_client = None
        if self.config.use_llm:
            self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client based on configuration."""
        if self.config.llm_provider == "openai" and OPENAI_AVAILABLE:
            self._llm_client = OpenAI()
        elif self.config.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self._llm_client = anthropic.Anthropic()

    def augment(self) -> list[AugmentedSample]:
        """Perform dataset augmentation.

        Returns:
            List of augmented samples
        """
        augmented: list[AugmentedSample] = []

        # Add original samples
        for i, sample in enumerate(self.samples):
            augmented.append(AugmentedSample(
                instruction=sample.get("instruction", ""),
                output=sample.get("output", ""),
                input=sample.get("input", ""),
                category=sample.get("category", ""),
                augmentation_type="original",
                source_index=i,
            ))

        # Calculate how many more samples we need
        samples_needed = self.config.target_samples - len(augmented)

        if samples_needed <= 0:
            return augmented

        # Distribute augmentation types
        per_type = samples_needed // 3

        # 1. Template variations
        if self.config.enable_paraphrasing:
            template_samples = self._augment_with_templates(per_type)
            augmented.extend(template_samples)

        # 2. Numerical variations
        if self.config.enable_numerical_variation:
            numerical_samples = self._augment_with_numerical_variation(per_type)
            augmented.extend(numerical_samples)

        # 3. LLM-based paraphrasing or synthetic generation
        remaining = self.config.target_samples - len(augmented)
        if remaining > 0:
            if self.config.use_llm and self._llm_client:
                llm_samples = self._augment_with_llm(remaining)
                augmented.extend(llm_samples)
            else:
                # Fallback to template-based augmentation
                fallback_samples = self._augment_with_templates(remaining)
                augmented.extend(fallback_samples)

        return augmented[:self.config.target_samples]

    def _augment_with_templates(
        self,
        count: int,
    ) -> list[AugmentedSample]:
        """Augment using instruction template variations.

        Args:
            count: Number of samples to generate

        Returns:
            List of augmented samples
        """
        augmented = []

        for _ in range(count):
            # Pick a random original sample
            idx = random.randint(0, len(self.samples) - 1)
            sample = self.samples[idx]

            category = sample.get("category", "")

            # Get templates for this category
            templates = self.INSTRUCTION_TEMPLATES.get(
                category,
                list(self.INSTRUCTION_TEMPLATES.values())[0],
            )

            # Pick a different instruction template
            original_instruction = sample.get("instruction", "")
            new_instruction = random.choice(templates)

            # Avoid identical instructions
            attempts = 0
            while new_instruction == original_instruction and attempts < 5:
                new_instruction = random.choice(templates)
                attempts += 1

            augmented.append(AugmentedSample(
                instruction=new_instruction,
                output=sample.get("output", ""),
                input=sample.get("input", ""),
                category=category,
                augmentation_type="template_variation",
                source_index=idx,
            ))

        return augmented

    def _augment_with_numerical_variation(
        self,
        count: int,
    ) -> list[AugmentedSample]:
        """Augment by varying numerical values.

        Args:
            count: Number of samples to generate

        Returns:
            List of augmented samples
        """
        augmented = []
        variation_range = self.config.numerical_variation_range

        for _ in range(count):
            idx = random.randint(0, len(self.samples) - 1)
            sample = self.samples[idx]

            input_text = sample.get("input", "")
            output_text = sample.get("output", "")

            # Vary numbers in input
            new_input = self._vary_numbers(input_text, variation_range)

            augmented.append(AugmentedSample(
                instruction=sample.get("instruction", ""),
                output=output_text,
                input=new_input,
                category=sample.get("category", ""),
                augmentation_type="numerical_variation",
                source_index=idx,
                metadata={"variation_range": variation_range},
            ))

        return augmented

    def _vary_numbers(self, text: str, variation_range: float) -> str:
        """Vary numerical values in text.

        Args:
            text: Text containing numbers
            variation_range: Range for variation (e.g., 0.3 for +/- 30%)

        Returns:
            Text with varied numbers
        """
        result = text

        for pattern, suffix in self.KOREAN_NUMBER_PATTERNS:
            def replacer(match: re.Match) -> str:
                original = int(match.group(1))
                variation = random.uniform(1 - variation_range, 1 + variation_range)
                new_value = int(original * variation)

                # Keep reasonable bounds
                new_value = max(1, new_value)

                return f"{new_value}{suffix}"

            result = re.sub(pattern, replacer, result)

        return result

    def _augment_with_llm(
        self,
        count: int,
    ) -> list[AugmentedSample]:
        """Augment using LLM-based generation.

        Args:
            count: Number of samples to generate

        Returns:
            List of augmented samples
        """
        augmented = []

        for _ in range(count):
            idx = random.randint(0, len(self.samples) - 1)
            sample = self.samples[idx]

            try:
                paraphrased = self._paraphrase_with_llm(sample)
                if paraphrased:
                    augmented.append(AugmentedSample(
                        instruction=paraphrased.get("instruction", sample["instruction"]),
                        output=paraphrased.get("output", sample["output"]),
                        input=paraphrased.get("input", sample.get("input", "")),
                        category=sample.get("category", ""),
                        augmentation_type="llm_paraphrase",
                        source_index=idx,
                        metadata={"llm_provider": self.config.llm_provider},
                    ))
            except Exception:
                # Fallback to template variation
                templates = list(self.INSTRUCTION_TEMPLATES.values())[0]
                augmented.append(AugmentedSample(
                    instruction=random.choice(templates),
                    output=sample.get("output", ""),
                    input=sample.get("input", ""),
                    category=sample.get("category", ""),
                    augmentation_type="template_fallback",
                    source_index=idx,
                ))

        return augmented

    def _paraphrase_with_llm(
        self,
        sample: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Paraphrase a sample using LLM.

        Args:
            sample: Original sample to paraphrase

        Returns:
            Paraphrased sample or None if failed
        """
        if not self._llm_client:
            return None

        prompt = f"""다음 금융 도메인 instruction-output 쌍을 자연스럽게 paraphrase해주세요.
의미는 동일하게 유지하되, 표현을 다르게 해주세요.

원본 Instruction: {sample.get('instruction', '')}
원본 Input: {sample.get('input', '')}
원본 Output: {sample.get('output', '')}

JSON 형식으로 응답해주세요:
{{"instruction": "paraphrased instruction", "input": "paraphrased input", "output": "paraphrased output"}}"""

        try:
            if self.config.llm_provider == "openai":
                response = self._llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response.choices[0].message.content
            elif self.config.llm_provider == "anthropic":
                response = self._llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
            else:
                return None

            # Parse JSON response
            if content:
                # Extract JSON from response
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception:
            pass

        return None

    def to_json(
        self,
        samples: list[AugmentedSample],
        output_path: str | Path,
    ) -> None:
        """Export augmented samples to JSON file.

        Args:
            samples: Augmented samples
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for sample in samples:
            data.append({
                "instruction": sample.instruction,
                "output": sample.output,
                "input": sample.input,
                "category": sample.category,
                "augmentation_type": sample.augmentation_type,
                "source_index": sample.source_index,
                "metadata": sample.metadata,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_statistics(
        self,
        samples: list[AugmentedSample],
    ) -> dict[str, Any]:
        """Get augmentation statistics.

        Args:
            samples: Augmented samples

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_samples": len(samples),
            "by_augmentation_type": {},
            "by_category": {},
            "generated_at": datetime.now().isoformat(),
        }

        for sample in samples:
            # Count by augmentation type
            aug_type = sample.augmentation_type
            stats["by_augmentation_type"][aug_type] = (
                stats["by_augmentation_type"].get(aug_type, 0) + 1
            )

            # Count by category
            category = sample.category or "unknown"
            stats["by_category"][category] = (
                stats["by_category"].get(category, 0) + 1
            )

        return stats


class SyntheticDataGenerator:
    """Generate synthetic financial instruction samples.

    This class generates entirely new samples rather than
    augmenting existing ones.
    """

    # Financial scenario templates for synthetic generation
    SCENARIO_TEMPLATES = {
        "fraud_detection": [
            {
                "input_template": "거래금액: {amount}만원, 거래시간: {time}, 거래위치: {location}, 최근 30일 평균 거래금액: {avg_amount}만원",
                "risk_factors": ["금액 이상", "시간 이상", "위치 이상", "빈도 이상"],
            },
        ],
        "investment_analysis": [
            {
                "input_template": "종목: {company}, 현재가: {price}원, PER: {per}, PBR: {pbr}, ROE: {roe}%, 배당수익률: {dividend}%",
            },
        ],
        "risk_assessment": [
            {
                "input_template": "포트폴리오: 주식 {stock}%, 채권 {bond}%, 대안투자 {alt}%, 현금 {cash}%",
            },
        ],
    }

    # Sample values for synthetic generation
    SAMPLE_VALUES = {
        "companies": ["삼성전자", "SK하이닉스", "NAVER", "카카오", "현대차", "LG에너지솔루션"],
        "locations": ["서울", "부산", "대구", "인천", "광주", "베트남 호치민", "필리핀 마닐라"],
        "times": ["새벽 2시", "새벽 3시", "오전 9시", "오후 2시", "오후 7시", "밤 11시"],
        "merchant_categories": ["대형마트", "편의점", "주유소", "해외직구", "게임아이템"],
    }

    def __init__(self, seed: int = 42):
        """Initialize generator.

        Args:
            seed: Random seed
        """
        self._seed = seed
        self._rng = random.Random(seed)

    def generate(
        self,
        category: str,
        count: int,
    ) -> list[dict[str, Any]]:
        """Generate synthetic samples for a category.

        Args:
            category: Category to generate
            count: Number of samples to generate

        Returns:
            List of generated samples
        """
        samples = []

        for _ in range(count):
            if category == "fraud_detection":
                sample = self._generate_fraud_sample()
            elif category == "investment_analysis":
                sample = self._generate_investment_sample()
            elif category == "risk_assessment":
                sample = self._generate_risk_sample()
            else:
                sample = self._generate_generic_sample(category)

            sample["category"] = category
            samples.append(sample)

        return samples

    def _generate_fraud_sample(self) -> dict[str, Any]:
        """Generate a synthetic fraud detection sample."""
        is_fraud = self._rng.random() < 0.4  # 40% fraud cases

        if is_fraud:
            amount = self._rng.randint(500, 10000)
            time = self._rng.choice(["새벽 2시", "새벽 3시", "밤 11시"])
            location = self._rng.choice(["베트남 호치민", "필리핀 마닐라", "태국 방콕"])
            avg_amount = self._rng.randint(30, 100)
        else:
            amount = self._rng.randint(10, 300)
            time = self._rng.choice(["오전 10시", "오후 2시", "오후 7시"])
            location = self._rng.choice(["서울", "부산", "인천"])
            avg_amount = self._rng.randint(50, 200)

        input_text = f"거래금액: {amount}만원, 거래시간: {time}, 거래위치: {location}, 최근 30일 평균 거래금액: {avg_amount}만원"

        if is_fraud:
            output = f"""이 거래는 **이상거래로 의심**됩니다.

**이상 징후:**
1. 거래금액({amount}만원)이 평균({avg_amount}만원)의 {amount//avg_amount}배입니다.
2. {time}은 비정상적인 거래 시간대입니다.
3. {location}에서의 해외 거래가 발생했습니다.

**리스크 점수:** {self._rng.randint(80, 99)}/100 (높음)

**권고 조치:** 거래 보류 및 본인 확인 필요"""
        else:
            output = f"""이 거래는 **정상 거래**로 판단됩니다.

**정상 판단 근거:**
1. 거래금액({amount}만원)이 평균 범위 내입니다.
2. {time}은 일반적인 거래 시간대입니다.
3. 국내({location}) 거래로 이상 없습니다.

**리스크 점수:** {self._rng.randint(10, 30)}/100 (낮음)"""

        return {
            "instruction": self._rng.choice([
                "다음 거래가 이상거래인지 분석해주세요.",
                "이 거래의 사기 가능성을 평가해주세요.",
            ]),
            "input": input_text,
            "output": output,
        }

    def _generate_investment_sample(self) -> dict[str, Any]:
        """Generate a synthetic investment analysis sample."""
        company = self._rng.choice(self.SAMPLE_VALUES["companies"])
        price = self._rng.randint(10000, 500000)
        per = round(self._rng.uniform(5, 50), 1)
        pbr = round(self._rng.uniform(0.5, 5), 2)
        roe = round(self._rng.uniform(5, 30), 1)
        dividend = round(self._rng.uniform(0, 5), 2)

        input_text = f"종목: {company}, 현재가: {price:,}원, PER: {per}, PBR: {pbr}, ROE: {roe}%, 배당수익률: {dividend}%"

        # Determine investment rating
        if per < 15 and roe > 15:
            rating = "매수"
            reason = "저평가 + 높은 수익성"
        elif per > 30 or roe < 10:
            rating = "관망"
            reason = "고평가 또는 낮은 수익성"
        else:
            rating = "중립"
            reason = "적정 가치 수준"

        output = f"""**{company} 투자 분석**

**밸류에이션:**
- PER {per}배: {'저평가' if per < 15 else '적정' if per < 25 else '고평가'}
- PBR {pbr}배: {'저평가' if pbr < 1 else '적정' if pbr < 2 else '고평가'}
- ROE {roe}%: {'우수' if roe > 15 else '보통' if roe > 10 else '개선 필요'}

**배당:**
- 배당수익률 {dividend}%: {'매력적' if dividend > 3 else '보통' if dividend > 1 else '낮음'}

**투자의견:** {rating} ({reason})

**목표가:** {int(price * (1.2 if rating == '매수' else 1.0 if rating == '중립' else 0.9)):,}원"""

        return {
            "instruction": self._rng.choice([
                "다음 종목에 대한 투자 분석을 해주세요.",
                "이 주식의 투자 가치를 평가해주세요.",
            ]),
            "input": input_text,
            "output": output,
        }

    def _generate_risk_sample(self) -> dict[str, Any]:
        """Generate a synthetic risk assessment sample."""
        stock = self._rng.randint(30, 70)
        bond = self._rng.randint(10, min(50, 100 - stock - 5))  # Ensure space for alt and cash
        remaining = 100 - stock - bond
        alt = self._rng.randint(0, max(0, remaining - 1))  # Ensure at least 1% for cash
        cash = remaining - alt

        input_text = f"포트폴리오: 주식 {stock}%, 채권 {bond}%, 대안투자 {alt}%, 현금 {cash}%"

        # Assess risk level
        if stock > 60:
            risk_level = "높음"
            risk_score = self._rng.randint(70, 90)
        elif stock > 40:
            risk_level = "중간"
            risk_score = self._rng.randint(40, 60)
        else:
            risk_level = "낮음"
            risk_score = self._rng.randint(20, 40)

        output = f"""**포트폴리오 리스크 평가**

**자산배분 분석:**
- 주식 {stock}%: {'공격적' if stock > 60 else '적정' if stock > 30 else '보수적'}
- 채권 {bond}%: 안정자산 역할
- 대안투자 {alt}%: 분산효과 기여
- 현금 {cash}%: 유동성 확보

**리스크 지표:**
- 리스크 레벨: {risk_level}
- 리스크 점수: {risk_score}/100
- 예상 변동성: {stock * 0.3:.1f}% (연율)

**권장사항:**
{'주식 비중 축소 검토' if stock > 70 else '현재 배분 유지' if 40 <= stock <= 60 else '성장성 제고를 위한 주식 비중 검토'}"""

        return {
            "instruction": self._rng.choice([
                "다음 포트폴리오의 리스크를 평가해주세요.",
                "이 자산배분의 위험도를 분석해주세요.",
            ]),
            "input": input_text,
            "output": output,
        }

    def _generate_generic_sample(self, category: str) -> dict[str, Any]:
        """Generate a generic sample for undefined categories."""
        return {
            "instruction": f"{category} 관련 분석을 해주세요.",
            "input": "",
            "output": f"{category}에 대한 분석 결과입니다.",
        }


def expand_dataset(
    original_samples: list[dict[str, Any]],
    target_size: int = 1000,
    use_llm: bool = False,
    output_path: str | Path | None = None,
) -> list[AugmentedSample]:
    """Convenience function to expand a dataset.

    Args:
        original_samples: Original dataset samples
        target_size: Target dataset size
        use_llm: Whether to use LLM for augmentation
        output_path: Optional path to save augmented dataset

    Returns:
        List of augmented samples
    """
    config = AugmentationConfig(
        target_samples=target_size,
        use_llm=use_llm,
    )

    augmentor = DatasetAugmentor(original_samples, config)
    augmented = augmentor.augment()

    if output_path:
        augmentor.to_json(augmented, output_path)

        # Also save statistics
        stats = augmentor.get_statistics(augmented)
        stats_path = Path(output_path).parent / "augmentation_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return augmented
