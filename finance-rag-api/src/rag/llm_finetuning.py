# -*- coding: utf-8 -*-
"""
LLM Fine-tuning 모듈

금융 도메인 QA에 특화된 LLM 학습을 지원합니다.
"""

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FineTuneMethod(Enum):
    """Fine-tuning 방법"""
    FULL = "full"  # 전체 파라미터 학습
    LORA = "lora"  # LoRA (Low-Rank Adaptation)
    QLORA = "qlora"  # QLoRA (Quantized LoRA)
    PREFIX = "prefix"  # Prefix Tuning


@dataclass
class FineTuneConfig:
    """Fine-tuning 설정

    Attributes:
        base_model: 기본 모델
        method: Fine-tuning 방법
        output_dir: 출력 디렉토리
        epochs: 학습 에폭
        batch_size: 배치 크기
        learning_rate: 학습률
        max_length: 최대 시퀀스 길이
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    base_model: str = "beomi/llama-2-ko-7b"
    method: FineTuneMethod = FineTuneMethod.LORA
    output_dir: str = "./models/finance-llm"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001

    # LoRA 설정
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 추천 모델 목록
    RECOMMENDED_MODELS = {
        "llama-2-ko-7b": {
            "name": "beomi/llama-2-ko-7b",
            "description": "한국어 LLaMA 2 7B",
            "size": "7B",
        },
        "polyglot-ko-5.8b": {
            "name": "EleutherAI/polyglot-ko-5.8b",
            "description": "한국어 Polyglot 5.8B",
            "size": "5.8B",
        },
        "kullm-polyglot-12.8b": {
            "name": "nlpai-lab/kullm-polyglot-12.8b-v2",
            "description": "KULLM Polyglot 12.8B",
            "size": "12.8B",
        },
        "mistral-7b": {
            "name": "mistralai/Mistral-7B-v0.1",
            "description": "Mistral 7B (다국어)",
            "size": "7B",
        },
    }


@dataclass
class QAExample:
    """QA 학습 예제

    Attributes:
        instruction: 지시문
        input: 입력 (컨텍스트)
        output: 출력 (답변)
        source: 출처
    """
    instruction: str
    input: str
    output: str
    source: str = ""

    def to_prompt(self, template: str = "alpaca") -> str:
        """프롬프트 형식으로 변환"""
        if template == "alpaca":
            return self._to_alpaca_prompt()
        elif template == "chatml":
            return self._to_chatml_prompt()
        else:
            return self._to_simple_prompt()

    def _to_alpaca_prompt(self) -> str:
        """Alpaca 프롬프트 형식"""
        if self.input:
            return f"""### Instruction:
{self.instruction}

### Input:
{self.input}

### Response:
{self.output}"""
        else:
            return f"""### Instruction:
{self.instruction}

### Response:
{self.output}"""

    def _to_chatml_prompt(self) -> str:
        """ChatML 프롬프트 형식"""
        return f"""<|im_start|>system
당신은 한국 금융 전문가입니다. 정확하고 신뢰할 수 있는 정보를 제공하세요.
<|im_end|>
<|im_start|>user
{self.instruction}

{self.input}
<|im_end|>
<|im_start|>assistant
{self.output}
<|im_end|>"""

    def _to_simple_prompt(self) -> str:
        """단순 프롬프트 형식"""
        return f"질문: {self.instruction}\n컨텍스트: {self.input}\n답변: {self.output}"


@dataclass
class FinanceQADataset:
    """금융 QA 데이터셋

    Attributes:
        examples: QA 예제 목록
        name: 데이터셋 이름
    """
    examples: list[QAExample] = field(default_factory=list)
    name: str = "finance-qa"

    def add_example(
        self,
        instruction: str,
        input: str,
        output: str,
        source: str = "",
    ) -> None:
        """예제 추가"""
        self.examples.append(QAExample(
            instruction=instruction,
            input=input,
            output=output,
            source=source,
        ))

    def __len__(self) -> int:
        return len(self.examples)

    def to_json(self, path: str) -> None:
        """JSON 파일로 저장"""
        data = [
            {
                "instruction": ex.instruction,
                "input": ex.input,
                "output": ex.output,
                "source": ex.source,
            }
            for ex in self.examples
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FinanceQADataset":
        """JSON 파일에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls()
        for item in data:
            dataset.add_example(
                instruction=item["instruction"],
                input=item.get("input", ""),
                output=item["output"],
                source=item.get("source", ""),
            )
        return dataset

    def split(
        self,
        train_ratio: float = 0.9,
    ) -> tuple["FinanceQADataset", "FinanceQADataset"]:
        """학습/검증 데이터셋 분할"""
        random.shuffle(self.examples)
        split_idx = int(len(self.examples) * train_ratio)

        train_dataset = FinanceQADataset(
            examples=self.examples[:split_idx],
            name=f"{self.name}-train",
        )
        val_dataset = FinanceQADataset(
            examples=self.examples[split_idx:],
            name=f"{self.name}-val",
        )

        return train_dataset, val_dataset


class FinanceQAGenerator:
    """금융 QA 데이터 생성기"""

    # 샘플 QA 데이터
    SAMPLE_QA = [
        # 실적 관련
        {
            "instruction": "삼성전자의 2024년 1분기 영업이익은 얼마인가요?",
            "input": "삼성전자는 2024년 1분기 매출 71조 9,200억원, 영업이익 6조 6,000억원을 기록했다. 전년 동기 대비 매출은 11% 증가, 영업이익은 흑자전환했다.",
            "output": "삼성전자의 2024년 1분기 영업이익은 6조 6,000억원입니다. 전년 동기 대비 흑자전환에 성공했습니다.",
        },
        {
            "instruction": "SK하이닉스의 HBM 매출 비중은?",
            "input": "SK하이닉스는 AI 반도체 수요 급증으로 HBM(고대역폭메모리) 매출이 전체 DRAM 매출의 30%를 차지했다. HBM 매출은 전년 대비 3배 이상 증가했다.",
            "output": "SK하이닉스의 HBM 매출은 전체 DRAM 매출의 30%를 차지하고 있으며, 전년 대비 3배 이상 증가했습니다.",
        },
        # 주가/시장 관련
        {
            "instruction": "코스피 시가총액 상위 기업을 알려주세요.",
            "input": "코스피 시가총액 순위: 1위 삼성전자(400조원), 2위 SK하이닉스(150조원), 3위 LG에너지솔루션(100조원), 4위 삼성바이오로직스(80조원), 5위 현대차(60조원).",
            "output": "코스피 시가총액 상위 5개 기업은 1위 삼성전자(400조원), 2위 SK하이닉스(150조원), 3위 LG에너지솔루션(100조원), 4위 삼성바이오로직스(80조원), 5위 현대차(60조원)입니다.",
        },
        {
            "instruction": "PER이 낮은 주식이 좋은 건가요?",
            "input": "PER(주가수익비율)은 주가를 주당순이익으로 나눈 값이다. PER이 낮으면 저평가, 높으면 고평가로 해석할 수 있다. 단, 업종별 평균 PER이 다르므로 동일 업종 내 비교가 필요하다.",
            "output": "PER이 낮다고 무조건 좋은 것은 아닙니다. PER은 상대적 지표이므로 동일 업종 내 비교가 중요합니다. 낮은 PER은 저평가를 의미할 수 있지만, 성장성이 낮거나 문제가 있을 수도 있습니다.",
        },
        # 공시/규제 관련
        {
            "instruction": "분기보고서 제출 기한은 언제인가요?",
            "input": "금융위원회 규정에 따르면, 분기보고서는 해당 분기 종료 후 45일 이내에 제출해야 한다. 예를 들어, 1분기(1-3월) 보고서는 5월 15일까지 제출해야 한다.",
            "output": "분기보고서는 해당 분기 종료 후 45일 이내에 제출해야 합니다. 1분기는 5월 15일, 2분기는 8월 14일, 3분기는 11월 14일이 제출 기한입니다.",
        },
        {
            "instruction": "유상증자가 주가에 미치는 영향은?",
            "input": "유상증자는 신주를 발행하여 자금을 조달하는 방법이다. 주식 수가 증가하므로 기존 주주 지분이 희석된다. 단, 자금 사용 목적에 따라 장기적 영향은 다를 수 있다.",
            "output": "유상증자는 단기적으로 주가에 부정적 영향을 줄 수 있습니다. 신주 발행으로 주식 수가 늘어나 기존 주주의 지분이 희석되기 때문입니다. 다만, 조달 자금의 사용 목적(시설 투자, R&D 등)에 따라 장기 주가에는 긍정적일 수 있습니다.",
        },
        # 산업/기술 관련
        {
            "instruction": "HBM 메모리란 무엇인가요?",
            "input": "HBM(High Bandwidth Memory)은 고대역폭 메모리로, 여러 DRAM 칩을 수직으로 적층하여 대역폭을 크게 높인 메모리다. AI, 고성능컴퓨팅에 필수적이며, SK하이닉스와 삼성전자가 주요 생산업체다.",
            "output": "HBM(High Bandwidth Memory)은 여러 DRAM 칩을 수직으로 적층하여 데이터 처리 속도를 크게 높인 고대역폭 메모리입니다. AI 반도체와 고성능 컴퓨팅에 필수적이며, 현재 SK하이닉스와 삼성전자가 글로벌 시장을 주도하고 있습니다.",
        },
        {
            "instruction": "전기차 배터리 시장 전망은?",
            "input": "글로벌 전기차 배터리 시장은 2023년 500GWh에서 2030년 2,000GWh로 4배 성장 전망. 주요 업체는 CATL(중국), LG에너지솔루션, BYD(중국), 파나소닉, 삼성SDI 순.",
            "output": "전기차 배터리 시장은 2030년까지 연평균 20% 이상 성장하여 2,000GWh 규모가 될 전망입니다. 현재 CATL(중국)이 1위이며, LG에너지솔루션이 2위입니다. 한국 기업들은 기술력과 안전성에서 경쟁력을 갖추고 있습니다.",
        },
        # 재무 분석 관련
        {
            "instruction": "ROE와 ROA의 차이점은?",
            "input": "ROE(자기자본이익률)는 순이익을 자기자본으로 나눈 값이고, ROA(총자산이익률)는 순이익을 총자산으로 나눈 값이다. ROE는 주주 수익성, ROA는 자산 효율성을 측정한다.",
            "output": "ROE(자기자본이익률)는 주주가 투자한 자본 대비 수익성을 나타내고, ROA(총자산이익률)는 기업이 보유한 총자산 대비 수익성을 나타냅니다. ROE가 높으면 주주 입장에서 효율적, ROA가 높으면 자산 활용이 효율적입니다. 부채가 많으면 ROE가 높아질 수 있어 함께 분석해야 합니다.",
        },
        {
            "instruction": "부채비율이 높으면 위험한가요?",
            "input": "부채비율은 부채를 자기자본으로 나눈 값이다. 업종별 적정 수준이 다르며, 제조업은 100~200%, 금융업은 300~500%가 일반적이다. 안정성과 성장성의 균형이 중요하다.",
            "output": "부채비율이 높다고 무조건 위험한 것은 아닙니다. 업종별로 적정 수준이 다르며, 제조업은 100~200%, 금융업은 300~500%가 일반적입니다. 중요한 것은 부채를 통해 창출하는 수익이 이자 비용보다 높은지, 그리고 현금 흐름이 안정적인지 여부입니다.",
        },
    ]

    def __init__(self):
        """초기화"""
        pass

    def generate_dataset(
        self,
        num_examples: int = 100,
    ) -> FinanceQADataset:
        """QA 데이터셋 생성

        Args:
            num_examples: 생성할 예제 수

        Returns:
            QA 데이터셋
        """
        dataset = FinanceQADataset(name="finance-qa-generated")

        for i in range(num_examples):
            sample = self.SAMPLE_QA[i % len(self.SAMPLE_QA)]
            dataset.add_example(
                instruction=sample["instruction"],
                input=sample["input"],
                output=sample["output"],
                source="generated",
            )

        return dataset

    def generate_from_documents(
        self,
        documents: list[dict[str, Any]],
        questions_per_doc: int = 3,
    ) -> FinanceQADataset:
        """문서에서 QA 데이터셋 생성

        Args:
            documents: 문서 목록
            questions_per_doc: 문서당 질문 수

        Returns:
            QA 데이터셋
        """
        dataset = FinanceQADataset(name="finance-qa-from-docs")

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            corp_name = metadata.get("corp_name", "회사")

            # 간단한 질문 생성 (실제로는 LLM 사용)
            questions = [
                f"{corp_name}의 주요 내용은 무엇인가요?",
                f"{corp_name} 관련 정보를 요약해주세요.",
                f"이 문서에서 {corp_name}에 대해 알 수 있는 것은?",
            ]

            for q in questions[:questions_per_doc]:
                dataset.add_example(
                    instruction=q,
                    input=content[:500],  # 길이 제한
                    output=f"{corp_name}에 대한 정보입니다. (자동 생성)",
                    source=metadata.get("source", ""),
                )

        return dataset


if TORCH_AVAILABLE:
    class FinanceQATorchDataset(Dataset):
        """PyTorch Dataset for Fine-tuning"""

        def __init__(
            self,
            dataset: FinanceQADataset,
            tokenizer: Any,
            max_length: int = 512,
            template: str = "alpaca",
        ):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.template = template

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            example = self.dataset.examples[idx]
            prompt = example.to_prompt(self.template)

            encoding = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze(),
            }


class FinanceLLMTrainer:
    """금융 LLM 학습기"""

    def __init__(
        self,
        config: Optional[FineTuneConfig] = None,
    ):
        """
        Args:
            config: Fine-tuning 설정
        """
        self.config = config or FineTuneConfig()
        self._model = None
        self._tokenizer = None

    def load_base_model(self) -> tuple[Any, Any]:
        """기본 모델 로드"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers와 peft가 필요합니다")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        # 패딩 토큰 설정
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if TORCH_AVAILABLE else None,
            device_map="auto",
            trust_remote_code=True,
        )

        return self._model, self._tokenizer

    def apply_lora(self) -> Any:
        """LoRA 적용"""
        if self._model is None:
            self.load_base_model()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )

        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()

        return self._model

    def train(
        self,
        train_dataset: FinanceQADataset,
        eval_dataset: Optional[FinanceQADataset] = None,
    ) -> str:
        """모델 학습

        Args:
            train_dataset: 학습 데이터셋
            eval_dataset: 평가 데이터셋

        Returns:
            저장된 모델 경로
        """
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError("transformers, peft, torch가 필요합니다")

        if self._model is None:
            if self.config.method == FineTuneMethod.LORA:
                self.apply_lora()
            else:
                self.load_base_model()

        # PyTorch Dataset 생성
        train_torch_dataset = FinanceQATorchDataset(
            train_dataset,
            self._tokenizer,
            self.config.max_length,
        )

        eval_torch_dataset = None
        if eval_dataset:
            eval_torch_dataset = FinanceQATorchDataset(
                eval_dataset,
                self._tokenizer,
                self.config.max_length,
            )

        # 학습 설정
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            fp16=True,
        )

        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_torch_dataset,
            eval_dataset=eval_torch_dataset,
            data_collator=data_collator,
        )

        # 학습 실행
        trainer.train()

        # 모델 저장
        trainer.save_model(self.config.output_dir)
        self._tokenizer.save_pretrained(self.config.output_dir)

        return self.config.output_dir

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도

        Returns:
            생성된 텍스트
        """
        if self._model is None or self._tokenizer is None:
            raise ValueError("모델이 로드되지 않았습니다")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        generated = self._tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return generated


class FinanceLLM:
    """금융 특화 LLM

    Fine-tuned 모델 또는 기본 모델을 사용하여 금융 QA를 수행합니다.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_api: bool = False,
        api_provider: str = "groq",
    ):
        """
        Args:
            model_path: Fine-tuned 모델 경로
            use_api: API 사용 여부
            api_provider: API 제공자 (groq, openai, anthropic)
        """
        self.model_path = model_path
        self.use_api = use_api
        self.api_provider = api_provider

        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """모델 로드"""
        if self.use_api:
            # API 사용 시 모델 로드 불필요
            return

        if self.model_path and TRANSFORMERS_AVAILABLE:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if TORCH_AVAILABLE else None,
                device_map="auto",
            )

    def answer(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 256,
    ) -> str:
        """질문에 답변

        Args:
            question: 질문
            context: 컨텍스트
            max_tokens: 최대 토큰 수

        Returns:
            답변
        """
        if self.use_api:
            return self._answer_with_api(question, context, max_tokens)
        else:
            return self._answer_with_local(question, context, max_tokens)

    def _answer_with_api(
        self,
        question: str,
        context: str,
        max_tokens: int,
    ) -> str:
        """API를 통한 답변"""
        # 실제 구현에서는 API 호출
        prompt = f"""당신은 한국 금융 전문가입니다.

컨텍스트:
{context}

질문: {question}

정확하고 간결하게 답변하세요."""

        # Placeholder - 실제로는 API 호출
        return f"[API 응답] {question}에 대한 답변입니다."

    def _answer_with_local(
        self,
        question: str,
        context: str,
        max_tokens: int,
    ) -> str:
        """로컬 모델을 통한 답변"""
        if self._model is None:
            self.load_model()

        if self._model is None:
            return f"[로컬 모델 미설정] {question}"

        prompt = f"""### Instruction:
{question}

### Input:
{context}

### Response:
"""

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
        )

        generated = self._tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Response 부분만 추출
        if "### Response:" in generated:
            return generated.split("### Response:")[-1].strip()

        return generated


# 편의 함수
def generate_qa_dataset(num_examples: int = 100) -> FinanceQADataset:
    """QA 데이터셋 생성

    Args:
        num_examples: 예제 수

    Returns:
        QA 데이터셋
    """
    generator = FinanceQAGenerator()
    return generator.generate_dataset(num_examples)


def create_finance_llm(
    model_path: Optional[str] = None,
    use_api: bool = True,
) -> FinanceLLM:
    """금융 LLM 생성

    Args:
        model_path: 모델 경로
        use_api: API 사용 여부

    Returns:
        금융 LLM
    """
    return FinanceLLM(model_path=model_path, use_api=use_api)
