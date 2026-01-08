# Inference Engine
"""
Fine-tuned 금융 LLM 추론 엔진
"""

from typing import Optional, Dict, Any, Generator
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel
from threading import Thread


class FinancialLLMInference:
    """
    Fine-tuned 금융 LLM 추론 클래스

    Usage:
        inference = FinancialLLMInference(
            base_model="beomi/Llama-3-Open-Ko-8B",
            adapter_path="./outputs/lora_adapter"
        )
        response = inference.generate("삼성전자 주식을 분석해주세요.")
    """

    def __init__(
        self,
        base_model: str = "beomi/Llama-3-Open-Ko-8B",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
    ):
        """
        Args:
            base_model: 베이스 모델 이름 또는 경로
            adapter_path: LoRA 어댑터 경로 (None이면 베이스 모델만 로드)
            load_in_4bit: 4-bit 양자화 사용 여부
            device_map: 디바이스 매핑 전략
            torch_dtype: 텐서 데이터 타입
        """
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype)

        self.model = None
        self.tokenizer = None

        # 기본 생성 설정
        self.default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }

    def load(self):
        """모델 로드"""
        print(f"Loading base model: {self.base_model_name}")

        # 양자화 설정
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 베이스 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        # LoRA 어댑터 적용
        if self.adapter_path and Path(self.adapter_path).exists():
            print(f"Loading LoRA adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
            )

        self.model.eval()
        print("Model loaded successfully!")

        return self

    def _format_prompt(self, instruction: str, input_text: str = "") -> str:
        """프롬프트 포맷팅"""
        if input_text:
            return f"""### 지시사항:
{instruction}

### 입력:
{input_text}

### 응답:
"""
        else:
            return f"""### 지시사항:
{instruction}

### 응답:
"""

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        **generation_kwargs,
    ) -> str:
        """
        텍스트 생성

        Args:
            instruction: 지시사항
            input_text: 입력 텍스트 (선택)
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            생성된 응답 텍스트
        """
        if self.model is None:
            self.load()

        # 프롬프트 구성
        prompt = self._format_prompt(instruction, input_text)

        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # 생성 설정
        gen_config = {**self.default_generation_config, **generation_kwargs}

        # 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 디코딩 (입력 부분 제외)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def generate_stream(
        self,
        instruction: str,
        input_text: str = "",
        **generation_kwargs,
    ) -> Generator[str, None, None]:
        """
        스트리밍 텍스트 생성

        Args:
            instruction: 지시사항
            input_text: 입력 텍스트 (선택)
            **generation_kwargs: 생성 파라미터 오버라이드

        Yields:
            생성된 토큰들
        """
        if self.model is None:
            self.load()

        # 프롬프트 구성
        prompt = self._format_prompt(instruction, input_text)

        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # 스트리머 설정
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # 생성 설정
        gen_config = {**self.default_generation_config, **generation_kwargs}
        gen_config["streamer"] = streamer

        # 백그라운드 스레드에서 생성
        generation_kwargs = {
            **inputs,
            **gen_config,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 토큰 스트리밍
        for token in streamer:
            yield token

        thread.join()

    def batch_generate(
        self,
        instructions: list,
        input_texts: Optional[list] = None,
        **generation_kwargs,
    ) -> list:
        """
        배치 텍스트 생성

        Args:
            instructions: 지시사항 리스트
            input_texts: 입력 텍스트 리스트 (선택)
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            생성된 응답 리스트
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)

        responses = []
        for instruction, input_text in zip(instructions, input_texts):
            response = self.generate(instruction, input_text, **generation_kwargs)
            responses.append(response)

        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if self.model is None:
            return {"status": "not loaded"}

        return {
            "base_model": self.base_model_name,
            "adapter_path": self.adapter_path,
            "load_in_4bit": self.load_in_4bit,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }


def load_inference_model(
    base_model: str = "beomi/Llama-3-Open-Ko-8B",
    adapter_path: Optional[str] = None,
    **kwargs,
) -> FinancialLLMInference:
    """
    추론 모델 로드 헬퍼 함수

    Args:
        base_model: 베이스 모델 이름
        adapter_path: LoRA 어댑터 경로
        **kwargs: 추가 설정

    Returns:
        로드된 추론 엔진
    """
    inference = FinancialLLMInference(
        base_model=base_model,
        adapter_path=adapter_path,
        **kwargs,
    )
    return inference.load()


# FastAPI 서버를 위한 엔드포인트 함수들
_global_inference: Optional[FinancialLLMInference] = None


def get_inference() -> FinancialLLMInference:
    """전역 추론 인스턴스 반환 (싱글톤)"""
    global _global_inference
    if _global_inference is None:
        raise RuntimeError("Inference not initialized. Call init_inference() first.")
    return _global_inference


def init_inference(
    base_model: str = "beomi/Llama-3-Open-Ko-8B",
    adapter_path: Optional[str] = None,
    **kwargs,
) -> FinancialLLMInference:
    """전역 추론 인스턴스 초기화"""
    global _global_inference
    _global_inference = load_inference_model(
        base_model=base_model,
        adapter_path=adapter_path,
        **kwargs,
    )
    return _global_inference


if __name__ == "__main__":
    # 테스트 실행
    print("=" * 50)
    print("Financial LLM Inference Test")
    print("=" * 50)

    # 실제 모델 로드 없이 구조 확인
    inference = FinancialLLMInference()
    print(f"Default generation config: {inference.default_generation_config}")
    print(f"Model info (before load): {inference.get_model_info()}")

    # 프롬프트 포맷 테스트
    prompt = inference._format_prompt(
        instruction="다음 금융 거래가 이상 거래인지 분석해주세요.",
        input_text="계좌번호: 123-456, 거래금액: 5억원, 거래시간: 새벽 3시"
    )
    print(f"\nFormatted prompt:\n{prompt}")
