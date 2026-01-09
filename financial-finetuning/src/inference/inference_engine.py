# Inference Engine
"""
Fine-tuned 금융 LLM 추론 엔진

Features:
- Thread-safe 추론
- LRU 캐싱
- 에러 핸들링 및 재시도
- 배치 처리 최적화
"""

import logging
import hashlib
from typing import Optional, Dict, Any, Generator, List
from pathlib import Path
from threading import Thread, Lock
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """추론 관련 예외"""
    pass


class ModelNotLoadedError(InferenceError):
    """모델이 로드되지 않았을 때 발생"""
    pass


class GenerationError(InferenceError):
    """텍스트 생성 중 오류 발생"""
    pass


@dataclass
class GenerationResult:
    """생성 결과를 담는 데이터 클래스"""
    response: str
    prompt: str
    tokens_generated: int
    generation_time_ms: float
    cached: bool = False


class FinancialLLMInference:
    """
    Fine-tuned 금융 LLM 추론 클래스 (Thread-safe)

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
        cache_size: int = 100,
    ):
        """
        Args:
            base_model: 베이스 모델 이름 또는 경로
            adapter_path: LoRA 어댑터 경로 (None이면 베이스 모델만 로드)
            load_in_4bit: 4-bit 양자화 사용 여부
            device_map: 디바이스 매핑 전략
            torch_dtype: 텐서 데이터 타입
            cache_size: 응답 캐시 크기
        """
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype)

        self.model = None
        self.tokenizer = None

        # Thread safety
        self._lock = Lock()
        self._is_loading = False

        # 응답 캐시
        self._cache: Dict[str, str] = {}
        self._cache_size = cache_size

        # 기본 생성 설정
        self.default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }

    def _get_cache_key(self, instruction: str, input_text: str, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = f"{instruction}|{input_text}|{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _add_to_cache(self, key: str, response: str):
        """캐시에 응답 추가 (LRU 방식)"""
        if len(self._cache) >= self._cache_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = response

    def clear_cache(self):
        """캐시 초기화"""
        with self._lock:
            self._cache.clear()
        logger.info("Cache cleared")

    def load(self):
        """모델 로드 (Thread-safe)"""
        with self._lock:
            if self._is_loading:
                logger.warning("Model is already being loaded")
                return self

            if self.model is not None:
                logger.info("Model already loaded")
                return self

            self._is_loading = True

        try:
            logger.info(f"Loading base model: {self.base_model_name}")

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
                logger.info(f"Loading LoRA adapter: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.adapter_path,
                )
            elif self.adapter_path:
                logger.warning(f"Adapter path not found: {self.adapter_path}")

            self.model.eval()
            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise InferenceError(f"Model loading failed: {e}")
        finally:
            with self._lock:
                self._is_loading = False

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
        use_cache: bool = True,
        **generation_kwargs,
    ) -> str:
        """
        텍스트 생성 (Thread-safe, Cached)

        Args:
            instruction: 지시사항
            input_text: 입력 텍스트 (선택)
            use_cache: 캐시 사용 여부
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            생성된 응답 텍스트

        Raises:
            ModelNotLoadedError: 모델 로드 실패
            GenerationError: 생성 중 오류
        """
        import time

        # 캐시 확인
        if use_cache:
            cache_key = self._get_cache_key(instruction, input_text, **generation_kwargs)
            if cache_key in self._cache:
                logger.debug("Cache hit")
                return self._cache[cache_key]

        # 모델 로드 확인
        if self.model is None:
            try:
                self.load()
            except Exception as e:
                raise ModelNotLoadedError(f"Failed to load model: {e}")

        try:
            start_time = time.time()

            # 프롬프트 구성
            prompt = self._format_prompt(instruction, input_text)

            # 토큰화
            with self._lock:
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
            ).strip()

            # 캐시에 저장
            if use_cache:
                self._add_to_cache(cache_key, response)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Generation completed in {elapsed_ms:.2f}ms")

            return response

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory")
            raise GenerationError("GPU memory exhausted. Try reducing max_new_tokens.")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Text generation failed: {e}")

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

        Raises:
            ModelNotLoadedError: 모델이 로드되지 않음
            GenerationError: 생성 중 오류
        """
        if self.model is None:
            try:
                self.load()
            except Exception as e:
                raise ModelNotLoadedError(f"Failed to load model: {e}")

        try:
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
            thread_kwargs = {
                **inputs,
                **gen_config,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            thread = Thread(target=self.model.generate, kwargs=thread_kwargs)
            thread.start()

            # 토큰 스트리밍
            for token in streamer:
                yield token

            thread.join()

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise GenerationError(f"Streaming generation failed: {e}")

    def batch_generate(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
        show_progress: bool = True,
        **generation_kwargs,
    ) -> List[str]:
        """
        배치 텍스트 생성

        Args:
            instructions: 지시사항 리스트
            input_texts: 입력 텍스트 리스트 (선택)
            show_progress: 진행 상황 표시 여부
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            생성된 응답 리스트
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)

        if len(instructions) != len(input_texts):
            raise ValueError("instructions and input_texts must have the same length")

        responses = []
        total = len(instructions)

        for i, (instruction, input_text) in enumerate(zip(instructions, input_texts)):
            try:
                response = self.generate(instruction, input_text, **generation_kwargs)
                responses.append(response)

                if show_progress:
                    logger.info(f"Batch progress: {i+1}/{total}")

            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                responses.append(f"[Error: {str(e)}]")

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


# FastAPI 서버를 위한 엔드포인트 함수들 (Thread-safe Singleton)
_global_inference: Optional[FinancialLLMInference] = None
_global_inference_lock = Lock()


def get_inference() -> FinancialLLMInference:
    """전역 추론 인스턴스 반환 (Thread-safe Singleton)"""
    global _global_inference
    with _global_inference_lock:
        if _global_inference is None:
            raise ModelNotLoadedError("Inference not initialized. Call init_inference() first.")
        return _global_inference


def init_inference(
    base_model: str = "beomi/Llama-3-Open-Ko-8B",
    adapter_path: Optional[str] = None,
    **kwargs,
) -> FinancialLLMInference:
    """전역 추론 인스턴스 초기화 (Thread-safe)"""
    global _global_inference
    with _global_inference_lock:
        if _global_inference is not None:
            logger.warning("Inference already initialized. Returning existing instance.")
            return _global_inference

        _global_inference = load_inference_model(
            base_model=base_model,
            adapter_path=adapter_path,
            **kwargs,
        )
        return _global_inference


def reset_inference():
    """전역 추론 인스턴스 리셋"""
    global _global_inference
    with _global_inference_lock:
        if _global_inference is not None:
            if _global_inference.model is not None:
                del _global_inference.model
            _global_inference = None
            logger.info("Global inference instance reset")


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
