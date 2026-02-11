# vLLM Inference Server
"""
vLLM 기반 고속 LLM 추론 서버

Features:
- PagedAttention을 통한 효율적 GPU 메모리 관리
- Continuous batching으로 높은 처리량
- Tensor parallelism 멀티 GPU 지원
- LoRA 어댑터 동적 로드
- AWQ/GPTQ 양자화 지원
- 벤치마크 비교 도구
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List, Optional

import yaml

logger = logging.getLogger(__name__)

# vLLM은 선택적 의존성
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    VLLM_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VLLMConfig:
    """vLLM 엔진 설정"""

    # 모델 설정
    model: str = "beomi/Llama-3-Open-Ko-8B"
    tokenizer: Optional[str] = None
    dtype: str = "auto"
    max_model_len: int = 2048
    trust_remote_code: bool = True

    # GPU / 병렬화
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    swap_space: int = 4  # GiB

    # 양자화
    quantization: Optional[str] = None  # "awq", "gptq", None

    # LoRA
    enable_lora: bool = False
    lora_adapter_path: Optional[str] = None
    max_loras: int = 1
    max_lora_rank: int = 64

    # 생성 기본값
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    repetition_penalty: float = 1.1

    # 서빙
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": self.swap_space,
            "quantization": self.quantization,
            "enable_lora": self.enable_lora,
            "lora_adapter_path": self.lora_adapter_path,
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "VLLMConfig":
        """YAML 파일에서 설정 로드"""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        engine = raw.get("engine", {})
        lora = raw.get("lora", {})
        generation = raw.get("generation", {})
        serving = raw.get("serving", {})

        return cls(
            model=engine.get("model", cls.model),
            tokenizer=engine.get("tokenizer"),
            dtype=engine.get("dtype", cls.dtype),
            max_model_len=engine.get("max_model_len", cls.max_model_len),
            trust_remote_code=engine.get("trust_remote_code", cls.trust_remote_code),
            tensor_parallel_size=engine.get("tensor_parallel_size", cls.tensor_parallel_size),
            gpu_memory_utilization=engine.get("gpu_memory_utilization", cls.gpu_memory_utilization),
            swap_space=engine.get("swap_space", cls.swap_space),
            quantization=engine.get("quantization"),
            enable_lora=lora.get("enabled", cls.enable_lora),
            lora_adapter_path=lora.get("adapter_path"),
            max_loras=lora.get("max_loras", cls.max_loras),
            max_lora_rank=lora.get("max_lora_rank", cls.max_lora_rank),
            temperature=generation.get("temperature", cls.temperature),
            top_p=generation.get("top_p", cls.top_p),
            top_k=generation.get("top_k", cls.top_k),
            max_tokens=generation.get("max_tokens", cls.max_tokens),
            repetition_penalty=generation.get("repetition_penalty", cls.repetition_penalty),
            max_num_seqs=serving.get("max_num_seqs", cls.max_num_seqs),
            max_num_batched_tokens=serving.get("max_num_batched_tokens"),
        )


@dataclass
class VLLMInferenceResult:
    """vLLM 추론 결과"""

    response: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time_ms: float
    tokens_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_time_ms": self.generation_time_ms,
            "tokens_per_second": self.tokens_per_second,
        }

    def summary(self) -> str:
        return (
            f"Tokens: {self.prompt_tokens}(in) + {self.completion_tokens}(out) = {self.total_tokens} | "
            f"Time: {self.generation_time_ms:.1f}ms | "
            f"Speed: {self.tokens_per_second:.1f} tok/s"
        )


@dataclass
class BenchmarkResult:
    """벤치마크 비교 결과"""

    num_requests: int
    total_tokens_generated: int
    total_time_seconds: float
    avg_latency_ms: float
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_requests": self.num_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_time_seconds": self.total_time_seconds,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "throughput_requests_per_sec": self.throughput_requests_per_sec,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }

    def summary(self) -> str:
        lines = [
            "=== vLLM Benchmark Results ===",
            f"Requests: {self.num_requests}",
            f"Total tokens: {self.total_tokens_generated}",
            f"Total time: {self.total_time_seconds:.2f}s",
            f"Avg latency: {self.avg_latency_ms:.1f}ms",
            f"Throughput: {self.throughput_tokens_per_sec:.1f} tok/s | {self.throughput_requests_per_sec:.2f} req/s",
            f"Latency P50/P95/P99: {self.p50_latency_ms:.1f} / {self.p95_latency_ms:.1f} / {self.p99_latency_ms:.1f} ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VLLMError(Exception):
    """vLLM 관련 예외"""
    pass


class VLLMEngineNotLoadedError(VLLMError):
    """엔진이 로드되지 않았을 때 발생"""
    pass


class VLLMGenerationError(VLLMError):
    """생성 중 오류 발생"""
    pass


# ---------------------------------------------------------------------------
# Core server
# ---------------------------------------------------------------------------

class VLLMInferenceServer:
    """
    vLLM 기반 고속 추론 서버

    PagedAttention을 활용한 효율적 메모리 관리와
    continuous batching으로 높은 처리량을 제공합니다.

    Usage:
        server = VLLMInferenceServer(config=VLLMConfig(
            model="beomi/Llama-3-Open-Ko-8B",
            tensor_parallel_size=1,
        ))
        server.load()
        result = server.generate("삼성전자 주식을 분석해주세요.")
    """

    def __init__(self, config: Optional[VLLMConfig] = None):
        """
        Args:
            config: vLLM 엔진 설정 (None이면 기본값 사용)
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required for vLLM inference. "
                "Install with: pip install 'vllm>=0.2.7'"
            )

        self.config = config or VLLMConfig()
        self.engine: Optional[LLM] = None
        self._lock = Lock()
        self._is_loading = False

        # 기본 샘플링 파라미터
        self.default_sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            repetition_penalty=self.config.repetition_penalty,
        )

    def load(self) -> "VLLMInferenceServer":
        """vLLM 엔진 로드 (Thread-safe)"""
        with self._lock:
            if self._is_loading:
                logger.warning("Engine is already being loaded")
                return self

            if self.engine is not None:
                logger.info("Engine already loaded")
                return self

            self._is_loading = True

        try:
            logger.info(f"Loading vLLM engine: {self.config.model}")
            logger.info(f"  tensor_parallel_size: {self.config.tensor_parallel_size}")
            logger.info(f"  gpu_memory_utilization: {self.config.gpu_memory_utilization}")
            logger.info(f"  quantization: {self.config.quantization or 'none'}")
            logger.info(f"  enable_lora: {self.config.enable_lora}")

            engine_kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "dtype": self.config.dtype,
                "max_model_len": self.config.max_model_len,
                "trust_remote_code": self.config.trust_remote_code,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "swap_space": self.config.swap_space,
                "max_num_seqs": self.config.max_num_seqs,
            }

            if self.config.tokenizer:
                engine_kwargs["tokenizer"] = self.config.tokenizer

            if self.config.quantization:
                engine_kwargs["quantization"] = self.config.quantization

            if self.config.enable_lora:
                engine_kwargs["enable_lora"] = True
                engine_kwargs["max_loras"] = self.config.max_loras
                engine_kwargs["max_lora_rank"] = self.config.max_lora_rank

            if self.config.max_num_batched_tokens is not None:
                engine_kwargs["max_num_batched_tokens"] = self.config.max_num_batched_tokens

            self.engine = LLM(**engine_kwargs)
            logger.info("vLLM engine loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load vLLM engine: {e}")
            self.engine = None
            raise VLLMError(f"Engine loading failed: {e}")
        finally:
            with self._lock:
                self._is_loading = False

        return self

    def _format_prompt(self, instruction: str, input_text: str = "") -> str:
        """프롬프트 포맷팅 (기존 inference_engine과 동일)"""
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

    def _create_sampling_params(self, **kwargs) -> "SamplingParams":
        """커스텀 샘플링 파라미터 생성"""
        params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
        }
        return SamplingParams(**params)

    def _get_lora_request(self) -> Optional["LoRARequest"]:
        """LoRA 어댑터 요청 생성"""
        if not self.config.enable_lora or not self.config.lora_adapter_path:
            return None

        adapter_path = Path(self.config.lora_adapter_path)
        if not adapter_path.exists():
            logger.warning(f"LoRA adapter path not found: {adapter_path}")
            return None

        return LoRARequest(
            lora_name="financial_lora",
            lora_int_id=1,
            lora_path=str(adapter_path),
        )

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        **sampling_kwargs,
    ) -> VLLMInferenceResult:
        """
        단일 텍스트 생성

        Args:
            instruction: 지시사항
            input_text: 입력 텍스트 (선택)
            **sampling_kwargs: 샘플링 파라미터 오버라이드

        Returns:
            VLLMInferenceResult

        Raises:
            VLLMEngineNotLoadedError: 엔진 미로드
            VLLMGenerationError: 생성 중 오류
        """
        if self.engine is None:
            raise VLLMEngineNotLoadedError("Engine not loaded. Call load() first.")

        try:
            start_time = time.time()

            prompt = self._format_prompt(instruction, input_text)
            sampling_params = self._create_sampling_params(**sampling_kwargs)
            lora_request = self._get_lora_request()

            outputs = self.engine.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                lora_request=lora_request,
            )

            output = outputs[0]
            response_text = output.outputs[0].text.strip()
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)

            elapsed_ms = (time.time() - start_time) * 1000
            tokens_per_sec = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

            return VLLMInferenceResult(
                response=response_text,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                generation_time_ms=elapsed_ms,
                tokens_per_second=tokens_per_sec,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise VLLMGenerationError(f"Text generation failed: {e}")

    def batch_generate(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
        **sampling_kwargs,
    ) -> List[VLLMInferenceResult]:
        """
        배치 텍스트 생성 (vLLM continuous batching 활용)

        Args:
            instructions: 지시사항 리스트
            input_texts: 입력 텍스트 리스트 (선택)
            **sampling_kwargs: 샘플링 파라미터 오버라이드

        Returns:
            VLLMInferenceResult 리스트
        """
        if self.engine is None:
            raise VLLMEngineNotLoadedError("Engine not loaded. Call load() first.")

        if input_texts is None:
            input_texts = [""] * len(instructions)

        if len(instructions) != len(input_texts):
            raise ValueError("instructions and input_texts must have the same length")

        try:
            start_time = time.time()

            prompts = [
                self._format_prompt(inst, inp)
                for inst, inp in zip(instructions, input_texts)
            ]

            sampling_params = self._create_sampling_params(**sampling_kwargs)
            lora_request = self._get_lora_request()

            outputs = self.engine.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )

            total_elapsed_ms = (time.time() - start_time) * 1000
            per_request_ms = total_elapsed_ms / len(outputs) if outputs else 0.0

            results = []
            for output in outputs:
                response_text = output.outputs[0].text.strip()
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)
                tokens_per_sec = (completion_tokens / (per_request_ms / 1000)) if per_request_ms > 0 else 0.0

                results.append(VLLMInferenceResult(
                    response=response_text,
                    prompt=output.prompt,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    generation_time_ms=per_request_ms,
                    tokens_per_second=tokens_per_sec,
                ))

            logger.info(
                f"Batch generation: {len(results)} requests in {total_elapsed_ms:.1f}ms "
                f"({total_elapsed_ms / len(results):.1f}ms/req)"
            )
            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise VLLMGenerationError(f"Batch generation failed: {e}")

    def benchmark(
        self,
        prompts: Optional[List[str]] = None,
        num_requests: int = 10,
        **sampling_kwargs,
    ) -> BenchmarkResult:
        """
        추론 성능 벤치마크

        Args:
            prompts: 벤치마크용 프롬프트 리스트 (None이면 기본 금융 프롬프트 사용)
            num_requests: 요청 수
            **sampling_kwargs: 샘플링 파라미터 오버라이드

        Returns:
            BenchmarkResult
        """
        if self.engine is None:
            raise VLLMEngineNotLoadedError("Engine not loaded. Call load() first.")

        if prompts is None:
            prompts = _default_benchmark_prompts()

        # 프롬프트를 num_requests에 맞게 반복
        test_prompts = []
        for i in range(num_requests):
            test_prompts.append(prompts[i % len(prompts)])

        instructions = [p for p in test_prompts]

        # 개별 요청 레이턴시 측정
        latencies_ms: List[float] = []
        total_tokens = 0

        start_time = time.time()

        for instruction in instructions:
            req_start = time.time()
            result = self.generate(instruction, **sampling_kwargs)
            req_elapsed = (time.time() - req_start) * 1000
            latencies_ms.append(req_elapsed)
            total_tokens += result.completion_tokens

        total_time = time.time() - start_time

        # 레이턴시 백분위수 계산
        sorted_latencies = sorted(latencies_ms)
        n = len(sorted_latencies)

        def percentile(pct: float) -> float:
            idx = int(pct / 100.0 * (n - 1))
            return sorted_latencies[min(idx, n - 1)]

        return BenchmarkResult(
            num_requests=num_requests,
            total_tokens_generated=total_tokens,
            total_time_seconds=total_time,
            avg_latency_ms=sum(latencies_ms) / n,
            throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0.0,
            throughput_requests_per_sec=num_requests / total_time if total_time > 0 else 0.0,
            p50_latency_ms=percentile(50),
            p95_latency_ms=percentile(95),
            p99_latency_ms=percentile(99),
        )

    def get_engine_info(self) -> Dict[str, Any]:
        """엔진 정보 반환"""
        if self.engine is None:
            return {"status": "not loaded"}

        return {
            "status": "loaded",
            "model": self.config.model,
            "dtype": self.config.dtype,
            "max_model_len": self.config.max_model_len,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "quantization": self.config.quantization,
            "enable_lora": self.config.enable_lora,
            "lora_adapter_path": self.config.lora_adapter_path,
            "max_num_seqs": self.config.max_num_seqs,
        }


# ---------------------------------------------------------------------------
# Global singleton (Thread-safe)
# ---------------------------------------------------------------------------

_global_vllm_server: Optional[VLLMInferenceServer] = None
_global_vllm_lock = Lock()


def get_vllm_server() -> VLLMInferenceServer:
    """전역 vLLM 서버 인스턴스 반환 (Thread-safe Singleton)"""
    global _global_vllm_server
    with _global_vllm_lock:
        if _global_vllm_server is None:
            raise VLLMEngineNotLoadedError(
                "vLLM server not initialized. Call init_vllm_server() first."
            )
        return _global_vllm_server


def init_vllm_server(
    config: Optional[VLLMConfig] = None,
    config_path: Optional[str] = None,
) -> VLLMInferenceServer:
    """
    전역 vLLM 서버 인스턴스 초기화 (Thread-safe)

    Args:
        config: VLLMConfig 객체 (우선)
        config_path: YAML 설정 파일 경로
    """
    global _global_vllm_server
    with _global_vllm_lock:
        if _global_vllm_server is not None:
            logger.warning("vLLM server already initialized. Returning existing instance.")
            return _global_vllm_server

        if config is None and config_path:
            config = VLLMConfig.from_yaml(config_path)

        _global_vllm_server = VLLMInferenceServer(config=config)
        _global_vllm_server.load()
        return _global_vllm_server


def reset_vllm_server():
    """전역 vLLM 서버 인스턴스 리셋"""
    global _global_vllm_server
    with _global_vllm_lock:
        if _global_vllm_server is not None:
            if _global_vllm_server.engine is not None:
                del _global_vllm_server.engine
            _global_vllm_server = None
            logger.info("Global vLLM server instance reset")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def _default_benchmark_prompts() -> List[str]:
    """기본 벤치마크 프롬프트 (금융 도메인)"""
    return [
        "삼성전자 2024년 4분기 실적을 분석해주세요.",
        "금리 인상이 부동산 시장에 미치는 영향을 설명해주세요.",
        "ETF와 개별 주식 투자의 장단점을 비교해주세요.",
        "다음 거래가 이상 거래인지 분석해주세요: 새벽 3시 해외 송금 5억원",
        "채권 투자의 기본 원리와 수익 구조를 설명해주세요.",
        "코스피 지수와 환율의 상관관계를 분석해주세요.",
        "퇴직연금 DC형과 DB형의 차이를 설명해주세요.",
        "주식 시장에서 PER, PBR, ROE 지표의 의미를 설명해주세요.",
    ]


def run_vllm_benchmark(
    config_path: str = "configs/vllm_config.yaml",
    num_requests: int = 10,
    output_dir: Optional[str] = None,
) -> BenchmarkResult:
    """
    vLLM 벤치마크 실행 원스텝 함수

    Args:
        config_path: vLLM 설정 YAML 경로
        num_requests: 벤치마크 요청 수
        output_dir: 결과 저장 경로

    Returns:
        BenchmarkResult
    """
    config = VLLMConfig.from_yaml(config_path)
    server = VLLMInferenceServer(config=config)
    server.load()

    result = server.benchmark(num_requests=num_requests)
    print(result.summary())

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result_file = out_path / "vllm_benchmark.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Benchmark results saved to {result_file}")

    return result
