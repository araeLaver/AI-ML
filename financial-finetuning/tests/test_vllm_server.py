# Tests for vLLM Inference Server
"""
vLLM 추론 서버 테스트 (엔진 로드 없이)
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

VLLM_MODULE_AVAILABLE = False
try:
    from src.inference.vllm_server import (
        VLLMConfig,
        VLLMInferenceResult,
        BenchmarkResult,
        VLLMInferenceServer,
        VLLMError,
        VLLMEngineNotLoadedError,
        VLLMGenerationError,
    )
    VLLM_MODULE_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not VLLM_MODULE_AVAILABLE, reason="vllm 의존성 없음")
class TestVLLMConfig:
    """VLLMConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = VLLMConfig()
        assert config.model == "beomi/Llama-3-Open-Ko-8B"
        assert config.dtype == "auto"
        assert config.max_model_len == 2048
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.90
        assert config.quantization is None
        assert config.enable_lora is False

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = VLLMConfig(
            model="custom/model",
            tensor_parallel_size=2,
            quantization="awq",
            enable_lora=True,
            lora_adapter_path="./custom/adapter",
        )
        assert config.model == "custom/model"
        assert config.tensor_parallel_size == 2
        assert config.quantization == "awq"
        assert config.enable_lora is True
        assert config.lora_adapter_path == "./custom/adapter"

    def test_to_dict(self):
        """설정 직렬화 테스트"""
        config = VLLMConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["model"] == "beomi/Llama-3-Open-Ko-8B"
        assert d["tensor_parallel_size"] == 1
        assert "temperature" in d
        assert "max_tokens" in d

    def test_from_yaml(self, tmp_path):
        """YAML 파일에서 설정 로드 테스트"""
        yaml_content = """
engine:
  model: "test/model"
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.85
  quantization: "awq"
lora:
  enabled: true
  adapter_path: "./test/adapter"
  max_lora_rank: 32
generation:
  temperature: 0.5
  max_tokens: 256
serving:
  max_num_seqs: 128
"""
        config_file = tmp_path / "test_vllm_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = VLLMConfig.from_yaml(str(config_file))
        assert config.model == "test/model"
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.85
        assert config.quantization == "awq"
        assert config.enable_lora is True
        assert config.lora_adapter_path == "./test/adapter"
        assert config.max_lora_rank == 32
        assert config.temperature == 0.5
        assert config.max_tokens == 256
        assert config.max_num_seqs == 128


@pytest.mark.skipif(not VLLM_MODULE_AVAILABLE, reason="vllm 의존성 없음")
class TestVLLMInferenceResult:
    """VLLMInferenceResult 테스트"""

    def test_result_creation(self):
        """결과 생성 테스트"""
        result = VLLMInferenceResult(
            response="테스트 응답",
            prompt="테스트 프롬프트",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            generation_time_ms=150.5,
            tokens_per_second=132.8,
        )
        assert result.response == "테스트 응답"
        assert result.total_tokens == 30
        assert result.tokens_per_second == 132.8

    def test_result_to_dict(self):
        """결과 직렬화 테스트"""
        result = VLLMInferenceResult(
            response="응답",
            prompt="프롬프트",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            generation_time_ms=100.0,
            tokens_per_second=100.0,
        )
        d = result.to_dict()
        assert d["response"] == "응답"
        assert d["total_tokens"] == 15

    def test_result_summary(self):
        """결과 요약 테스트"""
        result = VLLMInferenceResult(
            response="응답",
            prompt="프롬프트",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            generation_time_ms=100.0,
            tokens_per_second=100.0,
        )
        summary = result.summary()
        assert "5(in)" in summary
        assert "10(out)" in summary
        assert "100.0ms" in summary


@pytest.mark.skipif(not VLLM_MODULE_AVAILABLE, reason="vllm 의존성 없음")
class TestBenchmarkResult:
    """BenchmarkResult 테스트"""

    def test_benchmark_result(self):
        """벤치마크 결과 생성 테스트"""
        result = BenchmarkResult(
            num_requests=10,
            total_tokens_generated=500,
            total_time_seconds=5.0,
            avg_latency_ms=500.0,
            throughput_tokens_per_sec=100.0,
            throughput_requests_per_sec=2.0,
            p50_latency_ms=450.0,
            p95_latency_ms=800.0,
            p99_latency_ms=950.0,
        )
        assert result.num_requests == 10
        assert result.throughput_tokens_per_sec == 100.0

    def test_benchmark_summary(self):
        """벤치마크 요약 테스트"""
        result = BenchmarkResult(
            num_requests=10,
            total_tokens_generated=500,
            total_time_seconds=5.0,
            avg_latency_ms=500.0,
            throughput_tokens_per_sec=100.0,
            throughput_requests_per_sec=2.0,
            p50_latency_ms=450.0,
            p95_latency_ms=800.0,
            p99_latency_ms=950.0,
        )
        summary = result.summary()
        assert "Benchmark Results" in summary
        assert "100.0 tok/s" in summary
        assert "P50/P95/P99" in summary

    def test_benchmark_to_dict(self):
        """벤치마크 직렬화 테스트"""
        result = BenchmarkResult(
            num_requests=5,
            total_tokens_generated=250,
            total_time_seconds=2.5,
            avg_latency_ms=500.0,
            throughput_tokens_per_sec=100.0,
            throughput_requests_per_sec=2.0,
            p50_latency_ms=450.0,
            p95_latency_ms=800.0,
            p99_latency_ms=950.0,
        )
        d = result.to_dict()
        assert d["num_requests"] == 5
        assert "p99_latency_ms" in d


@pytest.mark.skipif(not VLLM_MODULE_AVAILABLE, reason="vllm 의존성 없음")
class TestVLLMInferenceServer:
    """VLLMInferenceServer 테스트 (엔진 로드 없이)"""

    def test_server_init(self):
        """서버 초기화 테스트"""
        server = VLLMInferenceServer()
        assert server.engine is None
        assert server.config.model == "beomi/Llama-3-Open-Ko-8B"

    def test_server_with_custom_config(self):
        """커스텀 설정 서버 초기화"""
        config = VLLMConfig(
            model="custom/model",
            tensor_parallel_size=2,
        )
        server = VLLMInferenceServer(config=config)
        assert server.config.model == "custom/model"
        assert server.config.tensor_parallel_size == 2

    def test_format_prompt_with_input(self):
        """프롬프트 포맷팅 테스트 (입력 있음)"""
        server = VLLMInferenceServer()
        prompt = server._format_prompt(
            instruction="테스트 지시사항",
            input_text="테스트 입력",
        )
        assert "### 지시사항:" in prompt
        assert "테스트 지시사항" in prompt
        assert "### 입력:" in prompt
        assert "테스트 입력" in prompt
        assert "### 응답:" in prompt

    def test_format_prompt_without_input(self):
        """프롬프트 포맷팅 테스트 (입력 없음)"""
        server = VLLMInferenceServer()
        prompt = server._format_prompt(
            instruction="테스트 지시사항",
            input_text="",
        )
        assert "### 지시사항:" in prompt
        assert "테스트 지시사항" in prompt
        assert "### 입력:" not in prompt
        assert "### 응답:" in prompt

    def test_engine_info_before_load(self):
        """엔진 로드 전 정보 조회"""
        server = VLLMInferenceServer()
        info = server.get_engine_info()
        assert info["status"] == "not loaded"

    def test_generate_without_load_raises(self):
        """엔진 미로드 상태에서 생성 시 예외"""
        server = VLLMInferenceServer()
        with pytest.raises(VLLMEngineNotLoadedError):
            server.generate("테스트")

    def test_batch_generate_without_load_raises(self):
        """엔진 미로드 상태에서 배치 생성 시 예외"""
        server = VLLMInferenceServer()
        with pytest.raises(VLLMEngineNotLoadedError):
            server.batch_generate(["테스트1", "테스트2"])

    def test_benchmark_without_load_raises(self):
        """엔진 미로드 상태에서 벤치마크 시 예외"""
        server = VLLMInferenceServer()
        with pytest.raises(VLLMEngineNotLoadedError):
            server.benchmark(num_requests=5)

    def test_batch_generate_length_mismatch(self):
        """배치 생성 시 입력 길이 불일치 예외"""
        server = VLLMInferenceServer()
        # engine을 Mock 처리하여 길이 검증만 테스트
        server.engine = "mock"
        with pytest.raises(ValueError, match="same length"):
            server.batch_generate(
                instructions=["a", "b"],
                input_texts=["x"],
            )
        server.engine = None


@pytest.mark.skipif(not VLLM_MODULE_AVAILABLE, reason="vllm 의존성 없음")
class TestGlobalVLLMServer:
    """전역 vLLM 서버 인스턴스 테스트"""

    def test_get_server_not_initialized(self):
        """초기화 전 get_vllm_server 호출"""
        from src.inference.vllm_server import get_vllm_server
        import src.inference.vllm_server as module
        module._global_vllm_server = None

        with pytest.raises(VLLMEngineNotLoadedError, match="not initialized"):
            get_vllm_server()

    def test_reset_server(self):
        """서버 리셋 테스트"""
        from src.inference.vllm_server import reset_vllm_server
        import src.inference.vllm_server as module
        module._global_vllm_server = None
        reset_vllm_server()  # None 상태에서도 에러 없이 실행


class TestVLLMExceptions:
    """vLLM 예외 클래스 테스트"""

    def test_exception_hierarchy(self):
        """예외 계층 구조 테스트"""
        from src.inference.vllm_server import (
            VLLMError,
            VLLMEngineNotLoadedError,
            VLLMGenerationError,
        )
        assert issubclass(VLLMEngineNotLoadedError, VLLMError)
        assert issubclass(VLLMGenerationError, VLLMError)
        assert issubclass(VLLMError, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
