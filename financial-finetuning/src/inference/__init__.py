# Inference Module
from .inference_engine import (
    FinancialLLMInference,
    InferenceError,
    ModelNotLoadedError,
    GenerationError,
    load_inference_model,
)

__all__ = [
    "FinancialLLMInference",
    "InferenceError",
    "ModelNotLoadedError",
    "GenerationError",
    "load_inference_model",
]

# vLLM (dataclasses와 예외 클래스는 항상 사용 가능)
from .vllm_server import VLLMConfig, VLLMInferenceResult, BenchmarkResult
from .vllm_server import VLLMError, VLLMEngineNotLoadedError, VLLMGenerationError
__all__.extend([
    "VLLMConfig",
    "VLLMInferenceResult",
    "BenchmarkResult",
    "VLLMError",
    "VLLMEngineNotLoadedError",
    "VLLMGenerationError",
])

try:
    from .vllm_server import (
        VLLMInferenceServer,
        get_vllm_server,
        init_vllm_server,
        reset_vllm_server,
        run_vllm_benchmark,
    )
    __all__.extend([
        "VLLMInferenceServer",
        "get_vllm_server",
        "init_vllm_server",
        "reset_vllm_server",
        "run_vllm_benchmark",
    ])
except ImportError:
    pass
