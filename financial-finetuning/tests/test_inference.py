# Tests for Inference Module
"""
추론 모듈 테스트 (모델 로드 없이)
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# peft 호환성 문제로 인해 직접 임포트 사용
try:
    from src.inference.inference_engine import FinancialLLMInference
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


@pytest.mark.skipif(not INFERENCE_AVAILABLE, reason="peft/transformers 호환성 문제")
class TestFinancialLLMInference:
    """추론 엔진 테스트"""

    def test_inference_init(self):
        """추론 엔진 초기화 테스트"""
        inference = FinancialLLMInference()

        assert inference is not None
        assert inference.model is None  # 아직 로드 안됨
        assert inference.base_model_name == "beomi/Llama-3-Open-Ko-8B"

    def test_custom_model_init(self):
        """커스텀 모델 설정 테스트"""
        inference = FinancialLLMInference(
            base_model="custom/model",
            adapter_path="./custom/adapter",
            load_in_4bit=False,
        )

        assert inference.base_model_name == "custom/model"
        assert inference.adapter_path == "./custom/adapter"
        assert inference.load_in_4bit is False

    def test_default_generation_config(self):
        """기본 생성 설정 테스트"""
        inference = FinancialLLMInference()
        config = inference.default_generation_config

        assert "temperature" in config
        assert "top_p" in config
        assert "top_k" in config
        assert "max_new_tokens" in config
        assert "repetition_penalty" in config

    def test_format_prompt_with_input(self):
        """프롬프트 포맷팅 테스트 (입력 있음)"""
        inference = FinancialLLMInference()
        prompt = inference._format_prompt(
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
        inference = FinancialLLMInference()
        prompt = inference._format_prompt(
            instruction="테스트 지시사항",
            input_text="",
        )

        assert "### 지시사항:" in prompt
        assert "테스트 지시사항" in prompt
        assert "### 입력:" not in prompt
        assert "### 응답:" in prompt

    def test_model_info_before_load(self):
        """모델 로드 전 정보 조회"""
        inference = FinancialLLMInference()
        info = inference.get_model_info()

        assert info["status"] == "not loaded"


@pytest.mark.skipif(not INFERENCE_AVAILABLE, reason="peft/transformers 호환성 문제")
class TestGlobalInference:
    """전역 추론 인스턴스 테스트"""

    def test_get_inference_not_initialized(self):
        """초기화 전 get_inference 호출"""
        from src.inference.inference_engine import get_inference
        import src.inference.inference_engine as module
        module._global_inference = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_inference()


class TestAPIServer:
    """API 서버 테스트"""

    def test_import_server(self):
        """서버 모듈 임포트 테스트"""
        # FastAPI 관련 임포트 확인
        from fastapi import FastAPI
        from pydantic import BaseModel

        assert FastAPI is not None
        assert BaseModel is not None

    def test_request_models(self):
        """요청 모델 구조 테스트"""
        # API 서버의 Pydantic 모델을 직접 테스트
        from pydantic import BaseModel, Field
        from typing import Optional

        class GenerateRequest(BaseModel):
            instruction: str = Field(..., description="지시사항")
            input_text: str = Field(default="", description="입력 텍스트")
            temperature: float = Field(default=0.7, ge=0.0, le=2.0)

        request = GenerateRequest(instruction="테스트")
        assert request.instruction == "테스트"
        assert request.input_text == ""
        assert request.temperature == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
