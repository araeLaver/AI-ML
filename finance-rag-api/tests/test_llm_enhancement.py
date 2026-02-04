# -*- coding: utf-8 -*-
"""
LLM 고도화 테스트 (Phase 16)

프롬프트 관리, 토큰 카운팅, 출력 검증, 멀티모델 지원 테스트
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


# ============================================================
# 프롬프트 템플릿 테스트
# ============================================================

class TestPromptTemplates:
    """프롬프트 템플릿 테스트"""

    def test_import_prompts_module(self):
        """프롬프트 모듈 임포트"""
        from src.rag.prompts import (
            PromptTemplate,
            PromptRegistry,
            PromptCategory,
            RAG_TEMPLATE,
            COT_TEMPLATE,
            SUMMARY_TEMPLATE,
        )
        assert PromptTemplate is not None
        assert PromptRegistry is not None

    def test_prompt_template_creation(self):
        """프롬프트 템플릿 생성"""
        from src.rag.prompts import PromptTemplate, PromptCategory

        template = PromptTemplate(
            name="test_template",
            version="1.0.0",
            category=PromptCategory.RAG,
            system_prompt="System: {role}",
            user_prompt_template="Question: {question}",
            description="Test template"
        )

        assert template.name == "test_template"
        assert template.version == "1.0.0"
        assert "role" in template.variables
        assert "question" in template.variables

    def test_prompt_template_format(self):
        """프롬프트 템플릿 포맷팅"""
        from src.rag.prompts import PromptTemplate, PromptCategory

        template = PromptTemplate(
            name="test",
            version="1.0.0",
            category=PromptCategory.RAG,
            system_prompt="You are a {role}",
            user_prompt_template="Answer: {question}",
        )

        result = template.format(role="expert", question="What is ETF?")

        assert result["system"] == "You are a expert"
        assert result["user"] == "Answer: What is ETF?"

    def test_prompt_template_missing_variable(self):
        """누락된 변수 에러"""
        from src.rag.prompts import PromptTemplate, PromptCategory

        template = PromptTemplate(
            name="test",
            version="1.0.0",
            category=PromptCategory.RAG,
            system_prompt="Role: {role}",
            user_prompt_template="Q: {question}",
        )

        with pytest.raises(ValueError, match="Missing variables"):
            template.format(role="expert")  # question 누락

    def test_prompt_registry_operations(self):
        """프롬프트 레지스트리 작업"""
        from src.rag.prompts import PromptTemplate, PromptRegistry, PromptCategory

        # 테스트 템플릿 등록
        test_template = PromptTemplate(
            name="registry_test",
            version="2.0.0",
            category=PromptCategory.ANALYSIS,
            system_prompt="Test",
            user_prompt_template="Test {input}",
        )
        PromptRegistry.register(test_template)

        # 조회
        retrieved = PromptRegistry.get("registry_test")
        assert retrieved is not None
        assert retrieved.version == "2.0.0"

    def test_default_templates_registered(self):
        """기본 템플릿 등록 확인"""
        from src.rag.prompts import PromptRegistry

        templates = PromptRegistry.list()
        names = [t["name"] for t in templates]

        assert "rag_default" in names
        assert "chain_of_thought" in names
        assert "summarization" in names

    def test_get_rag_prompt_helper(self):
        """RAG 프롬프트 헬퍼 함수"""
        from src.rag.prompts import get_rag_prompt

        result = get_rag_prompt(
            context="ETF는 펀드입니다.",
            question="ETF란 무엇인가요?"
        )

        assert "system" in result
        assert "user" in result
        assert "ETF" in result["user"]

    def test_few_shot_examples_creation(self):
        """Few-shot 예시 생성"""
        from src.rag.prompts import create_few_shot_examples

        examples = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]

        result = create_few_shot_examples(examples)

        assert "예시 1" in result
        assert "Q1?" in result
        assert "A1" in result


# ============================================================
# 토큰 카운팅 테스트
# ============================================================

class TestTokenCounter:
    """토큰 카운터 테스트"""

    def test_import_token_counter(self):
        """토큰 카운터 모듈 임포트"""
        from src.rag.token_counter import (
            TokenCounter,
            ContextManager,
            ModelConfig,
            ModelFamily,
        )
        assert TokenCounter is not None
        assert ContextManager is not None

    def test_token_counting_english(self):
        """영어 토큰 카운팅"""
        from src.rag.token_counter import TokenCounter

        counter = TokenCounter("llama-3.1-8b-instant")
        text = "This is a test sentence for token counting."
        tokens = counter.count_tokens(text)

        # 대략 10-15 토큰 예상
        assert 5 < tokens < 20

    def test_token_counting_korean(self):
        """한국어 토큰 카운팅"""
        from src.rag.token_counter import TokenCounter

        counter = TokenCounter("llama-3.1-8b-instant")
        text = "이것은 한국어 토큰 카운팅 테스트입니다."
        tokens = counter.count_tokens(text)

        # 한국어는 토큰이 더 많이 사용됨
        assert tokens > 0

    def test_model_config_lookup(self):
        """모델 설정 조회"""
        from src.rag.token_counter import TokenCounter

        counter = TokenCounter("gpt-4-turbo")

        assert counter.context_window == 128000
        assert counter.output_limit == 4096

    def test_context_manager_fit(self):
        """컨텍스트 맞춤"""
        from src.rag.token_counter import ContextManager

        manager = ContextManager("llama-3.1-8b-instant")

        documents = [
            "첫 번째 문서입니다. " * 50,
            "두 번째 문서입니다. " * 50,
            "세 번째 문서입니다. " * 50,
        ]
        question = "이 문서들의 내용은?"

        fitted_docs, used_tokens = manager.fit_context(
            documents, question, strategy="truncate_last"
        )

        assert len(fitted_docs) <= len(documents)
        assert used_tokens > 0

    def test_context_manager_strategies(self):
        """컨텍스트 조정 전략"""
        from src.rag.token_counter import ContextManager

        manager = ContextManager("llama-3.1-8b-instant")

        docs = ["문서 " * 100 for _ in range(5)]
        question = "질문?"

        # 각 전략 테스트
        result1, _ = manager.fit_context(docs, question, "truncate_last")
        result2, _ = manager.fit_context(docs, question, "drop_last")

        assert isinstance(result1, list)
        assert isinstance(result2, list)

    def test_response_quality_estimation(self):
        """응답 품질 예측"""
        from src.rag.token_counter import ContextManager

        manager = ContextManager("llama-3.1-8b-instant")

        quality = manager.estimate_response_quality(
            context_tokens=1000,
            question_tokens=50
        )

        assert "quality" in quality
        assert "utilization" in quality
        assert quality["quality"] in ["optimal", "good", "moderate", "limited"]

    def test_list_supported_models(self):
        """지원 모델 목록"""
        from src.rag.token_counter import list_supported_models

        models = list_supported_models()

        assert len(models) > 0
        assert any(m["name"] == "llama-3.1-8b-instant" for m in models)


# ============================================================
# 출력 검증 테스트
# ============================================================

class TestOutputValidator:
    """출력 검증기 테스트"""

    def test_import_validator(self):
        """검증 모듈 임포트"""
        from src.rag.output_validator import (
            OutputValidator,
            ValidationResult,
            ValidationStatus,
            ConfidenceCalibrator,
        )
        assert OutputValidator is not None
        assert ValidationResult is not None

    def test_length_validation(self):
        """길이 검증"""
        from src.rag.output_validator import OutputValidator

        validator = OutputValidator(min_answer_length=10, max_answer_length=100)

        # 너무 짧은 답변
        result = validator.validate("짧음", [], "질문?")
        assert not result.is_valid or result.score < 1.0

        # 적절한 답변
        result = validator.validate(
            "이것은 적절한 길이의 답변입니다. 충분한 정보를 포함합니다.",
            ["컨텍스트 문서"],
            "질문?"
        )
        assert result.score > 0.5

    def test_hallucination_detection(self):
        """환각 감지"""
        from src.rag.output_validator import OutputValidator

        validator = OutputValidator()

        # 환각 패턴 포함
        result = validator.validate(
            "제가 알기로는 이것은 사실입니다. 아마도 맞을 것입니다.",
            ["문서"],
            "질문?"
        )

        hallucination_check = next(
            (c for c in result.checks if c["check"] == "hallucination"),
            None
        )
        assert hallucination_check is not None
        assert len(hallucination_check.get("patterns_found", [])) > 0

    def test_risky_advice_detection(self):
        """위험한 조언 감지"""
        from src.rag.output_validator import OutputValidator

        validator = OutputValidator()

        result = validator.validate(
            "반드시 이 주식에 투자해야 합니다. 100% 수익 보장됩니다.",
            ["문서"],
            "투자 조언?"
        )

        risk_check = next(
            (c for c in result.checks if c["check"] == "risky_advice"),
            None
        )
        assert risk_check is not None
        assert not risk_check["passed"]

    def test_citation_validation(self):
        """인용 검증"""
        from src.rag.output_validator import OutputValidator

        validator = OutputValidator()

        context = ["ETF는 Exchange Traded Fund의 약자로, 거래소에서 거래되는 펀드입니다."]
        answer = "ETF는 Exchange Traded Fund로, 거래소에서 거래됩니다."

        result = validator.validate(answer, context, "ETF란?")

        citation_check = next(
            (c for c in result.checks if c["check"] == "citation"),
            None
        )
        assert citation_check is not None
        # 인용이 있으면 점수가 높아야 함
        assert citation_check["score"] > 0.3

    def test_relevance_validation(self):
        """관련성 검증"""
        from src.rag.output_validator import OutputValidator

        validator = OutputValidator()

        # 관련 있는 답변 (키워드가 충분히 겹치도록)
        result = validator.validate(
            "ETF means Exchange Traded Fund. ETF is a type of investment fund.",
            ["document"],
            "What is ETF fund investment?"
        )

        relevance_check = next(
            (c for c in result.checks if c["check"] == "relevance"),
            None
        )
        assert relevance_check is not None
        assert relevance_check["passed"]

    def test_confidence_calibration(self):
        """신뢰도 보정"""
        from src.rag.output_validator import (
            ConfidenceCalibrator,
            ValidationResult,
            ValidationStatus
        )

        calibrator = ConfidenceCalibrator()

        # Mock validation result
        validation_result = ValidationResult(
            status=ValidationStatus.PASSED,
            score=0.8,
            checks=[
                {"check": "citation", "score": 0.9},
                {"check": "hallucination", "score": 1.0}
            ]
        )

        confidence, score = calibrator.calibrate(
            base_confidence="medium",
            validation_result=validation_result,
            source_relevance_scores=[0.8, 0.7]
        )

        assert confidence in ["high", "medium", "low"]
        assert 0.0 <= score <= 1.0

    def test_json_validation(self):
        """JSON 검증"""
        from src.rag.output_validator import JSONValidator

        # 유효한 JSON
        valid_text = '```json\n{"name": "test", "value": 123}\n```'
        result = JSONValidator.validate_json(valid_text)
        assert result["valid"]
        assert result["parsed"]["name"] == "test"

        # 무효한 JSON
        invalid_text = "이것은 JSON이 아닙니다"
        result = JSONValidator.validate_json(invalid_text)
        assert not result["valid"]

    def test_validate_rag_response_helper(self):
        """RAG 응답 검증 헬퍼"""
        from src.rag.output_validator import validate_rag_response

        result = validate_rag_response(
            answer="ETF는 거래소에서 거래되는 펀드입니다. 분산 투자에 유리합니다.",
            context_documents=["ETF는 Exchange Traded Fund입니다."],
            question="ETF란 무엇인가요?",
            sources=[{"relevance_score": 0.8}],
            base_confidence="medium"
        )

        assert "is_valid" in result
        assert "calibrated_confidence" in result
        assert "validation_score" in result


# ============================================================
# LLM 프로바이더 테스트
# ============================================================

class TestLLMProvider:
    """LLM 프로바이더 테스트"""

    def test_import_llm_provider(self):
        """LLM 프로바이더 모듈 임포트"""
        from src.rag.llm_provider import (
            BaseLLMProvider,
            GroqProvider,
            OllamaProvider,
            OpenAIProvider,
            AnthropicProvider,
            LLMConfig,
            LLMProviderRegistry,
            ProviderType,
        )
        assert BaseLLMProvider is not None
        assert LLMProviderRegistry is not None

    def test_llm_config_defaults(self):
        """LLM 설정 기본값"""
        from src.rag.llm_provider import LLMConfig

        config = LLMConfig()

        assert config.provider == "groq"
        assert config.temperature == 0.2
        assert config.timeout == 60
        assert config.retry_count == 3

    def test_llm_config_with_options(self):
        """LLM 설정 옵션"""
        from src.rag.llm_provider import LLMConfig, ModelCapabilities

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.5,
            max_tokens=2048,
            capabilities=ModelCapabilities(
                streaming=True,
                vision=True,
                context_window=200000
            )
        )

        assert config.provider == "anthropic"
        assert config.max_tokens == 2048
        assert config.capabilities.vision is True

    def test_provider_registry_list(self):
        """프로바이더 레지스트리 목록"""
        from src.rag.llm_provider import LLMProviderRegistry

        providers = LLMProviderRegistry.list()

        assert "groq" in providers
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_provider_registry_get(self):
        """프로바이더 클래스 조회"""
        from src.rag.llm_provider import LLMProviderRegistry, GroqProvider

        provider_class = LLMProviderRegistry.get("groq")

        assert provider_class == GroqProvider

    def test_provider_env_keys(self):
        """프로바이더 환경변수 키"""
        from src.rag.llm_provider import LLMProviderRegistry

        assert LLMProviderRegistry.get_env_key("groq") == "GROQ_API_KEY"
        assert LLMProviderRegistry.get_env_key("openai") == "OPENAI_API_KEY"
        assert LLMProviderRegistry.get_env_key("anthropic") == "ANTHROPIC_API_KEY"

    @patch.dict('os.environ', {}, clear=True)
    def test_get_llm_provider_fallback(self):
        """프로바이더 fallback"""
        from src.rag.llm_provider import get_llm_provider, LLMConfig, OllamaProvider

        # API 키 없으면 Ollama로 fallback
        config = LLMConfig(provider="auto")

        # Ollama import 실패해도 에러만 발생
        try:
            provider = get_llm_provider(config)
            assert isinstance(provider, OllamaProvider)
        except ImportError:
            # Ollama가 설치되지 않은 환경에서는 pass
            pass

    def test_provider_type_enum(self):
        """프로바이더 타입 열거형"""
        from src.rag.llm_provider import ProviderType

        assert ProviderType.GROQ.value == "groq"
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"


# ============================================================
# 통합 테스트
# ============================================================

class TestLLMEnhancementIntegration:
    """LLM 고도화 통합 테스트"""

    def test_root_endpoint_version(self):
        """루트 엔드포인트 버전 확인"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "2.5.0"

    def test_prompt_with_token_counting(self):
        """프롬프트 + 토큰 카운팅 통합"""
        from src.rag.prompts import get_rag_prompt
        from src.rag.token_counter import TokenCounter

        prompt = get_rag_prompt(
            context="ETF는 거래소에서 거래되는 펀드입니다.",
            question="ETF란?"
        )

        counter = TokenCounter("llama-3.1-8b-instant")
        system_tokens = counter.count_tokens(prompt["system"])
        user_tokens = counter.count_tokens(prompt["user"])

        total_tokens = system_tokens + user_tokens
        assert total_tokens < counter.effective_context

    def test_full_validation_pipeline(self):
        """전체 검증 파이프라인"""
        from src.rag.prompts import get_rag_prompt
        from src.rag.token_counter import ContextManager
        from src.rag.output_validator import validate_rag_response

        # 1. 컨텍스트 관리
        manager = ContextManager("llama-3.1-8b-instant")
        documents = ["ETF는 펀드입니다.", "주식은 기업 소유권입니다."]
        question = "ETF란?"

        fitted_docs, _ = manager.fit_context(documents, question)

        # 2. 프롬프트 생성
        prompt = get_rag_prompt(
            context="\n".join(fitted_docs),
            question=question
        )

        # 3. (Mock) 답변 생성
        mock_answer = "ETF는 거래소에서 거래되는 펀드로, 분산 투자에 유리합니다."

        # 4. 응답 검증
        validation = validate_rag_response(
            answer=mock_answer,
            context_documents=fitted_docs,
            question=question,
            sources=[{"relevance_score": 0.8}],
            base_confidence="medium"
        )

        assert validation["is_valid"]
        assert validation["calibrated_confidence"] in ["high", "medium", "low"]


# ============================================================
# API 엔드포인트 테스트 (향후 추가 시)
# ============================================================

class TestLLMEnhancementAPI:
    """LLM 고도화 API 테스트"""

    def test_health_endpoint(self):
        """헬스체크 엔드포인트"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_stats_endpoint(self):
        """통계 엔드포인트 (LLM 연결 필요)"""
        import os

        # LLM API 키가 없으면 스킵
        if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            pytest.skip("LLM API key not available")

        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "llm_model" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
