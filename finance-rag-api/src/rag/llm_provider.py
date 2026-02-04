# -*- coding: utf-8 -*-
"""
LLM Provider - 다중 LLM 지원 모듈 (Phase 16 고도화)

로컬(Ollama)과 클라우드(Groq, OpenAI, Anthropic) LLM을 통합 인터페이스로 제공합니다.

지원 프로바이더:
- Groq: 빠른 추론, 무료 tier
- Ollama: 로컬 LLM
- OpenAI: GPT 모델 (옵션)
- Anthropic: Claude 모델 (옵션)
"""

import os
from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(Enum):
    """LLM 프로바이더 타입"""
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelCapabilities:
    """모델 기능"""
    streaming: bool = True
    function_calling: bool = False
    vision: bool = False
    json_mode: bool = False
    max_tokens: int = 4096
    context_window: int = 8192


@dataclass
class LLMConfig:
    """LLM 설정"""
    provider: str = "groq"  # "ollama", "groq", "openai", "anthropic"
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    retry_count: int = 3
    fallback_provider: Optional[str] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


class BaseLLMProvider(ABC):
    """LLM Provider 추상 클래스"""

    def __init__(self):
        self.last_usage: Dict[str, int] = {}
        self.total_tokens_used: int = 0

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """동기 생성"""
        pass

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """스트리밍 생성"""
        pass

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """메시지 리스트로 생성 (멀티턴 대화)"""
        # 기본 구현: 마지막 메시지 사용
        if not messages:
            return ""

        user_content = messages[-1].get("content", "")
        return self.generate(system_prompt or "", user_content)

    @property
    def provider_name(self) -> str:
        """프로바이더 이름"""
        return self.__class__.__name__

    def get_usage_stats(self) -> Dict[str, Any]:
        """사용량 통계"""
        return {
            "provider": self.provider_name,
            "last_usage": self.last_usage,
            "total_tokens": self.total_tokens_used
        }


class GroqProvider(BaseLLMProvider):
    """
    Groq API Provider

    - 무료 tier 제공
    - 빠른 응답 속도
    - Llama 3.1 지원
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None
    ):
        super().__init__()
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq 패키지가 필요합니다: pip install groq")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)

        # 사용량 추적
        if hasattr(response, 'usage') and response.usage:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            self.total_tokens_used += response.usage.total_tokens

        return response.choices[0].message.content

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "stream": True
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """멀티턴 대화 지원"""
        msg_list = []
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        msg_list.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": msg_list,
            "temperature": self.temperature
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class OllamaProvider(BaseLLMProvider):
    """
    Ollama Provider (로컬 LLM)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.2,
        base_url: Optional[str] = None
    ):
        super().__init__()
        try:
            import ollama
            self.ollama = ollama
            if base_url:
                self.ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("ollama 패키지가 필요합니다: pip install ollama")

        self.model = model
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": self.temperature}
        )
        return response.message.content

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        stream = self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": self.temperature},
            stream=True
        )

        for chunk in stream:
            if chunk.message.content:
                yield chunk.message.content

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """멀티턴 대화 지원"""
        msg_list = []
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        msg_list.extend(messages)

        response = self.ollama.chat(
            model=self.model,
            messages=msg_list,
            options={"temperature": self.temperature}
        )
        return response.message.content


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API Provider

    - GPT-4, GPT-3.5 지원
    - 함수 호출 지원
    - JSON 모드 지원
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None
    ):
        super().__init__()
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)

        # 사용량 추적
        if response.usage:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            self.total_tokens_used += response.usage.total_tokens

        return response.choices[0].message.content

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "stream": True
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude Provider

    - Claude 3 모델 지원
    - 긴 컨텍스트 (200K)
    - 고품질 추론
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.2,
        max_tokens: int = 4096
    ):
        super().__init__()
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )

        # 사용량 추적
        if hasattr(response, 'usage'):
            self.last_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            self.total_tokens_used += self.last_usage["total_tokens"]

        return response.content[0].text

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        ) as stream:
            for text in stream.text_stream:
                yield text

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """멀티턴 대화 지원"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt or "",
            messages=messages,
            temperature=self.temperature
        )
        return response.content[0].text


class LLMProviderRegistry:
    """LLM 프로바이더 레지스트리"""

    _providers: Dict[str, type] = {
        "groq": GroqProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    _env_keys: Dict[str, str] = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        """프로바이더 등록"""
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """프로바이더 클래스 조회"""
        return cls._providers.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """등록된 프로바이더 목록"""
        return list(cls._providers.keys())

    @classmethod
    def get_env_key(cls, name: str) -> Optional[str]:
        """프로바이더의 환경변수 키"""
        return cls._env_keys.get(name)

    @classmethod
    def detect_available(cls) -> List[str]:
        """사용 가능한 프로바이더 감지"""
        available = ["ollama"]  # Ollama는 항상 사용 가능 (로컬)

        for provider, env_key in cls._env_keys.items():
            if os.getenv(env_key):
                available.append(provider)

        return available


def get_llm_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """
    LLM Provider 팩토리

    우선순위:
    1. config에 명시된 provider
    2. 환경변수로 감지된 provider (GROQ > OPENAI > ANTHROPIC)
    3. 로컬 Ollama (fallback)

    Args:
        config: LLM 설정 (선택)

    Returns:
        BaseLLMProvider 인스턴스
    """
    if config is None:
        config = LLMConfig()

    # 명시적 프로바이더 지정
    if config.provider and config.provider != "auto":
        return _create_provider(config.provider, config)

    # 자동 감지
    if os.getenv("GROQ_API_KEY"):
        return _create_provider("groq", config)
    elif os.getenv("OPENAI_API_KEY"):
        return _create_provider("openai", config)
    elif os.getenv("ANTHROPIC_API_KEY"):
        return _create_provider("anthropic", config)
    else:
        # 로컬 Ollama fallback
        return _create_provider("ollama", config)


def _create_provider(provider_name: str, config: LLMConfig) -> BaseLLMProvider:
    """프로바이더 인스턴스 생성"""
    if provider_name == "groq":
        api_key = config.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key required")
        return GroqProvider(
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    elif provider_name == "openai":
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        return OpenAIProvider(
            api_key=api_key,
            model=config.model if "gpt" in config.model.lower() else "gpt-4-turbo-preview",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    elif provider_name == "anthropic":
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required")
        return AnthropicProvider(
            api_key=api_key,
            model=config.model if "claude" in config.model.lower() else "claude-3-sonnet-20240229",
            temperature=config.temperature,
            max_tokens=config.max_tokens or 4096
        )

    elif provider_name == "ollama":
        return OllamaProvider(
            model=config.model if config.model != "llama-3.1-8b-instant" else "llama3.2",
            temperature=config.temperature,
            base_url=config.base_url
        )

    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def get_provider_with_fallback(
    primary_config: LLMConfig,
    fallback_config: Optional[LLMConfig] = None
) -> BaseLLMProvider:
    """
    Fallback을 지원하는 프로바이더 생성

    Args:
        primary_config: 주 프로바이더 설정
        fallback_config: 대체 프로바이더 설정

    Returns:
        프로바이더 인스턴스 (실패 시 fallback으로 자동 전환)
    """
    try:
        return get_llm_provider(primary_config)
    except Exception as e:
        if fallback_config:
            return get_llm_provider(fallback_config)
        else:
            # 기본 fallback: Ollama
            return OllamaProvider(model="llama3.2", temperature=0.2)
