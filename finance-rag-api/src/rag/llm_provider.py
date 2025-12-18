# -*- coding: utf-8 -*-
"""
LLM Provider - 다중 LLM 지원 모듈

로컬(Ollama)과 클라우드(Groq) LLM을 통합 인터페이스로 제공합니다.
"""

import os
from abc import ABC, abstractmethod
from typing import Generator, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 설정"""
    provider: str = "groq"  # "ollama" or "groq"
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    api_key: Optional[str] = None


class BaseLLMProvider(ABC):
    """LLM Provider 추상 클래스"""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """동기 생성"""
        pass

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """스트리밍 생성"""
        pass


class GroqProvider(BaseLLMProvider):
    """
    Groq API Provider

    - 무료 tier 제공
    - 빠른 응답 속도
    - Llama 3.1 지원
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2):
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq 패키지가 필요합니다: pip install groq")

        self.model = model
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(BaseLLMProvider):
    """
    Ollama Provider (로컬 LLM)
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.2):
        try:
            import ollama
            self.ollama = ollama
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


def get_llm_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """
    LLM Provider 팩토리

    환경변수 또는 config로 provider 선택:
    - GROQ_API_KEY가 있으면 Groq 사용
    - 없으면 Ollama 사용 (로컬)
    """
    if config is None:
        config = LLMConfig()

    # API 키 확인 (환경변수 우선)
    api_key = config.api_key or os.getenv("GROQ_API_KEY")

    if api_key:
        return GroqProvider(
            api_key=api_key,
            model=config.model,
            temperature=config.temperature
        )
    else:
        # 로컬 Ollama fallback
        return OllamaProvider(
            model="llama3.2",
            temperature=config.temperature
        )
