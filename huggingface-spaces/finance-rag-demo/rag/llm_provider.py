# -*- coding: utf-8 -*-
"""
LLM Provider Module

Groq API (REST) + HuggingFace Inference API (Fallback)
groq 패키지 불필요 - 순수 REST API 사용
"""

import requests
import json
from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class BaseLLMProvider(ABC):
    """LLM Provider 추상 클래스"""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """동기 생성"""
        pass

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> Generator[str, None, None]:
        """스트리밍 생성"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 이름"""
        pass


class GroqProvider(BaseLLMProvider):
    """
    Groq API Provider (REST API 방식)

    - 무료 tier: 분당 30 요청, 일일 14,400 요청
    - 빠른 응답 속도 (LPU)
    - Llama 3.1 지원
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # 사용 가능한 모델
    MODELS = {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = self.MODELS.get(model, model)
        self.temperature = temperature
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """동기 생성"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage")

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groq API 요청 실패: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Groq API 응답 파싱 실패: {e}")

    def generate_stream(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> Generator[str, None, None]:
        """스트리밍 생성"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groq API 스트리밍 실패: {e}")


class HuggingFaceProvider(BaseLLMProvider):
    """
    HuggingFace Inference API Provider (Fallback)

    - 무료 사용 가능
    - 다양한 모델 지원
    - 속도 느림 (cold start)
    """

    API_URL = "https://api-inference.huggingface.co/models/"

    # 추천 모델
    MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "phi-2": "microsoft/phi-2",
    }

    def __init__(
        self,
        token: str = "",
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.7,
        timeout: int = 120
    ):
        self.token = token
        self.model = self.MODELS.get(model, model)
        self.temperature = temperature
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"HuggingFace ({self.model.split('/')[-1]})"

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """모델별 프롬프트 포맷팅"""
        if "mistral" in self.model.lower():
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        elif "zephyr" in self.model.lower():
            return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
        else:
            return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> LLMResponse:
        """동기 생성"""
        formatted_prompt = self._format_prompt(system_prompt, user_prompt)

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": self.temperature,
                "do_sample": True,
                "return_full_text": False
            },
            "options": {"wait_for_model": True}
        }

        try:
            response = requests.post(
                self.API_URL + self.model,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                content = data[0].get("generated_text", "").strip()
            else:
                content = ""

            return LLMResponse(
                content=content,
                model=self.model,
                usage=None
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HuggingFace API 요청 실패: {e}")

    def generate_stream(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Generator[str, None, None]:
        """스트리밍 생성 (HF는 스트리밍 미지원, 전체 응답 반환)"""
        response = self.generate(system_prompt, user_prompt, max_tokens)
        # HuggingFace Inference API는 스트리밍 미지원, 전체 텍스트를 한번에 yield
        yield response.content


def get_llm_provider(
    groq_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2
) -> BaseLLMProvider:
    """
    LLM Provider 팩토리

    우선순위:
    1. Groq API (빠름, 무료 tier)
    2. HuggingFace (느림, 무료)
    """
    if groq_api_key:
        return GroqProvider(
            api_key=groq_api_key,
            model=model,
            temperature=temperature
        )
    else:
        return HuggingFaceProvider(
            token=hf_token or "",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.7
        )
