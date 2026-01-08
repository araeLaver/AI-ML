# LLM API Client Module
"""
LLM API 클라이언트 (Step 2: OpenAI/Claude API 활용)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스"""
    content: str
    model: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMClient(ABC):
    """LLM 클라이언트 추상 클래스"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """채팅 완성 요청"""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """스트리밍 채팅 완성 요청"""
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI API 클라이언트

    Usage:
        client = OpenAIClient()
        response = client.chat([
            {"role": "user", "content": "안녕하세요"}
        ])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai 패키지가 필요합니다: pip install openai")
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        채팅 완성 요청

        Args:
            messages: 메시지 리스트
            system_prompt: 시스템 프롬프트
            tools: Function Calling 도구
            temperature: 창의성 조절
            max_tokens: 최대 토큰 수
        """
        client = self._get_client()

        # 시스템 프롬프트 추가
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        # API 호출
        params = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = client.chat.completions.create(**params)

        # 응답 파싱
        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            tool_calls=tool_calls,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """스트리밍 채팅 완성"""
        client = self._get_client()

        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        stream = client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ClaudeClient(LLMClient):
    """
    Anthropic Claude API 클라이언트

    Usage:
        client = ClaudeClient()
        response = client.chat([
            {"role": "user", "content": "안녕하세요"}
        ])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Anthropic 클라이언트 지연 초기화"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic")
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """채팅 완성 요청"""
        client = self._get_client()

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if system_prompt:
            params["system"] = system_prompt

        if temperature is not None:
            params["temperature"] = temperature

        if tools:
            # Claude 도구 형식으로 변환
            params["tools"] = self._convert_tools(tools)

        response = client.messages.create(**params)

        # 응답 파싱
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    }
                })

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            tool_calls=tool_calls if tool_calls else None,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """스트리밍 채팅 완성"""
        client = self._get_client()

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if system_prompt:
            params["system"] = system_prompt

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text

    def _convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAI 형식 도구를 Claude 형식으로 변환"""
        claude_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                claude_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
        return claude_tools


class MockLLMClient(LLMClient):
    """
    테스트용 Mock LLM 클라이언트
    API 키 없이 테스트 가능
    """

    def __init__(self, model: str = "mock-model"):
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Mock 응답 생성"""
        last_message = messages[-1]["content"] if messages else ""

        # 간단한 응답 생성
        if "이상거래" in last_message or "fraud" in last_message.lower():
            content = """**이상거래 분석 결과**

위험 수준: 높음

**분석 내용:**
1. 거래 금액이 평소 패턴과 크게 다릅니다
2. 거래 시간대가 비정상적입니다
3. 거래 위치가 평소와 다릅니다

**권장 조치:**
- 본인 확인 절차 진행
- 거래 보류 검토
- 추가 모니터링 필요"""
        else:
            content = f"Mock 응답: {last_message[:50]}..."

        return LLMResponse(
            content=content,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """Mock 스트리밍 응답"""
        response = self.chat(messages, **kwargs)
        for word in response.content.split():
            yield word + " "


def create_llm_client(provider: str = "auto", **kwargs) -> LLMClient:
    """
    LLM 클라이언트 팩토리 함수

    Args:
        provider: "openai", "claude", "mock", "auto"
        **kwargs: 클라이언트 설정

    Returns:
        LLMClient 인스턴스
    """
    if provider == "auto":
        # API 키 확인하여 자동 선택
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "claude"
        else:
            provider = "mock"

    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "claude":
        return ClaudeClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # 테스트
    print("Testing LLM Client...")

    # Mock 클라이언트 테스트
    client = create_llm_client("mock")
    response = client.chat([
        {"role": "user", "content": "이 거래가 이상거래인지 분석해주세요."}
    ])

    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")
    print(f"Response:\n{response.content}")
