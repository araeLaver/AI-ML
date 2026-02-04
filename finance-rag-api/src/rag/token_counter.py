# -*- coding: utf-8 -*-
"""
토큰 카운팅 및 컨텍스트 관리 모듈 (Phase 16)

다양한 모델의 토큰 수 추정 및 컨텍스트 윈도우 관리
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re


class ModelFamily(Enum):
    """모델 패밀리"""
    LLAMA = "llama"
    GPT = "gpt"
    CLAUDE = "claude"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """모델 설정"""
    name: str
    family: ModelFamily
    context_window: int  # 최대 컨텍스트 토큰 수
    output_limit: int    # 최대 출력 토큰 수
    chars_per_token: float = 4.0  # 평균 문자/토큰 비율 (추정용)

    @property
    def effective_context(self) -> int:
        """출력을 위한 여유를 뺀 실제 사용 가능 컨텍스트"""
        return self.context_window - self.output_limit


# 지원 모델 레지스트리
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # Llama 모델
    "llama-3.1-8b-instant": ModelConfig(
        name="llama-3.1-8b-instant",
        family=ModelFamily.LLAMA,
        context_window=8192,
        output_limit=2048,
        chars_per_token=3.5
    ),
    "llama-3.1-70b-versatile": ModelConfig(
        name="llama-3.1-70b-versatile",
        family=ModelFamily.LLAMA,
        context_window=32768,
        output_limit=4096,
        chars_per_token=3.5
    ),
    "llama3.2": ModelConfig(
        name="llama3.2",
        family=ModelFamily.LLAMA,
        context_window=8192,
        output_limit=2048,
        chars_per_token=3.5
    ),

    # GPT 모델
    "gpt-4": ModelConfig(
        name="gpt-4",
        family=ModelFamily.GPT,
        context_window=8192,
        output_limit=4096,
        chars_per_token=4.0
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        family=ModelFamily.GPT,
        context_window=128000,
        output_limit=4096,
        chars_per_token=4.0
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        family=ModelFamily.GPT,
        context_window=16385,
        output_limit=4096,
        chars_per_token=4.0
    ),

    # Claude 모델
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        family=ModelFamily.CLAUDE,
        context_window=200000,
        output_limit=4096,
        chars_per_token=3.5
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet",
        family=ModelFamily.CLAUDE,
        context_window=200000,
        output_limit=4096,
        chars_per_token=3.5
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku",
        family=ModelFamily.CLAUDE,
        context_window=200000,
        output_limit=4096,
        chars_per_token=3.5
    ),

    # Mistral 모델
    "mistral-7b": ModelConfig(
        name="mistral-7b",
        family=ModelFamily.MISTRAL,
        context_window=32768,
        output_limit=4096,
        chars_per_token=4.0
    ),
    "mixtral-8x7b": ModelConfig(
        name="mixtral-8x7b",
        family=ModelFamily.MISTRAL,
        context_window=32768,
        output_limit=4096,
        chars_per_token=4.0
    ),
}


class TokenCounter:
    """토큰 카운터"""

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_config = self._get_model_config(model_name)
        self._tokenizer = None

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """모델 설정 조회"""
        # 정확한 매칭
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name]

        # 부분 매칭
        for key, config in MODEL_CONFIGS.items():
            if key in model_name.lower() or model_name.lower() in key:
                return config

        # 기본값
        return ModelConfig(
            name=model_name,
            family=ModelFamily.UNKNOWN,
            context_window=8192,
            output_limit=2048,
            chars_per_token=4.0
        )

    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 추정

        정확한 토큰화 없이 휴리스틱으로 추정:
        - 영어: ~4 chars/token
        - 한국어: ~2-3 chars/token (유니코드 특성)
        """
        if not text:
            return 0

        # 한글 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(text)
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0

        # 한글 비율에 따른 chars_per_token 조정
        # 한글은 토큰당 문자 수가 적음 (더 많은 토큰 사용)
        if korean_ratio > 0.5:
            chars_per_token = 2.5  # 한글 위주
        elif korean_ratio > 0.2:
            chars_per_token = 3.0  # 혼합
        else:
            chars_per_token = self.model_config.chars_per_token  # 영어 위주

        return int(total_chars / chars_per_token)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """메시지 목록의 총 토큰 수"""
        total = 0
        for msg in messages:
            # 역할 토큰 오버헤드 (~4 tokens)
            total += 4
            total += self.count_tokens(msg.get("content", ""))
        return total

    @property
    def context_window(self) -> int:
        """전체 컨텍스트 윈도우 크기"""
        return self.model_config.context_window

    @property
    def effective_context(self) -> int:
        """사용 가능한 컨텍스트 크기 (출력 여유 제외)"""
        return self.model_config.effective_context

    @property
    def output_limit(self) -> int:
        """최대 출력 토큰 수"""
        return self.model_config.output_limit


class ContextManager:
    """컨텍스트 윈도우 관리자"""

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        reserve_for_output: int = 1024,
        reserve_for_system: int = 500
    ):
        self.token_counter = TokenCounter(model_name)
        self.reserve_for_output = reserve_for_output
        self.reserve_for_system = reserve_for_system

    @property
    def max_context_tokens(self) -> int:
        """컨텍스트에 사용 가능한 최대 토큰 수"""
        return (
            self.token_counter.context_window
            - self.reserve_for_output
            - self.reserve_for_system
        )

    def fit_context(
        self,
        documents: List[str],
        question: str,
        strategy: str = "truncate_last"
    ) -> Tuple[List[str], int]:
        """
        컨텍스트 윈도우에 맞게 문서 조정

        Args:
            documents: 문서 목록 (관련도 순으로 정렬됨)
            question: 사용자 질문
            strategy: 조정 전략
                - truncate_last: 마지막 문서부터 자르기
                - truncate_all: 모든 문서 균등하게 자르기
                - drop_last: 마지막 문서 완전히 제거

        Returns:
            (조정된 문서 목록, 사용된 토큰 수)
        """
        question_tokens = self.token_counter.count_tokens(question)
        available_tokens = self.max_context_tokens - question_tokens

        if available_tokens <= 0:
            return [], 0

        if strategy == "truncate_last":
            return self._truncate_last(documents, available_tokens)
        elif strategy == "truncate_all":
            return self._truncate_all(documents, available_tokens)
        elif strategy == "drop_last":
            return self._drop_last(documents, available_tokens)
        else:
            return self._truncate_last(documents, available_tokens)

    def _truncate_last(
        self,
        documents: List[str],
        available_tokens: int
    ) -> Tuple[List[str], int]:
        """마지막 문서부터 자르기"""
        result = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = self.token_counter.count_tokens(doc)

            if used_tokens + doc_tokens <= available_tokens:
                result.append(doc)
                used_tokens += doc_tokens
            else:
                # 남은 토큰으로 일부만 추가
                remaining = available_tokens - used_tokens
                if remaining > 100:  # 최소 100 토큰 이상일 때만
                    # 토큰을 문자로 변환하여 자르기
                    chars_to_keep = int(remaining * self.token_counter.model_config.chars_per_token)
                    truncated = doc[:chars_to_keep] + "..."
                    result.append(truncated)
                    used_tokens += remaining
                break

        return result, used_tokens

    def _truncate_all(
        self,
        documents: List[str],
        available_tokens: int
    ) -> Tuple[List[str], int]:
        """모든 문서 균등하게 자르기"""
        if not documents:
            return [], 0

        total_tokens = sum(self.token_counter.count_tokens(d) for d in documents)

        if total_tokens <= available_tokens:
            return documents, total_tokens

        # 비율 계산
        ratio = available_tokens / total_tokens
        result = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = self.token_counter.count_tokens(doc)
            target_tokens = int(doc_tokens * ratio)

            if target_tokens > 50:  # 최소 50 토큰
                chars_to_keep = int(target_tokens * self.token_counter.model_config.chars_per_token)
                truncated = doc[:chars_to_keep] + "..."
                result.append(truncated)
                used_tokens += target_tokens

        return result, used_tokens

    def _drop_last(
        self,
        documents: List[str],
        available_tokens: int
    ) -> Tuple[List[str], int]:
        """마지막 문서 완전히 제거"""
        result = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = self.token_counter.count_tokens(doc)

            if used_tokens + doc_tokens <= available_tokens:
                result.append(doc)
                used_tokens += doc_tokens
            else:
                break

        return result, used_tokens

    def estimate_response_quality(
        self,
        context_tokens: int,
        question_tokens: int
    ) -> Dict[str, Any]:
        """응답 품질 예측"""
        total_input = context_tokens + question_tokens + self.reserve_for_system
        utilization = total_input / self.token_counter.context_window

        if utilization < 0.3:
            quality = "optimal"
            recommendation = "충분한 여유가 있습니다"
        elif utilization < 0.6:
            quality = "good"
            recommendation = "적절한 수준입니다"
        elif utilization < 0.8:
            quality = "moderate"
            recommendation = "컨텍스트가 많습니다. 핵심 문서만 사용을 권장합니다"
        else:
            quality = "limited"
            recommendation = "컨텍스트가 너무 많습니다. 문서 수를 줄이세요"

        return {
            "quality": quality,
            "utilization": round(utilization * 100, 1),
            "input_tokens": total_input,
            "remaining_for_output": self.token_counter.context_window - total_input,
            "recommendation": recommendation
        }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """모델 정보 조회"""
    counter = TokenCounter(model_name)
    config = counter.model_config

    return {
        "name": config.name,
        "family": config.family.value,
        "context_window": config.context_window,
        "output_limit": config.output_limit,
        "effective_context": config.effective_context,
        "chars_per_token": config.chars_per_token
    }


def list_supported_models() -> List[Dict[str, Any]]:
    """지원 모델 목록"""
    return [
        {
            "name": config.name,
            "family": config.family.value,
            "context_window": config.context_window
        }
        for config in MODEL_CONFIGS.values()
    ]
