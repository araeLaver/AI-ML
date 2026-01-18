# -*- coding: utf-8 -*-
"""
Korean Tokenizer Module

HuggingFace Spaces 환경에 최적화된 경량 토크나이저
Kiwi 대신 2-gram 기반 (메모리 절약)
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseTokenizer(ABC):
    """토크나이저 추상 클래스"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """토크나이저 이름"""
        pass


class SimpleTokenizer(BaseTokenizer):
    """
    간단한 한국어 토크나이저 (2-gram)

    Kiwi 없이도 동작하는 경량 토크나이저
    - 공백 분리
    - 한글 2-gram 생성
    - 영문/숫자 단어 유지

    [특징]
    - 메모리 효율적 (추가 모델 불필요)
    - HuggingFace Spaces 2GB 제한 대응
    - 금융 용어 특화 처리
    """

    # 금융 도메인 중요 키워드 (분리하지 않음)
    FINANCIAL_KEYWORDS = {
        "삼성전자", "SK하이닉스", "LG에너지솔루션", "현대자동차",
        "네이버", "카카오", "삼성바이오로직스", "POSCO홀딩스",
        "영업이익", "순이익", "매출액", "시가총액",
        "PER", "PBR", "ROE", "EPS", "HBM", "HBM3E", "HBM4",
        "반도체", "파운드리", "AI반도체", "GPU",
        "기준금리", "금리인하", "배당수익률",
        "비트코인", "이더리움", "ETF",
    }

    def __init__(self, min_length: int = 2):
        """
        Args:
            min_length: 최소 토큰 길이 (기본 2)
        """
        self.min_length = min_length
        # 키워드를 소문자로 변환한 세트
        self._keywords_lower = {kw.lower() for kw in self.FINANCIAL_KEYWORDS}

    def tokenize(self, text: str) -> List[str]:
        """
        텍스트 토큰화

        1. 소문자 변환
        2. 특수문자 제거
        3. 공백 분리
        4. 한글 2-gram 생성
        5. 중요 키워드 보존
        """
        if not text:
            return []

        # 소문자 변환
        text_lower = text.lower()

        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        text_clean = re.sub(r'[^\w\s가-힣]', ' ', text_lower)

        # 공백 기준 분리
        words = text_clean.split()

        tokens = []
        seen = set()

        for word in words:
            if not word or len(word) < self.min_length:
                continue

            # 중요 키워드면 그대로 추가
            if word in self._keywords_lower:
                if word not in seen:
                    tokens.append(word)
                    seen.add(word)
                continue

            # 한글 단어: 2-gram + 원본
            if re.match(r'^[가-힣]+$', word):
                # 원본 단어 추가
                if word not in seen:
                    tokens.append(word)
                    seen.add(word)

                # 2-gram 생성 (길이 3 이상일 때)
                if len(word) >= 3:
                    for i in range(len(word) - 1):
                        gram = word[i:i+2]
                        if gram not in seen:
                            tokens.append(gram)
                            seen.add(gram)
            else:
                # 영문/숫자: 그대로 추가
                if word not in seen:
                    tokens.append(word)
                    seen.add(word)

        return tokens

    def get_name(self) -> str:
        return "Simple (2-gram)"


# 싱글톤 인스턴스
_tokenizer: Optional[BaseTokenizer] = None


def get_tokenizer() -> BaseTokenizer:
    """토크나이저 싱글톤 반환"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = SimpleTokenizer()
    return _tokenizer


def reset_tokenizer():
    """토크나이저 리셋 (테스트용)"""
    global _tokenizer
    _tokenizer = None
