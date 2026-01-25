# -*- coding: utf-8 -*-
"""
언어 감지 모듈

[기능]
- 문자 기반 언어 감지
- 통계 기반 언어 감지
- 다중 언어 혼합 감지
"""

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Language(Enum):
    """지원 언어"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    UNKNOWN = "unknown"

    @classmethod
    def from_code(cls, code: str) -> "Language":
        """코드로 언어 조회"""
        code = code.lower()
        for lang in cls:
            if lang.value == code:
                return lang
        return cls.UNKNOWN

    @property
    def name_native(self) -> str:
        """원어 이름"""
        names = {
            Language.KOREAN: "한국어",
            Language.ENGLISH: "English",
            Language.JAPANESE: "日本語",
            Language.CHINESE: "中文",
        }
        return names.get(self, "Unknown")


@dataclass
class DetectionResult:
    """언어 감지 결과"""
    language: Language
    confidence: float  # 0-1
    is_mixed: bool = False  # 여러 언어 혼합 여부
    language_ratios: Dict[Language, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language.value,
            "language_name": self.language.name_native,
            "confidence": round(self.confidence, 4),
            "is_mixed": self.is_mixed,
            "language_ratios": {
                k.value: round(v, 4)
                for k, v in self.language_ratios.items()
            },
        }


class LanguageDetector(ABC):
    """언어 감지기 기본 클래스"""

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """언어 감지"""
        pass

    @abstractmethod
    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """배치 감지"""
        pass


class CharacterBasedDetector(LanguageDetector):
    """
    문자 기반 언어 감지기

    유니코드 문자 범위를 기반으로 언어 감지
    """

    # 문자 범위 정의
    UNICODE_RANGES = {
        Language.KOREAN: [
            (0xAC00, 0xD7AF),  # 한글 음절
            (0x1100, 0x11FF),  # 한글 자모
            (0x3130, 0x318F),  # 호환용 한글 자모
        ],
        Language.JAPANESE: [
            (0x3040, 0x309F),  # 히라가나
            (0x30A0, 0x30FF),  # 가타카나
            (0x31F0, 0x31FF),  # 가타카나 확장
        ],
        Language.CHINESE: [
            (0x4E00, 0x9FFF),  # CJK 통합 한자
            (0x3400, 0x4DBF),  # CJK 확장 A
        ],
    }

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def detect(self, text: str) -> DetectionResult:
        """언어 감지"""
        if not text or not text.strip():
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
            )

        # 문자별 언어 카운트
        lang_counts: Dict[Language, int] = {lang: 0 for lang in Language}
        total_chars = 0

        for char in text:
            if char.isspace() or not char.isalpha():
                continue

            total_chars += 1
            char_lang = self._detect_char_language(char)
            lang_counts[char_lang] += 1

        if total_chars == 0:
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
            )

        # 비율 계산
        ratios = {
            lang: count / total_chars
            for lang, count in lang_counts.items()
            if count > 0
        }

        # 주요 언어 결정
        if not ratios:
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
            )

        primary_lang = max(ratios, key=ratios.get)
        confidence = ratios[primary_lang]

        # 혼합 언어 확인
        is_mixed = len([r for r in ratios.values() if r > 0.2]) > 1

        return DetectionResult(
            language=primary_lang,
            confidence=confidence,
            is_mixed=is_mixed,
            language_ratios=ratios,
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """배치 감지"""
        return [self.detect(text) for text in texts]

    def _detect_char_language(self, char: str) -> Language:
        """단일 문자의 언어 감지"""
        code_point = ord(char)

        for lang, ranges in self.UNICODE_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    # CJK 한자는 문맥에 따라 다르지만, 기본적으로 Chinese로 분류
                    if lang == Language.CHINESE:
                        # 하지만 한글/일본어와 함께 나타나면 해당 언어로 분류
                        pass
                    return lang

        # 라틴 문자는 영어로 분류
        if char.isascii() and char.isalpha():
            return Language.ENGLISH

        return Language.UNKNOWN

    def get_script_info(self, text: str) -> Dict[str, Any]:
        """스크립트 정보 반환"""
        scripts = Counter()

        for char in text:
            if char.isspace():
                continue

            try:
                script = unicodedata.name(char).split()[0]
                scripts[script] += 1
            except ValueError:
                scripts["UNKNOWN"] += 1

        total = sum(scripts.values())
        return {
            "scripts": dict(scripts),
            "ratios": {k: v / total for k, v in scripts.items()} if total > 0 else {},
        }


class StatisticalDetector(LanguageDetector):
    """
    통계 기반 언어 감지기

    N-gram 빈도를 기반으로 언어 감지
    """

    # 언어별 특징적인 N-gram
    LANGUAGE_PATTERNS = {
        Language.KOREAN: [
            "은", "는", "이", "가", "을", "를", "의", "에", "로",
            "다", "니다", "습니다", "입니다", "하다", "있다",
            "그", "그리고", "하지만", "때문", "대해",
        ],
        Language.ENGLISH: [
            "the", "is", "are", "was", "were", "have", "has",
            "ing", "tion", "and", "of", "to", "in", "for",
            "that", "this", "with", "from", "about",
        ],
        Language.JAPANESE: [
            "の", "は", "が", "を", "に", "で", "と", "も",
            "です", "ます", "した", "ない", "ある", "いる",
            "これ", "それ", "あれ", "どれ",
        ],
        Language.CHINESE: [
            "的", "是", "在", "有", "和", "了", "不", "人",
            "这", "那", "上", "下", "中", "大", "小",
            "可以", "什么", "怎么", "因为", "所以",
        ],
    }

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def detect(self, text: str) -> DetectionResult:
        """언어 감지"""
        if not text or not text.strip():
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
            )

        text_lower = text.lower()

        # 각 언어별 패턴 매칭 점수
        scores = {}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += text_lower.count(pattern)
            scores[lang] = score

        total_score = sum(scores.values())

        if total_score == 0:
            # 문자 기반 폴백
            char_detector = CharacterBasedDetector()
            return char_detector.detect(text)

        # 정규화된 점수
        ratios = {
            lang: score / total_score
            for lang, score in scores.items()
            if score > 0
        }

        primary_lang = max(ratios, key=ratios.get)
        confidence = ratios[primary_lang]

        is_mixed = len([r for r in ratios.values() if r > 0.2]) > 1

        return DetectionResult(
            language=primary_lang,
            confidence=confidence,
            is_mixed=is_mixed,
            language_ratios=ratios,
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """배치 감지"""
        return [self.detect(text) for text in texts]


class HybridDetector(LanguageDetector):
    """
    하이브리드 언어 감지기

    여러 감지 방법을 조합
    """

    def __init__(self):
        self.char_detector = CharacterBasedDetector()
        self.stat_detector = StatisticalDetector()

    def detect(self, text: str) -> DetectionResult:
        """언어 감지"""
        if not text or not text.strip():
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
            )

        # 두 방법으로 감지
        char_result = self.char_detector.detect(text)
        stat_result = self.stat_detector.detect(text)

        # 결과 병합
        combined_ratios: Dict[Language, float] = {}

        for lang in Language:
            char_ratio = char_result.language_ratios.get(lang, 0)
            stat_ratio = stat_result.language_ratios.get(lang, 0)
            # 가중 평균 (문자 기반 60%, 통계 기반 40%)
            combined_ratios[lang] = char_ratio * 0.6 + stat_ratio * 0.4

        # 최종 언어 결정
        if not combined_ratios or max(combined_ratios.values()) == 0:
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
            )

        primary_lang = max(combined_ratios, key=combined_ratios.get)
        confidence = combined_ratios[primary_lang]

        is_mixed = (
            char_result.is_mixed or stat_result.is_mixed or
            len([r for r in combined_ratios.values() if r > 0.15]) > 1
        )

        return DetectionResult(
            language=primary_lang,
            confidence=confidence,
            is_mixed=is_mixed,
            language_ratios={k: v for k, v in combined_ratios.items() if v > 0},
            metadata={
                "char_result": char_result.to_dict(),
                "stat_result": stat_result.to_dict(),
            },
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """배치 감지"""
        return [self.detect(text) for text in texts]


class LanguageRouter:
    """
    언어 라우터

    감지된 언어에 따라 적절한 처리기로 라우팅
    """

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        default_language: Language = Language.KOREAN,
    ):
        self.detector = detector or HybridDetector()
        self.default_language = default_language
        self._handlers: Dict[Language, Any] = {}

    def register_handler(self, language: Language, handler: Any) -> None:
        """언어별 핸들러 등록"""
        self._handlers[language] = handler

    def get_handler(self, text: str) -> Tuple[Any, DetectionResult]:
        """텍스트에 적합한 핸들러 반환"""
        result = self.detector.detect(text)

        # 신뢰도가 낮으면 기본 언어 사용
        if result.confidence < 0.3:
            lang = self.default_language
        else:
            lang = result.language

        handler = self._handlers.get(lang) or self._handlers.get(self.default_language)

        return handler, result

    def route(self, text: str) -> Dict[str, Any]:
        """텍스트 라우팅"""
        handler, detection = self.get_handler(text)

        return {
            "text": text,
            "detected_language": detection.language.value,
            "confidence": detection.confidence,
            "handler": handler,
        }
