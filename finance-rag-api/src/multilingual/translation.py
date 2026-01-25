# -*- coding: utf-8 -*-
"""
번역 서비스 모듈

[기능]
- 다국어 번역
- 번역 캐싱
- 번역 파이프라인
"""

import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detection import Language, LanguageDetector, HybridDetector

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """번역 결과"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_text: str = ""
    translated_text: str = ""
    source_language: Language = Language.UNKNOWN
    target_language: Language = Language.UNKNOWN
    confidence: float = 1.0
    is_cached: bool = False
    translation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_text": self.source_text[:100] if self.source_text else None,
            "translated_text": self.translated_text[:100] if self.translated_text else None,
            "source_language": self.source_language.value,
            "target_language": self.target_language.value,
            "confidence": round(self.confidence, 4),
            "is_cached": self.is_cached,
            "translation_time_ms": round(self.translation_time_ms, 2),
        }


class Translator(ABC):
    """번역기 기본 클래스"""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> TranslationResult:
        """번역"""
        pass

    @abstractmethod
    def supports(self, source_lang: Language, target_lang: Language) -> bool:
        """지원 언어쌍 확인"""
        pass


class DictionaryTranslator(Translator):
    """
    사전 기반 번역기

    간단한 단어/구문 매핑
    """

    # 기본 번역 사전 (예시)
    DICTIONARIES = {
        (Language.KOREAN, Language.ENGLISH): {
            "삼성전자": "Samsung Electronics",
            "주가": "stock price",
            "매출": "revenue",
            "영업이익": "operating profit",
            "분기": "quarter",
            "전년대비": "year-over-year",
            "증가": "increase",
            "감소": "decrease",
        },
        (Language.ENGLISH, Language.KOREAN): {
            "Samsung Electronics": "삼성전자",
            "stock price": "주가",
            "revenue": "매출",
            "operating profit": "영업이익",
            "quarter": "분기",
            "year-over-year": "전년대비",
            "increase": "증가",
            "decrease": "감소",
        },
        (Language.KOREAN, Language.JAPANESE): {
            "삼성전자": "サムスン電子",
            "주가": "株価",
            "매출": "売上",
            "영업이익": "営業利益",
        },
        (Language.JAPANESE, Language.KOREAN): {
            "サムスン電子": "삼성전자",
            "株価": "주가",
            "売上": "매출",
            "営業利益": "영업이익",
        },
    }

    def __init__(self, custom_dict: Optional[Dict] = None):
        self._dictionaries = self.DICTIONARIES.copy()
        if custom_dict:
            self._dictionaries.update(custom_dict)

    def translate(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> TranslationResult:
        """사전 기반 번역"""
        start_time = time.time()

        dictionary = self._dictionaries.get((source_lang, target_lang), {})

        translated = text
        matches = 0

        # 긴 구문부터 매칭
        for source, target in sorted(
            dictionary.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            if source in translated:
                translated = translated.replace(source, target)
                matches += 1

        translation_time = (time.time() - start_time) * 1000

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_lang,
            target_language=target_lang,
            confidence=min(1.0, matches * 0.2) if matches > 0 else 0.5,
            translation_time_ms=translation_time,
            metadata={"matches": matches, "method": "dictionary"},
        )

    def supports(self, source_lang: Language, target_lang: Language) -> bool:
        """지원 언어쌍 확인"""
        return (source_lang, target_lang) in self._dictionaries

    def add_translation(
        self,
        source: str,
        target: str,
        source_lang: Language,
        target_lang: Language,
    ) -> None:
        """번역 추가"""
        key = (source_lang, target_lang)
        if key not in self._dictionaries:
            self._dictionaries[key] = {}
        self._dictionaries[key][source] = target


class LLMTranslator(Translator):
    """
    LLM 기반 번역기

    LLM을 사용한 고품질 번역
    """

    def __init__(self, llm_func: Optional[Callable[[str], str]] = None):
        self.llm_func = llm_func or self._default_llm

    def _default_llm(self, prompt: str) -> str:
        """기본 LLM (시뮬레이션)"""
        # 실제로는 OpenAI API 등 호출
        return f"[Translated: {prompt[:50]}...]"

    def translate(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> TranslationResult:
        """LLM 기반 번역"""
        start_time = time.time()

        prompt = f"""Translate the following text from {source_lang.name_native} to {target_lang.name_native}.
Only output the translation, no explanations.

Text: {text}

Translation:"""

        translated = self.llm_func(prompt)
        translation_time = (time.time() - start_time) * 1000

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.9,  # LLM은 일반적으로 높은 품질
            translation_time_ms=translation_time,
            metadata={"method": "llm"},
        )

    def supports(self, source_lang: Language, target_lang: Language) -> bool:
        """LLM은 모든 언어쌍 지원"""
        return source_lang != Language.UNKNOWN and target_lang != Language.UNKNOWN


class CacheableTranslator(Translator):
    """
    캐싱 가능한 번역기 래퍼

    번역 결과를 캐싱하여 재사용
    """

    def __init__(
        self,
        translator: Translator,
        max_cache_size: int = 10000,
    ):
        self.translator = translator
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, TranslationResult] = {}

    def _cache_key(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> str:
        """캐시 키 생성"""
        content = f"{text}:{source_lang.value}:{target_lang.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def translate(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> TranslationResult:
        """캐시 확인 후 번역"""
        cache_key = self._cache_key(text, source_lang, target_lang)

        if cache_key in self._cache:
            result = self._cache[cache_key]
            result.is_cached = True
            return result

        result = self.translator.translate(text, source_lang, target_lang)

        # 캐시 저장
        if len(self._cache) >= self.max_cache_size:
            # 오래된 항목 제거 (간단히 첫 번째 항목)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = result
        return result

    def supports(self, source_lang: Language, target_lang: Language) -> bool:
        """지원 언어쌍 확인"""
        return self.translator.supports(source_lang, target_lang)

    def clear_cache(self) -> None:
        """캐시 삭제"""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self._cache),
            "max_size": self.max_cache_size,
            "utilization": len(self._cache) / self.max_cache_size,
        }


class TranslationPipeline:
    """
    번역 파이프라인

    여러 번역기를 조합하여 사용
    """

    def __init__(
        self,
        translators: Optional[List[Translator]] = None,
        detector: Optional[LanguageDetector] = None,
    ):
        self.translators = translators or [DictionaryTranslator()]
        self.detector = detector or HybridDetector()

    def translate(
        self,
        text: str,
        target_lang: Language,
        source_lang: Optional[Language] = None,
    ) -> TranslationResult:
        """번역 실행"""
        # 소스 언어 감지
        if source_lang is None:
            detection = self.detector.detect(text)
            source_lang = detection.language

        # 같은 언어면 그대로 반환
        if source_lang == target_lang:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0,
                metadata={"method": "same_language"},
            )

        # 적합한 번역기 찾기
        for translator in self.translators:
            if translator.supports(source_lang, target_lang):
                return translator.translate(text, source_lang, target_lang)

        # 폴백: 원본 반환
        logger.warning(
            f"No translator supports {source_lang.value} -> {target_lang.value}"
        )
        return TranslationResult(
            source_text=text,
            translated_text=text,
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.0,
            metadata={"method": "fallback"},
        )

    def translate_batch(
        self,
        texts: List[str],
        target_lang: Language,
        source_lang: Optional[Language] = None,
    ) -> List[TranslationResult]:
        """배치 번역"""
        return [
            self.translate(text, target_lang, source_lang)
            for text in texts
        ]


class TranslationService:
    """
    번역 서비스

    전체 번역 기능을 통합 제공
    """

    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        use_cache: bool = True,
        cache_size: int = 10000,
    ):
        # 번역기 초기화
        dict_translator = DictionaryTranslator()
        llm_translator = LLMTranslator(llm_func=llm_func)

        # 캐시 래핑
        if use_cache:
            dict_translator = CacheableTranslator(dict_translator, cache_size)
            llm_translator = CacheableTranslator(llm_translator, cache_size)

        self.pipeline = TranslationPipeline(
            translators=[dict_translator, llm_translator]
        )
        self.detector = HybridDetector()

    def translate(
        self,
        text: str,
        target_lang: Language,
        source_lang: Optional[Language] = None,
    ) -> TranslationResult:
        """번역"""
        return self.pipeline.translate(text, target_lang, source_lang)

    def translate_to_korean(self, text: str) -> TranslationResult:
        """한국어로 번역"""
        return self.translate(text, Language.KOREAN)

    def translate_to_english(self, text: str) -> TranslationResult:
        """영어로 번역"""
        return self.translate(text, Language.ENGLISH)

    def translate_to_japanese(self, text: str) -> TranslationResult:
        """일본어로 번역"""
        return self.translate(text, Language.JAPANESE)

    def detect_and_translate(
        self,
        text: str,
        target_lang: Language,
    ) -> Tuple[TranslationResult, Dict[str, Any]]:
        """언어 감지 후 번역"""
        detection = self.detector.detect(text)

        result = self.translate(
            text,
            target_lang,
            source_lang=detection.language,
        )

        return result, detection.to_dict()
