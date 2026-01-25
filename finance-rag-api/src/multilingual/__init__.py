# -*- coding: utf-8 -*-
"""
다국어 지원 모듈

[기능]
- 언어 감지
- 다국어 토크나이저
- 번역 서비스
- 다국어 임베딩
"""

from .detection import (
    LanguageDetector,
    Language,
    DetectionResult,
    CharacterBasedDetector,
    StatisticalDetector,
)
from .tokenizer import (
    MultilingualTokenizer,
    KoreanTokenizer,
    JapaneseTokenizer,
    EnglishTokenizer,
    ChineseTokenizer,
    TokenizerFactory,
)
from .translation import (
    TranslationService,
    Translator,
    TranslationResult,
    CacheableTranslator,
    TranslationPipeline,
)
from .embedding import (
    MultilingualEmbedding,
    CrossLingualEmbedding,
    LanguageAlignedEmbedding,
    EmbeddingRegistry,
)
from .i18n import (
    I18nManager,
    MessageCatalog,
    Locale,
    LocaleResolver,
)

__all__ = [
    # Detection
    "LanguageDetector",
    "Language",
    "DetectionResult",
    "CharacterBasedDetector",
    "StatisticalDetector",
    # Tokenizer
    "MultilingualTokenizer",
    "KoreanTokenizer",
    "JapaneseTokenizer",
    "EnglishTokenizer",
    "ChineseTokenizer",
    "TokenizerFactory",
    # Translation
    "TranslationService",
    "Translator",
    "TranslationResult",
    "CacheableTranslator",
    "TranslationPipeline",
    # Embedding
    "MultilingualEmbedding",
    "CrossLingualEmbedding",
    "LanguageAlignedEmbedding",
    "EmbeddingRegistry",
    # i18n
    "I18nManager",
    "MessageCatalog",
    "Locale",
    "LocaleResolver",
]
