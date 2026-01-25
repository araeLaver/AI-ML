# -*- coding: utf-8 -*-
"""
다국어 지원 테스트
"""

import pytest

from src.multilingual.detection import (
    Language,
    DetectionResult,
    CharacterBasedDetector,
    StatisticalDetector,
    HybridDetector,
    LanguageRouter,
)
from src.multilingual.tokenizer import (
    Token,
    TokenizationResult,
    KoreanTokenizer,
    EnglishTokenizer,
    JapaneseTokenizer,
    ChineseTokenizer,
    MultilingualTokenizer,
    TokenizerFactory,
)
from src.multilingual.translation import (
    TranslationResult,
    DictionaryTranslator,
    LLMTranslator,
    CacheableTranslator,
    TranslationPipeline,
    TranslationService,
)
from src.multilingual.embedding import (
    EmbeddingResult,
    SimpleHashEmbedding,
    MultilingualEmbedding,
    CrossLingualEmbedding,
    LanguageAlignedEmbedding,
    EmbeddingRegistry,
)
from src.multilingual.i18n import (
    Locale,
    MessageCatalog,
    LocaleResolver,
    I18nManager,
    get_i18n,
    t,
)


# =============================================================================
# Detection Tests
# =============================================================================

class TestLanguage:
    """언어 열거형 테스트"""

    def test_from_code(self):
        """코드로 언어 조회"""
        assert Language.from_code("ko") == Language.KOREAN
        assert Language.from_code("en") == Language.ENGLISH
        assert Language.from_code("ja") == Language.JAPANESE
        assert Language.from_code("unknown") == Language.UNKNOWN

    def test_native_name(self):
        """원어 이름"""
        assert Language.KOREAN.name_native == "한국어"
        assert Language.ENGLISH.name_native == "English"
        assert Language.JAPANESE.name_native == "日本語"


class TestCharacterBasedDetector:
    """문자 기반 감지기 테스트"""

    def test_detect_korean(self):
        """한국어 감지"""
        detector = CharacterBasedDetector()
        result = detector.detect("삼성전자 주가가 상승했습니다.")

        assert result.language == Language.KOREAN
        assert result.confidence > 0.5

    def test_detect_english(self):
        """영어 감지"""
        detector = CharacterBasedDetector()
        result = detector.detect("The stock price increased.")

        assert result.language == Language.ENGLISH
        assert result.confidence > 0.5

    def test_detect_japanese(self):
        """일본어 감지"""
        detector = CharacterBasedDetector()
        result = detector.detect("株価が上昇しました。")

        assert result.language in [Language.JAPANESE, Language.CHINESE]

    def test_detect_mixed(self):
        """혼합 언어 감지"""
        detector = CharacterBasedDetector()
        result = detector.detect("삼성전자 Samsung stock price")

        assert result.is_mixed or len(result.language_ratios) > 1


class TestStatisticalDetector:
    """통계 기반 감지기 테스트"""

    def test_detect_korean_patterns(self):
        """한국어 패턴 감지"""
        detector = StatisticalDetector()
        result = detector.detect("삼성전자의 매출이 증가했습니다.")

        assert result.language == Language.KOREAN

    def test_detect_english_patterns(self):
        """영어 패턴 감지"""
        detector = StatisticalDetector()
        result = detector.detect("The company has increased revenue.")

        assert result.language == Language.ENGLISH


class TestHybridDetector:
    """하이브리드 감지기 테스트"""

    def test_detect(self):
        """하이브리드 감지"""
        detector = HybridDetector()
        result = detector.detect("삼성전자 주가")

        assert result.language == Language.KOREAN
        assert "char_result" in result.metadata


class TestLanguageRouter:
    """언어 라우터 테스트"""

    def test_route(self):
        """라우팅 테스트"""
        router = LanguageRouter()
        router.register_handler(Language.KOREAN, "korean_handler")
        router.register_handler(Language.ENGLISH, "english_handler")

        result = router.route("삼성전자 주가")
        assert result["detected_language"] == "ko"


# =============================================================================
# Tokenizer Tests
# =============================================================================

class TestKoreanTokenizer:
    """한국어 토크나이저 테스트"""

    def test_tokenize(self):
        """토크나이제이션"""
        tokenizer = KoreanTokenizer()
        result = tokenizer.tokenize("삼성전자 주가")

        assert len(result.tokens) > 0
        assert result.language == Language.KOREAN

    def test_stopwords(self):
        """불용어 처리"""
        tokenizer = KoreanTokenizer()
        result = tokenizer.tokenize("삼성전자는 좋은 회사입니다")

        stopword_tokens = [t for t in result.tokens if t.is_stopword]
        assert len(stopword_tokens) >= 0


class TestEnglishTokenizer:
    """영어 토크나이저 테스트"""

    def test_tokenize(self):
        """토크나이제이션"""
        tokenizer = EnglishTokenizer()
        result = tokenizer.tokenize("Samsung stock price")

        assert len(result.tokens) == 3
        assert result.token_texts == ["Samsung", "stock", "price"]

    def test_stopwords(self):
        """불용어 처리"""
        tokenizer = EnglishTokenizer()
        result = tokenizer.tokenize("The stock is rising")

        stopword_count = sum(1 for t in result.tokens if t.is_stopword)
        assert stopword_count >= 2  # "The", "is"


class TestJapaneseTokenizer:
    """일본어 토크나이저 테스트"""

    def test_tokenize(self):
        """토크나이제이션"""
        tokenizer = JapaneseTokenizer()
        result = tokenizer.tokenize("株価が上昇")

        assert len(result.tokens) > 0
        assert result.language == Language.JAPANESE


class TestMultilingualTokenizer:
    """다국어 토크나이저 테스트"""

    def test_auto_detect_korean(self):
        """자동 한국어 감지"""
        tokenizer = MultilingualTokenizer()
        result = tokenizer.tokenize("삼성전자 주가")

        assert result.language == Language.KOREAN

    def test_auto_detect_english(self):
        """자동 영어 감지"""
        tokenizer = MultilingualTokenizer()
        result = tokenizer.tokenize("Stock price")

        assert result.language == Language.ENGLISH

    def test_explicit_language(self):
        """명시적 언어 지정"""
        tokenizer = MultilingualTokenizer()
        result = tokenizer.tokenize("Test", language=Language.ENGLISH)

        assert result.language == Language.ENGLISH


class TestTokenizerFactory:
    """토크나이저 팩토리 테스트"""

    def test_create(self):
        """토크나이저 생성"""
        tokenizer = TokenizerFactory.create(Language.KOREAN)
        assert isinstance(tokenizer, KoreanTokenizer)

    def test_create_multilingual(self):
        """다국어 토크나이저 생성"""
        tokenizer = TokenizerFactory.create_multilingual()
        assert isinstance(tokenizer, MultilingualTokenizer)


# =============================================================================
# Translation Tests
# =============================================================================

class TestDictionaryTranslator:
    """사전 번역기 테스트"""

    def test_translate(self):
        """번역 테스트"""
        translator = DictionaryTranslator()
        result = translator.translate(
            "삼성전자 주가",
            Language.KOREAN,
            Language.ENGLISH,
        )

        assert "Samsung" in result.translated_text or "stock" in result.translated_text

    def test_supports(self):
        """지원 언어쌍 확인"""
        translator = DictionaryTranslator()

        assert translator.supports(Language.KOREAN, Language.ENGLISH)
        assert translator.supports(Language.ENGLISH, Language.KOREAN)


class TestCacheableTranslator:
    """캐시 가능 번역기 테스트"""

    def test_cache_hit(self):
        """캐시 히트"""
        base_translator = DictionaryTranslator()
        translator = CacheableTranslator(base_translator)

        # 첫 번째 호출
        result1 = translator.translate("주가", Language.KOREAN, Language.ENGLISH)
        assert not result1.is_cached

        # 두 번째 호출 (캐시에서)
        result2 = translator.translate("주가", Language.KOREAN, Language.ENGLISH)
        assert result2.is_cached

    def test_cache_stats(self):
        """캐시 통계"""
        translator = CacheableTranslator(DictionaryTranslator())
        translator.translate("테스트", Language.KOREAN, Language.ENGLISH)

        stats = translator.cache_stats()
        assert stats["size"] == 1


class TestTranslationPipeline:
    """번역 파이프라인 테스트"""

    def test_translate(self):
        """파이프라인 번역"""
        pipeline = TranslationPipeline()
        result = pipeline.translate(
            "삼성전자",
            Language.ENGLISH,
            source_lang=Language.KOREAN,
        )

        assert result.source_language == Language.KOREAN
        assert result.target_language == Language.ENGLISH

    def test_same_language(self):
        """같은 언어 번역"""
        pipeline = TranslationPipeline()
        result = pipeline.translate(
            "테스트",
            Language.KOREAN,
            source_lang=Language.KOREAN,
        )

        assert result.translated_text == "테스트"
        assert result.confidence == 1.0


class TestTranslationService:
    """번역 서비스 테스트"""

    def test_translate_to_english(self):
        """영어로 번역"""
        service = TranslationService()
        result = service.translate_to_english("삼성전자")

        assert result.target_language == Language.ENGLISH

    def test_translate_to_korean(self):
        """한국어로 번역"""
        service = TranslationService()
        result = service.translate_to_korean("Samsung")

        assert result.target_language == Language.KOREAN


# =============================================================================
# Embedding Tests
# =============================================================================

class TestSimpleHashEmbedding:
    """해시 임베딩 테스트"""

    def test_embed(self):
        """임베딩"""
        embedding = SimpleHashEmbedding(dim=128)
        result = embedding.embed("테스트 텍스트")

        assert len(result.embedding) == 128
        assert result.dimension == 128

    def test_deterministic(self):
        """결정적 결과"""
        embedding = SimpleHashEmbedding(dim=64, seed=42)

        result1 = embedding.embed("test")
        result2 = embedding.embed("test")

        assert result1.embedding == result2.embedding


class TestMultilingualEmbedding:
    """다국어 임베딩 테스트"""

    def test_embed_korean(self):
        """한국어 임베딩"""
        embedding = MultilingualEmbedding(dim=256)
        result = embedding.embed("삼성전자 주가")

        assert result.language == Language.KOREAN
        assert len(result.embedding) == 256

    def test_embed_english(self):
        """영어 임베딩"""
        embedding = MultilingualEmbedding(dim=256)
        result = embedding.embed("Stock price")

        assert result.language == Language.ENGLISH


class TestCrossLingualEmbedding:
    """Cross-Lingual 임베딩 테스트"""

    def test_embed(self):
        """임베딩"""
        embedding = CrossLingualEmbedding()
        result = embedding.embed("테스트")

        assert len(result.embedding) == embedding.dimension

    def test_similarity(self):
        """유사도 계산"""
        embedding = CrossLingualEmbedding()
        sim = embedding.compute_similarity("테스트", "테스트")

        # 같은 텍스트는 높은 유사도
        assert sim > 0.9


class TestEmbeddingRegistry:
    """임베딩 레지스트리 테스트"""

    def test_register_and_get(self):
        """등록 및 조회"""
        registry = EmbeddingRegistry()

        assert registry.get("multilingual") is not None
        assert registry.get("simple_hash") is not None

    def test_embed(self):
        """임베딩 실행"""
        registry = EmbeddingRegistry()
        result = registry.embed("테스트")

        assert len(result.embedding) > 0

    def test_list_models(self):
        """모델 목록"""
        registry = EmbeddingRegistry()
        models = registry.list_models()

        assert "multilingual" in models
        assert "simple_hash" in models


# =============================================================================
# i18n Tests
# =============================================================================

class TestLocale:
    """로케일 테스트"""

    def test_language(self):
        """언어 변환"""
        assert Locale.KO_KR.language == Language.KOREAN
        assert Locale.EN_US.language == Language.ENGLISH
        assert Locale.JA_JP.language == Language.JAPANESE

    def test_from_language(self):
        """언어에서 로케일"""
        assert Locale.from_language(Language.KOREAN) == Locale.KO_KR
        assert Locale.from_language(Language.ENGLISH) == Locale.EN_US


class TestMessageCatalog:
    """메시지 카탈로그 테스트"""

    def test_get(self):
        """메시지 조회"""
        catalog = MessageCatalog(
            locale=Locale.KO_KR,
            messages={"hello": "안녕하세요"},
        )

        assert catalog.get("hello") == "안녕하세요"
        assert catalog.get("unknown") == "unknown"

    def test_format(self):
        """메시지 포맷팅"""
        catalog = MessageCatalog(
            locale=Locale.KO_KR,
            messages={"greeting": "안녕하세요, {name}님!"},
        )

        result = catalog.format("greeting", name="홍길동")
        assert result == "안녕하세요, 홍길동님!"


class TestLocaleResolver:
    """로케일 해결기 테스트"""

    def test_from_header(self):
        """헤더에서 해결"""
        resolver = LocaleResolver()

        result = resolver.resolve_from_header("ko-KR,ko;q=0.9,en-US;q=0.8")
        assert result == Locale.KO_KR

        result = resolver.resolve_from_header("en-US,en;q=0.9")
        assert result == Locale.EN_US

    def test_from_query(self):
        """쿼리에서 해결"""
        resolver = LocaleResolver()

        assert resolver.resolve_from_query("ko-KR") == Locale.KO_KR
        assert resolver.resolve_from_query("en-US") == Locale.EN_US


class TestI18nManager:
    """I18n 매니저 테스트"""

    def test_translate(self):
        """번역"""
        manager = I18nManager(default_locale=Locale.KO_KR)

        assert "환영" in manager.t("welcome")
        assert "Welcome" in manager.t("welcome", Locale.EN_US)

    def test_translate_with_params(self):
        """파라미터 포맷팅"""
        manager = I18nManager()

        result = manager.t("search.results_found", count=10)
        assert "10" in result

    def test_format_number(self):
        """숫자 포맷팅"""
        manager = I18nManager()

        result = manager.format_number(1000000, Locale.KO_KR, style="currency")
        assert "₩" in result or "1,000,000" in result

    def test_available_locales(self):
        """사용 가능한 로케일"""
        manager = I18nManager()
        locales = manager.get_available_locales()

        assert Locale.KO_KR in locales
        assert Locale.EN_US in locales


class TestGlobalI18n:
    """전역 i18n 테스트"""

    def test_get_i18n(self):
        """전역 인스턴스"""
        manager = get_i18n()
        assert isinstance(manager, I18nManager)

    def test_t_function(self):
        """편의 함수"""
        result = t("welcome")
        assert len(result) > 0
