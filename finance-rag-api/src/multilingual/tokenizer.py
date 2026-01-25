# -*- coding: utf-8 -*-
"""
다국어 토크나이저 모듈

[기능]
- 언어별 토크나이저
- 형태소 분석
- 서브워드 토크나이제이션
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detection import Language, LanguageDetector, HybridDetector

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """토큰"""
    text: str
    start: int  # 원본 텍스트에서의 시작 위치
    end: int  # 원본 텍스트에서의 끝 위치
    pos: Optional[str] = None  # 품사 태그
    lemma: Optional[str] = None  # 원형
    is_stopword: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "pos": self.pos,
            "lemma": self.lemma,
            "is_stopword": self.is_stopword,
        }


@dataclass
class TokenizationResult:
    """토크나이제이션 결과"""
    tokens: List[Token]
    original_text: str
    language: Language

    @property
    def token_texts(self) -> List[str]:
        """토큰 텍스트 리스트"""
        return [t.text for t in self.tokens]

    @property
    def non_stopword_tokens(self) -> List[Token]:
        """불용어 제외 토큰"""
        return [t for t in self.tokens if not t.is_stopword]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": [t.to_dict() for t in self.tokens],
            "token_count": len(self.tokens),
            "language": self.language.value,
        }


class BaseTokenizer(ABC):
    """토크나이저 기본 클래스"""

    @abstractmethod
    def tokenize(self, text: str) -> TokenizationResult:
        """토크나이제이션"""
        pass

    @property
    @abstractmethod
    def language(self) -> Language:
        """지원 언어"""
        pass


class KoreanTokenizer(BaseTokenizer):
    """
    한국어 토크나이저

    형태소 분석 기반 토크나이제이션
    """

    STOPWORDS = {
        "의", "가", "이", "은", "는", "을", "를", "에", "에서",
        "로", "으로", "와", "과", "도", "만", "까지", "부터",
        "이다", "있다", "하다", "되다", "그", "이", "저",
        "것", "수", "등", "들", "및", "더", "때", "데",
    }

    # 간단한 한국어 형태소 패턴
    PATTERNS = [
        (r"([가-힣]+(?:습니다|입니다|합니다|됩니다))", "VV"),  # 동사 종결형
        (r"([가-힣]+(?:는|은|이|가))", "JK"),  # 조사
        (r"([가-힣]+)", "NNG"),  # 일반 명사
        (r"([0-9]+)", "SN"),  # 숫자
        (r"([a-zA-Z]+)", "SL"),  # 외국어
    ]

    def __init__(self, use_morpheme: bool = True):
        self.use_morpheme = use_morpheme

    @property
    def language(self) -> Language:
        return Language.KOREAN

    def tokenize(self, text: str) -> TokenizationResult:
        """토크나이제이션"""
        tokens = []

        if self.use_morpheme:
            tokens = self._morpheme_tokenize(text)
        else:
            tokens = self._space_tokenize(text)

        return TokenizationResult(
            tokens=tokens,
            original_text=text,
            language=self.language,
        )

    def _space_tokenize(self, text: str) -> List[Token]:
        """공백 기반 토크나이제이션"""
        tokens = []
        pos = 0

        for word in text.split():
            start = text.find(word, pos)
            end = start + len(word)

            tokens.append(Token(
                text=word,
                start=start,
                end=end,
                is_stopword=word in self.STOPWORDS,
            ))
            pos = end

        return tokens

    def _morpheme_tokenize(self, text: str) -> List[Token]:
        """형태소 기반 토크나이제이션 (간소화)"""
        tokens = []
        remaining = text
        pos = 0

        while remaining:
            matched = False

            for pattern, tag in self.PATTERNS:
                match = re.match(pattern, remaining)
                if match:
                    word = match.group(1)
                    start = text.find(word, pos)
                    end = start + len(word)

                    tokens.append(Token(
                        text=word,
                        start=start,
                        end=end,
                        pos=tag,
                        is_stopword=word in self.STOPWORDS,
                    ))

                    remaining = remaining[match.end():]
                    pos = end
                    matched = True
                    break

            if not matched:
                # 매칭 안 되면 한 문자씩 스킵
                remaining = remaining[1:]
                pos += 1

        return tokens


class EnglishTokenizer(BaseTokenizer):
    """
    영어 토크나이저

    공백 및 구두점 기반 토크나이제이션
    """

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "of", "to", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "so", "yet", "both", "either",
        "neither", "not", "only", "own", "same", "than", "too",
        "very", "just", "also", "now", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "than", "too", "very",
        "it", "its", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "she", "him", "her",
        "they", "them", "their", "who", "which", "what", "whose",
    }

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    @property
    def language(self) -> Language:
        return Language.ENGLISH

    def tokenize(self, text: str) -> TokenizationResult:
        """토크나이제이션"""
        # 구두점 분리
        pattern = r"(\w+|[^\w\s])"
        tokens = []

        for match in re.finditer(pattern, text):
            word = match.group()
            if self.lowercase:
                word_check = word.lower()
            else:
                word_check = word

            tokens.append(Token(
                text=word,
                start=match.start(),
                end=match.end(),
                pos=self._get_pos(word),
                lemma=word.lower(),
                is_stopword=word_check.lower() in self.STOPWORDS,
            ))

        return TokenizationResult(
            tokens=tokens,
            original_text=text,
            language=self.language,
        )

    def _get_pos(self, word: str) -> str:
        """간단한 품사 태깅"""
        if not word.isalpha():
            return "PUNCT"
        elif word[0].isupper():
            return "NNP"  # 고유명사
        else:
            return "NN"  # 일반 명사


class JapaneseTokenizer(BaseTokenizer):
    """
    일본어 토크나이저

    문자 유형 기반 토크나이제이션
    """

    STOPWORDS = {
        "の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
        "だ", "です", "ます", "する", "いる", "ある", "なる", "れる",
        "これ", "それ", "あれ", "どれ", "この", "その", "あの", "どの",
        "ない", "なかった", "ました", "ません", "でした",
    }

    @property
    def language(self) -> Language:
        return Language.JAPANESE

    def tokenize(self, text: str) -> TokenizationResult:
        """토크나이제이션"""
        tokens = []

        # 문자 유형 변화 지점에서 분리
        current_type = None
        current_word = ""
        start_pos = 0

        for i, char in enumerate(text):
            char_type = self._get_char_type(char)

            if char_type != current_type:
                if current_word.strip():
                    tokens.append(Token(
                        text=current_word,
                        start=start_pos,
                        end=i,
                        pos=self._get_pos(current_word, current_type),
                        is_stopword=current_word in self.STOPWORDS,
                    ))
                current_word = char
                current_type = char_type
                start_pos = i
            else:
                current_word += char

        # 마지막 토큰
        if current_word.strip():
            tokens.append(Token(
                text=current_word,
                start=start_pos,
                end=len(text),
                pos=self._get_pos(current_word, current_type),
                is_stopword=current_word in self.STOPWORDS,
            ))

        return TokenizationResult(
            tokens=tokens,
            original_text=text,
            language=self.language,
        )

    def _get_char_type(self, char: str) -> str:
        """문자 유형 반환"""
        code = ord(char)

        if 0x3040 <= code <= 0x309F:
            return "hiragana"
        elif 0x30A0 <= code <= 0x30FF:
            return "katakana"
        elif 0x4E00 <= code <= 0x9FFF:
            return "kanji"
        elif char.isascii() and char.isalpha():
            return "romaji"
        elif char.isdigit():
            return "number"
        elif char.isspace():
            return "space"
        else:
            return "other"

    def _get_pos(self, word: str, char_type: Optional[str]) -> str:
        """간단한 품사 태깅"""
        if char_type == "hiragana":
            if word in self.STOPWORDS:
                return "助詞"
            return "動詞"
        elif char_type == "katakana":
            return "外来語"
        elif char_type == "kanji":
            return "名詞"
        return "記号"


class ChineseTokenizer(BaseTokenizer):
    """
    중국어 토크나이저

    단어 분리 기반 토크나이제이션
    """

    STOPWORDS = {
        "的", "是", "在", "有", "和", "了", "不", "也", "就", "都",
        "这", "那", "你", "我", "他", "她", "它", "们", "什么", "怎么",
        "为", "与", "以", "之", "到", "从", "把", "被", "让", "给",
    }

    # 간단한 단어 사전 (실제로는 더 큰 사전 필요)
    DICTIONARY = {
        "中国", "美国", "日本", "韩国", "公司", "市场", "价格", "股票",
        "投资", "经济", "金融", "银行", "技术", "发展", "增长",
    }

    @property
    def language(self) -> Language:
        return Language.CHINESE

    def tokenize(self, text: str) -> TokenizationResult:
        """토크나이제이션 (최대 매칭 방식)"""
        tokens = []
        i = 0

        while i < len(text):
            # 최대 길이 매칭 시도
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i + length]

                if word in self.DICTIONARY or length == 1:
                    if word.strip():
                        tokens.append(Token(
                            text=word,
                            start=i,
                            end=i + length,
                            is_stopword=word in self.STOPWORDS,
                        ))
                    i += length
                    matched = True
                    break

            if not matched:
                i += 1

        return TokenizationResult(
            tokens=tokens,
            original_text=text,
            language=self.language,
        )


class MultilingualTokenizer:
    """
    다국어 토크나이저

    자동 언어 감지 및 적절한 토크나이저 선택
    """

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        default_language: Language = Language.KOREAN,
    ):
        self.detector = detector or HybridDetector()
        self.default_language = default_language

        # 언어별 토크나이저
        self._tokenizers: Dict[Language, BaseTokenizer] = {
            Language.KOREAN: KoreanTokenizer(),
            Language.ENGLISH: EnglishTokenizer(),
            Language.JAPANESE: JapaneseTokenizer(),
            Language.CHINESE: ChineseTokenizer(),
        }

    def register_tokenizer(self, language: Language, tokenizer: BaseTokenizer) -> None:
        """토크나이저 등록"""
        self._tokenizers[language] = tokenizer

    def tokenize(
        self,
        text: str,
        language: Optional[Language] = None,
    ) -> TokenizationResult:
        """토크나이제이션"""
        if language is None:
            detection = self.detector.detect(text)
            language = detection.language if detection.confidence > 0.3 else self.default_language

        tokenizer = self._tokenizers.get(language) or self._tokenizers.get(self.default_language)

        if tokenizer:
            return tokenizer.tokenize(text)

        # 폴백: 공백 기반 토크나이제이션
        return self._fallback_tokenize(text, language)

    def _fallback_tokenize(self, text: str, language: Language) -> TokenizationResult:
        """폴백 토크나이제이션"""
        tokens = []
        for match in re.finditer(r'\S+', text):
            tokens.append(Token(
                text=match.group(),
                start=match.start(),
                end=match.end(),
            ))

        return TokenizationResult(
            tokens=tokens,
            original_text=text,
            language=language,
        )

    def tokenize_batch(
        self,
        texts: List[str],
        language: Optional[Language] = None,
    ) -> List[TokenizationResult]:
        """배치 토크나이제이션"""
        return [self.tokenize(text, language) for text in texts]


class TokenizerFactory:
    """토크나이저 팩토리"""

    _registry: Dict[Language, type] = {
        Language.KOREAN: KoreanTokenizer,
        Language.ENGLISH: EnglishTokenizer,
        Language.JAPANESE: JapaneseTokenizer,
        Language.CHINESE: ChineseTokenizer,
    }

    @classmethod
    def register(cls, language: Language, tokenizer_class: type) -> None:
        """토크나이저 클래스 등록"""
        cls._registry[language] = tokenizer_class

    @classmethod
    def create(cls, language: Language, **kwargs) -> BaseTokenizer:
        """토크나이저 생성"""
        tokenizer_class = cls._registry.get(language)
        if tokenizer_class:
            return tokenizer_class(**kwargs)
        raise ValueError(f"No tokenizer registered for {language}")

    @classmethod
    def create_multilingual(cls, **kwargs) -> MultilingualTokenizer:
        """다국어 토크나이저 생성"""
        return MultilingualTokenizer(**kwargs)
