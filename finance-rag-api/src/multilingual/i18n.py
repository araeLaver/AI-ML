# -*- coding: utf-8 -*-
"""
국제화 (i18n) 모듈

[기능]
- 메시지 번역
- 로케일 관리
- 다국어 포맷팅
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .detection import Language

logger = logging.getLogger(__name__)


class Locale(Enum):
    """로케일"""
    KO_KR = "ko-KR"
    EN_US = "en-US"
    EN_GB = "en-GB"
    JA_JP = "ja-JP"
    ZH_CN = "zh-CN"
    ZH_TW = "zh-TW"

    @property
    def language(self) -> Language:
        """로케일의 기본 언어"""
        mapping = {
            Locale.KO_KR: Language.KOREAN,
            Locale.EN_US: Language.ENGLISH,
            Locale.EN_GB: Language.ENGLISH,
            Locale.JA_JP: Language.JAPANESE,
            Locale.ZH_CN: Language.CHINESE,
            Locale.ZH_TW: Language.CHINESE,
        }
        return mapping.get(self, Language.UNKNOWN)

    @classmethod
    def from_language(cls, lang: Language) -> "Locale":
        """언어에서 기본 로케일"""
        mapping = {
            Language.KOREAN: Locale.KO_KR,
            Language.ENGLISH: Locale.EN_US,
            Language.JAPANESE: Locale.JA_JP,
            Language.CHINESE: Locale.ZH_CN,
        }
        return mapping.get(lang, Locale.EN_US)


@dataclass
class MessageCatalog:
    """메시지 카탈로그"""
    locale: Locale
    messages: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Optional[str] = None) -> str:
        """메시지 조회"""
        return self.messages.get(key, default or key)

    def format(self, key: str, **kwargs) -> str:
        """메시지 포맷팅"""
        template = self.get(key)
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def add(self, key: str, message: str) -> None:
        """메시지 추가"""
        self.messages[key] = message

    def update(self, messages: Dict[str, str]) -> None:
        """메시지 일괄 추가"""
        self.messages.update(messages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locale": self.locale.value,
            "message_count": len(self.messages),
        }


class LocaleResolver:
    """
    로케일 해결기

    요청에서 적절한 로케일 결정
    """

    def __init__(self, default_locale: Locale = Locale.KO_KR):
        self.default_locale = default_locale

    def resolve_from_header(self, accept_language: str) -> Locale:
        """Accept-Language 헤더에서 로케일 해결"""
        if not accept_language:
            return self.default_locale

        # 간단한 파싱: "ko-KR,ko;q=0.9,en-US;q=0.8"
        parts = accept_language.split(",")

        for part in parts:
            lang_part = part.split(";")[0].strip()

            # 정확한 매칭 시도
            for locale in Locale:
                if locale.value.lower() == lang_part.lower():
                    return locale

            # 언어 코드만으로 매칭
            lang_code = lang_part.split("-")[0].lower()
            for locale in Locale:
                if locale.value.split("-")[0].lower() == lang_code:
                    return locale

        return self.default_locale

    def resolve_from_query(self, locale_param: str) -> Locale:
        """쿼리 파라미터에서 로케일 해결"""
        for locale in Locale:
            if locale.value.lower() == locale_param.lower():
                return locale
            if locale.name.lower() == locale_param.lower():
                return locale

        return self.default_locale

    def resolve(
        self,
        accept_language: Optional[str] = None,
        locale_param: Optional[str] = None,
        user_preference: Optional[str] = None,
    ) -> Locale:
        """로케일 해결 (우선순위: 쿼리 > 사용자 > 헤더 > 기본)"""
        if locale_param:
            return self.resolve_from_query(locale_param)

        if user_preference:
            return self.resolve_from_query(user_preference)

        if accept_language:
            return self.resolve_from_header(accept_language)

        return self.default_locale


class I18nManager:
    """
    국제화 관리자

    다국어 메시지 및 포맷팅 관리
    """

    # 기본 메시지
    DEFAULT_MESSAGES = {
        Locale.KO_KR: {
            "welcome": "환영합니다!",
            "search.no_results": "검색 결과가 없습니다.",
            "search.results_found": "{count}개의 결과를 찾았습니다.",
            "error.not_found": "찾을 수 없습니다.",
            "error.server": "서버 오류가 발생했습니다.",
            "error.invalid_input": "잘못된 입력입니다.",
            "loading": "로딩 중...",
            "success": "성공!",
            "cancel": "취소",
            "confirm": "확인",
            "date.today": "오늘",
            "date.yesterday": "어제",
            "date.format": "%Y년 %m월 %d일",
            "number.currency": "₩{value:,}",
            "number.percent": "{value:.1f}%",
            "rag.answer_intro": "다음은 질문에 대한 답변입니다:",
            "rag.source_count": "{count}개의 출처에서 정보를 찾았습니다.",
            "rag.no_relevant": "관련 정보를 찾을 수 없습니다.",
        },
        Locale.EN_US: {
            "welcome": "Welcome!",
            "search.no_results": "No results found.",
            "search.results_found": "Found {count} results.",
            "error.not_found": "Not found.",
            "error.server": "Server error occurred.",
            "error.invalid_input": "Invalid input.",
            "loading": "Loading...",
            "success": "Success!",
            "cancel": "Cancel",
            "confirm": "OK",
            "date.today": "Today",
            "date.yesterday": "Yesterday",
            "date.format": "%B %d, %Y",
            "number.currency": "${value:,.2f}",
            "number.percent": "{value:.1f}%",
            "rag.answer_intro": "Here is the answer to your question:",
            "rag.source_count": "Found information from {count} sources.",
            "rag.no_relevant": "No relevant information found.",
        },
        Locale.JA_JP: {
            "welcome": "ようこそ！",
            "search.no_results": "検索結果がありません。",
            "search.results_found": "{count}件の結果が見つかりました。",
            "error.not_found": "見つかりません。",
            "error.server": "サーバーエラーが発生しました。",
            "error.invalid_input": "入力が無効です。",
            "loading": "読み込み中...",
            "success": "成功！",
            "cancel": "キャンセル",
            "confirm": "確認",
            "date.today": "今日",
            "date.yesterday": "昨日",
            "date.format": "%Y年%m月%d日",
            "number.currency": "¥{value:,}",
            "number.percent": "{value:.1f}%",
            "rag.answer_intro": "以下がご質問への回答です：",
            "rag.source_count": "{count}件のソースから情報を見つけました。",
            "rag.no_relevant": "関連情報が見つかりません。",
        },
        Locale.ZH_CN: {
            "welcome": "欢迎！",
            "search.no_results": "没有找到结果。",
            "search.results_found": "找到{count}个结果。",
            "error.not_found": "未找到。",
            "error.server": "服务器错误。",
            "error.invalid_input": "输入无效。",
            "loading": "加载中...",
            "success": "成功！",
            "cancel": "取消",
            "confirm": "确认",
            "date.today": "今天",
            "date.yesterday": "昨天",
            "date.format": "%Y年%m月%d日",
            "number.currency": "¥{value:,.2f}",
            "number.percent": "{value:.1f}%",
            "rag.answer_intro": "以下是您问题的答案：",
            "rag.source_count": "从{count}个来源找到信息。",
            "rag.no_relevant": "没有找到相关信息。",
        },
    }

    def __init__(self, default_locale: Locale = Locale.KO_KR):
        self.default_locale = default_locale
        self.resolver = LocaleResolver(default_locale)

        # 카탈로그 초기화
        self._catalogs: Dict[Locale, MessageCatalog] = {}
        self._load_default_messages()

    def _load_default_messages(self) -> None:
        """기본 메시지 로드"""
        for locale, messages in self.DEFAULT_MESSAGES.items():
            catalog = MessageCatalog(locale=locale, messages=messages)
            self._catalogs[locale] = catalog

    def get_catalog(self, locale: Locale) -> MessageCatalog:
        """카탈로그 조회"""
        return self._catalogs.get(locale) or self._catalogs.get(self.default_locale)

    def add_catalog(self, catalog: MessageCatalog) -> None:
        """카탈로그 추가"""
        self._catalogs[catalog.locale] = catalog

    def t(
        self,
        key: str,
        locale: Optional[Locale] = None,
        **kwargs,
    ) -> str:
        """메시지 번역 (translate)"""
        loc = locale or self.default_locale
        catalog = self.get_catalog(loc)

        if catalog:
            return catalog.format(key, **kwargs)
        return key

    def translate(
        self,
        key: str,
        locale: Optional[Locale] = None,
        **kwargs,
    ) -> str:
        """t의 별칭"""
        return self.t(key, locale, **kwargs)

    def format_date(
        self,
        date: datetime,
        locale: Optional[Locale] = None,
        format_key: str = "date.format",
    ) -> str:
        """날짜 포맷팅"""
        loc = locale or self.default_locale
        catalog = self.get_catalog(loc)

        format_str = catalog.get(format_key, "%Y-%m-%d")
        return date.strftime(format_str)

    def format_number(
        self,
        value: Union[int, float],
        locale: Optional[Locale] = None,
        style: str = "decimal",
    ) -> str:
        """숫자 포맷팅"""
        loc = locale or self.default_locale
        catalog = self.get_catalog(loc)

        if style == "currency":
            template = catalog.get("number.currency", "{value:,.2f}")
        elif style == "percent":
            template = catalog.get("number.percent", "{value:.1f}%")
        else:
            template = "{value:,}"

        return template.format(value=value)

    def get_available_locales(self) -> List[Locale]:
        """사용 가능한 로케일 목록"""
        return list(self._catalogs.keys())

    def set_messages(
        self,
        locale: Locale,
        messages: Dict[str, str],
    ) -> None:
        """메시지 설정"""
        catalog = self._catalogs.get(locale)
        if catalog:
            catalog.update(messages)
        else:
            self._catalogs[locale] = MessageCatalog(locale=locale, messages=messages)

    def get_all_keys(self) -> List[str]:
        """모든 메시지 키 목록"""
        keys = set()
        for catalog in self._catalogs.values():
            keys.update(catalog.messages.keys())
        return sorted(keys)


# 전역 인스턴스
_i18n_manager: Optional[I18nManager] = None


def get_i18n() -> I18nManager:
    """전역 I18n 매니저 반환"""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager


def t(key: str, locale: Optional[Locale] = None, **kwargs) -> str:
    """편의 함수: 번역"""
    return get_i18n().t(key, locale, **kwargs)


def set_default_locale(locale: Locale) -> None:
    """기본 로케일 설정"""
    get_i18n().default_locale = locale
