# -*- coding: utf-8 -*-
"""
Configuration Management

HuggingFace Spaces 환경에서 설정 관리
"""

import os
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class UIConfig:
    """UI 설정"""
    # 테마 색상 (GitHub-style Dark Theme)
    bg_primary: str = "#0d1117"
    bg_secondary: str = "#161b22"
    bg_elevated: str = "#21262d"
    text_primary: str = "#e6edf3"
    text_secondary: str = "#8b949e"
    accent_blue: str = "#58a6ff"
    accent_green: str = "#3fb950"
    accent_red: str = "#f85149"
    accent_orange: str = "#f0883e"
    border_color: str = "#30363d"

    # 대화 히스토리 설정
    chat_history_enabled: bool = True
    max_history_items: int = 50

    # 내보내기 설정
    export_enabled: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["csv", "pdf"])

    # 차트 설정
    chart_enabled: bool = True
    chart_default_period: str = "3mo"
    chart_periods: List[str] = field(default_factory=lambda: ["1mo", "3mo", "6mo", "1y"])


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    # LLM 설정
    groq_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.2

    # 검색 설정
    vector_weight: float = 0.5
    keyword_weight: float = 0.5
    top_k: int = 5

    # 임베딩 설정
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 청킹 설정
    chunk_size: int = 500
    chunk_overlap: int = 100

    # UI 설정
    ui: UIConfig = field(default_factory=UIConfig)


def get_secret(key: str, default: str = "") -> str:
    """
    시크릿 값 가져오기 (HuggingFace Spaces 호환)

    우선순위:
    1. Streamlit secrets (HF Spaces)
    2. 환경 변수
    3. 기본값
    """
    try:
        # HuggingFace Spaces에서는 st.secrets 사용
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
        return os.environ.get(key, default)
    except Exception:
        return os.environ.get(key, default)


def load_config() -> AppConfig:
    """설정 로드"""
    return AppConfig(
        groq_api_key=get_secret("GROQ_API_KEY"),
        hf_token=get_secret("HF_TOKEN"),
        llm_model=get_secret("LLM_MODEL", "llama-3.1-8b-instant"),
        llm_temperature=float(get_secret("LLM_TEMPERATURE", "0.2")),
        vector_weight=float(get_secret("VECTOR_WEIGHT", "0.5")),
        keyword_weight=float(get_secret("KEYWORD_WEIGHT", "0.5")),
        top_k=int(get_secret("TOP_K", "5")),
        embedding_model=get_secret("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=int(get_secret("CHUNK_SIZE", "500")),
        chunk_overlap=int(get_secret("CHUNK_OVERLAP", "100")),
        ui=UIConfig(
            chat_history_enabled=get_secret("CHAT_HISTORY_ENABLED", "true").lower() == "true",
            max_history_items=int(get_secret("MAX_HISTORY_ITEMS", "50")),
            export_enabled=get_secret("EXPORT_ENABLED", "true").lower() == "true",
            chart_enabled=get_secret("CHART_ENABLED", "true").lower() == "true",
            chart_default_period=get_secret("CHART_DEFAULT_PERIOD", "3mo"),
        ),
    )


# 싱글톤 설정 인스턴스
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """설정 싱글톤 반환"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """설정 다시 로드"""
    global _config
    _config = load_config()
    return _config
