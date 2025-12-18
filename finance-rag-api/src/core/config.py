"""
애플리케이션 설정 관리

백엔드 개발자로서 익숙한 패턴:
- 환경변수 기반 설정
- Pydantic으로 타입 검증
- 환경별 설정 분리 (dev/prod)
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """환경변수 기반 설정 클래스"""

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Application
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # LLM Settings (Ollama)
    ollama_api_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Vector DB (ChromaDB)
    chroma_db_path: str = "./db"
    chroma_collection_name: str = "finance_docs"
    chroma_persist_dir: str = "./data/chroma_db"  # Legacy, kept for compatibility

    # RAG Settings
    top_k: int = 5
    relevance_threshold: float = 0.3

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Security
    require_api_key: bool = False  # True면 항상 API Key 필요
    rate_limit_per_minute: int = 60
    allowed_origins: str = "*"  # CORS 허용 도메인 (쉼표 구분)

    # Pydantic V2 config
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # 정의되지 않은 환경변수 무시
    )


@lru_cache()
def get_settings() -> Settings:
    """
    싱글톤 패턴으로 설정 인스턴스 반환
    Spring의 @Configuration과 유사한 개념
    """
    return Settings()
