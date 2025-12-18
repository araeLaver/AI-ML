# -*- coding: utf-8 -*-
"""
pytest 설정 및 픽스처

[백엔드 개발자 관점]
- Spring의 @BeforeEach, @TestConfiguration과 유사
- 테스트 격리를 위한 픽스처 제공
- 의존성 모킹 지원
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# FastAPI 테스트 클라이언트
from fastapi.testclient import TestClient

# 프로젝트 모듈
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import app
from src.rag.vectorstore import VectorStoreService
from src.rag.document_loader import (
    ChunkingConfig,
    RecursiveTextSplitter,
    TextLoader,
    PDFLoader,
    Document
)


@pytest.fixture(scope="session")
def test_data_dir():
    """테스트 데이터 디렉토리"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir():
    """임시 디렉토리 (테스트 후 자동 삭제)"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_texts():
    """테스트용 샘플 텍스트"""
    return [
        "ETF는 Exchange Traded Fund의 약자로, 주식처럼 거래되는 펀드입니다.",
        "채권은 정부나 기업이 발행하는 빚 증서입니다.",
        "분산 투자는 리스크를 줄이는 기본 전략입니다.",
        "적립식 투자는 정기적으로 일정 금액을 투자하는 방법입니다.",
        "레버리지 ETF는 초보자에게 적합하지 않습니다."
    ]


@pytest.fixture
def sample_long_text():
    """청킹 테스트용 긴 텍스트"""
    return """
    금융 투자의 기본 원칙

    첫째, 분산 투자가 중요합니다. 모든 자산을 한 곳에 투자하면 위험이 집중됩니다.
    주식, 채권, 부동산 등 다양한 자산군에 나누어 투자해야 합니다.

    둘째, 장기 투자를 권장합니다. 단기적인 시장 변동에 흔들리지 마세요.
    역사적으로 주식 시장은 장기적으로 우상향했습니다.

    셋째, 비용을 최소화하세요. 수수료와 세금은 수익을 갉아먹습니다.
    ETF나 인덱스 펀드는 액티브 펀드보다 비용이 낮습니다.

    넷째, 자신만의 투자 원칙을 세우세요. 남의 말에 휩쓸리지 마세요.
    자신의 재무 상황과 위험 허용도를 파악하세요.
    """


@pytest.fixture
def chunking_config():
    """기본 청킹 설정"""
    return ChunkingConfig(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=50
    )


@pytest.fixture
def text_splitter(chunking_config):
    """텍스트 분할기"""
    return RecursiveTextSplitter(chunking_config)


@pytest.fixture
def vectorstore(temp_dir):
    """테스트용 벡터 스토어 (격리된 임시 DB)"""
    vs = VectorStoreService(
        persist_dir=temp_dir,
        collection_name="test_collection"
    )
    yield vs
    # 정리
    try:
        vs.delete_collection()
    except:
        pass


@pytest.fixture
def client():
    """FastAPI 테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
def sample_document():
    """테스트용 Document 객체"""
    return Document(
        content="테스트 문서 내용입니다.",
        metadata={
            "source": "테스트 출처",
            "chunk_index": 0
        }
    )
