# -*- coding: utf-8 -*-
"""
API 엔드포인트 테스트

[테스트 범위]
- 헬스체크
- 문서 추가/조회/삭제
- RAG 질의
- 파일 업로드
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


# 새로운 클라이언트 생성 (각 테스트 클래스마다 격리)
@pytest.fixture
def api_client():
    """API 테스트 클라이언트"""
    return TestClient(app)


class TestHealthEndpoint:
    """헬스체크 엔드포인트 테스트"""

    def test_health_check(self, api_client):
        """정상 응답 확인"""
        response = api_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "llm_model" in data


class TestStatsEndpoint:
    """통계 엔드포인트 테스트"""

    def test_get_stats(self, api_client):
        """통계 조회"""
        response = api_client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "llm_model" in data
        assert "top_k" in data
        assert "temperature" in data


class TestDocumentsEndpoint:
    """문서 관리 엔드포인트 테스트"""

    def test_add_documents(self, api_client):
        """문서 추가"""
        response = api_client.post(
            "/api/v1/documents",
            json={
                "documents": [
                    "테스트 문서 1입니다.",
                    "테스트 문서 2입니다."
                ],
                "source": "테스트 출처",
                "category": "테스트"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["added_count"] == 2

    def test_list_documents(self, api_client):
        """문서 목록 조회"""
        response = api_client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert "total_count" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_add_documents_validation(self, api_client):
        """문서 추가 유효성 검사"""
        # 빈 문서 리스트
        response = api_client.post(
            "/api/v1/documents",
            json={
                "documents": [],
                "source": "테스트"
            }
        )
        assert response.status_code == 422  # Validation Error


class TestQueryEndpoint:
    """RAG 질의 엔드포인트 테스트"""

    def test_query_success(self, api_client):
        """질의 성공"""
        response = api_client.post(
            "/api/v1/query",
            json={
                "question": "ETF가 뭔가요?",
                "top_k": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data

    def test_query_with_filter(self, api_client):
        """필터 적용 질의"""
        response = api_client.post(
            "/api/v1/query",
            json={
                "question": "투자 방법 알려줘",
                "top_k": 2,
                "filter_source": "투자 기초 가이드"
            }
        )

        assert response.status_code == 200

    def test_query_validation_short_question(self, api_client):
        """짧은 질문 유효성 검사"""
        response = api_client.post(
            "/api/v1/query",
            json={
                "question": "?",  # 너무 짧음
                "top_k": 3
            }
        )
        assert response.status_code == 422


class TestUploadEndpoint:
    """파일 업로드 엔드포인트 테스트"""

    def test_upload_txt_file(self, api_client):
        """TXT 파일 업로드"""
        content = """
        테스트 금융 문서입니다.
        ETF는 상장지수펀드입니다.
        분산 투자가 중요합니다.
        """

        response = api_client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", content.encode("utf-8"), "text/plain")},
            data={
                "source_name": "테스트 문서",
                "chunk_size": "200",
                "chunk_overlap": "50"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] > 0

    def test_upload_unsupported_format(self, api_client):
        """지원하지 않는 형식 업로드"""
        response = api_client.post(
            "/api/v1/upload",
            files={"file": ("test.xlsx", b"dummy", "application/vnd.ms-excel")}
        )

        assert response.status_code == 400
        assert "지원하지 않는" in response.json()["detail"]


class TestRootEndpoint:
    """루트 엔드포인트 테스트"""

    def test_root(self, api_client):
        """루트 경로 응답"""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
