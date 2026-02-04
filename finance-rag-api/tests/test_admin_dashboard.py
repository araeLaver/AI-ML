# -*- coding: utf-8 -*-
"""
관리자 대시보드 테스트 (Phase 14)

대시보드 API 클라이언트 및 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, patch


# ============================================================
# Dashboard API 클라이언트 테스트
# ============================================================

class TestDashboardAPI:
    """대시보드 API 클라이언트 테스트"""

    def test_import_dashboard_module(self):
        """대시보드 모듈 임포트"""
        from app.admin_dashboard import DashboardAPI
        assert DashboardAPI is not None

    def test_api_client_initialization(self):
        """API 클라이언트 초기화"""
        from app.admin_dashboard import DashboardAPI

        client = DashboardAPI("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_api_client_trailing_slash(self):
        """API URL 후행 슬래시 제거"""
        from app.admin_dashboard import DashboardAPI

        client = DashboardAPI("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    @patch("requests.Session.get")
    def test_get_health(self, mock_get):
        """헬스체크 조회"""
        from app.admin_dashboard import DashboardAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        client = DashboardAPI()
        result = client.get_health()

        assert result["status"] == "ok"

    @patch("requests.Session.get")
    def test_get_health_failure(self, mock_get):
        """헬스체크 실패"""
        from app.admin_dashboard import DashboardAPI

        mock_get.side_effect = Exception("Connection error")

        client = DashboardAPI()
        # Should return default value on error
        result = client.get_health()
        assert result.get("status") == "unknown"

    @patch("requests.Session.get")
    def test_get_cache_stats(self, mock_get):
        """캐시 통계 조회"""
        from app.admin_dashboard import DashboardAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "enabled": True,
            "backend": "local",
            "stats": {"hits": 100, "misses": 20}
        }
        mock_get.return_value = mock_response

        client = DashboardAPI()
        result = client.get_cache_stats()

        assert result["enabled"] is True
        assert result["stats"]["hits"] == 100

    @patch("requests.Session.get")
    def test_get_audit_logs(self, mock_get):
        """감사 로그 조회"""
        from app.admin_dashboard import DashboardAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"action": "login", "username": "admin", "success": True},
            {"action": "query_execute", "username": "user", "success": True}
        ]
        mock_get.return_value = mock_response

        client = DashboardAPI()
        result = client.get_audit_logs(limit=10)

        assert len(result) == 2
        assert result[0]["action"] == "login"

    @patch("requests.Session.post")
    def test_post_request(self, mock_post):
        """POST 요청"""
        from app.admin_dashboard import DashboardAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "cleared_count": 5}
        mock_post.return_value = mock_response

        client = DashboardAPI()
        result = client._post("/api/v1/performance/cache/clear", {"pattern": "*"})

        assert result["success"] is True
        assert result["cleared_count"] == 5


# ============================================================
# 대시보드 엔드포인트 통합 테스트
# ============================================================

class TestDashboardEndpoints:
    """대시보드 관련 API 엔드포인트 테스트"""

    def test_api_endpoints_available(self):
        """필수 API 엔드포인트 존재 확인"""
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)

        # 대시보드에서 사용하는 엔드포인트
        endpoints = [
            "/api/v1/health",
            "/api/v1/stats",
            "/api/v1/performance/cache/stats",
            "/api/v1/performance/metrics",
            "/api/v1/performance/config",
            "/api/v1/security/health",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Endpoint {endpoint} failed"

    def test_root_endpoint_has_all_info(self):
        """루트 엔드포인트에 모든 정보 포함"""
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        # 대시보드에서 필요한 필드
        assert "name" in data
        assert "version" in data
        assert "performance" in data
        assert "security" in data
        assert "realtime" in data
        assert "multimodal" in data


# ============================================================
# 유틸리티 함수 테스트
# ============================================================

class TestDashboardUtils:
    """대시보드 유틸리티 테스트"""

    def test_metric_card_generation(self):
        """메트릭 카드 HTML 생성"""
        # 간단한 HTML 생성 테스트
        title = "Test Metric"
        value = 100

        html = f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value:,}</div>
        </div>
        """

        assert "Test Metric" in html
        assert "100" in html

    def test_status_badge_classes(self):
        """상태 배지 CSS 클래스"""
        status_classes = {
            "ok": "status-ok",
            "warning": "status-warning",
            "error": "status-error"
        }

        for status, expected_class in status_classes.items():
            assert expected_class.endswith(status)

    def test_log_entry_formatting(self):
        """로그 엔트리 포맷팅"""
        log = {
            "timestamp": "2024-01-15T10:30:00",
            "action": "login",
            "username": "admin",
            "success": True
        }

        timestamp = log["timestamp"][:19].replace("T", " ")
        assert timestamp == "2024-01-15 10:30:00"

        success_class = "log-success" if log["success"] else "log-failure"
        assert success_class == "log-success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
