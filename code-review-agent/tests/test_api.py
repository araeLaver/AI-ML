"""Tests for API endpoints."""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from main import app
    return TestClient(app)


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Code Review Agent"
        assert data["status"] == "running"
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_includes_llm_check(self, client):
        response = client.get("/health")
        data = response.json()
        assert "checks" in data
        assert "llm" in data["checks"]


class TestQuickCheckEndpoint:
    """Tests for quick check endpoint (no LLM)."""

    def test_quick_check_python_code(self, client):
        code = """
def hello():
    print("Hello World")
"""
        response = client.post(
            "/api/quick-check",
            json={"code": code, "language": "python"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "issues" in data
        assert data["language"] == "python"

    def test_quick_check_returns_metrics(self, client):
        code = """
class MyClass:
    def method1(self):
        pass
    def method2(self):
        pass
"""
        response = client.post(
            "/api/quick-check",
            json={"code": code, "language": "python"}
        )
        data = response.json()
        assert data["metrics"]["classes"] >= 1
        assert data["metrics"]["functions"] >= 2

    def test_quick_check_empty_code_fails(self, client):
        response = client.post(
            "/api/quick-check",
            json={"code": "", "language": "python"}
        )
        assert response.status_code == 422  # Validation error


class TestSupportedLanguagesEndpoint:
    """Tests for supported languages endpoint."""

    def test_returns_language_list(self, client):
        response = client.get("/api/supported-languages")
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert len(data["languages"]) > 0

    def test_includes_common_languages(self, client):
        response = client.get("/api/supported-languages")
        data = response.json()
        language_codes = [lang["code"] for lang in data["languages"]]
        assert "python" in language_codes
        assert "javascript" in language_codes
        assert "typescript" in language_codes


class TestReviewEndpoint:
    """Tests for code review endpoint."""

    def test_review_requires_code(self, client):
        response = client.post("/api/review", json={})
        assert response.status_code == 422

    @patch("agents.ReviewOrchestrator")
    @patch("langchain_openai.ChatOpenAI")
    def test_review_returns_result(self, mock_openai, mock_orchestrator, client, mock_env):
        # Setup mocks
        mock_result = {
            "security": {"findings": [], "summary": "OK"},
            "performance": {"findings": [], "summary": "OK"},
            "style": {"findings": [], "summary": "OK"},
            "synthesis": {
                "verdict": "APPROVE",
                "health_score": 8,
                "summary": "Good code",
                "issue_counts": {"critical": 0, "high": 0, "medium": 0, "total": 0},
                "critical_issues": [],
                "important_improvements": [],
                "minor_suggestions": []
            }
        }
        mock_orchestrator.return_value.review.return_value = mock_result

        response = client.post(
            "/api/review",
            json={"code": "def hello(): pass", "language": "python"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "security" in data
        assert "final_report" in data

    def test_review_without_api_key_fails(self, client, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_URL", raising=False)

        response = client.post(
            "/api/review",
            json={"code": "def hello(): pass"}
        )
        assert response.status_code == 500


class TestAsyncReviewEndpoint:
    """Tests for async review endpoint."""

    def test_async_review_returns_job_id(self, client, mock_env):
        response = client.post(
            "/api/review/async",
            json={"code": "def hello(): pass", "language": "python"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "review_id" in data
        assert data["status"] == "pending"

    def test_get_review_status_not_found(self, client):
        response = client.get("/api/review/nonexistent-id")
        assert response.status_code == 404


class TestWebhookEndpoints:
    """Tests for webhook endpoints."""

    def test_webhook_ignores_non_pr_events(self, client):
        response = client.post(
            "/api/webhook/github",
            json={"action": "push"},
            headers={"X-GitHub-Event": "push"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ignored"

    def test_webhook_invalid_signature(self, client, monkeypatch):
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")

        response = client.post(
            "/api/webhook/github",
            json={"action": "opened"},
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "invalid"
            }
        )
        assert response.status_code == 401

    def test_list_jobs_returns_empty(self, client):
        response = client.get("/api/webhook/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
