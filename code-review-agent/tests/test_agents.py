"""Tests for review agents."""
import pytest
from unittest.mock import Mock, MagicMock

from agents.base import BaseReviewAgent
from agents.security_agent import SecurityAgent
from agents.performance_agent import PerformanceAgent
from agents.style_agent import StyleAgent
from agents.orchestrator import ReviewOrchestrator


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.invoke = MagicMock(return_value=Mock(content='{"findings": [], "summary": "No issues found"}'))
    return llm


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return '''
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item["price"]
    return total
'''


class TestSecurityAgent:
    """Tests for SecurityAgent."""

    def test_init(self, mock_llm):
        agent = SecurityAgent(mock_llm)
        assert agent.name == "SecurityAgent"

    def test_analyze_returns_dict(self, mock_llm, sample_code):
        agent = SecurityAgent(mock_llm)
        result = agent.analyze(sample_code)

        assert isinstance(result, dict)
        assert "agent" in result
        assert "type" in result
        assert result["type"] == "security"

    def test_system_prompt_exists(self, mock_llm):
        agent = SecurityAgent(mock_llm)
        assert len(agent.system_prompt) > 0
        assert "security" in agent.system_prompt.lower()


class TestPerformanceAgent:
    """Tests for PerformanceAgent."""

    def test_init(self, mock_llm):
        agent = PerformanceAgent(mock_llm)
        assert agent.name == "PerformanceAgent"

    def test_analyze_returns_dict(self, mock_llm, sample_code):
        agent = PerformanceAgent(mock_llm)
        result = agent.analyze(sample_code)

        assert isinstance(result, dict)
        assert result["type"] == "performance"


class TestStyleAgent:
    """Tests for StyleAgent."""

    def test_init(self, mock_llm):
        agent = StyleAgent(mock_llm)
        assert agent.name == "StyleAgent"

    def test_analyze_returns_dict(self, mock_llm, sample_code):
        agent = StyleAgent(mock_llm)
        result = agent.analyze(sample_code)

        assert isinstance(result, dict)
        assert result["type"] == "style"


class TestReviewOrchestrator:
    """Tests for ReviewOrchestrator."""

    def test_init(self, mock_llm):
        orchestrator = ReviewOrchestrator(mock_llm)
        assert orchestrator.security_agent is not None
        assert orchestrator.performance_agent is not None
        assert orchestrator.style_agent is not None

    def test_review_returns_all_analyses(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator.review(sample_code)

        assert "security" in result
        assert "performance" in result
        assert "style" in result
        assert "synthesis" in result

    def test_review_with_context(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        context = {"file_path": "test.py", "language": "python"}
        result = orchestrator.review(sample_code, context)

        assert result["context"] == context
