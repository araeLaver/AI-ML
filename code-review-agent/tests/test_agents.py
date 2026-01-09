"""Tests for review agents."""
import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from agents.base import BaseReviewAgent
from agents.security_agent import SecurityAgent
from agents.performance_agent import PerformanceAgent
from agents.style_agent import StyleAgent
from agents.orchestrator import ReviewOrchestrator, parse_json_safely


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.invoke = MagicMock(return_value=Mock(content='{"findings": [], "summary": "No issues found"}'))
    return llm


@pytest.fixture
def mock_llm_with_findings():
    """Create a mock LLM that returns findings."""
    llm = Mock()
    response = json.dumps({
        "findings": [
            {
                "severity": "HIGH",
                "location": "line 5",
                "title": "SQL Injection",
                "description": "User input not sanitized",
                "recommendation": "Use parameterized queries"
            }
        ],
        "summary": "Found 1 security issue"
    })
    llm.invoke = MagicMock(return_value=Mock(content=response))
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


@pytest.fixture
def vulnerable_code():
    """Sample code with security vulnerabilities."""
    return '''
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("db.sqlite")
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return conn.execute(query).fetchone()
'''


class TestParseJsonSafely:
    """Tests for JSON parsing utility."""

    def test_parse_valid_json(self):
        content = '{"findings": [], "summary": "test"}'
        result = parse_json_safely(content)
        assert result["findings"] == []
        assert result["summary"] == "test"

    def test_parse_json_in_code_block(self):
        content = '```json\n{"findings": [], "summary": "test"}\n```'
        result = parse_json_safely(content)
        assert result["findings"] == []

    def test_parse_json_in_generic_code_block(self):
        content = '```\n{"findings": [], "summary": "test"}\n```'
        result = parse_json_safely(content)
        assert result["findings"] == []

    def test_parse_invalid_json_returns_fallback(self):
        content = "This is not JSON"
        result = parse_json_safely(content, "TestAgent")
        assert result["parse_error"] is True
        assert result["findings"] == []
        assert "This is not JSON" in result["raw_content"]

    def test_parse_empty_string(self):
        result = parse_json_safely("", "TestAgent")
        assert result["parse_error"] is True


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

    def test_system_prompt_mentions_vulnerabilities(self, mock_llm):
        agent = SecurityAgent(mock_llm)
        prompt = agent.system_prompt.lower()
        assert "injection" in prompt
        assert "xss" in prompt

    def test_analyze_with_context(self, mock_llm, sample_code):
        agent = SecurityAgent(mock_llm)
        context = {"file_path": "test.py", "language": "python"}
        result = agent.analyze(sample_code, context)
        assert "analysis" in result


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

    def test_system_prompt_mentions_complexity(self, mock_llm):
        agent = PerformanceAgent(mock_llm)
        prompt = agent.system_prompt.lower()
        assert "complexity" in prompt
        assert "performance" in prompt


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

    def test_system_prompt_mentions_quality(self, mock_llm):
        agent = StyleAgent(mock_llm)
        prompt = agent.system_prompt.lower()
        assert "naming" in prompt or "style" in prompt


class TestReviewOrchestrator:
    """Tests for ReviewOrchestrator."""

    def test_init(self, mock_llm):
        orchestrator = ReviewOrchestrator(mock_llm)
        assert orchestrator.security_agent is not None
        assert orchestrator.performance_agent is not None
        assert orchestrator.style_agent is not None

    def test_init_with_timeout(self, mock_llm):
        orchestrator = ReviewOrchestrator(mock_llm, timeout=30.0)
        assert orchestrator.timeout == 30.0

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

    def test_review_synthesis_has_verdict(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator.review(sample_code)

        synthesis = result["synthesis"]
        assert "verdict" in synthesis
        assert synthesis["verdict"] in ["APPROVE", "REQUEST_CHANGES", "COMMENT"]

    def test_review_synthesis_has_health_score(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator.review(sample_code)

        synthesis = result["synthesis"]
        assert "health_score" in synthesis
        assert 1 <= synthesis["health_score"] <= 10

    def test_review_handles_agent_error(self, mock_llm, sample_code):
        """Test that orchestrator handles individual agent failures."""
        orchestrator = ReviewOrchestrator(mock_llm)
        # Make security agent fail
        orchestrator.security_agent.analyze = MagicMock(side_effect=Exception("API Error"))

        result = orchestrator.review(sample_code)

        # Should still return results from other agents
        assert "security" in result
        assert "error" in result["security"] or result.get("errors")

    def test_review_parallel_execution(self, mock_llm, sample_code):
        """Test that parallel execution works."""
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator._review_parallel(sample_code)

        assert "security" in result
        assert "performance" in result
        assert "style" in result

    def test_review_sequential_fallback(self, mock_llm, sample_code):
        """Test sequential execution fallback."""
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator._review_sequential(sample_code)

        assert "security" in result
        assert "synthesis" in result


class TestOrchestratorSynthesis:
    """Tests for orchestrator synthesis logic."""

    def test_synthesis_counts_issues(self, mock_llm_with_findings, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm_with_findings)
        result = orchestrator.review(sample_code)

        synthesis = result["synthesis"]
        assert "issue_counts" in synthesis

    def test_synthesis_generates_critical_issues_list(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator.review(sample_code)

        synthesis = result["synthesis"]
        assert "critical_issues" in synthesis
        assert isinstance(synthesis["critical_issues"], list)

    def test_synthesis_generates_improvements_list(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        result = orchestrator.review(sample_code)

        synthesis = result["synthesis"]
        assert "important_improvements" in synthesis
        assert isinstance(synthesis["important_improvements"], list)


@pytest.mark.asyncio
class TestAsyncReview:
    """Tests for async review functionality."""

    async def test_review_async_returns_results(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        result = await orchestrator.review_async(sample_code)

        assert "security" in result
        assert "performance" in result
        assert "style" in result

    async def test_review_async_with_context(self, mock_llm, sample_code):
        orchestrator = ReviewOrchestrator(mock_llm)
        context = {"file_path": "test.py", "language": "python"}
        result = await orchestrator.review_async(sample_code, context)

        assert result["context"] == context
