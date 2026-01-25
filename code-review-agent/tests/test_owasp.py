"""Tests for OWASP Top 10 agent and static analyzer."""
import pytest
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.owasp_agent import (
    OWASPAgent,
    OWASPStaticAnalyzer,
    OWASP_CATEGORIES,
    get_owasp_category,
    get_all_categories,
    get_language_patterns,
    LANGUAGE_PATTERNS,
)


# =============================================================================
# OWASP Categories Tests
# =============================================================================

class TestOWASPCategories:
    """Test OWASP category utilities."""

    def test_all_categories_exist(self):
        """Verify all 10 OWASP categories are defined."""
        assert len(OWASP_CATEGORIES) == 10

        expected = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10"]
        for cat_id in expected:
            assert cat_id in OWASP_CATEGORIES

    def test_category_structure(self):
        """Verify category structure."""
        for cat_id, category in OWASP_CATEGORIES.items():
            assert "id" in category
            assert "name" in category
            assert "description" in category
            assert "patterns" in category
            assert len(category["patterns"]) > 0

    def test_get_owasp_category(self):
        """Test getting category by ID."""
        cat = get_owasp_category("A01")
        assert cat is not None
        assert cat["name"] == "Broken Access Control"

        cat = get_owasp_category("A01:2021")
        assert cat is not None

        cat = get_owasp_category("a03")
        assert cat is not None
        assert cat["name"] == "Injection"

    def test_get_owasp_category_invalid(self):
        """Test getting invalid category."""
        cat = get_owasp_category("A99")
        assert cat is None

    def test_get_all_categories(self):
        """Test getting all categories list."""
        categories = get_all_categories()
        assert len(categories) == 10
        assert all("id" in cat for cat in categories)
        assert all("name" in cat for cat in categories)


# =============================================================================
# OWASPAgent Tests
# =============================================================================

class TestOWASPAgent:
    """Test OWASPAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        mock = Mock()
        mock.invoke.return_value = MagicMock(
            content='{"findings": [], "categories_checked": ["A01", "A02", "A03"], "risk_score": 0, "summary": "No issues found"}'
        )
        return mock

    @pytest.fixture
    def mock_llm_with_findings(self):
        """Create mock LLM with findings."""
        mock = Mock()
        mock.invoke.return_value = MagicMock(
            content='''{
                "findings": [
                    {
                        "owasp_id": "A03:2021",
                        "owasp_name": "Injection",
                        "severity": "HIGH",
                        "location": "line 10",
                        "title": "SQL Injection",
                        "description": "User input directly in query",
                        "code_snippet": "query = f\\"SELECT * FROM users WHERE id={user_id}\\"",
                        "recommendation": "Use parameterized queries",
                        "cwe_id": "CWE-89"
                    }
                ],
                "categories_checked": ["A01", "A02", "A03"],
                "risk_score": 7.5,
                "summary": "Found SQL injection vulnerability"
            }'''
        )
        return mock

    def test_agent_initialization(self, mock_llm):
        """Test agent initialization."""
        agent = OWASPAgent(mock_llm)

        assert agent.name == "OWASPAgent"
        assert agent.llm == mock_llm

    def test_system_prompt_contains_categories(self, mock_llm):
        """Test system prompt contains all OWASP categories."""
        agent = OWASPAgent(mock_llm)
        prompt = agent.system_prompt

        # Check all categories are mentioned
        for cat in OWASP_CATEGORIES.values():
            assert cat["id"] in prompt
            assert cat["name"] in prompt

    def test_analyze_returns_structure(self, mock_llm):
        """Test analyze returns correct structure."""
        agent = OWASPAgent(mock_llm)

        result = agent.analyze("def test(): pass", {"language": "python"})

        assert "agent" in result
        assert "type" in result
        assert "analysis" in result
        assert result["agent"] == "OWASPAgent"
        assert result["type"] == "owasp"

    def test_analyze_invokes_llm(self, mock_llm):
        """Test analyze invokes LLM."""
        agent = OWASPAgent(mock_llm)

        agent.analyze("def test(): pass")

        mock_llm.invoke.assert_called_once()

    def test_analyze_with_findings(self, mock_llm_with_findings):
        """Test analyze with findings."""
        agent = OWASPAgent(mock_llm_with_findings)

        result = agent.analyze("query = f'SELECT * FROM users WHERE id={user_id}'")

        assert "analysis" in result
        # The analysis should contain the JSON with findings
        assert "SQL Injection" in result["analysis"]

    def test_analyze_category_specific(self, mock_llm):
        """Test category-specific analysis."""
        agent = OWASPAgent(mock_llm)

        result = agent.analyze_category(
            "def test(): pass",
            "A03",
            {"language": "python"}
        )

        assert "category" in result
        assert result["category"] == "A03:2021"

    def test_analyze_category_invalid(self, mock_llm):
        """Test invalid category returns error."""
        agent = OWASPAgent(mock_llm)

        result = agent.analyze_category("code", "A99")

        assert "error" in result


# =============================================================================
# OWASPStaticAnalyzer Tests
# =============================================================================

class TestOWASPStaticAnalyzer:
    """Test OWASPStaticAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = OWASPStaticAnalyzer("python")

        assert analyzer.language == "python"
        assert len(analyzer.patterns) > 0

    def test_analyzer_unknown_language(self):
        """Test analyzer with unknown language."""
        analyzer = OWASPStaticAnalyzer("unknown_lang")

        assert analyzer.patterns == {}

    def test_analyze_empty_code(self):
        """Test analyzing empty code."""
        analyzer = OWASPStaticAnalyzer("python")

        result = analyzer.analyze("")

        assert result["findings"] == []
        assert result["language"] == "python"

    def test_analyze_safe_code(self):
        """Test analyzing safe code."""
        analyzer = OWASPStaticAnalyzer("python")

        safe_code = '''
def greet(name):
    """Greet a user."""
    return f"Hello, {name}!"
'''
        result = analyzer.analyze(safe_code)

        assert result["total_issues"] == 0

    def test_detect_sql_injection_python(self):
        """Test detecting SQL injection in Python."""
        analyzer = OWASPStaticAnalyzer("python")

        # Pattern matches execute() with + inside the call
        vulnerable_code = '''
def get_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id=" + user_id)
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any(f["owasp_id"] == "A03:2021" for f in result["findings"])

    def test_detect_weak_hash_python(self):
        """Test detecting weak hash algorithm."""
        analyzer = OWASPStaticAnalyzer("python")

        vulnerable_code = '''
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any("md5" in f["title"].lower() for f in result["findings"])

    def test_detect_command_injection_python(self):
        """Test detecting command injection."""
        analyzer = OWASPStaticAnalyzer("python")

        vulnerable_code = '''
import os
os.system("ls " + user_input)
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any("command" in f["title"].lower() for f in result["findings"])

    def test_detect_eval_python(self):
        """Test detecting eval usage."""
        analyzer = OWASPStaticAnalyzer("python")

        vulnerable_code = '''
result = eval(user_expression)
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any("eval" in f["code_snippet"].lower() for f in result["findings"])

    def test_detect_debug_mode_python(self):
        """Test detecting debug mode."""
        analyzer = OWASPStaticAnalyzer("python")

        vulnerable_code = '''
DEBUG = True
app.run(debug=True)
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0

    def test_detect_pickle_python(self):
        """Test detecting insecure deserialization."""
        analyzer = OWASPStaticAnalyzer("python")

        vulnerable_code = '''
import pickle
data = pickle.load(open("data.pkl", "rb"))
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any(f["owasp_id"] == "A08:2021" for f in result["findings"])


# =============================================================================
# JavaScript Static Analysis Tests
# =============================================================================

class TestOWASPStaticAnalyzerJavaScript:
    """Test OWASP static analysis for JavaScript."""

    def test_detect_innerHTML_xss(self):
        """Test detecting innerHTML XSS."""
        analyzer = OWASPStaticAnalyzer("javascript")

        vulnerable_code = '''
document.getElementById("content").innerHTML = userInput;
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any("innerHTML" in f["title"] for f in result["findings"])

    def test_detect_eval_js(self):
        """Test detecting eval in JavaScript."""
        analyzer = OWASPStaticAnalyzer("javascript")

        vulnerable_code = '''
const result = eval(userCode);
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0

    def test_detect_weak_random(self):
        """Test detecting Math.random for crypto."""
        analyzer = OWASPStaticAnalyzer("javascript")

        vulnerable_code = '''
const token = Math.random().toString(36);
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0


# =============================================================================
# Java Static Analysis Tests
# =============================================================================

class TestOWASPStaticAnalyzerJava:
    """Test OWASP static analysis for Java."""

    def test_detect_sql_injection_java(self):
        """Test detecting SQL injection in Java."""
        analyzer = OWASPStaticAnalyzer("java")

        # Pattern matches createQuery with + (HQL/JPA injection)
        vulnerable_code = '''
String query = "SELECT u FROM User u WHERE u.id = " + userId;
Query q = em.createQuery(query + " ORDER BY name");
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0
        assert any(f["owasp_id"] == "A03:2021" for f in result["findings"])

    def test_detect_weak_hash_java(self):
        """Test detecting weak hash in Java."""
        analyzer = OWASPStaticAnalyzer("java")

        vulnerable_code = '''
MessageDigest md = MessageDigest.getInstance("MD5");
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0

    def test_detect_insecure_random_java(self):
        """Test detecting insecure random in Java."""
        analyzer = OWASPStaticAnalyzer("java")

        vulnerable_code = '''
Random rand = new Random();
int token = rand.nextInt();
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0


# =============================================================================
# Go Static Analysis Tests
# =============================================================================

class TestOWASPStaticAnalyzerGo:
    """Test OWASP static analysis for Go."""

    def test_detect_sql_injection_go(self):
        """Test detecting SQL injection in Go."""
        analyzer = OWASPStaticAnalyzer("go")

        # Pattern matches db.Query with +
        vulnerable_code = '''
db.Query("SELECT * FROM users WHERE id=" + userId)
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0

    def test_detect_weak_hash_go(self):
        """Test detecting weak hash in Go."""
        analyzer = OWASPStaticAnalyzer("go")

        vulnerable_code = '''
import "crypto/md5"
hash := md5.Sum([]byte(data))
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0

    def test_detect_insecure_tls_go(self):
        """Test detecting insecure TLS in Go."""
        analyzer = OWASPStaticAnalyzer("go")

        vulnerable_code = '''
config := &tls.Config{
    InsecureSkipVerify: true,
}
'''
        result = analyzer.analyze(vulnerable_code)

        assert result["total_issues"] > 0


# =============================================================================
# Language Patterns Tests
# =============================================================================

class TestLanguagePatterns:
    """Test language pattern utilities."""

    def test_get_language_patterns_python(self):
        """Test getting Python patterns."""
        patterns = get_language_patterns("python")

        assert len(patterns) > 0
        assert "A03" in patterns  # Injection

    def test_get_language_patterns_javascript(self):
        """Test getting JavaScript patterns."""
        patterns = get_language_patterns("javascript")

        assert len(patterns) > 0

    def test_get_language_patterns_java(self):
        """Test getting Java patterns."""
        patterns = get_language_patterns("java")

        assert len(patterns) > 0

    def test_get_language_patterns_go(self):
        """Test getting Go patterns."""
        patterns = get_language_patterns("go")

        assert len(patterns) > 0

    def test_get_language_patterns_unknown(self):
        """Test getting patterns for unknown language."""
        patterns = get_language_patterns("cobol")

        assert patterns == {}

    def test_supported_languages(self):
        """Test all supported languages have patterns."""
        supported = ["python", "javascript", "java", "go"]

        for lang in supported:
            assert lang in LANGUAGE_PATTERNS
            assert len(LANGUAGE_PATTERNS[lang]) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestOWASPIntegration:
    """Integration tests for OWASP analysis."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        # Static analysis
        analyzer = OWASPStaticAnalyzer("python")

        vulnerable_code = '''
import hashlib
import os
import pickle

def process_user(user_id, data):
    # SQL Injection
    query = "SELECT * FROM users WHERE id=" + user_id
    cursor.execute(query)

    # Weak hash
    password_hash = hashlib.md5(data['password'].encode()).hexdigest()

    # Command injection
    os.system("process " + data['file'])

    # Insecure deserialization
    config = pickle.load(open("config.pkl", "rb"))

    # Debug mode
    DEBUG = True

    return {"hash": password_hash, "debug": DEBUG}
'''
        result = analyzer.analyze(vulnerable_code)

        # Should find multiple issues
        assert result["total_issues"] >= 4

        # Should find issues in multiple OWASP categories
        categories_found = set(f["owasp_id"] for f in result["findings"])
        assert "A02:2021" in categories_found  # Cryptographic Failures (MD5)
        assert "A03:2021" in categories_found  # Injection
        assert "A08:2021" in categories_found  # Insecure Deserialization

    def test_finding_structure(self):
        """Test finding structure is correct."""
        analyzer = OWASPStaticAnalyzer("python")

        code = 'os.system("ls " + user_input)'
        result = analyzer.analyze(code)

        assert result["total_issues"] > 0

        finding = result["findings"][0]
        assert "owasp_id" in finding
        assert "owasp_name" in finding
        assert "severity" in finding
        assert "location" in finding
        assert "title" in finding
        assert "code_snippet" in finding
        assert "source" in finding
        assert finding["source"] == "static_analysis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
