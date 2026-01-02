"""Tests for code analyzer."""
import pytest
from tools.code_analyzer import CodeAnalyzer


@pytest.fixture
def analyzer():
    return CodeAnalyzer()


@pytest.fixture
def python_code():
    return '''# Sample Python code
import os
from typing import List

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

def main():
    calc = Calculator()
    print(calc.add(1, 2))

if __name__ == "__main__":
    main()
'''


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer."""

    def test_detect_language_python(self, analyzer):
        assert analyzer.detect_language("test.py") == "python"

    def test_detect_language_javascript(self, analyzer):
        assert analyzer.detect_language("app.js") == "javascript"

    def test_detect_language_typescript(self, analyzer):
        assert analyzer.detect_language("component.tsx") == "typescript"

    def test_detect_language_unknown(self, analyzer):
        assert analyzer.detect_language("file.xyz") == "unknown"

    def test_extract_metrics_python(self, analyzer, python_code):
        metrics = analyzer.extract_metrics(python_code, "python")

        assert metrics.lines_total > 0
        assert metrics.functions >= 3  # add, multiply, main
        assert metrics.classes >= 1
        assert metrics.imports >= 1

    def test_extract_functions_python(self, analyzer, python_code):
        functions = analyzer.extract_functions(python_code, "python")

        assert len(functions) >= 3
        names = [f["name"] for f in functions]
        assert "add" in names
        assert "multiply" in names
        assert "main" in names

    def test_find_potential_issues_todo(self, analyzer):
        code = "# TODO: Fix this later\ndef foo(): pass"
        issues = analyzer.find_potential_issues(code, "python")

        assert len(issues) >= 1
        assert any("TODO" in i["message"] for i in issues)

    def test_find_potential_issues_hardcoded_password(self, analyzer):
        code = 'password = "secret123"'
        issues = analyzer.find_potential_issues(code, "python")

        assert len(issues) >= 1
        assert any("password" in i["message"].lower() for i in issues)

    def test_find_potential_issues_clean_code(self, analyzer):
        code = """
def add(a, b):
    return a + b
"""
        issues = analyzer.find_potential_issues(code, "python")
        # Should have minimal issues for clean code
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        assert len(critical_issues) == 0
