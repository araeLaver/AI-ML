"""Tests for custom YAML rules loader."""
import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rules_loader import (
    Rule,
    RuleSet,
    RulesLoader,
    CustomRulesAnalyzer,
    get_example_rules,
    create_default_ruleset,
)


# =============================================================================
# Rule Tests
# =============================================================================

class TestRule:
    """Tests for Rule dataclass."""

    def test_rule_creation(self):
        """Test creating a rule."""
        rule = Rule(
            id="TEST-001",
            name="Test Rule",
            description="A test rule",
            pattern=r"test_pattern",
            severity="high",
            languages=["python"],
        )

        assert rule.id == "TEST-001"
        assert rule.name == "Test Rule"
        assert rule.severity == "high"
        assert rule.enabled is True

    def test_rule_pattern_compilation(self):
        """Test that pattern is compiled correctly."""
        rule = Rule(
            id="TEST-001",
            name="Test Rule",
            description="Test",
            pattern=r"\beval\s*\(",
            severity="critical",
            languages=["python"],
        )

        assert rule.compiled_pattern is not None
        assert rule.matches("result = eval(user_input)")
        assert not rule.matches("result = safe_function()")

    def test_rule_invalid_pattern(self):
        """Test that invalid regex raises error."""
        with pytest.raises(ValueError):
            Rule(
                id="TEST-001",
                name="Test Rule",
                description="Test",
                pattern=r"[invalid",  # Invalid regex
                severity="high",
                languages=["python"],
            )

    def test_rule_to_dict(self):
        """Test converting rule to dictionary."""
        rule = Rule(
            id="TEST-001",
            name="Test Rule",
            description="Test",
            pattern=r"test",
            severity="medium",
            languages=["python"],
            owasp_id="A03:2021",
            tags=["security"],
        )

        d = rule.to_dict()
        assert d["id"] == "TEST-001"
        assert d["owasp_id"] == "A03:2021"
        assert d["tags"] == ["security"]


# =============================================================================
# RuleSet Tests
# =============================================================================

class TestRuleSet:
    """Tests for RuleSet class."""

    @pytest.fixture
    def sample_ruleset(self):
        """Create a sample ruleset for testing."""
        rules = [
            Rule(
                id="RULE-001",
                name="Python Eval",
                description="Detects eval usage",
                pattern=r"eval\s*\(",
                severity="critical",
                languages=["python"],
                tags=["security"],
            ),
            Rule(
                id="RULE-002",
                name="JS Console",
                description="Detects console.log",
                pattern=r"console\.log\s*\(",
                severity="info",
                languages=["javascript", "typescript"],
                tags=["debug"],
            ),
            Rule(
                id="RULE-003",
                name="All Lang TODO",
                description="Detects TODO comments",
                pattern=r"TODO",
                severity="info",
                languages=["all"],
                category="maintenance",
                tags=["todo"],
            ),
            Rule(
                id="RULE-004",
                name="Disabled Rule",
                description="This rule is disabled",
                pattern=r"disabled",
                severity="low",
                languages=["all"],
                enabled=False,
            ),
        ]
        return RuleSet(
            name="test-ruleset",
            version="1.0.0",
            description="Test ruleset",
            rules=rules,
        )

    def test_get_rules_for_language(self, sample_ruleset):
        """Test getting rules for a specific language."""
        python_rules = sample_ruleset.get_rules_for_language("python")

        # Should include Python-specific rule and "all" language rule
        assert len(python_rules) >= 2
        rule_ids = [r.id for r in python_rules]
        assert "RULE-001" in rule_ids  # Python eval
        assert "RULE-003" in rule_ids  # All lang TODO
        assert "RULE-002" not in rule_ids  # JS console
        assert "RULE-004" not in rule_ids  # Disabled

    def test_get_rules_for_javascript(self, sample_ruleset):
        """Test getting rules for JavaScript."""
        js_rules = sample_ruleset.get_rules_for_language("javascript")

        rule_ids = [r.id for r in js_rules]
        assert "RULE-002" in rule_ids  # JS console
        assert "RULE-003" in rule_ids  # All lang TODO

    def test_get_rules_by_severity(self, sample_ruleset):
        """Test getting rules by severity."""
        critical_rules = sample_ruleset.get_rules_by_severity("critical")

        assert len(critical_rules) == 1
        assert critical_rules[0].id == "RULE-001"

    def test_get_rules_by_category(self, sample_ruleset):
        """Test getting rules by category."""
        maintenance_rules = sample_ruleset.get_rules_by_category("maintenance")

        assert len(maintenance_rules) == 1
        assert maintenance_rules[0].id == "RULE-003"

    def test_get_rules_by_tag(self, sample_ruleset):
        """Test getting rules by tag."""
        security_rules = sample_ruleset.get_rules_by_tag("security")

        assert len(security_rules) == 1
        assert security_rules[0].id == "RULE-001"

    def test_disabled_rules_excluded(self, sample_ruleset):
        """Test that disabled rules are excluded."""
        all_rules = sample_ruleset.get_rules_for_language("python")
        rule_ids = [r.id for r in all_rules]
        assert "RULE-004" not in rule_ids


# =============================================================================
# RulesLoader Tests
# =============================================================================

class TestRulesLoader:
    """Tests for RulesLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a RulesLoader instance."""
        return RulesLoader()

    @pytest.fixture
    def sample_yaml(self):
        """Sample YAML content for testing."""
        return """
name: test-rules
version: "1.0.0"
description: Test rules for unit testing

rules:
  - id: TEST-001
    name: Test Eval
    description: Detects eval
    pattern: "eval\\\\s*\\\\("
    severity: critical
    languages:
      - python
    owasp_id: "A03:2021"
    tags:
      - security

  - id: TEST-002
    name: Test Print
    description: Detects print
    pattern: "print\\\\s*\\\\("
    severity: info
    languages:
      - python
"""

    def test_load_from_string(self, loader, sample_yaml):
        """Test loading rules from YAML string."""
        ruleset = loader.load_from_string(sample_yaml, "test")

        assert ruleset.name == "test-rules"
        assert ruleset.version == "1.0.0"
        assert len(ruleset.rules) == 2

    def test_load_from_file(self, loader, sample_yaml):
        """Test loading rules from a file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(sample_yaml)
            temp_path = f.name

        try:
            ruleset = loader.load_from_file(temp_path)
            assert ruleset.name == "test-rules"
            assert len(ruleset.rules) == 2
        finally:
            os.unlink(temp_path)

    def test_load_from_directory(self, loader):
        """Test loading rules from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test YAML files
            yaml1 = """
name: rules-1
version: "1.0"
description: Rules set 1
rules:
  - id: R1-001
    name: Rule 1
    pattern: "pattern1"
    severity: high
    languages: [python]
"""
            yaml2 = """
name: rules-2
version: "1.0"
description: Rules set 2
rules:
  - id: R2-001
    name: Rule 2
    pattern: "pattern2"
    severity: low
    languages: [javascript]
"""
            with open(os.path.join(temp_dir, "rules1.yaml"), "w") as f:
                f.write(yaml1)
            with open(os.path.join(temp_dir, "rules2.yml"), "w") as f:
                f.write(yaml2)

            rulesets = loader.load_from_directory(temp_dir)

            assert len(rulesets) == 2

    def test_load_file_not_found(self, loader):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("non_existent.yaml")

    def test_invalid_yaml_format(self, loader):
        """Test that invalid YAML format raises error."""
        invalid_yaml = "rules: 'not a list'"

        with pytest.raises(ValueError):
            loader.load_from_string(invalid_yaml)

    def test_missing_required_field(self, loader):
        """Test that missing required field is handled."""
        yaml_missing_id = """
name: test
version: "1.0"
rules:
  - name: No ID Rule
    pattern: "test"
    severity: high
    languages: [python]
"""
        # Should handle gracefully (with warning)
        ruleset = loader.load_from_string(yaml_missing_id)
        assert len(ruleset.rules) == 0  # Rule should be skipped

    def test_invalid_severity(self, loader):
        """Test that invalid severity is handled."""
        yaml_invalid_severity = """
name: test
version: "1.0"
rules:
  - id: TEST-001
    name: Test Rule
    pattern: "test"
    severity: super_high
    languages: [python]
"""
        # Should handle gracefully (with warning)
        ruleset = loader.load_from_string(yaml_invalid_severity)
        assert len(ruleset.rules) == 0  # Rule should be skipped


# =============================================================================
# CustomRulesAnalyzer Tests
# =============================================================================

class TestCustomRulesAnalyzer:
    """Tests for CustomRulesAnalyzer class."""

    @pytest.fixture
    def analyzer_with_rules(self):
        """Create an analyzer with sample rules."""
        rules = [
            Rule(
                id="EVAL-001",
                name="Python Eval",
                description="Detects eval usage",
                pattern=r"eval\s*\(",
                severity="critical",
                languages=["python"],
                owasp_id="A03:2021",
                recommendation="Use ast.literal_eval instead",
            ),
            Rule(
                id="PRINT-001",
                name="Debug Print",
                description="Detects print statements",
                pattern=r"print\s*\(",
                severity="info",
                languages=["python"],
            ),
            Rule(
                id="TODO-001",
                name="TODO Comment",
                description="Detects TODO",
                pattern=r"TODO",
                severity="info",
                languages=["all"],
            ),
        ]
        ruleset = RuleSet(
            name="test-rules",
            version="1.0.0",
            description="Test",
            rules=rules,
        )
        return CustomRulesAnalyzer([ruleset])

    def test_analyze_finds_issues(self, analyzer_with_rules):
        """Test that analyzer finds issues."""
        code = """
result = eval(user_input)
print("Debug output")
# TODO: Fix this
"""
        result = analyzer_with_rules.analyze(code, "python")

        assert result["total_issues"] == 3
        assert result["language"] == "python"

    def test_analyze_returns_finding_structure(self, analyzer_with_rules):
        """Test finding structure."""
        code = "result = eval(user_input)"
        result = analyzer_with_rules.analyze(code, "python")

        assert len(result["findings"]) >= 1
        finding = result["findings"][0]

        assert "rule_id" in finding
        assert "rule_name" in finding
        assert "severity" in finding
        assert "line_number" in finding
        assert "code_snippet" in finding
        assert finding["source"] == "custom_rules"

    def test_analyze_severity_summary(self, analyzer_with_rules):
        """Test severity summary."""
        code = """
result = eval(input)
print("test")
# TODO
"""
        result = analyzer_with_rules.analyze(code, "python")

        summary = result["severity_summary"]
        assert summary["critical"] >= 1  # eval
        assert summary["info"] >= 2  # print, TODO

    def test_analyze_language_specific_rules(self, analyzer_with_rules):
        """Test that language-specific rules are applied correctly."""
        code = """
print("test")
# TODO: something
"""
        # Analyze as JavaScript
        result = analyzer_with_rules.analyze(code, "javascript")

        # Should only find TODO (all languages) not print (python only)
        findings = result["findings"]
        rule_ids = [f["rule_id"] for f in findings]
        assert "TODO-001" in rule_ids
        assert "PRINT-001" not in rule_ids

    def test_analyze_empty_code(self, analyzer_with_rules):
        """Test analyzing empty code."""
        result = analyzer_with_rules.analyze("", "python")

        assert result["total_issues"] == 0
        assert result["findings"] == []

    def test_add_ruleset(self):
        """Test adding a ruleset to analyzer."""
        analyzer = CustomRulesAnalyzer()

        rules = [
            Rule(
                id="NEW-001",
                name="New Rule",
                pattern=r"new_pattern",
                description="Test",
                severity="low",
                languages=["all"],
            )
        ]
        ruleset = RuleSet(
            name="new-rules",
            version="1.0",
            description="New",
            rules=rules,
        )

        analyzer.add_ruleset(ruleset)
        assert len(analyzer.rulesets) == 1


# =============================================================================
# Example Rules Tests
# =============================================================================

class TestExampleRules:
    """Tests for example rules functionality."""

    def test_get_example_rules_returns_yaml(self):
        """Test that example rules returns valid YAML."""
        yaml_content = get_example_rules()

        assert yaml_content is not None
        assert "name:" in yaml_content
        assert "rules:" in yaml_content

    def test_create_default_ruleset(self):
        """Test creating default ruleset."""
        ruleset = create_default_ruleset()

        assert ruleset is not None
        assert ruleset.name == "my-custom-rules"
        assert len(ruleset.rules) > 0

    def test_default_rules_are_functional(self):
        """Test that default rules can detect issues."""
        ruleset = create_default_ruleset()
        analyzer = CustomRulesAnalyzer([ruleset])

        # Test with code containing issues
        code = """
AKIAIOSFODNN7EXAMPLE  # AWS key pattern
password = "supersecret123"
result = eval(user_input)
# TODO: fix this
"""
        result = analyzer.analyze(code, "python")

        assert result["total_issues"] > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestRulesIntegration:
    """Integration tests for rules system."""

    def test_load_and_analyze_workflow(self):
        """Test complete workflow from loading to analysis."""
        yaml_rules = """
name: integration-test
version: "1.0.0"
description: Integration test rules

rules:
  - id: INT-001
    name: SQL Injection
    pattern: 'execute\\s*\\([^)]*\\+'
    severity: critical
    languages: [python]
    owasp_id: "A03:2021"
    recommendation: Use parameterized queries

  - id: INT-002
    name: Hardcoded Password
    pattern: 'password\\s*=\\s*[''"][^''"]+[''""]'
    severity: critical
    languages: [all]
"""
        # Load rules
        loader = RulesLoader()
        ruleset = loader.load_from_string(yaml_rules, "integration")

        # Create analyzer
        analyzer = CustomRulesAnalyzer([ruleset])

        # Analyze vulnerable code
        vulnerable_code = '''
password = "secret123"
def get_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id=" + user_id)
'''
        result = analyzer.analyze(vulnerable_code, "python")

        # Verify findings
        assert result["total_issues"] >= 2
        findings = result["findings"]

        # Check SQL injection found
        sql_findings = [f for f in findings if f["rule_id"] == "INT-001"]
        assert len(sql_findings) >= 1

        # Check password found
        pwd_findings = [f for f in findings if f["rule_id"] == "INT-002"]
        assert len(pwd_findings) >= 1

    def test_load_actual_rules_files(self):
        """Test loading actual rules files from the rules directory."""
        rules_dir = Path(__file__).parent.parent / "rules"

        if rules_dir.exists():
            loader = RulesLoader()
            rulesets = loader.load_from_directory(rules_dir)

            assert len(rulesets) >= 1  # At least one rules file

            # Verify each ruleset
            for ruleset in rulesets:
                assert ruleset.name is not None
                assert len(ruleset.rules) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
