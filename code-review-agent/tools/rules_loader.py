"""Custom YAML rules loader for code review.

This module provides functionality to load, validate, and apply
custom code review rules defined in YAML format.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Rule:
    """A single code review rule."""
    id: str
    name: str
    description: str
    pattern: str
    severity: str  # critical, high, medium, low, info
    languages: list[str]
    category: str | None = None
    owasp_id: str | None = None
    cwe_id: str | None = None
    recommendation: str | None = None
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    _compiled_pattern: re.Pattern | None = field(default=None, repr=False)

    def __post_init__(self):
        """Compile the regex pattern after initialization."""
        if self.pattern:
            try:
                self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in rule {self.id}: {e}")

    @property
    def compiled_pattern(self) -> re.Pattern | None:
        """Get the compiled regex pattern."""
        return self._compiled_pattern

    def matches(self, line: str) -> bool:
        """Check if the line matches this rule's pattern."""
        if not self._compiled_pattern:
            return False
        return bool(self._compiled_pattern.search(line))

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "severity": self.severity,
            "languages": self.languages,
            "category": self.category,
            "owasp_id": self.owasp_id,
            "cwe_id": self.cwe_id,
            "recommendation": self.recommendation,
            "enabled": self.enabled,
            "tags": self.tags,
        }


@dataclass
class RuleSet:
    """A collection of code review rules."""
    name: str
    version: str
    description: str
    rules: list[Rule]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_rules_for_language(self, language: str) -> list[Rule]:
        """Get all enabled rules for a specific language."""
        return [
            rule for rule in self.rules
            if rule.enabled and (
                "all" in rule.languages or
                language.lower() in [lang.lower() for lang in rule.languages]
            )
        ]

    def get_rules_by_severity(self, severity: str) -> list[Rule]:
        """Get all enabled rules with a specific severity."""
        return [
            rule for rule in self.rules
            if rule.enabled and rule.severity.lower() == severity.lower()
        ]

    def get_rules_by_category(self, category: str) -> list[Rule]:
        """Get all enabled rules in a specific category."""
        return [
            rule for rule in self.rules
            if rule.enabled and rule.category and
            rule.category.lower() == category.lower()
        ]

    def get_rules_by_tag(self, tag: str) -> list[Rule]:
        """Get all enabled rules with a specific tag."""
        return [
            rule for rule in self.rules
            if rule.enabled and tag.lower() in [t.lower() for t in rule.tags]
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert ruleset to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "rules": [rule.to_dict() for rule in self.rules],
        }


class RulesLoader:
    """Loader for custom YAML rules."""

    VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}

    def __init__(self):
        """Initialize the rules loader."""
        self.rulesets: dict[str, RuleSet] = {}

    def load_from_file(self, file_path: str | Path) -> RuleSet:
        """Load rules from a YAML file.

        Args:
            file_path: Path to the YAML rules file

        Returns:
            RuleSet containing all loaded rules

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_ruleset(data, source=str(path))

    def load_from_string(self, yaml_content: str, name: str = "inline") -> RuleSet:
        """Load rules from a YAML string.

        Args:
            yaml_content: YAML content as a string
            name: Name for this ruleset

        Returns:
            RuleSet containing all loaded rules
        """
        data = yaml.safe_load(yaml_content)
        return self._parse_ruleset(data, source=name)

    def load_from_directory(self, directory: str | Path) -> list[RuleSet]:
        """Load all YAML rules from a directory.

        Args:
            directory: Path to directory containing YAML files

        Returns:
            List of RuleSets loaded from all YAML files
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        rulesets = []
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                ruleset = self.load_from_file(yaml_file)
                rulesets.append(ruleset)
                self.rulesets[ruleset.name] = ruleset
            except Exception as e:
                # Log but continue loading other files
                print(f"Warning: Failed to load {yaml_file}: {e}")

        for yml_file in dir_path.glob("*.yml"):
            try:
                ruleset = self.load_from_file(yml_file)
                rulesets.append(ruleset)
                self.rulesets[ruleset.name] = ruleset
            except Exception as e:
                print(f"Warning: Failed to load {yml_file}: {e}")

        return rulesets

    def _parse_ruleset(self, data: dict[str, Any], source: str = "unknown") -> RuleSet:
        """Parse a ruleset from dictionary data.

        Args:
            data: Dictionary containing ruleset data
            source: Source identifier for error messages

        Returns:
            Parsed RuleSet
        """
        if not isinstance(data, dict):
            raise ValueError(f"Invalid ruleset format in {source}: expected dict")

        # Get ruleset metadata
        name = data.get("name", source)
        version = data.get("version", "1.0.0")
        description = data.get("description", "")
        metadata = data.get("metadata", {})

        # Parse rules
        rules_data = data.get("rules", [])
        if not isinstance(rules_data, list):
            raise ValueError(f"Invalid rules format in {source}: expected list")

        rules = []
        for i, rule_data in enumerate(rules_data):
            try:
                rule = self._parse_rule(rule_data, f"{source}:rule[{i}]")
                rules.append(rule)
            except Exception as e:
                print(f"Warning: Failed to parse rule {i} in {source}: {e}")

        ruleset = RuleSet(
            name=name,
            version=version,
            description=description,
            rules=rules,
            metadata=metadata,
        )

        self.rulesets[name] = ruleset
        return ruleset

    def _parse_rule(self, data: dict[str, Any], source: str) -> Rule:
        """Parse a single rule from dictionary data.

        Args:
            data: Dictionary containing rule data
            source: Source identifier for error messages

        Returns:
            Parsed Rule
        """
        if not isinstance(data, dict):
            raise ValueError(f"Invalid rule format in {source}: expected dict")

        # Required fields
        rule_id = data.get("id")
        if not rule_id:
            raise ValueError(f"Missing required field 'id' in {source}")

        name = data.get("name")
        if not name:
            raise ValueError(f"Missing required field 'name' in {source}")

        pattern = data.get("pattern")
        if not pattern:
            raise ValueError(f"Missing required field 'pattern' in {source}")

        severity = data.get("severity", "medium").lower()
        if severity not in self.VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}' in {source}. "
                f"Valid values: {self.VALID_SEVERITIES}"
            )

        # Languages (default to "all")
        languages = data.get("languages", ["all"])
        if isinstance(languages, str):
            languages = [languages]

        # Optional fields
        description = data.get("description", "")
        category = data.get("category")
        owasp_id = data.get("owasp_id")
        cwe_id = data.get("cwe_id")
        recommendation = data.get("recommendation")
        enabled = data.get("enabled", True)
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        return Rule(
            id=rule_id,
            name=name,
            description=description,
            pattern=pattern,
            severity=severity,
            languages=languages,
            category=category,
            owasp_id=owasp_id,
            cwe_id=cwe_id,
            recommendation=recommendation,
            enabled=enabled,
            tags=tags,
        )

    def get_ruleset(self, name: str) -> RuleSet | None:
        """Get a loaded ruleset by name."""
        return self.rulesets.get(name)

    def get_all_rules(self) -> list[Rule]:
        """Get all rules from all loaded rulesets."""
        all_rules = []
        for ruleset in self.rulesets.values():
            all_rules.extend(ruleset.rules)
        return all_rules


class CustomRulesAnalyzer:
    """Analyzer that applies custom YAML rules to code."""

    def __init__(self, rulesets: list[RuleSet] | None = None):
        """Initialize the analyzer.

        Args:
            rulesets: List of RuleSets to use for analysis
        """
        self.rulesets = rulesets or []

    def add_ruleset(self, ruleset: RuleSet) -> None:
        """Add a ruleset to the analyzer."""
        self.rulesets.append(ruleset)

    def analyze(self, code: str, language: str) -> dict[str, Any]:
        """Analyze code using custom rules.

        Args:
            code: Source code to analyze
            language: Programming language of the code

        Returns:
            Analysis results with findings
        """
        findings = []
        lines = code.split("\n")

        # Collect all applicable rules
        applicable_rules = []
        for ruleset in self.rulesets:
            applicable_rules.extend(ruleset.get_rules_for_language(language))

        # Apply each rule
        for i, line in enumerate(lines, 1):
            for rule in applicable_rules:
                if rule.matches(line):
                    findings.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "description": rule.description,
                        "severity": rule.severity,
                        "category": rule.category,
                        "owasp_id": rule.owasp_id,
                        "cwe_id": rule.cwe_id,
                        "recommendation": rule.recommendation,
                        "location": f"line {i}",
                        "line_number": i,
                        "code_snippet": line.strip()[:100],
                        "tags": rule.tags,
                        "source": "custom_rules",
                    })

        # Generate summary
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for finding in findings:
            sev = finding["severity"].lower()
            if sev in severity_counts:
                severity_counts[sev] += 1

        return {
            "language": language,
            "total_issues": len(findings),
            "findings": findings,
            "severity_summary": severity_counts,
            "rulesets_used": [rs.name for rs in self.rulesets],
            "rules_applied": len(applicable_rules),
        }


# Example YAML rule format
EXAMPLE_YAML_RULES = """
# Custom Code Review Rules
# Format: YAML

name: my-custom-rules
version: "1.0.0"
description: Custom security and quality rules

metadata:
  author: Your Name
  created: 2024-01-01
  updated: 2024-01-01

rules:
  - id: CUSTOM-001
    name: Hardcoded AWS Key
    description: Detects hardcoded AWS access keys
    pattern: 'AKIA[0-9A-Z]{16}'
    severity: critical
    languages:
      - all
    category: security
    owasp_id: "A02:2021"
    cwe_id: "CWE-798"
    recommendation: Use environment variables or AWS IAM roles instead of hardcoding credentials
    tags:
      - aws
      - credentials
      - secrets

  - id: CUSTOM-002
    name: Debug Print Statement
    description: Detects debug print statements
    pattern: 'print\\s*\\('
    severity: info
    languages:
      - python
    category: code-quality
    recommendation: Remove debug print statements before production
    tags:
      - debug
      - cleanup

  - id: CUSTOM-003
    name: Console Log
    description: Detects console.log statements
    pattern: 'console\\.(log|debug|info|warn|error)\\s*\\('
    severity: info
    languages:
      - javascript
      - typescript
    category: code-quality
    recommendation: Remove console statements or use a proper logging framework
    tags:
      - debug
      - cleanup

  - id: CUSTOM-004
    name: SQL String Concatenation
    description: Detects potential SQL injection via string concatenation
    pattern: '(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE).*\\+.*[$\\{\\[]'
    severity: critical
    languages:
      - python
      - javascript
      - java
    category: security
    owasp_id: "A03:2021"
    cwe_id: "CWE-89"
    recommendation: Use parameterized queries instead of string concatenation
    tags:
      - sql
      - injection
      - security

  - id: CUSTOM-005
    name: TODO Comment
    description: Detects TODO comments
    pattern: 'TODO|FIXME|XXX|HACK'
    severity: info
    languages:
      - all
    category: code-quality
    recommendation: Address TODO items before production release
    tags:
      - todo
      - maintenance
"""


def get_example_rules() -> str:
    """Get example YAML rules content."""
    return EXAMPLE_YAML_RULES


def create_default_ruleset() -> RuleSet:
    """Create a default ruleset with common rules."""
    loader = RulesLoader()
    return loader.load_from_string(EXAMPLE_YAML_RULES, "default")
