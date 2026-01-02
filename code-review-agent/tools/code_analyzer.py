"""Static code analysis utilities."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeMetrics:
    """Basic code metrics."""
    lines_total: int
    lines_code: int
    lines_comment: int
    lines_blank: int
    functions: int
    classes: int
    imports: int


class CodeAnalyzer:
    """Utility for basic static code analysis."""

    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
    }

    def detect_language(self, filename: str) -> str:
        """Detect programming language from filename."""
        ext = Path(filename).suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext, "unknown")

    def extract_metrics(self, code: str, language: str) -> CodeMetrics:
        """Extract basic metrics from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            CodeMetrics with counts
        """
        lines = code.split("\n")
        lines_total = len(lines)
        lines_blank = sum(1 for line in lines if not line.strip())

        # Language-specific patterns
        if language == "python":
            comment_pattern = r'^\s*#'
            function_pattern = r'^\s*def\s+'
            class_pattern = r'^\s*class\s+'
            import_pattern = r'^\s*(import|from)\s+'
        elif language in ("javascript", "typescript"):
            comment_pattern = r'^\s*//'
            function_pattern = r'(function\s+\w+|const\s+\w+\s*=\s*\(|^\s*\w+\s*\([^)]*\)\s*{)'
            class_pattern = r'^\s*class\s+'
            import_pattern = r'^\s*(import|require)\s*'
        else:
            comment_pattern = r'^\s*//'
            function_pattern = r'function|def|func\s+'
            class_pattern = r'class\s+'
            import_pattern = r'import|require|use\s+'

        lines_comment = sum(1 for line in lines if re.match(comment_pattern, line))
        lines_code = lines_total - lines_blank - lines_comment

        functions = sum(1 for line in lines if re.search(function_pattern, line))
        classes = sum(1 for line in lines if re.search(class_pattern, line))
        imports = sum(1 for line in lines if re.search(import_pattern, line))

        return CodeMetrics(
            lines_total=lines_total,
            lines_code=lines_code,
            lines_comment=lines_comment,
            lines_blank=lines_blank,
            functions=functions,
            classes=classes,
            imports=imports
        )

    def extract_functions(self, code: str, language: str) -> list[dict]:
        """Extract function definitions from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of function info dicts
        """
        functions = []
        lines = code.split("\n")

        if language == "python":
            pattern = r'^\s*def\s+(\w+)\s*\('
        elif language in ("javascript", "typescript"):
            pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()'
        else:
            pattern = r'function\s+(\w+)'

        for i, line in enumerate(lines, 1):
            match = re.search(pattern, line)
            if match:
                name = match.group(1) or (match.group(2) if match.lastindex > 1 else None)
                if name:
                    functions.append({
                        "name": name,
                        "line": i,
                        "signature": line.strip()
                    })

        return functions

    def find_potential_issues(self, code: str, language: str) -> list[dict]:
        """Find common code issues using pattern matching.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of potential issues
        """
        issues = []
        lines = code.split("\n")

        # Common patterns to flag
        patterns = [
            (r'TODO|FIXME|XXX|HACK', "TODO/FIXME comment found", "info"),
            (r'print\(|console\.log\(', "Debug statement found", "warning"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "critical"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "critical"),
            (r'eval\(|exec\(', "Dangerous eval/exec usage", "critical"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message, severity in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "line": i,
                        "message": message,
                        "severity": severity,
                        "code": line.strip()[:100]
                    })

        return issues
