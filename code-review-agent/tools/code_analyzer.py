"""Static code analysis utilities."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    interfaces: int = 0
    structs: int = 0
    enums: int = 0


@dataclass
class LanguageConfig:
    """Language-specific configuration for code analysis."""
    name: str
    extensions: list[str]
    comment_single: str
    comment_multi_start: str | None = None
    comment_multi_end: str | None = None
    function_patterns: list[str] = field(default_factory=list)
    class_patterns: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)
    interface_patterns: list[str] = field(default_factory=list)
    struct_patterns: list[str] = field(default_factory=list)
    enum_patterns: list[str] = field(default_factory=list)
    issue_patterns: list[tuple[str, str, str]] = field(default_factory=list)


# Language configurations
LANGUAGE_CONFIGS = {
    "python": LanguageConfig(
        name="python",
        extensions=[".py"],
        comment_single=r'^\s*#',
        comment_multi_start=r'^\s*"""',
        comment_multi_end=r'"""',
        function_patterns=[r'^\s*def\s+(\w+)\s*\('],
        class_patterns=[r'^\s*class\s+(\w+)'],
        import_patterns=[r'^\s*(import|from)\s+'],
        issue_patterns=[
            (r'print\(', "Debug print statement", "info"),
            (r'eval\(|exec\(', "Dangerous eval/exec usage", "critical"),
            (r'pickle\.load', "Insecure deserialization", "warning"),
            (r'assert\s+', "Assert in production code", "info"),
        ],
    ),
    "javascript": LanguageConfig(
        name="javascript",
        extensions=[".js", ".jsx"],
        comment_single=r'^\s*//',
        comment_multi_start=r'/\*',
        comment_multi_end=r'\*/',
        function_patterns=[
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*(?:async\s*)?\(',
            r'(\w+)\s*:\s*(?:async\s*)?function',
            r'(\w+)\s*\([^)]*\)\s*=>',
        ],
        class_patterns=[r'class\s+(\w+)'],
        import_patterns=[r'import\s+', r'require\s*\('],
        issue_patterns=[
            (r'console\.log\(', "Debug console.log", "info"),
            (r'eval\(', "Dangerous eval usage", "critical"),
            (r'innerHTML\s*=', "Potential XSS vulnerability", "warning"),
            (r'document\.write', "Unsafe document.write", "warning"),
        ],
    ),
    "typescript": LanguageConfig(
        name="typescript",
        extensions=[".ts", ".tsx"],
        comment_single=r'^\s*//',
        comment_multi_start=r'/\*',
        comment_multi_end=r'\*/',
        function_patterns=[
            r'function\s+(\w+)\s*[<(]',
            r'const\s+(\w+)\s*=\s*(?:async\s*)?\(',
            r'(\w+)\s*\([^)]*\)\s*:\s*\w+',
        ],
        class_patterns=[r'class\s+(\w+)'],
        import_patterns=[r'import\s+'],
        interface_patterns=[r'interface\s+(\w+)'],
        enum_patterns=[r'enum\s+(\w+)'],
        issue_patterns=[
            (r'console\.log\(', "Debug console.log", "info"),
            (r'any\s*[;,)]', "Usage of 'any' type", "warning"),
            (r'@ts-ignore', "TypeScript ignore directive", "warning"),
        ],
    ),
    "java": LanguageConfig(
        name="java",
        extensions=[".java"],
        comment_single=r'^\s*//',
        comment_multi_start=r'/\*',
        comment_multi_end=r'\*/',
        function_patterns=[
            r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
            r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:[\w<>\[\]]+)\s+(\w+)\s*\(',
        ],
        class_patterns=[
            r'(?:public|private|protected)?\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
        ],
        import_patterns=[r'import\s+(?:static\s+)?[\w.]+;'],
        interface_patterns=[r'(?:public\s+)?interface\s+(\w+)'],
        enum_patterns=[r'(?:public\s+)?enum\s+(\w+)'],
        issue_patterns=[
            (r'System\.out\.print', "Debug System.out", "info"),
            (r'e\.printStackTrace\(\)', "printStackTrace in production", "warning"),
            (r'catch\s*\(\s*Exception\s+', "Catching generic Exception", "warning"),
            (r'@SuppressWarnings', "Suppressed warnings", "info"),
            (r'new\s+Random\(\)', "Use SecureRandom for security", "warning"),
            (r'MessageDigest\.getInstance\(["\']MD5', "Weak hash MD5", "critical"),
            (r'MessageDigest\.getInstance\(["\']SHA-?1', "Weak hash SHA1", "warning"),
            (r'\.equals\(\s*null\s*\)', "Null comparison with equals", "warning"),
            (r'synchronized\s*\(this\)', "Synchronizing on this", "warning"),
        ],
    ),
    "go": LanguageConfig(
        name="go",
        extensions=[".go"],
        comment_single=r'^\s*//',
        comment_multi_start=r'/\*',
        comment_multi_end=r'\*/',
        function_patterns=[
            r'func\s+(\w+)\s*\(',
            r'func\s+\([^)]+\)\s+(\w+)\s*\(',  # Method with receiver
        ],
        class_patterns=[],  # Go doesn't have classes
        import_patterns=[r'import\s+[("]+'],
        interface_patterns=[r'type\s+(\w+)\s+interface\s*\{'],
        struct_patterns=[r'type\s+(\w+)\s+struct\s*\{'],
        issue_patterns=[
            (r'fmt\.Print', "Debug fmt.Print", "info"),
            (r'panic\(', "Panic usage", "warning"),
            (r'_\s*=\s*err', "Ignored error", "warning"),
            (r'md5\.', "Weak hash MD5", "critical"),
            (r'sha1\.', "Weak hash SHA1", "warning"),
            (r'rand\.(Int|Intn|Float)', "Use crypto/rand for security", "warning"),
            (r'InsecureSkipVerify:\s*true', "TLS verification disabled", "critical"),
            (r'defer\s+\w+\.Close\(\)\s*$', "Deferred close without error check", "info"),
        ],
    ),
    "rust": LanguageConfig(
        name="rust",
        extensions=[".rs"],
        comment_single=r'^\s*//',
        comment_multi_start=r'/\*',
        comment_multi_end=r'\*/',
        function_patterns=[r'fn\s+(\w+)\s*[<(]'],
        class_patterns=[],
        import_patterns=[r'use\s+[\w:]+'],
        struct_patterns=[r'struct\s+(\w+)'],
        enum_patterns=[r'enum\s+(\w+)'],
        issue_patterns=[
            (r'println!\(', "Debug println", "info"),
            (r'unwrap\(\)', "Unwrap without error handling", "warning"),
            (r'expect\(["\']', "Expect with message", "info"),
            (r'unsafe\s*\{', "Unsafe block", "warning"),
        ],
    ),
}


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
        ".kt": "kotlin",
        ".swift": "swift",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
    }

    def detect_language(self, filename: str) -> str:
        """Detect programming language from filename."""
        ext = Path(filename).suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext, "unknown")

    def get_language_config(self, language: str) -> LanguageConfig | None:
        """Get language configuration."""
        return LANGUAGE_CONFIGS.get(language)

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

        config = self.get_language_config(language)

        if config:
            comment_pattern = config.comment_single
            function_patterns = config.function_patterns
            class_patterns = config.class_patterns
            import_patterns = config.import_patterns
            interface_patterns = config.interface_patterns
            struct_patterns = config.struct_patterns
            enum_patterns = config.enum_patterns
        else:
            # Fallback for unknown languages
            comment_pattern = r'^\s*//'
            function_patterns = [r'function|def|func\s+']
            class_patterns = [r'class\s+']
            import_patterns = [r'import|require|use\s+']
            interface_patterns = []
            struct_patterns = []
            enum_patterns = []

        # Count comments (single-line only for simplicity)
        lines_comment = sum(1 for line in lines if re.match(comment_pattern, line))
        lines_code = lines_total - lines_blank - lines_comment

        # Count definitions
        functions = sum(
            1 for line in lines
            for pattern in function_patterns
            if re.search(pattern, line)
        )
        classes = sum(
            1 for line in lines
            for pattern in class_patterns
            if re.search(pattern, line)
        )
        imports = sum(
            1 for line in lines
            for pattern in import_patterns
            if re.search(pattern, line)
        )
        interfaces = sum(
            1 for line in lines
            for pattern in interface_patterns
            if re.search(pattern, line)
        )
        structs = sum(
            1 for line in lines
            for pattern in struct_patterns
            if re.search(pattern, line)
        )
        enums = sum(
            1 for line in lines
            for pattern in enum_patterns
            if re.search(pattern, line)
        )

        return CodeMetrics(
            lines_total=lines_total,
            lines_code=lines_code,
            lines_comment=lines_comment,
            lines_blank=lines_blank,
            functions=functions,
            classes=classes,
            imports=imports,
            interfaces=interfaces,
            structs=structs,
            enums=enums,
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
        config = self.get_language_config(language)

        if config and config.function_patterns:
            patterns = config.function_patterns
        elif language == "python":
            patterns = [r'^\s*def\s+(\w+)\s*\(']
        elif language in ("javascript", "typescript"):
            patterns = [r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()']
        elif language == "java":
            patterns = [r'(?:public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(']
        elif language == "go":
            patterns = [r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(']
        else:
            patterns = [r'function\s+(\w+)']

        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # Get the first non-None group
                    name = None
                    for g in match.groups():
                        if g:
                            name = g
                            break
                    if name:
                        functions.append({
                            "name": name,
                            "line": i,
                            "signature": line.strip()
                        })
                        break  # Only match one pattern per line

        return functions

    def extract_interfaces(self, code: str, language: str) -> list[dict]:
        """Extract interface definitions (for languages that support them).

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of interface info dicts
        """
        interfaces = []
        lines = code.split("\n")
        config = self.get_language_config(language)

        if not config or not config.interface_patterns:
            return interfaces

        for i, line in enumerate(lines, 1):
            for pattern in config.interface_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) if match.groups() else None
                    if name:
                        interfaces.append({
                            "name": name,
                            "line": i,
                            "signature": line.strip()
                        })

        return interfaces

    def extract_structs(self, code: str, language: str) -> list[dict]:
        """Extract struct definitions (for Go, Rust, etc.).

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of struct info dicts
        """
        structs = []
        lines = code.split("\n")
        config = self.get_language_config(language)

        if not config or not config.struct_patterns:
            return structs

        for i, line in enumerate(lines, 1):
            for pattern in config.struct_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) if match.groups() else None
                    if name:
                        structs.append({
                            "name": name,
                            "line": i,
                            "signature": line.strip()
                        })

        return structs

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

        # Common patterns for all languages
        common_patterns = [
            (r'TODO|FIXME|XXX|HACK', "TODO/FIXME comment found", "info"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "critical"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "critical"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret", "critical"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token", "warning"),
        ]

        # Get language-specific patterns
        config = self.get_language_config(language)
        language_patterns = config.issue_patterns if config else []

        # Combine patterns
        all_patterns = common_patterns + list(language_patterns)

        for i, line in enumerate(lines, 1):
            for pattern, message, severity in all_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "line": i,
                        "message": message,
                        "severity": severity,
                        "code": line.strip()[:100],
                        "language": language,
                    })

        return issues

    def get_full_analysis(self, code: str, language: str) -> dict[str, Any]:
        """Get complete code analysis including metrics, functions, and issues.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Complete analysis results
        """
        metrics = self.extract_metrics(code, language)
        functions = self.extract_functions(code, language)
        interfaces = self.extract_interfaces(code, language)
        structs = self.extract_structs(code, language)
        issues = self.find_potential_issues(code, language)

        return {
            "language": language,
            "metrics": {
                "lines_total": metrics.lines_total,
                "lines_code": metrics.lines_code,
                "lines_comment": metrics.lines_comment,
                "lines_blank": metrics.lines_blank,
                "functions": metrics.functions,
                "classes": metrics.classes,
                "interfaces": metrics.interfaces,
                "structs": metrics.structs,
                "enums": metrics.enums,
                "imports": metrics.imports,
            },
            "functions": functions,
            "interfaces": interfaces,
            "structs": structs,
            "issues": issues,
            "issue_summary": {
                "critical": sum(1 for i in issues if i["severity"] == "critical"),
                "warning": sum(1 for i in issues if i["severity"] == "warning"),
                "info": sum(1 for i in issues if i["severity"] == "info"),
                "total": len(issues),
            }
        }
