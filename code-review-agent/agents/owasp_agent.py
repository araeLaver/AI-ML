"""OWASP Top 10 security analysis agent."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseReviewAgent


# OWASP Top 10 (2021) Categories
OWASP_CATEGORIES = {
    "A01": {
        "id": "A01:2021",
        "name": "Broken Access Control",
        "description": "Restrictions on authenticated users are not properly enforced",
        "patterns": [
            "Missing authorization checks",
            "IDOR (Insecure Direct Object References)",
            "Path traversal",
            "CORS misconfiguration",
            "Privilege escalation",
            "JWT manipulation",
        ]
    },
    "A02": {
        "id": "A02:2021",
        "name": "Cryptographic Failures",
        "description": "Failures related to cryptography which often lead to data exposure",
        "patterns": [
            "Weak encryption algorithms (MD5, SHA1, DES)",
            "Hardcoded encryption keys",
            "Missing encryption for sensitive data",
            "Weak random number generation",
            "Deprecated crypto libraries",
        ]
    },
    "A03": {
        "id": "A03:2021",
        "name": "Injection",
        "description": "User-supplied data is not validated, filtered, or sanitized",
        "patterns": [
            "SQL injection",
            "NoSQL injection",
            "Command injection",
            "LDAP injection",
            "Expression Language injection",
            "XPath injection",
        ]
    },
    "A04": {
        "id": "A04:2021",
        "name": "Insecure Design",
        "description": "Missing or ineffective security controls in design",
        "patterns": [
            "Missing rate limiting",
            "Lack of input validation architecture",
            "No defense in depth",
            "Trust boundary violations",
            "Business logic flaws",
        ]
    },
    "A05": {
        "id": "A05:2021",
        "name": "Security Misconfiguration",
        "description": "Missing security hardening or improperly configured permissions",
        "patterns": [
            "Default credentials",
            "Unnecessary features enabled",
            "Verbose error messages",
            "Missing security headers",
            "Outdated configurations",
            "Debug mode in production",
        ]
    },
    "A06": {
        "id": "A06:2021",
        "name": "Vulnerable and Outdated Components",
        "description": "Using components with known vulnerabilities",
        "patterns": [
            "Outdated libraries/frameworks",
            "Unpatched dependencies",
            "Unsupported software versions",
            "Missing security patches",
        ]
    },
    "A07": {
        "id": "A07:2021",
        "name": "Identification and Authentication Failures",
        "description": "Authentication and session management weaknesses",
        "patterns": [
            "Weak password policies",
            "Missing brute force protection",
            "Credential stuffing vulnerabilities",
            "Session fixation",
            "Insecure session management",
            "Missing MFA",
        ]
    },
    "A08": {
        "id": "A08:2021",
        "name": "Software and Data Integrity Failures",
        "description": "Code and infrastructure that does not protect against integrity violations",
        "patterns": [
            "Insecure deserialization",
            "Missing code signing",
            "Untrusted CI/CD pipelines",
            "Auto-update without verification",
            "Unsigned or unverified data",
        ]
    },
    "A09": {
        "id": "A09:2021",
        "name": "Security Logging and Monitoring Failures",
        "description": "Insufficient logging, detection, monitoring, and response",
        "patterns": [
            "Missing audit logs",
            "Logs not monitored",
            "Sensitive data in logs",
            "No alerting mechanism",
            "Insufficient log retention",
        ]
    },
    "A10": {
        "id": "A10:2021",
        "name": "Server-Side Request Forgery (SSRF)",
        "description": "Web application fetches remote resource without validating URL",
        "patterns": [
            "Unvalidated URL redirects",
            "URL parameter manipulation",
            "Internal network access",
            "Cloud metadata access",
            "Protocol smuggling",
        ]
    },
}


def get_owasp_category(category_id: str) -> dict | None:
    """Get OWASP category information by ID."""
    return OWASP_CATEGORIES.get(category_id.upper().replace(":2021", ""))


def get_all_categories() -> list[dict]:
    """Get all OWASP Top 10 categories."""
    return [
        {"id": cat["id"], "name": cat["name"], "description": cat["description"]}
        for cat in OWASP_CATEGORIES.values()
    ]


class OWASPAgent(BaseReviewAgent):
    """Agent specialized in OWASP Top 10 vulnerability detection."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, "OWASPAgent")

    @property
    def system_prompt(self) -> str:
        categories_text = "\n".join([
            f"{i+1}. {cat['id']} - {cat['name']}: {cat['description']}"
            for i, cat in enumerate(OWASP_CATEGORIES.values())
        ])

        patterns_text = "\n".join([
            f"\n**{cat['id']} - {cat['name']}**:\n" + "\n".join(f"  - {p}" for p in cat['patterns'])
            for cat in OWASP_CATEGORIES.values()
        ])

        return f"""You are a security expert specialized in OWASP Top 10 vulnerability detection.
Your task is to identify security vulnerabilities according to the OWASP Top 10 (2021) categories.

## OWASP Top 10 Categories:
{categories_text}

## Vulnerability Patterns to Look For:
{patterns_text}

## Analysis Guidelines:
1. Examine the code for each OWASP category systematically
2. Consider the language and framework being used
3. Look for both obvious and subtle vulnerabilities
4. Consider the context and how the code might be used
5. Prioritize findings by risk level

## Severity Levels:
- CRITICAL: Immediate exploitation possible, severe impact
- HIGH: Exploitation likely, significant impact
- MEDIUM: Exploitation possible with conditions, moderate impact
- LOW: Minor security concern, limited impact

## Response Format:
Respond ONLY in valid JSON format:
{{
    "findings": [
        {{
            "owasp_id": "A01:2021",
            "owasp_name": "Broken Access Control",
            "severity": "HIGH",
            "location": "line 42-45",
            "title": "Missing Authorization Check",
            "description": "The function accesses user data without verifying permissions",
            "code_snippet": "user_data = get_user(user_id)  # No auth check",
            "recommendation": "Add authorization check before accessing user data",
            "cwe_id": "CWE-862"
        }}
    ],
    "categories_checked": ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10"],
    "risk_score": 7.5,
    "summary": "Brief overall OWASP assessment with key risks identified"
}}

If no vulnerabilities are found, return:
{{
    "findings": [],
    "categories_checked": ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10"],
    "risk_score": 0,
    "summary": "No OWASP Top 10 vulnerabilities detected in the analyzed code"
}}"""

    def analyze(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze code for OWASP Top 10 vulnerabilities.

        Args:
            code: Source code to analyze
            context: Optional context with file info

        Returns:
            OWASP analysis results with categorized findings
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_analysis_prompt(code, context))
        ]

        response = self.llm.invoke(messages)

        return {
            "agent": self.name,
            "type": "owasp",
            "analysis": response.content
        }

    def analyze_category(
        self,
        code: str,
        category_id: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analyze code for a specific OWASP category.

        Args:
            code: Source code to analyze
            category_id: OWASP category ID (e.g., "A01", "A03:2021")
            context: Optional context with file info

        Returns:
            Analysis results for the specific category
        """
        category = get_owasp_category(category_id)
        if not category:
            return {
                "agent": self.name,
                "type": "owasp",
                "error": f"Unknown OWASP category: {category_id}"
            }

        focused_prompt = f"""You are analyzing code specifically for {category['id']} - {category['name']}.

Description: {category['description']}

Patterns to look for:
{chr(10).join(f'- {p}' for p in category['patterns'])}

Analyze the code and report any findings related to this specific category.

Respond in JSON format:
{{
    "findings": [...],
    "category": "{category['id']}",
    "summary": "Assessment for {category['name']}"
}}"""

        messages = [
            SystemMessage(content=focused_prompt),
            HumanMessage(content=self._create_analysis_prompt(code, context))
        ]

        response = self.llm.invoke(messages)

        return {
            "agent": self.name,
            "type": "owasp",
            "category": category["id"],
            "analysis": response.content
        }


# Language-specific OWASP patterns for static analysis
LANGUAGE_PATTERNS = {
    "python": {
        "A01": [  # Broken Access Control
            (r"@app\.route.*methods=\[.*POST.*\](?!.*@login_required)", "Missing authentication decorator"),
            (r"user_id\s*=\s*request\.(args|form|json)\[", "Potential IDOR - user ID from request"),
            (r"open\([^)]*\+[^)]*\)", "Potential path traversal"),
        ],
        "A02": [  # Cryptographic Failures
            (r"hashlib\.md5\(", "Weak hash algorithm (MD5)"),
            (r"hashlib\.sha1\(", "Weak hash algorithm (SHA1)"),
            (r"DES\.|Blowfish\.", "Weak encryption algorithm"),
            (r"random\.(random|randint|choice)\(", "Insecure random for crypto"),
        ],
        "A03": [  # Injection
            (r"execute\([^)]*%[^)]*\)", "Potential SQL injection (string formatting)"),
            (r"execute\([^)]*\+[^)]*\)", "Potential SQL injection (concatenation)"),
            (r"f['\"].*\{.*\}.*['\"].*execute", "Potential SQL injection (f-string)"),
            (r"subprocess\.(call|run|Popen)\([^)]*shell=True", "Command injection risk"),
            (r"os\.system\(", "Command injection risk"),
            (r"eval\(", "Code injection risk"),
            (r"exec\(", "Code injection risk"),
        ],
        "A05": [  # Security Misconfiguration
            (r"DEBUG\s*=\s*True", "Debug mode enabled"),
            (r"SECRET_KEY\s*=\s*['\"][^'\"]{1,20}['\"]", "Weak secret key"),
            (r"CORS\(.*origins=\[?['\"]?\*", "CORS allows all origins"),
            (r"verify\s*=\s*False", "SSL verification disabled"),
        ],
        "A07": [  # Identification and Authentication Failures
            (r"password\s*==\s*['\"]", "Hardcoded password comparison"),
            (r"session\[.*\]\s*=.*user", "Potential session fixation"),
        ],
        "A08": [  # Software and Data Integrity Failures
            (r"pickle\.load\(", "Insecure deserialization (pickle)"),
            (r"yaml\.load\([^)]*\)", "Insecure YAML deserialization"),
            (r"marshal\.load\(", "Insecure deserialization (marshal)"),
        ],
        "A10": [  # SSRF
            (r"requests\.(get|post|put|delete)\([^)]*\+", "Potential SSRF (URL concatenation)"),
            (r"urllib\.request\.urlopen\([^)]*\+", "Potential SSRF"),
        ],
    },
    "javascript": {
        "A01": [
            (r"req\.(params|query|body)\[.*\](?!.*auth)", "Potential IDOR"),
            (r"\.findById\(req\.(params|query|body)", "Direct object reference from request"),
        ],
        "A02": [
            (r"crypto\.createHash\(['\"]md5['\"]", "Weak hash algorithm (MD5)"),
            (r"crypto\.createHash\(['\"]sha1['\"]", "Weak hash algorithm (SHA1)"),
            (r"Math\.random\(\)", "Insecure random for crypto"),
        ],
        "A03": [
            (r"\$\{.*\}.*query|query.*\$\{", "Potential SQL injection (template literal)"),
            (r"innerHTML\s*=", "Potential XSS (innerHTML)"),
            (r"document\.write\(", "Potential XSS (document.write)"),
            (r"eval\(", "Code injection risk"),
            (r"new Function\(", "Code injection risk"),
            (r"exec\(.*\+", "Command injection risk"),
        ],
        "A05": [
            (r"NODE_ENV.*development", "Development mode check"),
            (r"cors\(\s*\)", "CORS with default (permissive) settings"),
            (r"'Access-Control-Allow-Origin'.*\*", "CORS allows all origins"),
        ],
        "A08": [
            (r"JSON\.parse\(.*\)", "JSON parsing (verify source)"),
            (r"deserialize\(", "Deserialization (verify source)"),
        ],
        "A10": [
            (r"fetch\([^)]*\+", "Potential SSRF (URL concatenation)"),
            (r"axios\.(get|post)\([^)]*\+", "Potential SSRF"),
            (r"http\.get\([^)]*\+", "Potential SSRF"),
        ],
    },
    "java": {
        "A01": [
            (r"@RequestMapping(?!.*@PreAuthorize)", "Missing authorization annotation"),
            (r"request\.getParameter\(['\"].*id", "Potential IDOR"),
        ],
        "A02": [
            (r"MessageDigest\.getInstance\(['\"]MD5['\"]", "Weak hash (MD5)"),
            (r"MessageDigest\.getInstance\(['\"]SHA-?1['\"]", "Weak hash (SHA1)"),
            (r"DESKeySpec|DESedeKeySpec", "Weak encryption (DES)"),
            (r"new Random\(\)", "Insecure random (use SecureRandom)"),
        ],
        "A03": [
            (r"Statement.*execute.*\+", "SQL injection (Statement)"),
            (r"createQuery\([^)]*\+", "SQL/HQL injection"),
            (r"Runtime\.getRuntime\(\)\.exec\(", "Command injection risk"),
            (r"ProcessBuilder\([^)]*\+", "Command injection risk"),
            (r"ScriptEngine.*eval\(", "Code injection risk"),
        ],
        "A05": [
            (r"@CrossOrigin\([^)]*\*", "CORS allows all origins"),
            (r"setAllowedOrigins\([^)]*\*", "CORS allows all origins"),
            (r"TrustAllCerts|X509TrustManager", "SSL trust all certs"),
        ],
        "A08": [
            (r"ObjectInputStream", "Deserialization risk"),
            (r"XMLDecoder", "XML deserialization risk"),
            (r"readObject\(\)", "Deserialization (verify source)"),
        ],
        "A10": [
            (r"new URL\([^)]*\+", "Potential SSRF"),
            (r"HttpURLConnection.*\+", "Potential SSRF"),
        ],
    },
    "go": {
        "A01": [
            (r"r\.URL\.Query\(\)\.Get\(['\"].*id", "Potential IDOR"),
            (r"mux\.Vars\(r\)\[['\"]id", "Potential IDOR"),
        ],
        "A02": [
            (r"md5\.New\(\)|md5\.Sum\(", "Weak hash (MD5)"),
            (r"sha1\.New\(\)|sha1\.Sum\(", "Weak hash (SHA1)"),
            (r"des\.", "Weak encryption (DES)"),
            (r"rand\.(Int|Intn|Float)\(", "Insecure random (use crypto/rand)"),
        ],
        "A03": [
            (r"db\.Query\([^)]*\+", "SQL injection (concatenation)"),
            (r"fmt\.Sprintf.*db\.(Query|Exec)", "SQL injection (Sprintf)"),
            (r"exec\.Command\([^)]*\+", "Command injection"),
            (r"template\.HTML\(", "Potential XSS (unescaped HTML)"),
        ],
        "A05": [
            (r"InsecureSkipVerify:\s*true", "SSL verification disabled"),
            (r"AllowAllOrigins:\s*true", "CORS allows all origins"),
        ],
        "A08": [
            (r"gob\.Decode|json\.Unmarshal", "Deserialization (verify source)"),
            (r"encoding/gob", "Gob deserialization risk"),
        ],
        "A10": [
            (r"http\.Get\([^)]*\+", "Potential SSRF"),
            (r"http\.NewRequest.*\+", "Potential SSRF"),
        ],
    },
}


def get_language_patterns(language: str) -> dict[str, list]:
    """Get OWASP patterns for a specific language."""
    return LANGUAGE_PATTERNS.get(language.lower(), {})


class OWASPStaticAnalyzer:
    """Static analyzer for OWASP patterns without LLM."""

    def __init__(self, language: str = "python"):
        self.language = language.lower()
        self.patterns = get_language_patterns(self.language)

    def analyze(self, code: str) -> dict[str, Any]:
        """Perform static analysis for OWASP vulnerabilities.

        Args:
            code: Source code to analyze

        Returns:
            Static analysis findings
        """
        import re

        findings = []
        lines = code.split("\n")

        for category_id, patterns in self.patterns.items():
            category = OWASP_CATEGORIES.get(category_id, {})

            for pattern, description in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            "owasp_id": category.get("id", category_id),
                            "owasp_name": category.get("name", "Unknown"),
                            "severity": self._estimate_severity(category_id),
                            "location": f"line {line_num}",
                            "title": description,
                            "description": f"Pattern matched: {pattern}",
                            "code_snippet": line.strip()[:100],
                            "recommendation": f"Review for {category.get('name', 'security issues')}",
                            "source": "static_analysis"
                        })

        return {
            "findings": findings,
            "language": self.language,
            "patterns_checked": len(self.patterns),
            "total_issues": len(findings)
        }

    def _estimate_severity(self, category_id: str) -> str:
        """Estimate severity based on OWASP category."""
        high_severity = {"A01", "A02", "A03", "A07", "A08", "A10"}
        medium_severity = {"A04", "A05", "A06", "A09"}

        if category_id in high_severity:
            return "HIGH"
        elif category_id in medium_severity:
            return "MEDIUM"
        return "LOW"
