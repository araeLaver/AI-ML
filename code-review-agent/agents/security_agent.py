"""Security analysis agent for detecting vulnerabilities."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseReviewAgent


class SecurityAgent(BaseReviewAgent):
    """Agent specialized in security vulnerability detection."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, "SecurityAgent")

    @property
    def system_prompt(self) -> str:
        return """You are a security expert specialized in code review.
Your task is to identify security vulnerabilities in the provided code.

Focus on:
1. Injection vulnerabilities (SQL, Command, XSS)
2. Authentication/Authorization issues
3. Sensitive data exposure
4. Insecure configurations
5. Cryptographic weaknesses

For each finding, provide:
- Severity: CRITICAL, HIGH, MEDIUM, LOW
- Location: Line number or code section
- Description: What the vulnerability is
- Recommendation: How to fix it

If no vulnerabilities are found, explicitly state that the code appears secure
from your analysis perspective.

Respond in JSON format:
{
    "findings": [
        {
            "severity": "HIGH",
            "location": "line 42",
            "title": "SQL Injection",
            "description": "...",
            "recommendation": "..."
        }
    ],
    "summary": "Brief overall security assessment"
}"""

    def analyze(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze code for security vulnerabilities.

        Args:
            code: Source code to analyze
            context: Optional context with file info

        Returns:
            Security analysis results
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_analysis_prompt(code, context))
        ]

        response = self.llm.invoke(messages)

        return {
            "agent": self.name,
            "type": "security",
            "analysis": response.content
        }
