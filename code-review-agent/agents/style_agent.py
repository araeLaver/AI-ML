"""Code style and quality analysis agent."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseReviewAgent


class StyleAgent(BaseReviewAgent):
    """Agent specialized in code style and quality review."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, "StyleAgent")

    @property
    def system_prompt(self) -> str:
        return """You are a code quality expert focused on maintainability and best practices.
Your task is to review code for style, readability, and maintainability issues.

Focus on:
1. Naming conventions (variables, functions, classes)
2. Code organization and structure
3. Documentation and comments
4. DRY principle violations
5. SOLID principles adherence
6. Error handling patterns
7. Code complexity (cognitive and cyclomatic)

For each finding, provide:
- Priority: HIGH, MEDIUM, LOW
- Location: Line number or code section
- Issue: What the quality problem is
- Suggestion: How to improve

Respond in JSON format:
{
    "findings": [
        {
            "priority": "MEDIUM",
            "location": "line 10",
            "title": "Unclear Variable Name",
            "issue": "...",
            "suggestion": "..."
        }
    ],
    "metrics": {
        "readability": "7/10",
        "maintainability": "6/10",
        "complexity": "moderate"
    },
    "summary": "Brief overall code quality assessment"
}"""

    def analyze(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze code for style and quality issues.

        Args:
            code: Source code to analyze
            context: Optional context with file info

        Returns:
            Style analysis results
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_analysis_prompt(code, context))
        ]

        response = self.llm.invoke(messages)

        return {
            "agent": self.name,
            "type": "style",
            "analysis": response.content
        }
