"""Performance analysis agent for detecting optimization opportunities."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseReviewAgent


class PerformanceAgent(BaseReviewAgent):
    """Agent specialized in performance analysis and optimization."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, "PerformanceAgent")

    @property
    def system_prompt(self) -> str:
        return """You are a performance optimization expert.
Your task is to identify performance issues and optimization opportunities in code.

Focus on:
1. Algorithm complexity (time and space)
2. Inefficient loops or iterations
3. Memory leaks or excessive allocations
4. Database query optimization (N+1 problems, missing indexes)
5. Caching opportunities
6. Async/parallel processing opportunities

For each finding, provide:
- Impact: HIGH, MEDIUM, LOW
- Location: Line number or code section
- Issue: What the performance problem is
- Recommendation: How to optimize

Respond in JSON format:
{
    "findings": [
        {
            "impact": "HIGH",
            "location": "line 15-25",
            "title": "N+1 Query Problem",
            "issue": "...",
            "recommendation": "..."
        }
    ],
    "complexity_analysis": {
        "time": "O(n^2)",
        "space": "O(n)",
        "explanation": "..."
    },
    "summary": "Brief overall performance assessment"
}"""

    def analyze(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze code for performance issues.

        Args:
            code: Source code to analyze
            context: Optional context with file info

        Returns:
            Performance analysis results
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_analysis_prompt(code, context))
        ]

        response = self.llm.invoke(messages)

        return {
            "agent": self.name,
            "type": "performance",
            "analysis": response.content
        }
