"""Orchestrator for coordinating multiple review agents."""
import json
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .security_agent import SecurityAgent
from .performance_agent import PerformanceAgent
from .style_agent import StyleAgent


class ReviewOrchestrator:
    """Orchestrates multiple agents to produce comprehensive code review."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.security_agent = SecurityAgent(llm)
        self.performance_agent = PerformanceAgent(llm)
        self.style_agent = StyleAgent(llm)

    def review(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all agents and synthesize results.

        Args:
            code: Source code to review
            context: Optional context with file info

        Returns:
            Comprehensive review with all agent findings
        """
        # Run all agents
        security_result = self.security_agent.analyze(code, context)
        performance_result = self.performance_agent.analyze(code, context)
        style_result = self.style_agent.analyze(code, context)

        # Synthesize results
        synthesis = self._synthesize_results(
            security_result,
            performance_result,
            style_result
        )

        return {
            "security": security_result,
            "performance": performance_result,
            "style": style_result,
            "synthesis": synthesis,
            "context": context
        }

    def _synthesize_results(
        self,
        security: dict[str, Any],
        performance: dict[str, Any],
        style: dict[str, Any]
    ) -> dict[str, Any]:
        """Synthesize all agent results into a final summary."""
        synthesis_prompt = f"""You are a senior code reviewer synthesizing analysis from multiple experts.

Security Analysis:
{security.get('analysis', 'No analysis')}

Performance Analysis:
{performance.get('analysis', 'No analysis')}

Style Analysis:
{style.get('analysis', 'No analysis')}

Create a prioritized summary with:
1. Critical issues requiring immediate attention
2. Important improvements to consider
3. Minor suggestions for enhancement
4. Overall code health score (1-10)

Respond in JSON format:
{{
    "critical_issues": [...],
    "important_improvements": [...],
    "minor_suggestions": [...],
    "health_score": 7,
    "verdict": "APPROVE" | "REQUEST_CHANGES" | "COMMENT",
    "summary": "2-3 sentence overall assessment"
}}"""

        messages = [
            SystemMessage(content="You synthesize code review findings into actionable summaries."),
            HumanMessage(content=synthesis_prompt)
        ]

        response = self.llm.invoke(messages)

        return {
            "analysis": response.content
        }
