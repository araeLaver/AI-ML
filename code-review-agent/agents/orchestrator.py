"""Orchestrator for coordinating multiple review agents with parallel execution."""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .security_agent import SecurityAgent
from .performance_agent import PerformanceAgent
from .style_agent import StyleAgent

logger = logging.getLogger(__name__)


def parse_json_safely(content: str, agent_name: str = "Agent") -> dict:
    """Parse JSON from LLM response with robust error handling.

    Args:
        content: Raw LLM response content
        agent_name: Name of the agent for logging

    Returns:
        Parsed JSON dict or fallback dict
    """
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"{agent_name}: Failed to parse JSON: {e}")
        return {
            "parse_error": True,
            "raw_content": content,
            "findings": [],
            "summary": content[:500] if content else "No response"
        }


class ReviewOrchestrator:
    """Orchestrates multiple agents to produce comprehensive code review with parallel execution."""

    def __init__(self, llm: BaseChatModel, timeout: float = 60.0):
        """Initialize orchestrator.

        Args:
            llm: Language model to use
            timeout: Timeout for each agent in seconds
        """
        self.llm = llm
        self.timeout = timeout
        self.security_agent = SecurityAgent(llm)
        self.performance_agent = PerformanceAgent(llm)
        self.style_agent = StyleAgent(llm)

    def review(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all agents and synthesize results with parallel execution.

        Args:
            code: Source code to review
            context: Optional context with file info

        Returns:
            Comprehensive review with all agent findings
        """
        # Try parallel execution, fallback to sequential
        try:
            return self._review_parallel(code, context)
        except Exception as e:
            logger.warning(f"Parallel review failed, falling back to sequential: {e}")
            return self._review_sequential(code, context)

    def _review_parallel(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all agents in parallel using ThreadPoolExecutor."""
        results = {}
        errors = []

        def run_agent(agent, name: str) -> tuple[str, dict]:
            try:
                result = agent.analyze(code, context)
                parsed = parse_json_safely(result.get("analysis", ""), name)
                return name, {
                    "agent": result.get("agent", name),
                    "type": result.get("type", name.lower()),
                    "findings": parsed.get("findings", []),
                    "summary": parsed.get("summary", ""),
                    "metrics": parsed.get("metrics", {}),
                    "complexity_analysis": parsed.get("complexity_analysis", {}),
                    "raw": result.get("analysis") if parsed.get("parse_error") else None
                }
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                return name, {
                    "agent": name,
                    "type": name.lower(),
                    "error": str(e),
                    "findings": [],
                    "summary": f"Analysis failed: {e}"
                }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_agent, self.security_agent, "security"),
                executor.submit(run_agent, self.performance_agent, "performance"),
                executor.submit(run_agent, self.style_agent, "style")
            ]

            for future in futures:
                try:
                    name, result = future.result(timeout=self.timeout)
                    results[name] = result
                    if result.get("error"):
                        errors.append(f"{name}: {result['error']}")
                except TimeoutError:
                    logger.error(f"Agent timed out")
                    errors.append("Agent timeout")
                except Exception as e:
                    logger.error(f"Agent execution failed: {e}")
                    errors.append(str(e))

        # Ensure all results exist
        security_result = results.get("security", {"findings": [], "summary": "Not available"})
        performance_result = results.get("performance", {"findings": [], "summary": "Not available"})
        style_result = results.get("style", {"findings": [], "summary": "Not available"})

        # Synthesize results
        synthesis = self._synthesize_results(security_result, performance_result, style_result)

        return {
            "security": security_result,
            "performance": performance_result,
            "style": style_result,
            "synthesis": synthesis,
            "context": context,
            "errors": errors if errors else None
        }

    def _review_sequential(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all agents sequentially (fallback)."""
        errors = []

        # Security
        try:
            sec_raw = self.security_agent.analyze(code, context)
            sec_parsed = parse_json_safely(sec_raw.get("analysis", ""), "security")
            security_result = {
                "agent": sec_raw.get("agent"),
                "type": sec_raw.get("type"),
                "findings": sec_parsed.get("findings", []),
                "summary": sec_parsed.get("summary", ""),
                "raw": sec_raw.get("analysis") if sec_parsed.get("parse_error") else None
            }
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            errors.append(f"security: {e}")
            security_result = {"findings": [], "summary": f"Failed: {e}", "error": str(e)}

        # Performance
        try:
            perf_raw = self.performance_agent.analyze(code, context)
            perf_parsed = parse_json_safely(perf_raw.get("analysis", ""), "performance")
            performance_result = {
                "agent": perf_raw.get("agent"),
                "type": perf_raw.get("type"),
                "findings": perf_parsed.get("findings", []),
                "summary": perf_parsed.get("summary", ""),
                "complexity_analysis": perf_parsed.get("complexity_analysis", {}),
                "raw": perf_raw.get("analysis") if perf_parsed.get("parse_error") else None
            }
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            errors.append(f"performance: {e}")
            performance_result = {"findings": [], "summary": f"Failed: {e}", "error": str(e)}

        # Style
        try:
            style_raw = self.style_agent.analyze(code, context)
            style_parsed = parse_json_safely(style_raw.get("analysis", ""), "style")
            style_result = {
                "agent": style_raw.get("agent"),
                "type": style_raw.get("type"),
                "findings": style_parsed.get("findings", []),
                "summary": style_parsed.get("summary", ""),
                "metrics": style_parsed.get("metrics", {}),
                "raw": style_raw.get("analysis") if style_parsed.get("parse_error") else None
            }
        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            errors.append(f"style: {e}")
            style_result = {"findings": [], "summary": f"Failed: {e}", "error": str(e)}

        # Synthesize results
        synthesis = self._synthesize_results(security_result, performance_result, style_result)

        return {
            "security": security_result,
            "performance": performance_result,
            "style": style_result,
            "synthesis": synthesis,
            "context": context,
            "errors": errors if errors else None
        }

    async def review_async(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all agents asynchronously with true parallel execution.

        Args:
            code: Source code to review
            context: Optional context with file info

        Returns:
            Comprehensive review with all agent findings
        """
        async def run_agent(agent, name: str) -> tuple[str, dict]:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: agent.analyze(code, context)),
                    timeout=self.timeout
                )
                parsed = parse_json_safely(result.get("analysis", ""), name)
                return name, {
                    "agent": result.get("agent", name),
                    "type": result.get("type", name.lower()),
                    "findings": parsed.get("findings", []),
                    "summary": parsed.get("summary", ""),
                    "metrics": parsed.get("metrics", {}),
                    "complexity_analysis": parsed.get("complexity_analysis", {}),
                    "raw": result.get("analysis") if parsed.get("parse_error") else None
                }
            except asyncio.TimeoutError:
                logger.error(f"{name} timed out")
                return name, {"agent": name, "error": "Timeout", "findings": [], "summary": "Timed out"}
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                return name, {"agent": name, "error": str(e), "findings": [], "summary": f"Failed: {e}"}

        # Run all agents in parallel
        results = await asyncio.gather(
            run_agent(self.security_agent, "security"),
            run_agent(self.performance_agent, "performance"),
            run_agent(self.style_agent, "style"),
            return_exceptions=True
        )

        # Process results
        result_dict = {}
        errors = []
        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
            else:
                name, data = r
                result_dict[name] = data
                if data.get("error"):
                    errors.append(f"{name}: {data['error']}")

        security_result = result_dict.get("security", {"findings": [], "summary": "Not available"})
        performance_result = result_dict.get("performance", {"findings": [], "summary": "Not available"})
        style_result = result_dict.get("style", {"findings": [], "summary": "Not available"})

        # Synthesize results
        synthesis = self._synthesize_results(security_result, performance_result, style_result)

        return {
            "security": security_result,
            "performance": performance_result,
            "style": style_result,
            "synthesis": synthesis,
            "context": context,
            "errors": errors if errors else None
        }

    def _synthesize_results(
        self,
        security: dict[str, Any],
        performance: dict[str, Any],
        style: dict[str, Any]
    ) -> dict[str, Any]:
        """Synthesize all agent results into a final summary."""
        # Count issues by severity
        critical_count = sum(1 for f in security.get("findings", []) if f.get("severity", "").upper() == "CRITICAL")
        high_count = sum(1 for f in security.get("findings", []) if f.get("severity", "").upper() == "HIGH")
        high_count += sum(1 for f in performance.get("findings", []) if f.get("impact", "").upper() == "HIGH")
        high_count += sum(1 for f in style.get("findings", []) if f.get("priority", "").upper() == "HIGH")
        medium_count = sum(1 for f in security.get("findings", []) if f.get("severity", "").upper() == "MEDIUM")
        medium_count += sum(1 for f in performance.get("findings", []) if f.get("impact", "").upper() == "MEDIUM")
        medium_count += sum(1 for f in style.get("findings", []) if f.get("priority", "").upper() == "MEDIUM")

        total_findings = (
            len(security.get("findings", [])) +
            len(performance.get("findings", [])) +
            len(style.get("findings", []))
        )

        synthesis_prompt = f"""You are a senior code reviewer synthesizing analysis from multiple experts.

## Security Analysis
Summary: {security.get('summary', 'No analysis')}
Findings: {len(security.get('findings', []))} issues

## Performance Analysis
Summary: {performance.get('summary', 'No analysis')}
Findings: {len(performance.get('findings', []))} issues

## Style Analysis
Summary: {style.get('summary', 'No analysis')}
Findings: {len(style.get('findings', []))} issues

## Issue Summary
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}
- Total: {total_findings}

Create a prioritized JSON summary:
{{
    "critical_issues": ["list of critical issues requiring immediate attention"],
    "important_improvements": ["list of important improvements"],
    "minor_suggestions": ["list of minor suggestions"],
    "health_score": 7,
    "verdict": "APPROVE | REQUEST_CHANGES | COMMENT",
    "summary": "2-3 sentence overall assessment"
}}"""

        messages = [
            SystemMessage(content="You synthesize code review findings into actionable JSON summaries. Respond only with valid JSON."),
            HumanMessage(content=synthesis_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            parsed = parse_json_safely(response.content, "Synthesizer")

            return {
                "critical_issues": parsed.get("critical_issues", []),
                "important_improvements": parsed.get("important_improvements", []),
                "minor_suggestions": parsed.get("minor_suggestions", []),
                "health_score": parsed.get("health_score", 5),
                "verdict": parsed.get("verdict", "COMMENT"),
                "summary": parsed.get("summary", ""),
                "issue_counts": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "total": total_findings
                },
                "raw": response.content if parsed.get("parse_error") else None
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback synthesis
            if critical_count > 0:
                verdict = "REQUEST_CHANGES"
                score = 3
            elif high_count > 2:
                verdict = "REQUEST_CHANGES"
                score = 5
            elif total_findings > 5:
                verdict = "COMMENT"
                score = 6
            else:
                verdict = "APPROVE"
                score = 8

            return {
                "critical_issues": [],
                "important_improvements": [],
                "minor_suggestions": [],
                "health_score": score,
                "verdict": verdict,
                "summary": f"Found {total_findings} issues ({critical_count} critical, {high_count} high priority)",
                "issue_counts": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "total": total_findings
                },
                "error": str(e)
            }
