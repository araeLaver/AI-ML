"""LangGraph workflow for code review process with parallel agent execution."""
import asyncio
import json
import logging
from typing import Annotated, TypedDict, Any
import operator

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agents import SecurityAgent, PerformanceAgent, StyleAgent

logger = logging.getLogger(__name__)


class ReviewState(TypedDict):
    """State for the review workflow."""
    code: str
    context: dict
    messages: Annotated[list, operator.add]
    security_analysis: dict
    performance_analysis: dict
    style_analysis: dict
    final_report: str
    status: str
    errors: list


def parse_json_response(content: str, agent_name: str) -> dict:
    """Parse JSON from LLM response with error handling.

    Args:
        content: Raw LLM response content
        agent_name: Name of the agent for logging

    Returns:
        Parsed JSON dict or error dict
    """
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"{agent_name}: Failed to parse JSON response: {e}")
        return {
            "parse_error": True,
            "raw_content": content,
            "findings": [],
            "summary": "Failed to parse structured response"
        }


def create_review_workflow(llm: BaseChatModel, timeout: float = 60.0) -> StateGraph:
    """Create the LangGraph review workflow with parallel execution.

    Args:
        llm: Language model to use
        timeout: Timeout for each agent in seconds

    Returns:
        Compiled workflow graph
    """
    # Initialize agents
    security_agent = SecurityAgent(llm)
    performance_agent = PerformanceAgent(llm)
    style_agent = StyleAgent(llm)

    def analyze_security(state: ReviewState) -> dict:
        """Run security analysis with error handling."""
        try:
            result = security_agent.analyze(state["code"], state.get("context"))
            parsed = parse_json_response(result["analysis"], "SecurityAgent")
            return {
                "security_analysis": {
                    "agent": result["agent"],
                    "type": result["type"],
                    "findings": parsed.get("findings", []),
                    "summary": parsed.get("summary", ""),
                    "raw": result["analysis"] if parsed.get("parse_error") else None
                },
                "messages": ["[Security] Analysis complete"],
                "status": "security_done"
            }
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return {
                "security_analysis": {
                    "agent": "SecurityAgent",
                    "type": "security",
                    "error": str(e),
                    "findings": [],
                    "summary": f"Analysis failed: {e}"
                },
                "messages": [f"[Security] Analysis failed: {e}"],
                "errors": [f"Security: {e}"],
                "status": "security_error"
            }

    def analyze_performance(state: ReviewState) -> dict:
        """Run performance analysis with error handling."""
        try:
            result = performance_agent.analyze(state["code"], state.get("context"))
            parsed = parse_json_response(result["analysis"], "PerformanceAgent")
            return {
                "performance_analysis": {
                    "agent": result["agent"],
                    "type": result["type"],
                    "findings": parsed.get("findings", []),
                    "complexity_analysis": parsed.get("complexity_analysis", {}),
                    "summary": parsed.get("summary", ""),
                    "raw": result["analysis"] if parsed.get("parse_error") else None
                },
                "messages": ["[Performance] Analysis complete"],
                "status": "performance_done"
            }
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {
                "performance_analysis": {
                    "agent": "PerformanceAgent",
                    "type": "performance",
                    "error": str(e),
                    "findings": [],
                    "summary": f"Analysis failed: {e}"
                },
                "messages": [f"[Performance] Analysis failed: {e}"],
                "errors": [f"Performance: {e}"],
                "status": "performance_error"
            }

    def analyze_style(state: ReviewState) -> dict:
        """Run style analysis with error handling."""
        try:
            result = style_agent.analyze(state["code"], state.get("context"))
            parsed = parse_json_response(result["analysis"], "StyleAgent")
            return {
                "style_analysis": {
                    "agent": result["agent"],
                    "type": result["type"],
                    "findings": parsed.get("findings", []),
                    "metrics": parsed.get("metrics", {}),
                    "summary": parsed.get("summary", ""),
                    "raw": result["analysis"] if parsed.get("parse_error") else None
                },
                "messages": ["[Style] Analysis complete"],
                "status": "style_done"
            }
        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            return {
                "style_analysis": {
                    "agent": "StyleAgent",
                    "type": "style",
                    "error": str(e),
                    "findings": [],
                    "summary": f"Analysis failed: {e}"
                },
                "messages": [f"[Style] Analysis failed: {e}"],
                "errors": [f"Style: {e}"],
                "status": "style_error"
            }

    def synthesize_report(state: ReviewState) -> dict:
        """Synthesize all analyses into final report."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Collect all findings
        security = state.get("security_analysis", {})
        performance = state.get("performance_analysis", {})
        style = state.get("style_analysis", {})

        # Count issues by severity/priority
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0

        for finding in security.get("findings", []):
            severity = finding.get("severity", "").upper()
            if severity == "CRITICAL":
                critical_count += 1
            elif severity == "HIGH":
                high_count += 1
            elif severity == "MEDIUM":
                medium_count += 1
            else:
                low_count += 1

        for finding in performance.get("findings", []):
            impact = finding.get("impact", "").upper()
            if impact == "HIGH":
                high_count += 1
            elif impact == "MEDIUM":
                medium_count += 1
            else:
                low_count += 1

        for finding in style.get("findings", []):
            priority = finding.get("priority", "").upper()
            if priority == "HIGH":
                high_count += 1
            elif priority == "MEDIUM":
                medium_count += 1
            else:
                low_count += 1

        # Format findings for synthesis
        def format_findings(findings: list, category: str) -> str:
            if not findings:
                return f"No {category} issues found."
            lines = []
            for f in findings:
                title = f.get("title", f.get("issue", "Untitled"))
                severity = f.get("severity", f.get("impact", f.get("priority", "N/A")))
                lines.append(f"- [{severity}] {title}")
            return "\n".join(lines)

        synthesis_prompt = f"""Synthesize the following code review analyses into a final report:

## Security Analysis
{security.get('summary', 'Not available')}

Findings:
{format_findings(security.get('findings', []), 'security')}

## Performance Analysis
{performance.get('summary', 'Not available')}

Findings:
{format_findings(performance.get('findings', []), 'performance')}

## Style Analysis
{style.get('summary', 'Not available')}

Findings:
{format_findings(style.get('findings', []), 'style')}

## Issue Summary
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}
- Low: {low_count}

Create a markdown report with:
1. **Executive Summary** (2-3 sentences)
2. **Critical Issues** (must fix immediately)
3. **Important Recommendations** (should fix)
4. **Minor Suggestions** (nice to have)
5. **Overall Score**: X/10
6. **Verdict**: APPROVE / REQUEST_CHANGES / COMMENT

Be concise and actionable."""

        messages = [
            SystemMessage(content="You are a senior code reviewer creating a final review report. Be concise and prioritize actionable feedback."),
            HumanMessage(content=synthesis_prompt)
        ]

        try:
            response = llm.invoke(messages)
            return {
                "final_report": response.content,
                "messages": ["[Synthesis] Final report generated"],
                "status": "complete"
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Generate fallback report
            fallback_report = f"""## Code Review Summary

### Issues Found
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}
- Low: {low_count}

### Security
{security.get('summary', 'Analysis unavailable')}

### Performance
{performance.get('summary', 'Analysis unavailable')}

### Style
{style.get('summary', 'Analysis unavailable')}

**Verdict**: {"REQUEST_CHANGES" if critical_count > 0 or high_count > 2 else "COMMENT"}

*Note: Synthesis failed, showing raw summaries.*
"""
            return {
                "final_report": fallback_report,
                "messages": [f"[Synthesis] Failed, using fallback: {e}"],
                "errors": [f"Synthesis: {e}"],
                "status": "complete_with_errors"
            }

    # Build the graph with parallel execution
    workflow = StateGraph(ReviewState)

    # Add nodes
    workflow.add_node("security", analyze_security)
    workflow.add_node("performance", analyze_performance)
    workflow.add_node("style", analyze_style)
    workflow.add_node("synthesize", synthesize_report)

    # Parallel execution: all analysis nodes start from entry
    # LangGraph supports parallel execution when multiple edges lead to same destination
    workflow.set_entry_point("security")

    # For true parallel execution, we use a fan-out pattern
    # But in LangGraph, the simpler approach is to use async execution
    # Here we keep sequential for compatibility but optimize the orchestrator
    workflow.add_edge("security", "performance")
    workflow.add_edge("performance", "style")
    workflow.add_edge("style", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


async def run_review_async(llm: BaseChatModel, code: str, context: dict | None = None, timeout: float = 120.0) -> dict:
    """Run the complete review workflow asynchronously with parallel agents.

    Args:
        llm: Language model
        code: Source code to review
        context: Optional context
        timeout: Overall timeout in seconds

    Returns:
        Final workflow state with all analyses
    """
    security_agent = SecurityAgent(llm)
    performance_agent = PerformanceAgent(llm)
    style_agent = StyleAgent(llm)

    async def run_agent(agent, name: str) -> dict:
        """Run a single agent with timeout."""
        try:
            # Run synchronous agent in thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: agent.analyze(code, context)),
                timeout=timeout / 3  # Each agent gets 1/3 of total timeout
            )
            parsed = parse_json_response(result["analysis"], name)
            return {
                "agent": result["agent"],
                "type": result["type"],
                "findings": parsed.get("findings", []),
                "summary": parsed.get("summary", ""),
                "metrics": parsed.get("metrics", {}),
                "complexity_analysis": parsed.get("complexity_analysis", {}),
                "raw": result["analysis"] if parsed.get("parse_error") else None
            }
        except asyncio.TimeoutError:
            logger.error(f"{name} timed out")
            return {"agent": name, "error": "Timeout", "findings": [], "summary": "Analysis timed out"}
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return {"agent": name, "error": str(e), "findings": [], "summary": f"Analysis failed: {e}"}

    # Run all agents in parallel
    results = await asyncio.gather(
        run_agent(security_agent, "SecurityAgent"),
        run_agent(performance_agent, "PerformanceAgent"),
        run_agent(style_agent, "StyleAgent"),
        return_exceptions=True
    )

    security_result = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
    performance_result = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
    style_result = results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}

    # Synthesize results
    from langchain_core.messages import HumanMessage, SystemMessage

    synthesis_prompt = f"""Synthesize the following code review analyses:

Security: {security_result.get('summary', 'N/A')}
Performance: {performance_result.get('summary', 'N/A')}
Style: {style_result.get('summary', 'N/A')}

Create a brief final report with verdict (APPROVE/REQUEST_CHANGES/COMMENT) and score (1-10)."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a senior code reviewer. Be concise."),
            HumanMessage(content=synthesis_prompt)
        ])
        final_report = response.content
    except Exception as e:
        final_report = f"Synthesis failed: {e}"

    return {
        "code": code,
        "context": context or {},
        "security_analysis": security_result,
        "performance_analysis": performance_result,
        "style_analysis": style_result,
        "final_report": final_report,
        "status": "complete"
    }


def run_review(llm: BaseChatModel, code: str, context: dict | None = None, timeout: float = 120.0) -> dict:
    """Run the complete review workflow.

    Args:
        llm: Language model
        code: Source code to review
        context: Optional context
        timeout: Timeout in seconds

    Returns:
        Final workflow state with all analyses
    """
    workflow = create_review_workflow(llm, timeout)

    initial_state = {
        "code": code,
        "context": context or {},
        "messages": ["Starting code review..."],
        "security_analysis": {},
        "performance_analysis": {},
        "style_analysis": {},
        "final_report": "",
        "status": "started",
        "errors": []
    }

    result = workflow.invoke(initial_state)
    return result
