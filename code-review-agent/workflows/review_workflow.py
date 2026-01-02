"""LangGraph workflow for code review process."""
from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agents import SecurityAgent, PerformanceAgent, StyleAgent


class ReviewState(TypedDict):
    """State for the review workflow."""
    code: str
    context: dict
    messages: Annotated[list, operator.add]
    security_analysis: str
    performance_analysis: str
    style_analysis: str
    final_report: str
    status: str


def create_review_workflow(llm: BaseChatModel) -> StateGraph:
    """Create the LangGraph review workflow.

    Args:
        llm: Language model to use

    Returns:
        Compiled workflow graph
    """
    # Initialize agents
    security_agent = SecurityAgent(llm)
    performance_agent = PerformanceAgent(llm)
    style_agent = StyleAgent(llm)

    # Node functions
    def analyze_security(state: ReviewState) -> dict:
        """Run security analysis."""
        result = security_agent.analyze(state["code"], state.get("context"))
        return {
            "security_analysis": result["analysis"],
            "messages": [f"[Security] Analysis complete"],
            "status": "security_done"
        }

    def analyze_performance(state: ReviewState) -> dict:
        """Run performance analysis."""
        result = performance_agent.analyze(state["code"], state.get("context"))
        return {
            "performance_analysis": result["analysis"],
            "messages": [f"[Performance] Analysis complete"],
            "status": "performance_done"
        }

    def analyze_style(state: ReviewState) -> dict:
        """Run style analysis."""
        result = style_agent.analyze(state["code"], state.get("context"))
        return {
            "style_analysis": result["analysis"],
            "messages": [f"[Style] Analysis complete"],
            "status": "style_done"
        }

    def synthesize_report(state: ReviewState) -> dict:
        """Synthesize all analyses into final report."""
        from langchain_core.messages import HumanMessage, SystemMessage

        synthesis_prompt = f"""Synthesize the following code review analyses into a final report:

## Security Analysis
{state.get('security_analysis', 'Not available')}

## Performance Analysis
{state.get('performance_analysis', 'Not available')}

## Style Analysis
{state.get('style_analysis', 'Not available')}

Create a markdown report with:
1. Executive Summary
2. Critical Issues (must fix)
3. Recommendations (should fix)
4. Minor Suggestions
5. Overall Score (1-10)
6. Final Verdict: APPROVE / REQUEST_CHANGES / COMMENT
"""

        messages = [
            SystemMessage(content="You are a senior code reviewer creating a final review report."),
            HumanMessage(content=synthesis_prompt)
        ]

        response = llm.invoke(messages)

        return {
            "final_report": response.content,
            "messages": [f"[Synthesis] Final report generated"],
            "status": "complete"
        }

    # Build the graph
    workflow = StateGraph(ReviewState)

    # Add nodes
    workflow.add_node("security", analyze_security)
    workflow.add_node("performance", analyze_performance)
    workflow.add_node("style", analyze_style)
    workflow.add_node("synthesize", synthesize_report)

    # Define edges (sequential for now, can be parallelized)
    workflow.set_entry_point("security")
    workflow.add_edge("security", "performance")
    workflow.add_edge("performance", "style")
    workflow.add_edge("style", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def run_review(llm: BaseChatModel, code: str, context: dict | None = None) -> dict:
    """Run the complete review workflow.

    Args:
        llm: Language model
        code: Source code to review
        context: Optional context

    Returns:
        Final workflow state with all analyses
    """
    workflow = create_review_workflow(llm)

    initial_state = {
        "code": code,
        "context": context or {},
        "messages": ["Starting code review..."],
        "security_analysis": "",
        "performance_analysis": "",
        "style_analysis": "",
        "final_report": "",
        "status": "started"
    }

    result = workflow.invoke(initial_state)
    return result
