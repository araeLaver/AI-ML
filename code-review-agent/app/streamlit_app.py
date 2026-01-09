"""Streamlit demo app for Code Review Agent with improved UX."""
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from agents import ReviewOrchestrator
from tools.code_analyzer import CodeAnalyzer

# Page config
st.set_page_config(
    page_title="Code Review Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1a1a2e;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .severity-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .severity-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .severity-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }
    .severity-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }
    .verdict-approve {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .verdict-changes {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .verdict-comment {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "review_history" not in st.session_state:
    st.session_state.review_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

# Header
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<p class="main-header">Code Review Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered multi-agent code review system</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="text-align: right; padding-top: 1rem;">
        <span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;">v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

# Sidebar - Settings
with st.sidebar:
    st.header("Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "Ollama (Local)"],
        help="Select the AI model provider"
    )

    if llm_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        if api_key:
            st.session_state.api_key = api_key

        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="gpt-4o-mini is recommended for balance of speed and quality"
        )
    else:
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="URL of your Ollama server"
        )
        model = st.selectbox(
            "Model",
            ["llama3.2", "codellama", "mistral", "deepseek-coder"],
            help="Select the local model to use"
        )

    st.divider()

    # Agent Info
    st.header("Agent Info")
    with st.expander("View Agents", expanded=True):
        st.markdown("""
        **Security Agent**
        - SQL/Command Injection
        - XSS vulnerabilities
        - Auth issues
        - Data exposure

        **Performance Agent**
        - Algorithm complexity
        - N+1 queries
        - Caching opportunities
        - Async optimization

        **Style Agent**
        - Naming conventions
        - Code structure
        - SOLID principles
        - Documentation
        """)

    st.divider()

    # Review History
    if st.session_state.review_history:
        st.header("History")
        for i, hist in enumerate(st.session_state.review_history[-5:]):
            with st.expander(f"{hist['timestamp'][:16]}"):
                st.write(f"**Score:** {hist.get('score', 'N/A')}/10")
                st.write(f"**Verdict:** {hist.get('verdict', 'N/A')}")
                if st.button("Load", key=f"load_{i}"):
                    st.session_state.current_result = hist.get("result")
                    st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Code Input")

    # File upload option
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "cpp", "c", "rb", "php"],
        help="Or paste code below"
    )

    if uploaded_file:
        code_input = uploaded_file.read().decode("utf-8")
        detected_lang = uploaded_file.name.split(".")[-1]
    else:
        code_input = st.text_area(
            "Paste your code here",
            height=350,
            placeholder="""def calculate_total(items):
    total = 0
    for item in items:
        # TODO: Add validation
        total = total + item['price']
    return total""",
            help="Paste the code you want to review"
        )
        detected_lang = None

    col_lang, col_timeout = st.columns(2)
    with col_lang:
        language = st.selectbox(
            "Language",
            ["auto-detect", "python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"],
            help="Select or auto-detect the programming language"
        )
    with col_timeout:
        timeout = st.slider(
            "Timeout (sec)",
            min_value=30,
            max_value=120,
            value=60,
            help="Maximum time for analysis"
        )

    # Quick analysis
    if code_input:
        analyzer = CodeAnalyzer()
        lang = detected_lang if detected_lang else (language if language != "auto-detect" else "python")
        metrics = analyzer.extract_metrics(code_input, lang)
        quick_issues = analyzer.find_potential_issues(code_input, lang)

        st.markdown("**Quick Metrics**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lines", metrics.lines_total)
        m2.metric("Functions", metrics.functions)
        m3.metric("Classes", metrics.classes)
        m4.metric("Quick Issues", len(quick_issues))

        if quick_issues:
            with st.expander(f"Quick Issues ({len(quick_issues)})", expanded=False):
                for issue in quick_issues[:5]:
                    st.warning(f"Line {issue.get('line', '?')}: {issue.get('message', issue.get('issue', 'Unknown'))}")

    # Review button
    review_btn = st.button(
        "Run AI Review",
        type="primary",
        use_container_width=True,
        disabled=not code_input
    )

with col2:
    st.subheader("Review Results")

    if review_btn and code_input:
        # Validate API key
        if llm_provider == "OpenAI" and not st.session_state.api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
            st.stop()

        try:
            # Initialize LLM
            if llm_provider == "OpenAI":
                llm = ChatOpenAI(
                    model=model,
                    api_key=st.session_state.api_key,
                    temperature=0,
                    request_timeout=timeout
                )
            else:
                llm = ChatOllama(model=model, base_url=ollama_url)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Run review with progress updates
            status_text.text("Initializing agents...")
            progress_bar.progress(10)

            start_time = time.time()
            orchestrator = ReviewOrchestrator(llm, timeout=float(timeout))

            status_text.text("Running security analysis...")
            progress_bar.progress(30)

            lang = detected_lang if detected_lang else (language if language != "auto-detect" else "python")
            result = orchestrator.review(code_input, {"language": lang})

            status_text.text("Synthesizing results...")
            progress_bar.progress(90)

            duration = time.time() - start_time
            progress_bar.progress(100)
            status_text.text(f"Completed in {duration:.1f}s")

            # Store result
            st.session_state.current_result = result
            st.session_state.review_history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "score": result.get("synthesis", {}).get("health_score", "N/A"),
                "verdict": result.get("synthesis", {}).get("verdict", "N/A"),
                "result": result
            })

            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"Review failed: {str(e)}")
            st.stop()

    # Display results
    result = st.session_state.current_result
    if result:
        synthesis = result.get("synthesis", {})

        # Verdict and Score
        verdict = synthesis.get("verdict", "COMMENT")
        score = synthesis.get("health_score", 5)

        if verdict == "APPROVE":
            verdict_class = "verdict-approve"
            verdict_icon = ""
        elif verdict == "REQUEST_CHANGES":
            verdict_class = "verdict-changes"
            verdict_icon = ""
        else:
            verdict_class = "verdict-comment"
            verdict_icon = ""

        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f'<div class="{verdict_class}">{verdict_icon} {verdict}</div>', unsafe_allow_html=True)
        with v2:
            st.metric("Health Score", f"{score}/10")

        # Issue counts
        issue_counts = synthesis.get("issue_counts", {})
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Critical", issue_counts.get("critical", 0))
        ic2.metric("High", issue_counts.get("high", 0))
        ic3.metric("Medium", issue_counts.get("medium", 0))
        ic4.metric("Total", issue_counts.get("total", 0))

        st.divider()

        # Summary
        if synthesis.get("summary"):
            st.markdown("**Summary**")
            st.info(synthesis.get("summary"))

        # Detailed tabs
        tabs = st.tabs(["Security", "Performance", "Style", "Full Report", "Export"])

        with tabs[0]:
            security = result.get("security", {})
            findings = security.get("findings", [])
            if findings:
                for finding in findings:
                    severity = finding.get("severity", "LOW")
                    with st.expander(f"[{severity}] {finding.get('title', 'Finding')}", expanded=severity in ["CRITICAL", "HIGH"]):
                        st.write(f"**Location:** {finding.get('location', 'N/A')}")
                        st.write(f"**Description:** {finding.get('description', 'N/A')}")
                        st.write(f"**Recommendation:** {finding.get('recommendation', 'N/A')}")
            else:
                st.success("No security issues found!")
            if security.get("summary"):
                st.markdown(f"**Summary:** {security.get('summary')}")

        with tabs[1]:
            performance = result.get("performance", {})
            findings = performance.get("findings", [])
            if findings:
                for finding in findings:
                    impact = finding.get("impact", "LOW")
                    with st.expander(f"[{impact}] {finding.get('title', 'Finding')}", expanded=impact == "HIGH"):
                        st.write(f"**Location:** {finding.get('location', 'N/A')}")
                        st.write(f"**Issue:** {finding.get('issue', 'N/A')}")
                        st.write(f"**Recommendation:** {finding.get('recommendation', 'N/A')}")
            else:
                st.success("No performance issues found!")
            complexity = performance.get("complexity_analysis", {})
            if complexity:
                st.markdown("**Complexity Analysis**")
                st.write(f"Time: {complexity.get('time', 'N/A')}, Space: {complexity.get('space', 'N/A')}")

        with tabs[2]:
            style = result.get("style", {})
            findings = style.get("findings", [])
            if findings:
                for finding in findings:
                    priority = finding.get("priority", "LOW")
                    with st.expander(f"[{priority}] {finding.get('title', 'Finding')}", expanded=False):
                        st.write(f"**Location:** {finding.get('location', 'N/A')}")
                        st.write(f"**Issue:** {finding.get('issue', 'N/A')}")
                        st.write(f"**Suggestion:** {finding.get('suggestion', 'N/A')}")
            else:
                st.success("No style issues found!")
            metrics = style.get("metrics", {})
            if metrics:
                st.markdown("**Code Metrics**")
                st.write(f"Readability: {metrics.get('readability', 'N/A')}")
                st.write(f"Maintainability: {metrics.get('maintainability', 'N/A')}")

        with tabs[3]:
            st.markdown("### Full Report")
            report = _generate_markdown_report(result)
            st.markdown(report)

        with tabs[4]:
            st.markdown("### Export Results")

            # Markdown export
            report = _generate_markdown_report(result)
            st.download_button(
                "Download Markdown",
                data=report,
                file_name="code_review_report.md",
                mime="text/markdown"
            )

            # JSON export
            import json
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="code_review_report.json",
                mime="application/json"
            )

    elif not review_btn:
        st.info("Paste code and click 'Run AI Review' to start analysis")


def _generate_markdown_report(result: dict) -> str:
    """Generate a markdown report from review results."""
    synthesis = result.get("synthesis", {})
    verdict = synthesis.get("verdict", "COMMENT")
    score = synthesis.get("health_score", 5)
    summary = synthesis.get("summary", "")
    issue_counts = synthesis.get("issue_counts", {})

    report = f"""# Code Review Report

## Summary
{summary}

**Verdict:** {verdict}
**Health Score:** {score}/10

## Issue Summary
| Severity | Count |
|----------|-------|
| Critical | {issue_counts.get('critical', 0)} |
| High | {issue_counts.get('high', 0)} |
| Medium | {issue_counts.get('medium', 0)} |
| **Total** | **{issue_counts.get('total', 0)}** |

"""

    # Security findings
    security = result.get("security", {})
    if security.get("findings"):
        report += "## Security Issues\n"
        for f in security["findings"]:
            report += f"- **[{f.get('severity', 'N/A')}]** {f.get('title', 'N/A')}: {f.get('description', 'N/A')}\n"
        report += "\n"

    # Performance findings
    performance = result.get("performance", {})
    if performance.get("findings"):
        report += "## Performance Issues\n"
        for f in performance["findings"]:
            report += f"- **[{f.get('impact', 'N/A')}]** {f.get('title', 'N/A')}: {f.get('issue', 'N/A')}\n"
        report += "\n"

    # Style findings
    style = result.get("style", {})
    if style.get("findings"):
        report += "## Style Issues\n"
        for f in style["findings"]:
            report += f"- **[{f.get('priority', 'N/A')}]** {f.get('title', 'N/A')}: {f.get('issue', 'N/A')}\n"
        report += "\n"

    report += "\n---\n*Generated by Code Review Agent*"
    return report


# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Code Review Agent v1.0.0 | "
    "Built with LangChain, LangGraph & Streamlit</p>",
    unsafe_allow_html=True
)
