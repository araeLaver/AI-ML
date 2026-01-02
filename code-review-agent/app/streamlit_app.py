"""Streamlit demo app for Code Review Agent."""
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import ReviewOrchestrator
from tools.code_analyzer import CodeAnalyzer

# Page config
st.set_page_config(
    page_title="Code Review Agent",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .severity-critical { color: #dc3545; font-weight: bold; }
    .severity-high { color: #fd7e14; font-weight: bold; }
    .severity-medium { color: #ffc107; }
    .severity-low { color: #28a745; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Code Review Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered multi-agent code review system</p>', unsafe_allow_html=True)

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "Ollama (Local)"]
    )

    if llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    else:
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        model = st.selectbox("Model", ["llama3.2", "codellama", "mistral"])

    st.divider()

    st.header("📊 Agent Info")
    st.info("""
    **Agents:**
    - 🔐 Security Agent
    - ⚡ Performance Agent
    - 🎨 Style Agent

    Each agent specializes in different aspects of code quality.
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Code Input")

    code_input = st.text_area(
        "Paste your code here",
        height=400,
        placeholder="""def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total"""
    )

    language = st.selectbox(
        "Language",
        ["python", "javascript", "typescript", "java", "go", "auto-detect"]
    )

    review_btn = st.button("🚀 Run Review", type="primary", use_container_width=True)

with col2:
    st.subheader("📋 Review Results")

    if review_btn and code_input:
        # Initialize LLM
        try:
            if llm_provider == "OpenAI":
                if not api_key:
                    st.error("Please enter your OpenAI API key")
                    st.stop()
                llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
            else:
                llm = ChatOllama(model=model, base_url=ollama_url)

            # Analyze code
            analyzer = CodeAnalyzer()
            detected_lang = analyzer.detect_language(f"code.{language}") if language != "auto-detect" else "python"
            metrics = analyzer.extract_metrics(code_input, detected_lang)

            # Show metrics
            st.markdown("**📊 Code Metrics**")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Lines", metrics.lines_total)
            mcol2.metric("Functions", metrics.functions)
            mcol3.metric("Classes", metrics.classes)

            # Quick issues check
            quick_issues = analyzer.find_potential_issues(code_input, detected_lang)
            if quick_issues:
                st.warning(f"⚠️ {len(quick_issues)} potential issues found in quick scan")

            st.divider()

            # Run full review
            with st.spinner("🔍 Analyzing code with AI agents..."):
                orchestrator = ReviewOrchestrator(llm)
                result = orchestrator.review(code_input, {"language": detected_lang})

            # Display results
            tabs = st.tabs(["🔐 Security", "⚡ Performance", "🎨 Style", "📑 Summary"])

            with tabs[0]:
                st.markdown("### Security Analysis")
                st.markdown(result["security"]["analysis"])

            with tabs[1]:
                st.markdown("### Performance Analysis")
                st.markdown(result["performance"]["analysis"])

            with tabs[2]:
                st.markdown("### Style Analysis")
                st.markdown(result["style"]["analysis"])

            with tabs[3]:
                st.markdown("### Synthesis Report")
                st.markdown(result["synthesis"]["analysis"])

        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif review_btn:
        st.warning("Please enter some code to review")
    else:
        st.info("👈 Paste code and click 'Run Review' to start")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Code Review Agent v0.1.0 | "
    "Built with LangChain & LangGraph</p>",
    unsafe_allow_html=True
)
