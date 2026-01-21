# -*- coding: utf-8 -*-
"""
Finance RAG Pro - Premium Portfolio Dashboard
HuggingFace Spaces Version

Features:
- Premium Dark Theme (GitHub-style)
- Chat History with Export (PDF/CSV)
- Interactive Plotly Charts
- Hybrid Search (Vector + BM25 + RRF)
- Groq API LLM (Fast Response)
"""

import streamlit as st
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Config & Modules
from config import get_config, get_secret
from rag.llm_provider import get_llm_provider, BaseLLMProvider
from rag.hybrid_search import HybridSearcher, SearchResult
from rag.reranker import KeywordReranker, RankedDocument
from data.sample_docs import get_all_documents, get_categories, get_document_count
from data.document_loader import DocumentLoader, ChunkingConfig
from utils.session_manager import SessionManager, ChatSession
from utils.export_utils import export_to_csv, export_to_pdf, export_sources_to_csv

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Finance RAG Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Premium Dark Theme CSS
# ============================================================
def get_premium_css() -> str:
    """프리미엄 다크 테마 CSS"""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-elevated: #21262d;
    --bg-hover: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-orange: #f0883e;
    --accent-purple: #a371f7;
    --border-color: #30363d;
    --border-subtle: #21262d;
    --shadow: 0 8px 24px rgba(0,0,0,0.4);
    --shadow-sm: 0 4px 12px rgba(0,0,0,0.3);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace;
}

/* Base App Styling */
.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide default elements */
header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Premium Header */
.premium-header {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    animation: fadeIn 0.5s ease-out;
}

.premium-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.premium-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

/* Badges */
.badge-container {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.badge:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}

.badge-groq {
    background: rgba(249, 115, 22, 0.15);
    color: #fb923c;
    border-color: rgba(249, 115, 22, 0.3);
}

.badge-hybrid {
    background: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    border-color: rgba(139, 92, 246, 0.3);
}

.badge-rag {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border-color: rgba(59, 130, 246, 0.3);
}

/* Premium Cards */
.premium-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.3s ease;
    animation: fadeIn 0.4s ease-out;
}

.premium-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);
    transform: translateY(-2px);
}

/* Stock Cards */
.stock-card {
    background: linear-gradient(145deg, var(--bg-secondary), var(--bg-elevated));
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stock-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
}

.stock-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow);
    border-color: var(--accent-blue);
}

.stock-symbol {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.25rem;
}

.stock-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.stock-price {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.stock-change {
    font-size: 0.9rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

.stock-up { color: var(--accent-green); }
.stock-down { color: var(--accent-red); }

.stock-volume {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

/* Chat Messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin: 1rem 0;
}

.chat-message {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    border-radius: 12px;
    animation: slideIn 0.3s ease-out;
}

.chat-user {
    background: var(--bg-elevated);
    border: 1px solid var(--border-color);
    margin-left: 2rem;
}

.chat-assistant {
    background: linear-gradient(145deg, rgba(88, 166, 255, 0.08), rgba(163, 113, 247, 0.08));
    border: 1px solid rgba(88, 166, 255, 0.2);
    margin-right: 2rem;
}

.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.chat-avatar-user {
    background: var(--accent-blue);
}

.chat-avatar-assistant {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
}

.chat-content {
    flex: 1;
}

.chat-timestamp {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

/* Source Cards */
.source-card {
    background: var(--bg-elevated);
    border-left: 3px solid var(--accent-blue);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    transition: all 0.2s ease;
}

.source-card:hover {
    background: var(--bg-hover);
    border-left-color: var(--accent-purple);
}

/* Search Type Badges */
.search-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.search-hybrid {
    background: rgba(139, 92, 246, 0.2);
    color: #a78bfa;
}

.search-vector {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
}

.search-keyword {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
}

/* Stat Boxes */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-box {
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-elevated));
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-box:hover {
    border-color: var(--accent-blue);
    transform: translateY(-2px);
}

.stat-number {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
    border: none;
    padding: 0.6rem 1.25rem;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
}

.btn-secondary {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 0.6rem 1.25rem;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-secondary:hover {
    background: var(--bg-hover);
    border-color: var(--accent-blue);
}

.btn-ghost {
    background: transparent;
    color: var(--text-secondary);
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-ghost:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
}

/* History Panel */
.history-item {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.history-item:hover {
    border-color: var(--accent-blue);
    background: var(--bg-elevated);
}

.history-question {
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.history-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    display: flex;
    gap: 1rem;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--text-muted);
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state-title {
    font-size: 1.25rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.empty-state-desc {
    font-size: 0.9rem;
}

/* Export Panel */
.export-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    margin-top: 1rem;
}

.export-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.export-buttons {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary);
}

/* Input Styling */
.stTextInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

.stTextInput input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
}

/* Select Box */
.stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 0.25rem;
    gap: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 8px;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* Metrics */
.stMetric {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem;
}

.stMetric label {
    color: var(--text-secondary) !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}

/* Spinner */
.stSpinner > div {
    border-color: var(--accent-blue) transparent transparent transparent !important;
}

/* Links */
a {
    color: var(--accent-blue);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .premium-title {
        font-size: 1.75rem;
    }

    .chat-user, .chat-assistant {
        margin-left: 0;
        margin-right: 0;
    }
}
</style>
"""

st.markdown(get_premium_css(), unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
def init_session_state():
    """세션 상태 초기화"""
    config = get_config()

    if "searcher" not in st.session_state:
        st.session_state.searcher = HybridSearcher(
            hf_token=config.hf_token or "",
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight
        )
        load_sample_documents()

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []

    if "reranker" not in st.session_state:
        st.session_state.reranker = KeywordReranker()

    # 대화 히스토리 초기화
    SessionManager.init(max_history=config.ui.max_history_items)


def load_sample_documents():
    """샘플 문서를 검색 엔진에 로드"""
    docs = get_all_documents()
    documents = [f"{d['title']}\n{d['content']}" for d in docs]
    doc_ids = [d['id'] for d in docs]
    metadatas = [
        {
            "title": d.get("title", ""),
            "date": d.get("date", ""),
            "source": d.get("source", ""),
            "category": d.get("category", ""),
            "company": d.get("company", ""),
        }
        for d in docs
    ]
    st.session_state.searcher.index_documents(documents, doc_ids, metadatas)


def get_llm() -> Optional[BaseLLMProvider]:
    """LLM Provider 가져오기"""
    config = get_config()
    try:
        return get_llm_provider(
            groq_api_key=config.groq_api_key,
            hf_token=config.hf_token,
            model=config.llm_model,
            temperature=config.llm_temperature
        )
    except Exception:
        return None


# ============================================================
# RAG Pipeline
# ============================================================
def rag_query(
    question: str,
    search_mode: str = "hybrid",
    use_rerank: bool = True,
    top_k: int = 5
) -> Tuple[str, List[Dict[str, Any]], float]:
    """RAG 파이프라인 실행"""
    start_time = time.time()

    # 1. 검색
    searcher = st.session_state.searcher
    search_results = searcher.search(question, top_k=top_k * 2, search_mode=search_mode)

    # 2. Re-ranking (선택)
    if use_rerank and search_results:
        reranker = st.session_state.reranker
        docs_for_rerank = [
            {
                "doc_id": r.doc_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
                "search_type": r.search_type
            }
            for r in search_results
        ]
        reranked = reranker.rerank(question, docs_for_rerank, top_k=top_k)
        retrieved = [
            {
                "doc_id": r.doc_id,
                "content": r.content,
                "score": r.rerank_score,
                "metadata": r.metadata,
                "search_type": docs_for_rerank[r.original_rank - 1]["search_type"] if r.original_rank <= len(docs_for_rerank) else "unknown"
            }
            for r in reranked
        ]
    else:
        retrieved = [
            {
                "doc_id": r.doc_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
                "search_type": r.search_type
            }
            for r in search_results[:top_k]
        ]

    # 3. 컨텍스트 구성
    context_parts = []
    for doc in retrieved:
        meta = doc.get("metadata", {})
        title = meta.get("title", "")
        source = meta.get("source", "")
        date = meta.get("date", "")
        content = doc.get("content", "")

        if title:
            context_parts.append(f"[{title}] ({source}, {date})\n{content}")
        else:
            context_parts.append(content)

    context = "\n\n---\n\n".join(context_parts)

    # 4. LLM 답변 생성
    llm = get_llm()

    if llm:
        system_prompt = """당신은 금융 전문 AI 어시스턴트입니다.
주어진 참고 문서를 바탕으로 정확하고 구체적으로 답변해주세요.
문서에 없는 내용은 추측하지 마세요.
답변은 한국어로 작성하세요."""

        user_prompt = f"""### 참고 문서:
{context}

### 질문:
{question}

### 답변:"""

        try:
            response = llm.generate(system_prompt, user_prompt, max_tokens=1024)
            answer = response.content
        except Exception:
            answer = generate_fallback_answer(question, retrieved)
    else:
        answer = generate_fallback_answer(question, retrieved)

    elapsed = time.time() - start_time

    return answer, retrieved, elapsed


def generate_fallback_answer(question: str, retrieved: List[Dict[str, Any]]) -> str:
    """API 실패 시 Fallback 답변"""
    if not retrieved:
        return "관련 문서를 찾을 수 없습니다."

    top_doc = retrieved[0]
    meta = top_doc.get("metadata", {})
    title = meta.get("title", "관련 문서")
    content = top_doc.get("content", "")
    score = top_doc.get("score", 0)

    return f"""**{title}**에서 관련 정보를 찾았습니다.

{content[:500]}{"..." if len(content) > 500 else ""}

---
📊 관련도: {score:.1%} | 📅 날짜: {meta.get('date', 'N/A')} | 📌 출처: {meta.get('source', 'N/A')}

💡 더 정확한 답변을 위해 GROQ_API_KEY를 설정해주세요."""


# ============================================================
# Stock Data & Charts
# ============================================================
@dataclass
class StockQuote:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    history: Optional[Any] = None


@st.cache_data(ttl=300)
def get_stock_data() -> Dict[str, StockQuote]:
    """실시간 주식 데이터 (yfinance)"""
    try:
        import yfinance as yf
        stocks = {
            "삼성전자": "005930.KS",
            "SK하이닉스": "000660.KS",
            "NAVER": "035420.KS",
            "카카오": "035720.KS",
        }
        result = {}
        for name, symbol in stocks.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
                if not hist.empty and len(hist) >= 1:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - prev
                    change_pct = (change / prev * 100) if prev else 0
                    result[name] = StockQuote(
                        symbol=symbol,
                        name=name,
                        price=round(current, 0),
                        change=round(change, 0),
                        change_percent=round(change_pct, 2),
                        volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                        history=hist
                    )
            except Exception:
                pass
        if result:
            return result
    except ImportError:
        pass

    # Fallback
    return {
        "삼성전자": StockQuote("005930.KS", "삼성전자", 71500, 1200, 1.71, 12500000),
        "SK하이닉스": StockQuote("000660.KS", "SK하이닉스", 178000, 3500, 2.01, 3200000),
        "NAVER": StockQuote("035420.KS", "NAVER", 185000, -2000, -1.07, 580000),
        "카카오": StockQuote("035720.KS", "카카오", 42000, -500, -1.18, 2100000),
    }


def create_candlestick_chart(stock: StockQuote, period: str = "3mo") -> Optional[go.Figure]:
    """Plotly 캔들스틱 차트 생성"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(stock.symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return None

        # 서브플롯 생성 (가격 + 거래량)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{stock.name} ({stock.symbol})', 'Volume')
        )

        # 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price',
                increasing_line_color='#3fb950',
                decreasing_line_color='#f85149'
            ),
            row=1, col=1
        )

        # 거래량 차트
        colors = ['#3fb950' if c >= o else '#f85149'
                  for c, o in zip(hist['Close'], hist['Open'])]
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

        # 다크 테마 레이아웃
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            font=dict(family='Inter', color='#e6edf3'),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=30),
            height=500
        )

        # 축 스타일
        fig.update_xaxes(
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False
        )

        return fig
    except Exception:
        return None


# ============================================================
# UI Components
# ============================================================
def render_header():
    """프리미엄 헤더 렌더링"""
    st.markdown("""
    <div class="premium-header">
        <div class="premium-title">Finance RAG Pro</div>
        <div class="premium-subtitle">
            Production-Grade Financial AI Assistant with Hybrid RAG
        </div>
        <div class="badge-container">
            <span class="badge badge-groq">⚡ Groq LLM</span>
            <span class="badge badge-hybrid">🔀 Hybrid Search</span>
            <span class="badge badge-rag">🎯 RAG Pipeline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats():
    """통계 박스 렌더링"""
    config = get_config()
    llm_name = "Groq" if config.groq_api_key else "Fallback"
    history_count = SessionManager.get_count()

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-number">{get_document_count()}+</div>
            <div class="stat-label">Sample Documents</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">Vector+BM25</div>
            <div class="stat-label">Hybrid Search</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{llm_name}</div>
            <div class="stat-label">LLM Provider</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{history_count}</div>
            <div class="stat-label">Chat History</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_message(role: str, content: str, timestamp: Optional[datetime] = None):
    """채팅 메시지 렌더링"""
    if role == "user":
        avatar_class = "chat-avatar-user"
        message_class = "chat-user"
        avatar = "👤"
    else:
        avatar_class = "chat-avatar-assistant"
        message_class = "chat-assistant"
        avatar = "🤖"

    time_str = timestamp.strftime("%H:%M") if timestamp else ""

    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="chat-avatar {avatar_class}">{avatar}</div>
        <div class="chat-content">
            {content}
            <div class="chat-timestamp">{time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_source_card(doc: Dict[str, Any], index: int):
    """소스 카드 렌더링"""
    meta = doc.get("metadata", {})
    title = meta.get("title", f"문서 {index + 1}")
    score = doc.get("score", 0)
    search_type = doc.get("search_type", "unknown")

    badge_class = {
        "hybrid": "search-hybrid",
        "vector": "search-vector",
        "keyword": "search-keyword"
    }.get(search_type, "search-keyword")

    with st.expander(f"📄 {title} ({score:.1%})"):
        st.markdown(f'<span class="search-badge {badge_class}">{search_type}</span>', unsafe_allow_html=True)
        st.markdown(f"**카테고리**: {meta.get('category', 'N/A')} | **기업**: {meta.get('company', 'N/A')}")
        st.markdown(doc.get("content", "")[:500] + "...")
        st.caption(f"출처: {meta.get('source', 'N/A')} | 날짜: {meta.get('date', 'N/A')}")


def render_stock_card(name: str, stock: StockQuote):
    """주식 카드 렌더링"""
    change_class = "stock-up" if stock.change >= 0 else "stock-down"
    change_sign = "+" if stock.change >= 0 else ""
    arrow = "▲" if stock.change >= 0 else "▼"

    st.markdown(f"""
    <div class="stock-card">
        <div class="stock-symbol">{stock.symbol}</div>
        <div class="stock-name">{name}</div>
        <div class="stock-price">₩{stock.price:,.0f}</div>
        <div class="stock-change {change_class}">
            {arrow} {change_sign}{stock.change:,.0f} ({change_sign}{stock.change_percent}%)
        </div>
        <div class="stock-volume">Vol: {stock.volume:,}</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(icon: str, title: str, description: str):
    """빈 상태 렌더링"""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_export_panel(sessions: List[Dict[str, Any]]):
    """내보내기 패널 렌더링"""
    if not sessions:
        return

    st.markdown("""
    <div class="export-title">📥 Export Chat History</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_to_csv(sessions)
        st.download_button(
            "📊 CSV",
            csv_data,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        try:
            pdf_data = export_to_pdf(sessions)
            st.download_button(
                "📄 PDF",
                pdf_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.button("📄 PDF (N/A)", disabled=True, use_container_width=True)

    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            SessionManager.clear_history()
            st.rerun()


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # API 상태
        config = get_config()
        if config.groq_api_key:
            st.success("✅ Groq API Connected")
        else:
            st.warning("⚠️ Groq API Not Set (Fallback Mode)")
            st.caption("Add GROQ_API_KEY to Secrets")

        st.markdown("---")

        # 검색 설정
        st.markdown("### 🔍 Search Settings")

        search_mode = st.selectbox(
            "Search Mode",
            options=["hybrid", "vector", "keyword"],
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid (Recommended)",
                "vector": "🎯 Vector Search",
                "keyword": "📝 Keyword Search"
            }[x]
        )

        use_rerank = st.checkbox("Use Re-ranking", value=True)
        top_k = st.slider("Result Count", 3, 10, 5)

        st.markdown("---")

        # 문서 업로드
        st.markdown("### 📤 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF/TXT",
            type=["pdf", "txt", "md"],
            help="Uploaded documents will be added to RAG search"
        )

        if uploaded_file:
            if st.button("Add Document", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        loader = DocumentLoader(ChunkingConfig(chunk_size=500, chunk_overlap=100))
                        docs = loader.load_from_uploaded_file(uploaded_file)

                        documents = [d.content for d in docs]
                        doc_ids = [d.id for d in docs]
                        metadatas = [d.metadata for d in docs]

                        st.session_state.searcher.index_documents(documents, doc_ids, metadatas)
                        st.session_state.uploaded_docs.append({
                            "filename": uploaded_file.name,
                            "chunks": len(docs)
                        })
                        st.success(f"✅ {len(docs)} chunks added")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        if st.session_state.uploaded_docs:
            st.markdown("**Uploaded:**")
            for doc in st.session_state.uploaded_docs:
                st.caption(f"📄 {doc['filename']} ({doc['chunks']} chunks)")

        st.markdown("---")

        # 시스템 정보
        st.markdown("### 📊 System Info")
        stats = st.session_state.searcher.get_stats()
        st.metric("Total Documents", stats["total_documents"])
        st.caption(f"Vector: {stats['vector_store']['model'].split('/')[-1]}")
        st.caption(f"BM25 Vocab: {stats['bm25']['vocab_size']}")

    return search_mode, use_rerank, top_k


# ============================================================
# Main App
# ============================================================
def main():
    init_session_state()

    # Header
    render_header()

    # Sidebar
    search_mode, use_rerank, top_k = render_sidebar()

    # Stats
    render_stats()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 RAG Chat",
        "📊 Stock Charts",
        "📚 Documents",
        "🕐 History"
    ])

    # ============ TAB 1: RAG Chat ============
    with tab1:
        st.markdown("### 🤖 Financial AI Assistant")

        # Sample Questions
        st.markdown("**Quick Questions:**")
        sample_qs = [
            "삼성전자 4분기 실적",
            "HBM 시장 전망",
            "2025년 금리 전망",
            "네이버 AI 사업",
            "비트코인 전망",
        ]

        cols = st.columns(5)
        selected_q = None
        for col, q in zip(cols, sample_qs):
            if col.button(q, use_container_width=True, key=f"sample_{q}"):
                selected_q = q

        # Query Input
        query = st.text_input(
            "Ask a question about finance",
            value=selected_q if selected_q else "",
            placeholder="예: 삼성전자 HBM 전략은?"
        )

        if query:
            with st.spinner("🔍 Searching and generating answer..."):
                answer, retrieved, elapsed = rag_query(
                    query,
                    search_mode=search_mode,
                    use_rerank=use_rerank,
                    top_k=top_k
                )

            # 히스토리에 저장
            SessionManager.add_session(
                question=query,
                answer=answer,
                sources=retrieved,
                search_mode=search_mode,
                elapsed_time=elapsed
            )

            # Chat UI
            st.markdown("---")
            render_chat_message("user", query, datetime.now())
            render_chat_message("assistant", answer, datetime.now())

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Response Time", f"{elapsed:.2f}s")
            col2.metric("Search Mode", search_mode.upper())
            col3.metric("Sources Found", f"{len(retrieved)}")

            # Sources
            st.markdown("### 📚 Reference Sources")
            for i, doc in enumerate(retrieved):
                render_source_card(doc, i)

            # Export current result
            if retrieved:
                st.markdown("---")
                csv_sources = export_sources_to_csv(retrieved)
                st.download_button(
                    "📥 Export Sources (CSV)",
                    csv_sources,
                    file_name=f"sources_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        else:
            render_empty_state(
                "💬",
                "Start a conversation",
                "Ask questions about Korean financial markets, companies, and economic trends"
            )

    # ============ TAB 2: Stock Charts ============
    with tab2:
        st.markdown("### 📈 Interactive Stock Charts")

        stocks = get_stock_data()

        # Stock selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_stock = st.selectbox(
                "Select Stock",
                options=list(stocks.keys()),
                format_func=lambda x: f"{x} ({stocks[x].symbol})"
            )
        with col2:
            period = st.selectbox(
                "Period",
                options=["1mo", "3mo", "6mo", "1y"],
                index=1
            )

        # Stock cards
        st.markdown("#### Current Prices")
        cols = st.columns(len(stocks))
        for col, (name, stock) in zip(cols, stocks.items()):
            with col:
                render_stock_card(name, stock)

        # Chart
        st.markdown("#### Price Chart")
        stock = stocks[selected_stock]
        chart = create_candlestick_chart(stock, period)

        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            render_empty_state(
                "📊",
                "Chart unavailable",
                "Unable to load stock data. Please try again later."
            )

    # ============ TAB 3: Documents ============
    with tab3:
        st.markdown("### 📚 RAG Knowledge Base")

        docs = get_all_documents()
        st.markdown(f"Total **{len(docs)}** documents in the knowledge base.")

        # Category filter
        categories = ["All"] + sorted(get_categories())
        selected_cat = st.selectbox("Category Filter", categories)

        # Search within documents
        doc_search = st.text_input("Search documents", placeholder="Enter keywords...")

        # Document list
        filtered_docs = docs
        if selected_cat != "All":
            filtered_docs = [d for d in filtered_docs if d.get("category") == selected_cat]
        if doc_search:
            doc_search_lower = doc_search.lower()
            filtered_docs = [d for d in filtered_docs
                          if doc_search_lower in d.get("title", "").lower()
                          or doc_search_lower in d.get("content", "").lower()]

        st.caption(f"Showing {len(filtered_docs)} documents")

        for doc in filtered_docs[:20]:  # Limit display
            with st.expander(f"📄 {doc['title']} [{doc.get('category', '')}]"):
                st.markdown(doc.get("content", ""))
                st.caption(f"Source: {doc.get('source', '')} | Date: {doc.get('date', '')} | Company: {doc.get('company', '')}")

    # ============ TAB 4: History ============
    with tab4:
        st.markdown("### 🕐 Chat History")

        history = SessionManager.get_history()

        if history:
            # Export panel
            sessions_data = SessionManager.export_to_list()
            render_export_panel(sessions_data)

            st.markdown("---")

            # History list
            for session in history:
                col1, col2 = st.columns([6, 1])

                with col1:
                    st.markdown(f"""
                    <div class="history-item">
                        <div class="history-question">{session.question[:100]}...</div>
                        <div class="history-meta">
                            <span>🕐 {session.timestamp.strftime('%m/%d %H:%M')}</span>
                            <span>⚡ {session.elapsed_time:.2f}s</span>
                            <span>🔍 {session.search_mode}</span>
                            <span>📚 {len(session.sources)} sources</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("🗑️", key=f"del_{session.id}"):
                        SessionManager.delete_session(session.id)
                        st.rerun()

                # Show details on click
                with st.expander("View Details"):
                    st.markdown(f"**Question:** {session.question}")
                    st.markdown(f"**Answer:** {session.answer}")

                    if session.sources:
                        st.markdown("**Sources:**")
                        for i, src in enumerate(session.sources[:3]):
                            meta = src.get("metadata", {})
                            st.caption(f"- {meta.get('title', f'Source {i+1}')}")
        else:
            render_empty_state(
                "🕐",
                "No chat history yet",
                "Your conversation history will appear here"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 1rem;">
        <p><strong>🛠️ Tech Stack</strong>: Streamlit + Groq API + Hybrid Search (Vector + BM25 + RRF) + Plotly</p>
        <p><strong>🤖 Models</strong>: Llama 3.1 (LLM) + all-MiniLM-L6-v2 (Embeddings)</p>
        <p>Made with ❤️ by <a href="https://github.com/araeLaver" target="_blank">Kim Dawoon</a> |
        <a href="https://github.com/araeLaver/AI-ML" target="_blank">Full Project on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
