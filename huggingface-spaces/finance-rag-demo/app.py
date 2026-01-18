# -*- coding: utf-8 -*-
"""
Finance RAG Pro - Production-Grade RAG System
HuggingFace Spaces Version

Features:
- Hybrid Search (Vector + BM25 + RRF)
- Groq API LLM (Fast Response)
- Document Upload (PDF/TXT)
- Re-ranking (Keyword-based)
- 50+ DART-style Sample Documents
"""

import streamlit as st
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Config & Modules
from config import get_config, get_secret
from rag.llm_provider import get_llm_provider, BaseLLMProvider
from rag.hybrid_search import HybridSearcher, SearchResult
from rag.reranker import KeywordReranker, RankedDocument
from data.sample_docs import get_all_documents, get_categories, get_document_count
from data.document_loader import DocumentLoader, ChunkingConfig

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
# CSS Styles
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --black: #0a0a0a;
    --white: #fafafa;
    --gray: #888;
    --accent: #ff4d00;
    --green: #26a69a;
    --red: #ef5350;
    --blue: #2196f3;
}

* { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: var(--white); }

header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.main-header {
    text-align: center;
    padding: 1.5rem 0;
    border-bottom: 1px solid #eee;
    margin-bottom: 1rem;
}
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--black);
    margin-bottom: 0.25rem;
}
.main-subtitle {
    font-size: 0.9rem;
    color: var(--gray);
}
.badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    padding: 3px 10px;
    border-radius: 15px;
    font-size: 0.7rem;
    margin-left: 8px;
    vertical-align: middle;
}
.badge-groq { background: #f97316; }
.badge-hybrid { background: #8b5cf6; }

.stock-card {
    background: white;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
}
.stock-name { font-size: 0.8rem; color: var(--gray); margin-bottom: 0.2rem; }
.stock-price { font-size: 1.5rem; font-weight: 700; color: var(--black); }
.stock-change { font-size: 0.85rem; font-weight: 500; }
.stock-up { color: var(--green); }
.stock-down { color: var(--red); }

.source-card {
    background: #f8f9fa;
    border-left: 3px solid var(--accent);
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
}

.search-type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 500;
}
.search-hybrid { background: #ede9fe; color: #7c3aed; }
.search-vector { background: #dbeafe; color: #2563eb; }
.search-keyword { background: #fef3c7; color: #d97706; }

.stat-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-number { font-size: 1.5rem; font-weight: 700; }
.stat-label { font-size: 0.75rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
def init_session_state():
    """세션 상태 초기화"""
    if "searcher" not in st.session_state:
        config = get_config()
        st.session_state.searcher = HybridSearcher(
            hf_token=config.hf_token or "",
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight
        )
        # 샘플 문서 로드
        load_sample_documents()

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []

    if "reranker" not in st.session_state:
        st.session_state.reranker = KeywordReranker()


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
    except Exception as e:
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
    """
    RAG 파이프라인 실행

    Returns:
        answer, retrieved_docs, elapsed_time
    """
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
        except Exception as e:
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
# Stock Data
# ============================================================
@dataclass
class StockQuote:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int


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
                hist = ticker.history(period="2d")
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
                        volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
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


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("### ⚙️ 설정")

        # API 상태 표시
        config = get_config()
        if config.groq_api_key:
            st.success("✅ Groq API 연결됨")
        else:
            st.warning("⚠️ Groq API 미설정 (Fallback 모드)")
            st.caption("Secrets에 GROQ_API_KEY를 추가하세요")

        st.markdown("---")

        # 검색 설정
        st.markdown("### 🔍 검색 설정")

        search_mode = st.selectbox(
            "검색 모드",
            options=["hybrid", "vector", "keyword"],
            format_func=lambda x: {
                "hybrid": "🔀 하이브리드 (추천)",
                "vector": "🎯 벡터 검색",
                "keyword": "📝 키워드 검색"
            }[x]
        )

        use_rerank = st.checkbox("Re-ranking 사용", value=True)
        top_k = st.slider("검색 결과 수", 3, 10, 5)

        st.markdown("---")

        # 문서 업로드
        st.markdown("### 📤 문서 업로드")
        uploaded_file = st.file_uploader(
            "PDF/TXT 파일 업로드",
            type=["pdf", "txt", "md"],
            help="업로드된 문서는 RAG 검색에 추가됩니다"
        )

        if uploaded_file:
            if st.button("문서 추가", type="primary"):
                with st.spinner("문서 처리 중..."):
                    try:
                        loader = DocumentLoader(ChunkingConfig(chunk_size=500, chunk_overlap=100))
                        docs = loader.load_from_uploaded_file(uploaded_file)

                        # 검색 엔진에 추가
                        documents = [d.content for d in docs]
                        doc_ids = [d.id for d in docs]
                        metadatas = [d.metadata for d in docs]

                        st.session_state.searcher.index_documents(documents, doc_ids, metadatas)
                        st.session_state.uploaded_docs.append({
                            "filename": uploaded_file.name,
                            "chunks": len(docs)
                        })
                        st.success(f"✅ {len(docs)}개 청크 추가됨")
                    except Exception as e:
                        st.error(f"❌ 오류: {e}")

        # 업로드된 문서 목록
        if st.session_state.uploaded_docs:
            st.markdown("**업로드된 문서:**")
            for doc in st.session_state.uploaded_docs:
                st.caption(f"📄 {doc['filename']} ({doc['chunks']} chunks)")

        st.markdown("---")

        # 통계
        st.markdown("### 📊 시스템 정보")
        stats = st.session_state.searcher.get_stats()
        st.metric("총 문서 수", stats["total_documents"])
        st.caption(f"Vector: {stats['vector_store']['model'].split('/')[-1]}")
        st.caption(f"BM25: {stats['bm25']['vocab_size']} 어휘")

    return search_mode, use_rerank, top_k


# ============================================================
# Main App
# ============================================================
def main():
    # 초기화
    init_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            Finance RAG Pro
            <span class="badge badge-groq">Groq</span>
            <span class="badge badge-hybrid">Hybrid</span>
        </div>
        <div class="main-subtitle">
            Production-Grade RAG | 하이브리드 검색 + Groq LLM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    search_mode, use_rerank, top_k = render_sidebar()

    # Stats Row
    col1, col2, col3, col4 = st.columns(4)

    config = get_config()
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{get_document_count()}+</div>
            <div class="stat-label">샘플 문서</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">Vector+BM25</div>
            <div class="stat-label">하이브리드 검색</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        llm_name = "Groq" if config.groq_api_key else "Fallback"
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{llm_name}</div>
            <div class="stat-label">LLM Provider</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">2-3초</div>
            <div class="stat-label">예상 응답시간</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 RAG Q&A", "📊 실시간 시세", "📚 문서 목록"])

    # ============ TAB 1: RAG Q&A ============
    with tab1:
        st.markdown("### 🤖 금융 AI 어시스턴트")

        # Sample Questions
        st.markdown("**📝 예시 질문:**")
        sample_qs = [
            "삼성전자 4분기 실적은?",
            "HBM 시장 전망은?",
            "2025년 금리 전망",
            "네이버 AI 사업 현황",
            "비트코인 전망",
        ]

        cols = st.columns(5)
        selected_q = None
        for col, q in zip(cols, sample_qs):
            if col.button(q, use_container_width=True, key=f"sample_{q}"):
                selected_q = q

        # Query Input
        query = st.text_input(
            "질문을 입력하세요",
            value=selected_q if selected_q else "",
            placeholder="예: 삼성전자 HBM 전략은?"
        )

        if query:
            with st.spinner("🔍 검색 및 답변 생성 중..."):
                answer, retrieved, elapsed = rag_query(
                    query,
                    search_mode=search_mode,
                    use_rerank=use_rerank,
                    top_k=top_k
                )

            # Answer
            st.markdown("---")
            st.markdown("### 📝 AI 답변")
            st.markdown(answer)

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("응답 시간", f"{elapsed:.2f}초")
            col2.metric("검색 모드", search_mode.upper())
            col3.metric("검색 결과", f"{len(retrieved)}개")

            # Sources
            st.markdown("### 📚 참조 문서")
            for i, doc in enumerate(retrieved):
                meta = doc.get("metadata", {})
                title = meta.get("title", f"문서 {i+1}")
                score = doc.get("score", 0)
                search_type = doc.get("search_type", "unknown")

                # Search type badge
                badge_class = {
                    "hybrid": "search-hybrid",
                    "vector": "search-vector",
                    "keyword": "search-keyword"
                }.get(search_type, "search-keyword")

                with st.expander(f"📄 {title} (관련도: {score:.1%})"):
                    st.markdown(f"""
                    <span class="search-type-badge {badge_class}">{search_type.upper()}</span>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**카테고리**: {meta.get('category', 'N/A')}")
                    st.markdown(f"**기업**: {meta.get('company', 'N/A')}")
                    st.markdown(doc.get("content", "")[:500] + "...")
                    st.caption(f"출처: {meta.get('source', 'N/A')} | 날짜: {meta.get('date', 'N/A')}")

    # ============ TAB 2: Stock Data ============
    with tab2:
        st.markdown("### 📈 실시간 주요 종목")

        stocks = get_stock_data()
        cols = st.columns(2)

        for i, (name, stock) in enumerate(stocks.items()):
            with cols[i % 2]:
                change_class = "stock-up" if stock.change >= 0 else "stock-down"
                change_sign = "+" if stock.change >= 0 else ""

                st.markdown(f"""
                <div class="stock-card">
                    <div class="stock-name">{stock.symbol}</div>
                    <div class="stock-price">{name}</div>
                    <div class="stock-price">₩{stock.price:,.0f}</div>
                    <div class="stock-change {change_class}">
                        {change_sign}{stock.change:,.0f} ({change_sign}{stock.change_percent}%)
                    </div>
                    <div style="color: #888; font-size: 0.8rem;">
                        거래량: {stock.volume:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.caption("* 데이터: yfinance | 5분마다 갱신")

    # ============ TAB 3: Documents ============
    with tab3:
        st.markdown("### 📚 RAG 지식베이스")

        docs = get_all_documents()
        st.markdown(f"총 **{len(docs)}개** 문서가 등록되어 있습니다.")

        # Category filter
        categories = ["전체"] + sorted(get_categories())
        selected_cat = st.selectbox("카테고리 필터", categories)

        # Document list
        for doc in docs:
            if selected_cat != "전체" and doc.get("category") != selected_cat:
                continue

            with st.expander(f"📄 {doc['title']} [{doc.get('category', '')}]"):
                st.markdown(doc.get("content", ""))
                st.caption(f"출처: {doc.get('source', '')} | 날짜: {doc.get('date', '')} | 기업: {doc.get('company', '')}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem;">
        <p><strong>🛠️ Tech Stack</strong>: Streamlit + Groq API + Hybrid Search (Vector + BM25 + RRF)</p>
        <p><strong>🤖 Models</strong>: Llama 3.1 (LLM) + all-MiniLM-L6-v2 (Embeddings)</p>
        <p>Made with ❤️ by <a href="https://github.com/araeLaver" target="_blank">Kim Dawoon</a> |
        <a href="https://github.com/araeLaver/AI-ML" target="_blank">Full Project on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
