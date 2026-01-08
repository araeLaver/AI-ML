# -*- coding: utf-8 -*-
"""
Finance RAG - Enhanced Edition
Real-time Data + Charts + Streaming UI
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Finance RAG Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS 스타일
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --black: #0a0a0a;
    --white: #fafafa;
    --gray: #888;
    --light-gray: #f0f0f0;
    --accent: #ff4d00;
    --green: #26a69a;
    --red: #ef5350;
}

* { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: var(--white); }

header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* 네비게이션 */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 3rem;
    border-bottom: 1px solid #eee;
    background: var(--white);
}
.nav-logo {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--black);
}
.nav-badge {
    font-size: 0.6rem;
    background: var(--accent);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}
.nav-links {
    display: flex;
    gap: 1.5rem;
    font-size: 0.75rem;
}
.nav-link {
    color: var(--gray);
    text-decoration: none;
    cursor: pointer;
    transition: color 0.2s;
}
.nav-link:hover { color: var(--black); }
.nav-link.active { color: var(--black); font-weight: 600; }

/* 메인 콘텐츠 */
.main-content {
    padding: 2rem 3rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* 시장 개요 카드 */
.market-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.market-card {
    background: var(--black);
    border-radius: 12px;
    padding: 1.25rem;
    color: white;
}
.market-card-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.market-card-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.market-card-change {
    font-size: 0.85rem;
    font-weight: 500;
}
.market-card-change.up { color: var(--green); }
.market-card-change.down { color: var(--red); }

/* 섹션 헤더 */
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #eee;
}
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--black);
}
.section-meta {
    font-size: 0.7rem;
    color: var(--gray);
}

/* 차트 컨테이너 */
.chart-container {
    background: var(--black);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 2rem;
}

/* 채팅 영역 */
.chat-container {
    background: var(--black);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.chat-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #222;
}
.chat-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #333;
}
.chat-dot.active { background: var(--accent); }
.chat-label {
    font-size: 0.65rem;
    color: #666;
    margin-left: auto;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.chat-area {
    min-height: 300px;
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.msg-user {
    background: var(--accent);
    color: white;
    padding: 0.875rem 1rem;
    border-radius: 12px 12px 4px 12px;
    margin-left: 30%;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
}
.msg-ai {
    background: #1a1a1a;
    color: #ddd;
    padding: 0.875rem 1rem;
    border-radius: 12px 12px 12px 4px;
    margin-right: 20%;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid #333;
}
.msg-sources {
    background: #0d0d0d;
    border-left: 2px solid var(--accent);
    padding: 0.5rem 0.75rem;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    color: #888;
}
.msg-typing {
    color: var(--accent);
    font-size: 0.85rem;
}

/* 입력 필드 */
.stTextInput > div > div > input {
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 0.875rem !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input::placeholder { color: #666 !important; }
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

/* 버튼 */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.875rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
.stButton > button:hover { background: #e64500 !important; }

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--light-gray);
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: white !important;
}

/* 파일 업로더 */
.stFileUploader > div {
    background: var(--light-gray);
    border: 2px dashed #ccc;
    border-radius: 12px;
    padding: 2rem;
}

/* 종목 검색 결과 */
.stock-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--light-gray);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: background 0.2s;
}
.stock-result:hover { background: #e0e0e0; }
.stock-name {
    font-weight: 600;
    color: var(--black);
}
.stock-symbol {
    font-size: 0.8rem;
    color: var(--gray);
}
.stock-price {
    font-weight: 600;
    font-size: 1.1rem;
}
.stock-change { font-size: 0.85rem; }

/* 푸터 */
.footer {
    display: flex;
    justify-content: space-between;
    padding: 2rem 3rem;
    border-top: 1px solid #eee;
    margin-top: 3rem;
    font-size: 0.75rem;
    color: var(--gray);
}

/* 반응형 */
@media (max-width: 768px) {
    .nav, .main-content, .footer { padding-left: 1rem; padding-right: 1rem; }
    .market-grid { grid-template-columns: repeat(2, 1fr); }
    .msg-user { margin-left: 10%; }
    .msg-ai { margin-right: 5%; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 서비스 초기화
# ============================================================
@st.cache_resource
def init_services():
    """서비스 초기화 (캐싱)"""
    from src.rag.realtime_data import get_realtime_service
    from src.rag.charts import FinanceChartBuilder
    return {
        "realtime": get_realtime_service(),
        "charts": FinanceChartBuilder,
    }

class SimpleSearch:
    """간단한 문서 검색"""
    def __init__(self):
        self.documents = self._load_docs()

    def _load_docs(self):
        docs = []
        docs_path = project_root / "data" / "sample_docs"
        if docs_path.exists():
            for f in docs_path.glob("*.txt"):
                try:
                    content = f.read_text(encoding='utf-8')
                    docs.append({
                        'id': f.stem,
                        'title': f.stem.replace('_', ' ').title(),
                        'content': content
                    })
                except:
                    pass
        if not docs:
            # 기본 금융 데이터
            from src.rag.financial_data import get_all_documents
            for doc in get_all_documents():
                docs.append({
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content
                })
        return docs

    def search(self, query, top_k=3):
        import re
        keywords = re.findall(r'[가-힣]+|[a-zA-Z]+', query.lower())
        results = []
        for doc in self.documents:
            text = (doc['content'] + doc['title']).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append({'doc': doc, 'score': score})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

class GroqLLM:
    """Groq LLM 클라이언트"""
    def __init__(self):
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
            self.client = Groq(api_key=key) if key else None
        except:
            self.client = None

    def generate(self, system, user):
        if not self.client:
            return "API 키가 설정되지 않았습니다. GROQ_API_KEY를 확인하세요."
        try:
            r = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"오류: {e}"

    def generate_stream(self, system, user):
        """스트리밍 응답"""
        if not self.client:
            yield "API 키가 설정되지 않았습니다."
            return
        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"오류: {e}"

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search" not in st.session_state:
    st.session_state.search = SimpleSearch()
if "llm" not in st.session_state:
    st.session_state.llm = GroqLLM()
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

services = init_services()

# ============================================================
# 네비게이션
# ============================================================
st.markdown("""
<div class="nav">
    <div>
        <span class="nav-logo">FINANCE RAG</span>
        <span class="nav-badge">PRO</span>
    </div>
    <div class="nav-links">
        <span class="nav-link">Chat</span>
        <span class="nav-link">Market</span>
        <span class="nav-link">Charts</span>
        <a href="https://github.com/araeLaver/AI-ML" target="_blank" class="nav-link">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 메인 콘텐츠
# ============================================================
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# 탭 네비게이션
tab1, tab2, tab3, tab4 = st.tabs(["RAG Chat", "Market Data", "Stock Charts", "Documents"])

# ============================================================
# TAB 1: RAG 채팅
# ============================================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">AI Financial Assistant</span>
            <span class="section-meta">Powered by LLaMA 3.1 + RAG</span>
        </div>
        """, unsafe_allow_html=True)

        # 채팅 컨테이너
        st.markdown("""
        <div class="chat-container">
            <div class="chat-header">
                <span class="chat-dot"></span>
                <span class="chat-dot"></span>
                <span class="chat-dot active"></span>
                <span class="chat-label">Finance RAG</span>
            </div>
        """, unsafe_allow_html=True)

        # 채팅 영역
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div class="msg-ai">
                    금융 문서에 대해 질문하세요.<br>
                    <br>
                    <b>예시 질문:</b><br>
                    - "삼성전자 3분기 실적 알려줘"<br>
                    - "HBM 시장 전망은?"<br>
                    - "ETF 투자 방법 설명해줘"
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    html = f'<div class="msg-ai">{msg["content"]}'
                    if msg.get("sources"):
                        html += '<div class="msg-sources">Sources: ' + ' / '.join(msg["sources"][:3]) + '</div>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # 입력 영역
        input_col1, input_col2 = st.columns([5, 1])
        with input_col1:
            user_input = st.text_input(
                "query",
                placeholder="금융 관련 질문을 입력하세요...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with input_col2:
            send_btn = st.button("Send", use_container_width=True, key="send_btn")

        # 예시 질문 버튼
        st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
        example_cols = st.columns(4)
        examples = ["삼성전자 실적", "HBM 시장 전망", "금리와 주식", "ETF 투자법"]
        selected_example = None
        for i, ex in enumerate(examples):
            with example_cols[i]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    selected_example = ex
        st.markdown("</div>", unsafe_allow_html=True)

        # 질문 처리
        query = selected_example or (user_input if send_btn else None)
        if query and query.strip():
            st.session_state.messages.append({"role": "user", "content": query})

            # 검색
            results = st.session_state.search.search(query)

            if results:
                context = "\n\n".join([
                    f"[{r['doc']['title']}]\n{r['doc']['content'][:800]}"
                    for r in results
                ])
                sources = [r['doc']['title'] for r in results]
            else:
                context = "관련 문서 없음"
                sources = []

            system = """금융 전문 AI 어시스턴트입니다.
규칙:
1. 제공된 문서 기반으로만 답변
2. 문서에 없는 내용은 "해당 정보가 문서에 없습니다"라고 답변
3. 숫자와 수치는 정확히 인용
4. 투자 조언이 아닌 정보 제공임을 명시
한국어로 답변하세요."""

            user_prompt = f"[참고문서]\n{context}\n\n[질문]\n{query}"

            # 스트리밍 응답
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner(""):
                for token in st.session_state.llm.generate_stream(system, user_prompt):
                    full_response += token
                    response_placeholder.markdown(
                        f'<div class="msg-ai msg-typing">{full_response}...</div>',
                        unsafe_allow_html=True
                    )

            response_placeholder.empty()
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
            st.rerun()

    with col2:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Market Overview</span>
        </div>
        """, unsafe_allow_html=True)

        # 실시간 시장 데이터
        realtime = services["realtime"]
        market_summary = realtime.get_market_summary()

        for name, data in market_summary.get("indices", {}).items():
            change_class = "up" if data["change_percent"] >= 0 else "down"
            change_sign = "+" if data["change_percent"] >= 0 else ""
            st.markdown(f"""
            <div class="market-card">
                <div class="market-card-label">{name}</div>
                <div class="market-card-value">{data['value']:,.2f}</div>
                <div class="market-card-change {change_class}">
                    {change_sign}{data['change']:.2f} ({change_sign}{data['change_percent']:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 환율
        st.markdown('<div class="market-card-label" style="color: var(--gray);">EXCHANGE RATES</div>', unsafe_allow_html=True)
        for pair, data in market_summary.get("exchange_rates", {}).items():
            change_class = "up" if data["change_percent"] >= 0 else "down"
            change_sign = "+" if data["change_percent"] >= 0 else ""
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #eee;">
                <span style="font-size: 0.85rem;">{pair}</span>
                <span style="font-weight: 600;">{data['rate']:,.2f}
                    <span class="market-card-change {change_class}" style="font-size: 0.75rem;">
                        ({change_sign}{data['change_percent']:.2f}%)
                    </span>
                </span>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: 시장 데이터
# ============================================================
with tab2:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Real-time Market Data</span>
        <span class="section-meta">Data updates every 5 minutes</span>
    </div>
    """, unsafe_allow_html=True)

    # 시장 지수 차트
    market_summary = services["realtime"].get_market_summary()
    if market_summary.get("indices"):
        fig = services["charts"].create_market_overview_chart(
            market_summary["indices"],
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # 주요 종목
    st.markdown("""
    <div class="section-header" style="margin-top: 2rem;">
        <span class="section-title">Top Korean Stocks</span>
    </div>
    """, unsafe_allow_html=True)

    stock_cols = st.columns(3)
    top_stocks = ["삼성전자", "SK하이닉스", "NAVER", "카카오", "현대차", "LG에너지솔루션"]

    for i, stock_name in enumerate(top_stocks):
        with stock_cols[i % 3]:
            quote = services["realtime"].get_stock_quote(stock_name)
            if quote:
                change_class = "up" if quote.change_percent >= 0 else "down"
                change_sign = "+" if quote.change_percent >= 0 else ""
                st.markdown(f"""
                <div class="stock-result">
                    <div>
                        <div class="stock-name">{quote.name}</div>
                        <div class="stock-symbol">{quote.symbol}</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="stock-price">{quote.price:,.0f}</div>
                        <div class="stock-change market-card-change {change_class}">
                            {change_sign}{quote.change:,.0f} ({change_sign}{quote.change_percent:.2f}%)
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 3: 차트
# ============================================================
with tab3:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Stock Charts</span>
    </div>
    """, unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1, 3])

    with chart_col1:
        # 종목 선택
        stock_options = list(services["realtime"].KOREAN_STOCKS.keys())
        selected_stock = st.selectbox(
            "종목 선택",
            stock_options,
            index=0,
            key="stock_select"
        )

        period_options = {
            "1개월": "1mo",
            "3개월": "3mo",
            "6개월": "6mo",
            "1년": "1y",
            "2년": "2y"
        }
        selected_period = st.selectbox(
            "기간",
            list(period_options.keys()),
            index=0,
            key="period_select"
        )

        chart_type = st.radio(
            "차트 유형",
            ["캔들스틱", "라인"],
            horizontal=True,
            key="chart_type"
        )

        show_volume = st.checkbox("거래량 표시", value=True, key="show_volume")

        # 종목 비교
        st.markdown("<br>", unsafe_allow_html=True)
        compare_stocks = st.multiselect(
            "비교 종목 (최대 3개)",
            [s for s in stock_options if s != selected_stock],
            max_selections=3,
            key="compare_stocks"
        )

    with chart_col2:
        # 주가 데이터 가져오기
        history = services["realtime"].get_stock_history(
            selected_stock,
            period_options[selected_period]
        )

        if history:
            # 메인 차트
            if chart_type == "캔들스틱":
                fig = services["charts"].create_candlestick_chart(
                    history,
                    show_volume=show_volume,
                    height=500
                )
            else:
                fig = services["charts"].create_line_chart(
                    history,
                    show_volume=show_volume,
                    height=500
                )

            st.plotly_chart(fig, use_container_width=True)

            # 비교 차트
            if compare_stocks:
                datasets = [history]
                for comp_stock in compare_stocks:
                    comp_history = services["realtime"].get_stock_history(
                        comp_stock,
                        period_options[selected_period]
                    )
                    if comp_history:
                        datasets.append(comp_history)

                if len(datasets) > 1:
                    st.markdown("""
                    <div class="section-header" style="margin-top: 2rem;">
                        <span class="section-title">Performance Comparison</span>
                    </div>
                    """, unsafe_allow_html=True)

                    comp_fig = services["charts"].create_comparison_chart(datasets, height=400)
                    st.plotly_chart(comp_fig, use_container_width=True)

            # 종목 정보
            quote = services["realtime"].get_stock_quote(selected_stock)
            if quote:
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("현재가", f"{quote.price:,.0f}")
                with info_cols[1]:
                    st.metric(
                        "등락",
                        f"{quote.change:+,.0f}",
                        f"{quote.change_percent:+.2f}%"
                    )
                with info_cols[2]:
                    if quote.pe_ratio:
                        st.metric("PER", f"{quote.pe_ratio:.1f}")
                    else:
                        st.metric("거래량", f"{quote.volume:,}")
                with info_cols[3]:
                    if quote.market_cap:
                        st.metric("시가총액", f"{quote.market_cap/1e12:.1f}조")
                    else:
                        st.metric("거래량", f"{quote.volume:,}")

# ============================================================
# TAB 4: 문서 관리
# ============================================================
with tab4:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Document Management</span>
    </div>
    """, unsafe_allow_html=True)

    doc_col1, doc_col2 = st.columns([1, 1])

    with doc_col1:
        st.markdown("### Upload Documents")
        uploaded_file = st.file_uploader(
            "금융 문서 업로드 (PDF, TXT)",
            type=["pdf", "txt"],
            help="업로드된 문서는 RAG 검색에 사용됩니다."
        )

        if uploaded_file:
            st.success(f"'{uploaded_file.name}' 업로드 완료!")

            # 파일 내용 미리보기
            if uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode("utf-8")
                st.text_area("미리보기", content[:1000] + "...", height=200)

                if st.button("문서 추가", key="add_doc"):
                    # 세션 검색에 추가
                    st.session_state.search.documents.append({
                        'id': uploaded_file.name,
                        'title': uploaded_file.name,
                        'content': content
                    })
                    st.success("문서가 추가되었습니다!")

    with doc_col2:
        st.markdown("### Current Documents")
        st.markdown(f"**총 {len(st.session_state.search.documents)}개 문서**")

        for doc in st.session_state.search.documents[:10]:
            with st.expander(doc['title']):
                st.text(doc['content'][:500] + "...")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 푸터
# ============================================================
st.markdown(f"""
<div class="footer">
    <div>
        Built by araeLaver · {datetime.now().year}<br>
        <a href="https://github.com/araeLaver/AI-ML" style="color: #888;">github.com/araeLaver</a>
    </div>
    <div style="text-align: right;">
        <span style="font-size: 0.7rem; color: #888;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
</div>
""", unsafe_allow_html=True)
