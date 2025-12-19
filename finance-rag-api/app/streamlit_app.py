# -*- coding: utf-8 -*-
"""
Finance RAG - Creative Portfolio
Inspired by bpco.kr
"""

import streamlit as st
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import re

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Finance RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# bpco.kr 스타일 CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

/* 전역 리셋 */
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg: #E8E4DF;
    --bg-light: #F5F3F0;
    --text-dark: #1a1a1a;
    --text-gray: #666;
    --accent: #FF4D00;
    --border: #ccc;
}

/* Streamlit 요소 숨김 */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

.stApp {
    background: var(--bg);
    font-family: 'Noto Sans KR', sans-serif;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ===== 네비게이션 ===== */
.nav-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 3rem;
    background: var(--bg);
    z-index: 1000;
    border-bottom: 1px solid var(--border);
}

.nav-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-dark);
    letter-spacing: -0.5px;
}

.nav-links {
    display: flex;
    gap: 2.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.nav-link {
    color: var(--text-dark);
    text-decoration: none;
    position: relative;
}

.nav-link::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 0;
    width: 0;
    height: 1px;
    background: var(--text-dark);
    transition: width 0.3s ease;
}

.nav-link:hover::after {
    width: 100%;
}

.nav-time {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-gray);
}

/* ===== 히어로 섹션 ===== */
.hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 6rem 2rem 4rem;
    position: relative;
    text-align: center;
}

.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: var(--text-gray);
    text-transform: uppercase;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.hero-label::before,
.hero-label::after {
    content: '✦';
    font-size: 0.6rem;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(3rem, 12vw, 8rem);
    font-weight: 700;
    color: var(--text-dark);
    line-height: 0.95;
    margin-bottom: 1rem;
    letter-spacing: -3px;
}

.hero-title-line {
    display: block;
}

.hero-title .outline {
    -webkit-text-stroke: 1.5px var(--text-dark);
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-gray);
    margin-top: 2rem;
    max-width: 500px;
    line-height: 1.8;
}

.scroll-hint {
    position: absolute;
    bottom: 3rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: var(--text-gray);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.scroll-arrow {
    width: 1px;
    height: 40px;
    background: linear-gradient(to bottom, var(--text-gray), transparent);
    animation: scrollDown 1.5s ease-in-out infinite;
}

@keyframes scrollDown {
    0%, 100% { transform: translateY(0); opacity: 1; }
    50% { transform: translateY(10px); opacity: 0.5; }
}

/* ===== 섹션 공통 ===== */
.section {
    padding: 6rem 4rem;
    border-top: 1px solid var(--border);
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 4rem;
}

.section-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-gray);
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dark);
}

/* ===== 서비스 그리드 ===== */
.services-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
}

.service-card {
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 0;
    padding: 2.5rem 2rem;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}

.service-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: var(--accent);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.4s ease;
}

.service-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.08);
}

.service-card:hover::before {
    transform: scaleX(1);
}

.service-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    margin-bottom: 1.5rem;
}

.service-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}

.service-desc {
    font-size: 0.9rem;
    color: var(--text-gray);
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

.service-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-gray);
    border: 1px solid var(--border);
    padding: 0.4rem 0.8rem;
    display: inline-block;
}

/* ===== 스탯 섹션 ===== */
.stats-row {
    display: flex;
    justify-content: space-around;
    padding: 4rem 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}

.stat-item {
    text-align: center;
}

.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1;
}

.stat-label {
    font-size: 0.85rem;
    color: var(--text-gray);
    margin-top: 0.5rem;
}

/* ===== 데모 섹션 ===== */
.demo-section {
    padding: 6rem 4rem;
    background: var(--bg-light);
    border-top: 1px solid var(--border);
}

.demo-container {
    max-width: 900px;
    margin: 0 auto;
}

.demo-window {
    background: white;
    border: 1px solid var(--border);
    overflow: hidden;
}

.demo-titlebar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    background: #fafafa;
}

.demo-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1px solid #ddd;
}

.demo-dot.red { background: #ff5f57; border-color: #e0443e; }
.demo-dot.yellow { background: #febc2e; border-color: #dea123; }
.demo-dot.green { background: #28c840; border-color: #1aab29; }

.demo-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-gray);
    margin-left: 1rem;
}

.demo-body {
    padding: 2rem;
    min-height: 400px;
}

/* 채팅 스타일 */
.chat-messages {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.msg {
    max-width: 80%;
    padding: 1rem 1.25rem;
    font-size: 0.95rem;
    line-height: 1.6;
}

.msg-user {
    align-self: flex-end;
    background: var(--text-dark);
    color: white;
    border-radius: 20px 20px 4px 20px;
}

.msg-ai {
    align-self: flex-start;
    background: var(--bg);
    color: var(--text-dark);
    border-radius: 20px 20px 20px 4px;
    border: 1px solid var(--border);
}

.msg-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
    opacity: 0.7;
}

.sources-box {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 1rem;
    margin-top: 1rem;
    font-size: 0.8rem;
}

.sources-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
    color: var(--accent);
}

/* 입력 필드 */
.stTextInput > div > div > input {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 1rem 1.25rem !important;
    font-size: 1rem !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--text-dark) !important;
    box-shadow: none !important;
}

.stButton > button {
    background: var(--text-dark) !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 1rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: var(--accent) !important;
    transform: none !important;
}

/* ===== 테크 스택 ===== */
.tech-list {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 2rem;
}

.tech-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 0.6rem 1.2rem;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dark);
    transition: all 0.3s ease;
}

.tech-tag:hover {
    background: var(--text-dark);
    color: white;
    border-color: var(--text-dark);
}

/* ===== 푸터 ===== */
.footer {
    padding: 4rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-left {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
}

.footer-right {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-gray);
}

/* ===== 마퀴 ===== */
.marquee-container {
    overflow: hidden;
    padding: 2rem 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}

.marquee-track {
    display: flex;
    animation: marquee 20s linear infinite;
}

.marquee-text {
    font-family: 'Space Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: var(--text-dark);
    white-space: nowrap;
    padding-right: 4rem;
    -webkit-text-stroke: 1px var(--text-dark);
    -webkit-text-fill-color: transparent;
}

@keyframes marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* 반응형 */
@media (max-width: 768px) {
    .nav-bar { padding: 1rem; }
    .nav-links { display: none; }
    .hero-title { font-size: 2.5rem; }
    .services-grid { grid-template-columns: 1fr; }
    .stats-row { flex-direction: column; gap: 2rem; }
    .section { padding: 3rem 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 금융 데이터
# ============================================================
@dataclass
class FinancialDocument:
    id: str
    title: str
    content: str
    category: str
    date: str

FINANCIAL_DOCUMENTS = [
    FinancialDocument(
        id="samsung_q3_2024",
        title="삼성전자 2024년 3분기 실적",
        content="""삼성전자가 2024년 3분기에 영업이익 9조 1,834억원을 기록했다.
        이는 전년동기대비 274.5% 증가한 수치다. 매출액은 79조 1,024억원으로 17.3% 증가했다.
        반도체 부문이 실적 개선을 주도했으며, HBM 수요 증가가 주요 요인이다.""",
        category="실적",
        date="2024-10-31"
    ),
    FinancialDocument(
        id="fed_rate_2024",
        title="미국 연준 금리 동결",
        content="""미국 연방준비제도(Fed)가 2024년 9월 FOMC에서 기준금리를 5.25-5.50%로 동결했다.
        파월 의장은 인플레이션이 목표치인 2%로 수렴하고 있다고 평가했다.
        시장에서는 2024년 4분기 중 금리 인하 가능성을 50% 이상으로 전망하고 있다.""",
        category="금리",
        date="2024-09-20"
    ),
    FinancialDocument(
        id="nvidia_hbm",
        title="NVIDIA HBM 수요 급증",
        content="""NVIDIA의 AI 가속기 수요 급증으로 HBM(고대역폭메모리) 시장이 폭발적으로 성장하고 있다.
        SK하이닉스가 HBM3E 시장을 선도하고 있으며, 삼성전자도 HBM3E 양산을 시작했다.
        2024년 HBM 시장 규모는 전년대비 2배 이상 성장할 것으로 예상된다.""",
        category="반도체",
        date="2024-08-15"
    ),
]

# ============================================================
# 검색 & LLM
# ============================================================
class SimpleSearch:
    def __init__(self, documents):
        self.documents = documents

    def search(self, query: str, top_k: int = 3):
        query_lower = query.lower()
        keywords = re.findall(r'[가-힣]+|[a-zA-Z]+', query_lower)

        results = []
        for doc in self.documents:
            content_lower = doc.content.lower() + doc.title.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                results.append({"doc": doc, "score": score / len(keywords) if keywords else 0})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

class GroqLLM:
    def __init__(self):
        try:
            from groq import Groq
            from dotenv import load_dotenv
            load_dotenv()
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.available = True
        except:
            self.client = None
            self.available = False

    def generate_stream(self, system_prompt: str, user_prompt: str):
        if not self.available:
            yield "Groq API 키가 설정되지 않았습니다."
            return
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"오류: {str(e)}"

# ============================================================
# 세션 상태
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search" not in st.session_state:
    st.session_state.search = SimpleSearch(FINANCIAL_DOCUMENTS)
if "llm" not in st.session_state:
    st.session_state.llm = GroqLLM()

# ============================================================
# 네비게이션
# ============================================================
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">FINANCE_RAG</div>
    <div class="nav-links">
        <a href="#" class="nav-link">HOME</a>
        <a href="#" class="nav-link">FEATURES</a>
        <a href="#" class="nav-link">DEMO</a>
        <a href="#" class="nav-link">GITHUB</a>
    </div>
    <div class="nav-time">SEOUL, KR</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 히어로 섹션
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">AI-Powered Financial Intelligence</div>
    <h1 class="hero-title">
        <span class="hero-title-line">FINANCE</span>
        <span class="hero-title-line outline">RAG_</span>
    </h1>
    <p class="hero-subtitle">
        금융 문서 기반 AI 질의응답 시스템<br>
        하이브리드 검색과 LLM을 결합한 차세대 분석 도구
    </p>
    <div class="scroll-hint">
        <span>SCROLL DOWN</span>
        <div class="scroll-arrow"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 마퀴 텍스트
# ============================================================
st.markdown("""
<div class="marquee-container">
    <div class="marquee-track">
        <span class="marquee-text">HYBRID SEARCH ✦ RE-RANKING ✦ MULTI-TURN ✦ STREAMING ✦ RAG EVALUATION ✦ </span>
        <span class="marquee-text">HYBRID SEARCH ✦ RE-RANKING ✦ MULTI-TURN ✦ STREAMING ✦ RAG EVALUATION ✦ </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 스탯 섹션
# ============================================================
st.markdown("""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-num">5+</div>
        <div class="stat-label">데이터 소스</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">3</div>
        <div class="stat-label">검색 알고리즘</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">&lt;2s</div>
        <div class="stat-label">응답 시간</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">∞</div>
        <div class="stat-label">대화 컨텍스트</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 서비스 섹션
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-header">
        <span class="section-num">01</span>
        <span class="section-title">CORE_FEATURES</span>
    </div>

    <div class="services-grid">
        <div class="service-card">
            <div class="service-num">01</div>
            <div class="service-title">Hybrid_Search</div>
            <div class="service-desc">
                벡터 검색(의미 기반)과 BM25(키워드 기반)를
                RRF 알고리즘으로 결합하여 검색 정확도를 극대화합니다.
            </div>
            <span class="service-tag">VECTOR + BM25 + RRF</span>
        </div>

        <div class="service-card">
            <div class="service-num">02</div>
            <div class="service-title">Re_Ranking</div>
            <div class="service-desc">
                Cross-Encoder 또는 LLM 기반 재정렬로
                초기 검색 결과를 정교하게 평가합니다.
            </div>
            <span class="service-tag">TWO-STAGE RETRIEVAL</span>
        </div>

        <div class="service-card">
            <div class="service-num">03</div>
            <div class="service-title">Multi_Turn</div>
            <div class="service-desc">
                대화 히스토리와 엔티티 추적으로
                "그 회사"와 같은 대명사를 정확히 해석합니다.
            </div>
            <span class="service-tag">CONTEXT MEMORY</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 데모 섹션
# ============================================================
st.markdown("""
<div class="demo-section">
    <div class="section-header">
        <span class="section-num">02</span>
        <span class="section-title">LIVE_DEMO</span>
    </div>

    <div class="demo-container">
        <div class="demo-window">
            <div class="demo-titlebar">
                <span class="demo-dot red"></span>
                <span class="demo-dot yellow"></span>
                <span class="demo-dot green"></span>
                <span class="demo-title">finance-rag-terminal</span>
            </div>
            <div class="demo-body">
""", unsafe_allow_html=True)

# 채팅 메시지 표시
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg msg-user">
            <div class="msg-label">YOU</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg msg-ai">
            <div class="msg-label">AI</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

        if "sources" in msg and msg["sources"]:
            sources_html = "<div class='sources-box'><div class='sources-title'>SOURCES</div>"
            for src in msg["sources"][:2]:
                sources_html += f"<div>→ {src['title']}</div>"
            sources_html += "</div>"
            st.markdown(sources_html, unsafe_allow_html=True)

st.markdown("</div></div></div>", unsafe_allow_html=True)

# 입력
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "질문",
        placeholder="예: 삼성전자 3분기 실적이 어떻게 되나요?",
        key="user_input",
        label_visibility="collapsed"
    )
with col2:
    send = st.button("SEND", use_container_width=True)

# 예시 버튼
st.markdown("</div>", unsafe_allow_html=True)

ex_cols = st.columns(3)
examples = ["삼성전자 3분기 실적", "HBM 시장 현황", "연준 금리 정책"]
for i, ex in enumerate(examples):
    with ex_cols[i]:
        if st.button(f"→ {ex}", key=f"ex_{i}"):
            user_input = ex

# 질문 처리
if (send or user_input) and user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    results = st.session_state.search.search(user_input)
    context = "\n\n".join([f"[{r['doc'].title}]\n{r['doc'].content}" for r in results])

    system_prompt = "당신은 금융 전문 AI입니다. 제공된 문서를 기반으로 정확하게 답하세요."
    user_prompt = f"[문서]\n{context}\n\n[질문]\n{user_input}"

    with st.spinner("분석 중..."):
        full_response = ""
        placeholder = st.empty()
        for chunk in st.session_state.llm.generate_stream(system_prompt, user_prompt):
            full_response += chunk
            placeholder.markdown(f"<div class='msg msg-ai'>{full_response}</div>", unsafe_allow_html=True)

    sources = [{"title": r["doc"].title} for r in results]
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
    st.rerun()

# ============================================================
# 테크 스택
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-header">
        <span class="section-num">03</span>
        <span class="section-title">TECH_STACK</span>
    </div>

    <div class="tech-list">
        <span class="tech-tag">LLaMA 3.1</span>
        <span class="tech-tag">Groq</span>
        <span class="tech-tag">Python</span>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">BM25</span>
        <span class="tech-tag">Vector DB</span>
        <span class="tech-tag">RAGAS</span>
        <span class="tech-tag">Hybrid Search</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 푸터
# ============================================================
st.markdown("""
<div class="footer">
    <div class="footer-left">FINANCE_RAG ✦</div>
    <div class="footer-right">
        Built by 김다운 · AI/ML Portfolio · 2024
    </div>
</div>
""", unsafe_allow_html=True)
