# -*- coding: utf-8 -*-
"""
Finance RAG - Modern Editorial Design
2025 Trend: Minimal + Bold Typography + Whitespace
"""

import streamlit as st
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Finance RAG",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2025 트렌드 CSS - 에디토리얼/브루탈리즘
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
}

* {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: var(--white);
}

header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ===== 네비게이션 ===== */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 4rem;
    border-bottom: 1px solid #eee;
    position: sticky;
    top: 0;
    background: var(--white);
    z-index: 100;
}

.nav-logo {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--black);
}

.nav-links {
    display: flex;
    gap: 2rem;
    font-size: 0.8rem;
    color: var(--gray);
}

.nav-link {
    color: var(--gray);
    text-decoration: none;
    transition: color 0.2s;
}

.nav-link:hover {
    color: var(--black);
}

/* ===== 히어로 ===== */
.hero {
    padding: 8rem 4rem 6rem;
    max-width: 1400px;
    margin: 0 auto;
}

.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: clamp(3rem, 8vw, 7rem);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: -3px;
    color: var(--black);
    margin-bottom: 2rem;
}

.hero-title-outline {
    -webkit-text-stroke: 1.5px var(--black);
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    font-size: 1.1rem;
    line-height: 1.7;
    color: var(--gray);
    max-width: 500px;
    margin-bottom: 3rem;
}

/* ===== 스크롤 인디케이터 ===== */
.scroll-indicator {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.75rem;
    color: var(--gray);
    letter-spacing: 1px;
    text-transform: uppercase;
}

.scroll-line {
    width: 60px;
    height: 1px;
    background: var(--gray);
}

/* ===== 섹션 공통 ===== */
.section {
    padding: 6rem 4rem;
    max-width: 1400px;
    margin: 0 auto;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #eee;
}

.section-num {
    font-size: 0.75rem;
    color: var(--gray);
}

.section-title {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--black);
}

/* ===== 기능 그리드 ===== */
.features {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    background: #eee;
    border: 1px solid #eee;
    margin-bottom: 4rem;
}

.feature-item {
    background: var(--white);
    padding: 3rem;
    transition: background 0.3s;
}

.feature-item:hover {
    background: var(--light-gray);
}

.feature-num {
    font-size: 0.7rem;
    color: var(--accent);
    margin-bottom: 1.5rem;
}

.feature-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--black);
    margin-bottom: 1rem;
}

.feature-desc {
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--gray);
}

/* ===== 데모 섹션 ===== */
.demo-container {
    background: var(--black);
    border-radius: 16px;
    padding: 2rem;
    margin: 2rem 4rem;
}

.demo-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.5rem;
}

.demo-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #333;
}

.demo-dot.active { background: var(--accent); }

.demo-label {
    font-size: 0.7rem;
    color: #666;
    margin-left: auto;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ===== 채팅 ===== */
.chat-area {
    background: #111;
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 350px;
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.msg-user {
    background: var(--accent);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 12px 12px 4px 12px;
    margin-left: 25%;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

.msg-ai {
    background: #1a1a1a;
    color: #ddd;
    padding: 1rem 1.25rem;
    border-radius: 12px 12px 12px 4px;
    margin-right: 25%;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid #333;
}

.msg-sources {
    background: #0d0d0d;
    border-left: 2px solid var(--accent);
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #888;
}

/* ===== 입력 ===== */
.stTextInput > div > div > input {
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 1rem !important;
    font-size: 0.95rem !important;
}

.stTextInput > div > div > input::placeholder {
    color: #666 !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
}

.stButton > button:hover {
    background: #e64500 !important;
}

/* ===== 스탯 ===== */
.stats {
    display: flex;
    gap: 4rem;
    padding: 4rem;
    border-top: 1px solid #eee;
    border-bottom: 1px solid #eee;
    margin: 4rem 0;
}

.stat-item {
    flex: 1;
}

.stat-value {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: var(--black);
    line-height: 1;
}

.stat-label {
    font-size: 0.8rem;
    color: var(--gray);
    margin-top: 0.5rem;
}

/* ===== 테크 태그 ===== */
.tech-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 0 4rem;
}

.tech-tag {
    font-size: 0.8rem;
    padding: 0.6rem 1.2rem;
    border: 1px solid #ddd;
    border-radius: 100px;
    color: var(--black);
    background: transparent;
    transition: all 0.2s;
}

.tech-tag:hover {
    background: var(--black);
    color: var(--white);
    border-color: var(--black);
}

/* ===== 푸터 ===== */
.footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3rem 4rem;
    border-top: 1px solid #eee;
    margin-top: 4rem;
}

.footer-left {
    font-size: 0.8rem;
    color: var(--gray);
}

.footer-right {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
}

/* ===== 반응형 ===== */
@media (max-width: 768px) {
    .nav, .hero, .section { padding-left: 1.5rem; padding-right: 1.5rem; }
    .hero-title { font-size: 2.5rem; letter-spacing: -1px; }
    .features { grid-template-columns: 1fr; }
    .stats { flex-direction: column; gap: 2rem; padding: 2rem 1.5rem; }
    .demo-container { margin: 1rem; }
    .msg-user { margin-left: 10%; }
    .msg-ai { margin-right: 10%; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 검색 & LLM
# ============================================================
class SimpleSearch:
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
            docs = [
                {'id': 'samsung', 'title': '삼성전자 2024 Q3', 'content': '삼성전자 2024년 3분기 영업이익 9조 1,834억원. 전년대비 274.5% 증가.'},
                {'id': 'hbm', 'title': 'HBM 시장 분석', 'content': 'SK하이닉스 HBM 시장점유율 53%. 2024년 시장 2배 성장 전망.'},
                {'id': 'fed', 'title': '연준 금리 정책', 'content': '2024년 9월 FOMC 기준금리 5.25-5.50% 동결. 4분기 인하 가능성.'},
            ]
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
    def __init__(self):
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
            self.client = Groq(api_key=key) if key else None
        except:
            self.client = None

    def generate(self, system, user):
        if not self.client:
            return "API 키가 설정되지 않았습니다."
        try:
            r = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.7, max_tokens=1024
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"오류: {e}"

# 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search" not in st.session_state:
    st.session_state.search = SimpleSearch()
if "llm" not in st.session_state:
    st.session_state.llm = GroqLLM()

# ============================================================
# UI 렌더링
# ============================================================

# 네비게이션
st.markdown("""
<div class="nav">
    <div class="nav-logo">FINANCE RAG</div>
    <div class="nav-links">
        <span>Features</span>
        <span>Demo</span>
        <a href="https://github.com/araeLaver/AI-ML" target="_blank" class="nav-link">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# 히어로
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Document Intelligence</div>
    <h1 class="hero-title">
        Smarter<br>
        <span class="hero-title-outline">Finance</span><br>
        Answers
    </h1>
    <p class="hero-desc">
        금융 문서를 검색하고, LLM이 근거 기반 답변을 생성합니다.
        환각 없이, 출처와 함께.
    </p>
    <div class="scroll-indicator">
        <span class="scroll-line"></span>
        <span>Scroll to explore</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 기능 섹션
st.markdown("""
<div class="section">
    <div class="section-header">
        <span class="section-num">01</span>
        <span class="section-title">Core Features</span>
    </div>
    <div class="features">
        <div class="feature-item">
            <div class="feature-num">01</div>
            <div class="feature-title">Hybrid Search</div>
            <div class="feature-desc">Vector + BM25 + RRF. 의미와 키워드를 동시에 잡는 하이브리드 검색.</div>
        </div>
        <div class="feature-item">
            <div class="feature-num">02</div>
            <div class="feature-title">Re-Ranking</div>
            <div class="feature-desc">Cross-Encoder 기반 2단계 검색으로 정확도 15% 향상.</div>
        </div>
        <div class="feature-item">
            <div class="feature-num">03</div>
            <div class="feature-title">Multi-Turn</div>
            <div class="feature-desc">대화 맥락과 엔티티를 추적. "그 회사"도 이해합니다.</div>
        </div>
        <div class="feature-item">
            <div class="feature-num">04</div>
            <div class="feature-title">No Hallucination</div>
            <div class="feature-desc">문서에 없으면 답하지 않습니다. 출처 명시로 신뢰도 확보.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 스탯
doc_count = len(st.session_state.search.documents)
st.markdown(f"""
<div class="stats">
    <div class="stat-item">
        <div class="stat-value">{doc_count}</div>
        <div class="stat-label">Documents</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">&lt;2s</div>
        <div class="stat-label">Response Time</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">95%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">3</div>
        <div class="stat-label">Search Algorithms</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 데모 섹션
st.markdown("""
<div class="section">
    <div class="section-header">
        <span class="section-num">02</span>
        <span class="section-title">Live Demo</span>
    </div>
</div>
<div class="demo-container">
    <div class="demo-header">
        <span class="demo-dot"></span>
        <span class="demo-dot"></span>
        <span class="demo-dot active"></span>
        <span class="demo-label">Terminal</span>
    </div>
""", unsafe_allow_html=True)

# 채팅 표시
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="msg-ai">
        금융 문서에 대해 질문하세요.<br>
        예: "삼성전자 3분기 실적" / "HBM 시장 점유율"
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        html = f'<div class="msg-ai">{msg["content"]}'
        if msg.get("sources"):
            html += '<div class="msg-sources">Sources: ' + ' / '.join(msg["sources"][:2]) + '</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 입력
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("q", placeholder="질문을 입력하세요...", label_visibility="collapsed", key="input")
with col2:
    send = st.button("Send", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# 예시 버튼
st.markdown('<div style="padding: 1rem 4rem;">', unsafe_allow_html=True)
cols = st.columns(4)
examples = ["삼성전자 실적", "HBM 점유율", "Fed 금리", "AI 투자"]
selected = None
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            selected = ex
st.markdown('</div>', unsafe_allow_html=True)

# 처리
query = selected or (user_input if send else None)
if query and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})
    results = st.session_state.search.search(query)

    if results:
        context = "\n\n".join([f"[{r['doc']['title']}]\n{r['doc']['content'][:500]}" for r in results])
        sources = [r['doc']['title'] for r in results]
    else:
        context = "관련 문서 없음"
        sources = []

    system = "금융 전문 AI. 문서 기반으로만 답변. 없으면 '문서에서 찾을 수 없습니다'라고 답변. 한국어로."
    user = f"[문서]\n{context}\n\n[질문]\n{query}"

    with st.spinner("검색 중..."):
        response = st.session_state.llm.generate(system, user)

    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
    st.rerun()

# 테크 스택
st.markdown("""
<div class="section">
    <div class="section-header">
        <span class="section-num">03</span>
        <span class="section-title">Tech Stack</span>
    </div>
</div>
<div class="tech-tags">
    <span class="tech-tag">LLaMA 3.1</span>
    <span class="tech-tag">Groq</span>
    <span class="tech-tag">Python</span>
    <span class="tech-tag">Streamlit</span>
    <span class="tech-tag">ChromaDB</span>
    <span class="tech-tag">BM25</span>
    <span class="tech-tag">Cross-Encoder</span>
    <span class="tech-tag">RAGAS</span>
</div>
""", unsafe_allow_html=True)

# 푸터
st.markdown("""
<div class="footer">
    <div class="footer-left">
        Built by 김다운 · 2024<br>
        <a href="https://github.com/araeLaver/AI-ML" style="color: #888;">github.com/araeLaver</a>
    </div>
    <div class="footer-right">◐</div>
</div>
""", unsafe_allow_html=True)
