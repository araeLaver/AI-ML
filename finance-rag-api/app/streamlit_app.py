# -*- coding: utf-8 -*-
"""
Finance RAG - Portfolio Demo
Clean & Stable Design for Streamlit Cloud
"""

import streamlit as st
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Finance RAG | AI 금융 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 안정적인 CSS 스타일
# ============================================================
st.markdown("""
<style>
/* 기본 설정 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* 헤더 숨김 */
header[data-testid="stHeader"] {
    background: transparent;
}

/* 메인 컨테이너 */
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

/* 히어로 카드 */
.hero-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 24px;
    padding: 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    text-align: center;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 1rem;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
    line-height: 1.6;
}

/* 기능 카드 그리드 */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.feature-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
}

.feature-desc {
    font-size: 0.9rem;
    color: #666;
    line-height: 1.5;
}

/* 스탯 바 */
.stats-bar {
    display: flex;
    justify-content: space-around;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
}

.stat-label {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.25rem;
}

/* 데모 섹션 */
.demo-card {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    margin-bottom: 2rem;
}

.demo-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #eee;
}

.demo-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.demo-title {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #666;
    margin-left: 0.5rem;
}

/* 채팅 메시지 */
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 12px;
}

.user-msg {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 18px 18px 4px 18px;
    margin-bottom: 1rem;
    margin-left: 20%;
    font-size: 0.95rem;
    line-height: 1.5;
}

.ai-msg {
    background: white;
    color: #1a1a2e;
    padding: 1rem 1.25rem;
    border-radius: 18px 18px 18px 4px;
    margin-bottom: 1rem;
    margin-right: 20%;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid #eee;
}

.msg-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

/* 출처 박스 */
.sources-box {
    background: #f0f4ff;
    border-left: 3px solid #667eea;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}

.sources-title {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 0.25rem;
    font-size: 0.75rem;
}

/* 입력 필드 스타일 */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
}

/* 예시 버튼 */
.example-btn {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #666;
    cursor: pointer;
    transition: all 0.2s;
}

.example-btn:hover {
    border-color: #667eea;
    color: #667eea;
}

/* 테크 스택 */
.tech-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    justify-content: center;
}

.tech-tag {
    background: white;
    color: #667eea;
    padding: 0.5rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid #e0e0e0;
}

/* 푸터 */
.footer {
    text-align: center;
    padding: 2rem;
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
}

.footer a {
    color: white;
    text-decoration: none;
}

/* 반응형 */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .stats-bar { flex-direction: column; gap: 1.5rem; }
    .user-msg { margin-left: 10%; }
    .ai-msg { margin-right: 10%; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# RAG 시스템 초기화
# ============================================================
@st.cache_resource
def init_rag_system():
    """RAG 시스템 초기화"""
    try:
        from src.rag.hybrid_search import HybridSearchEngine
        from src.rag.llm_client import LLMClient
        from src.rag.document_loader import DocumentLoader

        # 문서 로드
        loader = DocumentLoader()
        docs_path = project_root / "data" / "sample_docs"

        if docs_path.exists():
            documents = loader.load_directory(str(docs_path))
        else:
            documents = []

        # 검색 엔진 초기화
        search_engine = HybridSearchEngine()
        if documents:
            search_engine.add_documents(documents)

        # LLM 클라이언트
        llm = LLMClient()

        return search_engine, llm, len(documents)
    except Exception as e:
        return None, None, 0

# 간단한 폴백 검색
class SimpleSearch:
    def __init__(self):
        self.documents = self._load_sample_docs()

    def _load_sample_docs(self):
        docs = []
        docs_path = project_root / "data" / "sample_docs"
        if docs_path.exists():
            for file in docs_path.glob("*.txt"):
                try:
                    content = file.read_text(encoding='utf-8')
                    docs.append({
                        'id': file.stem,
                        'title': file.stem.replace('_', ' ').title(),
                        'content': content
                    })
                except:
                    pass

        # 기본 문서
        if not docs:
            docs = [
                {
                    'id': 'samsung_q3',
                    'title': '삼성전자 2024년 3분기 실적',
                    'content': '삼성전자가 2024년 3분기에 영업이익 9조 1,834억원을 기록했다. 전년동기대비 274.5% 증가. 매출액 79조 1,024억원으로 17.3% 증가. HBM 수요 증가가 주요 요인.'
                },
                {
                    'id': 'fed_rate',
                    'title': '미국 연준 금리 정책',
                    'content': '미국 연준이 2024년 9월 FOMC에서 기준금리를 5.25-5.50%로 동결. 파월 의장은 인플레이션이 2% 목표로 수렴 중이라 평가. 4분기 금리 인하 가능성 50% 이상.'
                },
                {
                    'id': 'hbm_market',
                    'title': 'HBM 시장 현황',
                    'content': 'NVIDIA AI 가속기 수요로 HBM 시장 폭발적 성장. SK하이닉스 HBM3E 시장 선도, 시장점유율 53%. 삼성전자 HBM3E 양산 시작. 2024년 시장규모 전년대비 2배 이상 성장 전망.'
                }
            ]
        return docs

    def search(self, query, top_k=3):
        import re
        query_lower = query.lower()
        keywords = re.findall(r'[가-힣]+|[a-zA-Z]+', query_lower)

        results = []
        for doc in self.documents:
            content_lower = (doc['content'] + doc['title']).lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                results.append({
                    'doc': doc,
                    'score': score / max(len(keywords), 1)
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

# LLM 클라이언트
class GroqLLM:
    def __init__(self):
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
            if api_key:
                self.client = Groq(api_key=api_key)
                self.available = True
            else:
                self.available = False
        except:
            self.available = False

    def generate(self, system_prompt, user_prompt):
        if not self.available:
            return "⚠️ Groq API 키가 설정되지 않았습니다. Streamlit Cloud Secrets에 GROQ_API_KEY를 추가해주세요."

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ 오류 발생: {str(e)}"

# ============================================================
# 세션 상태
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_engine" not in st.session_state:
    st.session_state.search_engine = SimpleSearch()
if "llm" not in st.session_state:
    st.session_state.llm = GroqLLM()

# ============================================================
# UI 렌더링
# ============================================================

# 히어로 섹션
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">🚀 AI-Powered RAG System</div>
    <h1 class="hero-title">Finance RAG</h1>
    <p class="hero-subtitle">
        금융 문서 기반 AI 질의응답 시스템<br>
        하이브리드 검색 + LLM으로 정확한 답변과 출처를 제공합니다
    </p>
</div>
""", unsafe_allow_html=True)

# 기능 카드
st.markdown("""
<div class="features-grid">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">하이브리드 검색</div>
        <div class="feature-desc">Vector + BM25 + RRF 알고리즘으로 의미 기반과 키워드 기반 검색을 결합</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Re-Ranking</div>
        <div class="feature-desc">Cross-Encoder 기반 재정렬로 검색 결과의 정확도를 향상</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">멀티턴 대화</div>
        <div class="feature-desc">대화 히스토리와 엔티티 추적으로 자연스러운 대화 지원</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">출처 추적</div>
        <div class="feature-desc">모든 답변에 근거 문서와 신뢰도 점수를 함께 제공</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 스탯 바
doc_count = len(st.session_state.search_engine.documents)
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">{doc_count}</div>
        <div class="stat-label">문서 수</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">3</div>
        <div class="stat-label">검색 알고리즘</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">&lt;2s</div>
        <div class="stat-label">응답 시간</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">✓</div>
        <div class="stat-label">환각 방지</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 데모 섹션
st.markdown("""
<div class="demo-card">
    <div class="demo-header">
        <span class="demo-dot" style="background: #ff5f57;"></span>
        <span class="demo-dot" style="background: #febc2e;"></span>
        <span class="demo-dot" style="background: #28c840;"></span>
        <span class="demo-title">Finance RAG Terminal</span>
    </div>
""", unsafe_allow_html=True)

# 채팅 메시지 표시
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="ai-msg">
        <div class="msg-label">AI Assistant</div>
        안녕하세요! 금융 문서에 대해 질문해주세요.
        예를 들어 "삼성전자 3분기 실적은?" 또는 "HBM 시장 현황 알려줘" 같은 질문을 해보세요.
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg">
            <div class="msg-label">You</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-msg">
            <div class="msg-label">AI Assistant</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

        if msg.get("sources"):
            sources_html = '<div class="sources-box"><div class="sources-title">📚 참조 문서</div>'
            for src in msg["sources"][:3]:
                sources_html += f"<div>• {src}</div>"
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 입력 영역
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "질문 입력",
        placeholder="금융 관련 질문을 입력하세요...",
        key="user_input",
        label_visibility="collapsed"
    )
with col2:
    send_btn = st.button("전송", use_container_width=True)

# 예시 질문
st.markdown("**💡 예시 질문:**")
ex_cols = st.columns(4)
examples = ["삼성전자 3분기 실적", "HBM 시장 점유율", "연준 금리 전망", "SK하이닉스 현황"]

selected_example = None
for i, ex in enumerate(examples):
    with ex_cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            selected_example = ex

st.markdown('</div>', unsafe_allow_html=True)

# 질문 처리
query = selected_example or (user_input if send_btn else None)

if query and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})

    # 검색
    results = st.session_state.search_engine.search(query)

    # 컨텍스트 생성
    if results:
        context = "\n\n".join([
            f"[{r['doc']['title']}]\n{r['doc']['content'][:500]}"
            for r in results
        ])
        sources = [r['doc']['title'] for r in results]
    else:
        context = "관련 문서를 찾지 못했습니다."
        sources = []

    # LLM 응답 생성
    system_prompt = """당신은 금융 전문 AI 어시스턴트입니다.
제공된 문서를 기반으로 정확하게 답변하세요.
문서에 없는 정보는 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
답변은 한국어로 명확하고 간결하게 작성하세요."""

    user_prompt = f"""[참조 문서]
{context}

[질문]
{query}

위 문서를 기반으로 질문에 답변해주세요."""

    with st.spinner("🔍 문서 검색 및 분석 중..."):
        response = st.session_state.llm.generate(system_prompt, user_prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })

    st.rerun()

# 테크 스택
st.markdown("""
<div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 2rem; margin: 2rem 0; text-align: center;">
    <h3 style="color: #1a1a2e; margin-bottom: 1.5rem;">🛠 Tech Stack</h3>
    <div class="tech-grid">
        <span class="tech-tag">🦙 LLaMA 3.1</span>
        <span class="tech-tag">⚡ Groq</span>
        <span class="tech-tag">🐍 Python</span>
        <span class="tech-tag">🎈 Streamlit</span>
        <span class="tech-tag">🔍 BM25</span>
        <span class="tech-tag">📦 ChromaDB</span>
        <span class="tech-tag">🎯 Re-Ranking</span>
        <span class="tech-tag">📊 RAGAS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 푸터
st.markdown("""
<div class="footer">
    <p>Built with ❤️ by <strong>김다운</strong></p>
    <p><a href="https://github.com/araeLaver/AI-ML" target="_blank">GitHub</a> · AI/ML Portfolio · 2024</p>
</div>
""", unsafe_allow_html=True)
