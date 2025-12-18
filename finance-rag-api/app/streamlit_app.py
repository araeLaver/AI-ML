# -*- coding: utf-8 -*-
"""
Finance RAG API - 포트폴리오 웹 데모

[포트폴리오 핵심]
- 프로젝트 소개 및 아키텍처 설명
- 스트리밍 RAG Q&A 데모
- 검색 결과 시각화

실행:
    streamlit run app/streamlit_app.py --server.port 8501
"""

import streamlit as st
import requests
import time
import json
import sys
from pathlib import Path
from typing import Optional, Generator

# 프로젝트 루트를 path에 추가 (로컬 모듈 import용)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# API 서버 설정
API_BASE_URL = "http://localhost:8000/api/v1"

# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="Finance RAG - AI 금융 Q&A",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 커스텀 CSS
# ============================================================

st.markdown("""
<style>
    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }

    /* 카드 스타일 */
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    /* 출처 카드 */
    .source-card {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #17a2b8;
    }

    /* 신뢰도 배지 */
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }

    /* 아키텍처 박스 */
    .arch-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }

    /* 기술 스택 태그 */
    .tech-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }

    /* 스트리밍 텍스트 커서 */
    .streaming-cursor {
        display: inline-block;
        width: 10px;
        height: 20px;
        background: #667eea;
        animation: blink 1s infinite;
    }

    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 세션 상태 초기화
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "intro"

# ============================================================
# 유틸리티 함수
# ============================================================

def check_api_health() -> bool:
    """API 서버 상태 확인"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def get_api_stats() -> Optional[dict]:
    """API 통계 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def stream_rag_response(question: str, top_k: int = 3) -> Generator:
    """
    RAG 스트리밍 응답 - 로컬 모듈 직접 호출
    """
    try:
        from src.rag.vectorstore import VectorStoreService
        from src.rag.rag_service import RAGService
        from src.core.config import get_settings

        settings = get_settings()
        vectorstore = VectorStoreService(
            persist_dir=settings.chroma_persist_dir,
            collection_name="finance_docs"
        )
        rag_service = RAGService(
            vectorstore=vectorstore,
            llm_model="llama3.2",
            top_k=top_k,
            temperature=0.2
        )

        for chunk in rag_service.query_stream(question):
            yield chunk

    except Exception as e:
        yield {"type": "error", "content": str(e)}


def get_documents(limit: int = 20) -> Optional[dict]:
    """문서 목록 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents?limit={limit}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def upload_document(file, source_name: str, chunk_size: int) -> Optional[dict]:
    """문서 업로드"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"source_name": source_name, "chunk_size": str(chunk_size), "chunk_overlap": "100"}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data, timeout=60)
        if response.status_code == 201:
            return response.json()
        else:
            return {"error": True, "detail": response.json().get("detail", "업로드 실패")}
    except Exception as e:
        return {"error": True, "detail": str(e)}


# ============================================================
# 사이드바 - 네비게이션
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=80)
    st.title("Finance RAG")
    st.caption("AI 금융 문서 Q&A 시스템")

    st.divider()

    # 네비게이션 메뉴
    st.subheader("📑 메뉴")

    menu_options = {
        "intro": "🏠 프로젝트 소개",
        "demo": "💬 Q&A 데모",
        "docs": "📄 문서 관리",
        "tech": "🔧 기술 상세"
    }

    for key, label in menu_options.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key
            st.rerun()

    st.divider()

    # API 상태
    st.subheader("⚡ 시스템 상태")
    api_healthy = check_api_health()

    if api_healthy:
        st.success("🟢 API 서버 연결됨")
        stats = get_api_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("문서 수", stats.get("total_documents", 0))
            with col2:
                st.metric("Top-K", stats.get("top_k", 3))
    else:
        st.error("🔴 API 서버 오프라인")
        st.caption("```\nuvicorn src.main:app --reload\n```")

    st.divider()
    st.caption("Made with ❤️ by 김다운")


# ============================================================
# 페이지: 프로젝트 소개
# ============================================================

def render_intro_page():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>💰 Finance RAG API</h1>
        <p>금융 문서 기반 AI Q&A 시스템 - LLM 환각 방지 RAG 구현</p>
    </div>
    """, unsafe_allow_html=True)

    # 프로젝트 개요
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 왜 만들었나요?")
        st.markdown("""
        <div class="info-card">
        <h4>문제 인식</h4>
        LLM(대규모 언어 모델)은 강력하지만 <b>환각(Hallucination)</b> 문제가 있습니다.
        금융 분야에서는 잘못된 정보가 실제 투자 손실로 이어질 수 있어 특히 위험합니다.

        <h4>해결 방안</h4>
        <b>RAG(Retrieval-Augmented Generation)</b>를 적용하여:
        <ul>
            <li>검증된 금융 문서에서만 정보를 검색</li>
            <li>LLM이 검색된 문서 기반으로만 답변</li>
            <li>답변의 출처와 신뢰도를 명시</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🏗️ 시스템 아키텍처")
        st.markdown("""
        <div class="arch-box">
┌─────────────────────────────────────────────────────────────────────┐
│                         Finance RAG API                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   👤 User                                                            │
│     │                                                                │
│     ▼                                                                │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │   Streamlit  │───▶│   FastAPI    │───▶│  RAG Service │         │
│   │   (Web UI)   │    │   (REST)     │    │  (Pipeline)  │         │
│   └──────────────┘    └──────────────┘    └──────┬───────┘         │
│                                                   │                  │
│                              ┌────────────────────┼────────────┐    │
│                              │                    │            │    │
│                              ▼                    ▼            │    │
│                        ┌──────────┐        ┌──────────┐       │    │
│                        │ ChromaDB │        │  Ollama  │       │    │
│                        │ (Vector) │        │  (LLM)   │       │    │
│                        └──────────┘        └──────────┘       │    │
│                                                                │    │
└─────────────────────────────────────────────────────────────────────┘

📊 RAG 파이프라인:
1. 질문 입력 → 2. 벡터 검색 → 3. 컨텍스트 구성 → 4. LLM 생성 → 5. 출처와 반환
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🛠️ 기술 스택")

        tech_stack = {
            "Backend": ["FastAPI", "Python 3.11", "Pydantic"],
            "AI/ML": ["Ollama", "LLama 3.2", "ChromaDB"],
            "Testing": ["pytest", "35 tests"],
            "DevOps": ["Docker", "Docker Compose"],
            "Frontend": ["Streamlit"]
        }

        for category, techs in tech_stack.items():
            st.markdown(f"**{category}**")
            tags = " ".join([f'<span class="tech-tag">{t}</span>' for t in techs])
            st.markdown(tags, unsafe_allow_html=True)
            st.write("")

        st.subheader("✨ 주요 기능")
        features = [
            "🔍 벡터 유사도 검색",
            "🤖 스트리밍 LLM 응답",
            "📄 PDF/텍스트 파싱",
            "📊 신뢰도 점수 제공",
            "🔒 API Key 인증",
            "⚡ Rate Limiting"
        ]
        for f in features:
            st.markdown(f"- {f}")

    st.divider()

    # 핵심 코드 하이라이트
    st.subheader("💡 핵심 구현 포인트")

    tab1, tab2, tab3 = st.tabs(["환각 방지 프롬프트", "벡터 검색", "청킹 전략"])

    with tab1:
        st.code('''
# 환각 방지를 위한 시스템 프롬프트
SYSTEM_PROMPT = """당신은 금융 전문 상담 AI입니다.

규칙:
1. 문서에 없는 내용은 "해당 정보가 제공된 문서에 없습니다"라고 답하세요
2. 추측하거나 지어내지 마세요
3. 숫자나 수치는 문서 그대로 인용하세요
4. 답변 마지막에 참조 문서를 표시하세요

주의:
- 이 정보는 투자 권유가 아닙니다
- 실제 투자 결정은 전문가와 상담하세요"""
        ''', language="python")

    with tab2:
        st.code('''
# ChromaDB 벡터 검색
def search(self, query: str, top_k: int = 5):
    results = self.collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 거리 → 유사도 변환
    relevance = 1 - distance / 2  # 0~1 정규화
    return results
        ''', language="python")

    with tab3:
        st.code('''
# LangChain 스타일 재귀적 텍스트 분할
class RecursiveTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.separators = ["\\n\\n", "\\n", ". ", " "]

    def split(self, text: str) -> List[str]:
        # 문단 → 문장 → 단어 순으로 분할 시도
        # 의미 단위를 최대한 유지
        ''', language="python")


# ============================================================
# 페이지: Q&A 데모
# ============================================================

def render_demo_page():
    st.markdown("""
    <div class="main-header">
        <h1>💬 RAG Q&A 데모</h1>
        <p>금융 문서를 검색하여 AI가 답변합니다 (스트리밍 응답)</p>
    </div>
    """, unsafe_allow_html=True)

    # 설정
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        top_k = st.slider("검색 문서 수", 1, 10, 3)
    with col2:
        st.write("")  # 스페이서
    with col3:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 출처 시각화
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 참조 문서 및 관련도", expanded=False):
                    for source in message["sources"]:
                        score = source.get("relevance_score", 0)
                        score_pct = int(score * 100)

                        st.markdown(f"""
                        <div class="source-card">
                            <b>{source['source']}</b>
                            <div style="background: #ddd; border-radius: 10px; height: 8px; margin: 5px 0;">
                                <div style="background: linear-gradient(90deg, #667eea, #764ba2);
                                            width: {score_pct}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                            <small>관련도: {score_pct}%</small><br>
                            <small style="color: #666;">{source['content_preview']}</small>
                        </div>
                        """, unsafe_allow_html=True)

    # 예시 질문 버튼
    if not st.session_state.messages:
        st.markdown("#### 💡 예시 질문을 클릭해보세요")
        example_questions = [
            "ETF가 뭔가요?",
            "분산 투자의 장점은?",
            "초보자 투자 방법 알려주세요",
            "레버리지 ETF는 왜 위험한가요?"
        ]

        cols = st.columns(4)
        for i, q in enumerate(example_questions):
            with cols[i]:
                if st.button(q, key=f"example_{i}"):
                    st.session_state.pending_question = q
                    st.rerun()

    # 질문 입력
    if prompt := st.chat_input("금융에 대해 무엇이든 물어보세요..."):
        process_question(prompt, top_k)

    # 예시 질문 처리
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        process_question(question, top_k)


def process_question(question: str, top_k: int):
    """질문 처리 및 스트리밍 응답"""
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # AI 응답 (스트리밍)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_response = ""
        sources = []
        confidence = "medium"

        # 스트리밍 응답
        with st.spinner("🔍 관련 문서 검색 중..."):
            for chunk in stream_rag_response(question, top_k):
                if chunk["type"] == "sources":
                    sources = chunk["content"]
                    # 검색 완료 표시
                    sources_placeholder.info(f"📚 {len(sources)}개 관련 문서 발견")
                    time.sleep(0.3)
                    sources_placeholder.empty()

                elif chunk["type"] == "token":
                    full_response += chunk["content"]
                    response_placeholder.markdown(full_response + "▌")

                elif chunk["type"] == "done":
                    confidence = chunk.get("confidence", "medium")

                elif chunk["type"] == "error":
                    st.error(f"오류: {chunk['content']}")
                    return

        # 최종 응답 표시
        response_placeholder.markdown(full_response)

        # 신뢰도 표시
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        confidence_text = {"high": "높음", "medium": "보통", "low": "낮음"}
        st.caption(f"{confidence_emoji.get(confidence, '⚪')} 신뢰도: {confidence_text.get(confidence, '알 수 없음')}")

        # 출처 시각화
        if sources:
            with st.expander("📚 참조 문서 및 관련도", expanded=True):
                for source in sources:
                    score = source.get("relevance_score", 0)
                    score_pct = int(score * 100)

                    st.markdown(f"""
                    <div class="source-card">
                        <b>{source['source']}</b>
                        <div style="background: #ddd; border-radius: 10px; height: 8px; margin: 5px 0;">
                            <div style="background: linear-gradient(90deg, #667eea, #764ba2);
                                        width: {score_pct}%; height: 100%; border-radius: 10px;"></div>
                        </div>
                        <small>관련도: {score_pct}%</small><br>
                        <small style="color: #666;">{source['content_preview']}</small>
                    </div>
                    """, unsafe_allow_html=True)

        # 세션에 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "confidence": confidence
        })


# ============================================================
# 페이지: 문서 관리
# ============================================================

def render_docs_page():
    st.markdown("""
    <div class="main-header">
        <h1>📄 문서 관리</h1>
        <p>RAG 시스템에 금융 문서를 추가하고 관리합니다</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 문서 업로드")

        uploaded_file = st.file_uploader(
            "PDF 또는 텍스트 파일",
            type=["pdf", "txt", "md"],
            help="지원 형식: PDF, TXT, Markdown"
        )

        source_name = st.text_input("출처명", placeholder="예: 투자가이드 2024")
        chunk_size = st.slider("청크 크기", 200, 1000, 500, help="문서를 나누는 단위 (글자 수)")

        if uploaded_file and st.button("📤 업로드", type="primary"):
            with st.spinner("업로드 및 처리 중..."):
                result = upload_document(
                    uploaded_file,
                    source_name or uploaded_file.name,
                    chunk_size
                )

            if result and not result.get("error"):
                st.success(f"✅ {result['chunks_created']}개 청크 생성!")
                st.balloons()
            else:
                st.error(f"❌ {result.get('detail', '업로드 실패')}")

    with col2:
        st.subheader("📋 등록된 문서")

        if st.button("🔄 새로고침"):
            st.rerun()

        docs = get_documents(50)
        if docs and docs.get("documents"):
            st.metric("총 문서 수", docs.get("total_count", 0))

            for doc in docs["documents"][:15]:
                meta = doc.get("metadata", {})
                source = meta.get("source", "Unknown")

                with st.expander(f"📄 {source}"):
                    st.markdown(f"**ID**: `{doc.get('id', 'N/A')[:30]}...`")
                    st.markdown(f"**내용**: {doc.get('content_preview', '미리보기 없음')}")
        else:
            st.info("등록된 문서가 없습니다.")


# ============================================================
# 페이지: 기술 상세
# ============================================================

def render_tech_page():
    st.markdown("""
    <div class="main-header">
        <h1>🔧 기술 상세</h1>
        <p>프로젝트의 기술적 구현 세부사항</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📁 프로젝트 구조", "🔄 RAG 파이프라인", "🧪 테스트", "🐳 배포"])

    with tab1:
        st.subheader("프로젝트 구조")
        st.code('''
finance-rag-api/
├── src/
│   ├── api/                    # API Layer
│   │   ├── routes.py           # REST 엔드포인트 (7개)
│   │   ├── schemas.py          # Pydantic 스키마
│   │   ├── security.py         # API Key + Rate Limiting
│   │   ├── middleware.py       # 요청 로깅
│   │   └── exception_handlers.py
│   │
│   ├── rag/                    # RAG Core
│   │   ├── rag_service.py      # RAG 파이프라인 (핵심!)
│   │   ├── vectorstore.py      # ChromaDB 래퍼
│   │   └── document_loader.py  # PDF/텍스트 파싱 + 청킹
│   │
│   ├── core/                   # 공통 모듈
│   │   ├── config.py           # 환경변수 설정
│   │   ├── logging.py          # 구조화된 로깅
│   │   └── exceptions.py       # 커스텀 예외 (15개)
│   │
│   └── main.py                 # FastAPI 앱 진입점
│
├── tests/                      # 테스트 (35개)
│   ├── test_api.py
│   ├── test_vectorstore.py
│   └── test_document_loader.py
│
├── app/
│   └── streamlit_app.py        # 웹 데모 (현재 페이지)
│
├── Dockerfile                  # 멀티스테이지 빌드
├── docker-compose.yml          # API + Ollama 오케스트레이션
└── Makefile                    # 편의 명령어
        ''', language="text")

        st.subheader("레이어드 아키텍처")
        st.markdown("""
        | Layer | 역할 | Spring 비유 |
        |-------|------|-------------|
        | **API Layer** | HTTP 요청/응답 처리 | `@Controller` |
        | **Service Layer** | 비즈니스 로직 | `@Service` |
        | **Repository Layer** | 데이터 액세스 | `@Repository` |
        """)

    with tab2:
        st.subheader("RAG 파이프라인 상세")

        st.markdown("""
        ```
        [질문] "ETF가 뭔가요?"
              │
              ▼
        ┌─────────────────────────────────────────┐
        │ 1. Retrieval (검색)                     │
        │    - 질문을 임베딩 벡터로 변환           │
        │    - ChromaDB에서 유사 문서 검색         │
        │    - top_k개 문서 + 거리(유사도) 반환    │
        └─────────────────────────────────────────┘
              │
              ▼
        ┌─────────────────────────────────────────┐
        │ 2. Augmentation (증강)                  │
        │    - 검색된 문서로 컨텍스트 구성         │
        │    - 시스템 프롬프트 + 컨텍스트 + 질문   │
        └─────────────────────────────────────────┘
              │
              ▼
        ┌─────────────────────────────────────────┐
        │ 3. Generation (생성)                    │
        │    - Ollama (llama3.2)로 답변 생성      │
        │    - 스트리밍으로 토큰 단위 반환         │
        │    - 환각 방지 프롬프트 적용             │
        └─────────────────────────────────────────┘
              │
              ▼
        [응답] + 출처 + 신뢰도
        ```
        """)

        st.subheader("신뢰도 계산")
        st.code('''
# 평균 유사도 기반 신뢰도 판단
avg_relevance = sum(1 - dist/2 for dist in distances) / len(distances)

if avg_relevance > 0.6:
    confidence = "high"    # 검색 결과가 질문과 매우 관련
elif avg_relevance > 0.4:
    confidence = "medium"  # 어느 정도 관련
else:
    confidence = "low"     # 관련성 낮음 (주의 필요)
        ''', language="python")

    with tab3:
        st.subheader("테스트 현황")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 테스트", "35개")
        with col2:
            st.metric("통과율", "100%")
        with col3:
            st.metric("커버리지", "~85%")

        st.markdown("""
        | 테스트 파일 | 테스트 수 | 범위 |
        |------------|----------|------|
        | `test_api.py` | 11개 | 모든 REST 엔드포인트 |
        | `test_document_loader.py` | 16개 | 청킹, PDF 파싱 |
        | `test_vectorstore.py` | 8개 | 검색, 필터링 |
        """)

        st.code('''
# 테스트 실행
pytest tests/ -v

# 커버리지 포함
pytest tests/ -v --cov=src --cov-report=html
        ''', language="bash")

    with tab4:
        st.subheader("Docker 배포")

        st.code('''
# 전체 스택 실행 (API + Ollama)
docker-compose up -d

# 서비스 확인
docker-compose ps

# 로그 확인
docker-compose logs -f api
        ''', language="bash")

        st.subheader("Dockerfile (멀티스테이지 빌드)")
        st.code('''
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# 보안: non-root user
RUN useradd --uid 1000 appuser
USER appuser

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]
        ''', language="dockerfile")


# ============================================================
# 메인 라우팅
# ============================================================

page = st.session_state.current_page

if page == "intro":
    render_intro_page()
elif page == "demo":
    render_demo_page()
elif page == "docs":
    render_docs_page()
elif page == "tech":
    render_tech_page()
else:
    render_intro_page()


# ============================================================
# 푸터
# ============================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    💰 Finance RAG API | FastAPI + Ollama + ChromaDB + Streamlit<br>
    <small>© 2024 김다운 - AI/ML 포트폴리오 프로젝트</small>
</div>
""", unsafe_allow_html=True)
