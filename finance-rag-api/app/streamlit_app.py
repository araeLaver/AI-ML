# -*- coding: utf-8 -*-
"""
Finance RAG - 금융 문서 기반 AI Q&A 시스템

포트폴리오 웹 데모
- Groq API (클라우드 LLM)
- ChromaDB (벡터 검색)
- 스트리밍 응답
"""

import streamlit as st
import os
from typing import List, Dict, Any, Generator, Optional
from dataclasses import dataclass

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
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }

    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0f3460;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .highlight-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .source-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1976d2;
    }

    .tech-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .flow-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
        line-height: 1.6;
    }

    .decision-card {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card h3 { color: #0f3460; margin: 0; font-size: 2rem; }
    .metric-card p { color: #666; margin: 0.5rem 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 세션 상태 초기화
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "intro"
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False

# ============================================================
# 핵심 클래스들 (Self-contained)
# ============================================================

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    id: str


class SimpleVectorStore:
    """간단한 ChromaDB 래퍼"""

    def __init__(self, collection_name: str = "finance_docs"):
        try:
            import chromadb
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"ChromaDB 초기화 실패: {e}")
            self.collection = None

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        if self.collection:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def search(self, query: str, top_k: int = 3) -> Dict:
        if not self.collection or self.collection.count() == 0:
            return {"documents": [], "metadatas": [], "distances": []}

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def count(self) -> int:
        return self.collection.count() if self.collection else 0


class GroqLLM:
    """Groq API 래퍼"""

    SYSTEM_PROMPT = """당신은 금융 전문 상담 AI입니다.

역할:
- 제공된 문서만을 기반으로 정확하게 답변합니다
- 금융 용어를 쉽게 설명합니다

규칙:
1. 문서에 없는 내용은 "해당 정보가 제공된 문서에 없습니다"라고 답하세요
2. 추측하거나 지어내지 마세요
3. 숫자나 수치는 문서 그대로 인용하세요

주의: 이 정보는 투자 권유가 아닙니다."""

    def __init__(self, api_key: str):
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = "llama-3.1-8b-instant"
        except ImportError:
            st.error("groq 패키지 필요: pip install groq")
            self.client = None

    def generate_stream(self, context: str, question: str) -> Generator[str, None, None]:
        if not self.client:
            yield "LLM 클라이언트가 초기화되지 않았습니다."
            return

        user_prompt = f"""[참고 문서]
{context}

[질문]
{question}

[답변]"""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"오류 발생: {str(e)}"


# ============================================================
# 샘플 금융 데이터
# ============================================================

SAMPLE_FINANCE_DATA = [
    {
        "content": """ETF(Exchange Traded Fund)란?
ETF는 '상장지수펀드'로, 주식처럼 거래소에서 실시간 매매가 가능한 펀드입니다.

주요 특징:
1. 분산투자: 하나의 ETF로 여러 종목에 투자 가능
2. 낮은 비용: 일반 펀드 대비 운용보수가 저렴 (0.1~0.5%)
3. 투명성: 구성 종목이 매일 공개됨
4. 유동성: 주식처럼 실시간 매매 가능

대표적인 ETF 유형:
- 지수 추종 ETF: KOSPI200, S&P500 등 지수를 따라감
- 섹터 ETF: 반도체, 2차전지 등 특정 산업에 투자
- 채권 ETF: 국채, 회사채 등에 투자
- 원자재 ETF: 금, 원유 등에 투자""",
        "source": "ETF 투자 가이드",
        "id": "etf_guide_1"
    },
    {
        "content": """분산투자의 원칙과 방법

분산투자란 '계란을 한 바구니에 담지 않는다'는 투자 원칙입니다.

분산투자의 장점:
1. 리스크 감소: 한 종목 하락 시 전체 손실 제한
2. 안정적 수익: 변동성 완화로 꾸준한 수익 추구
3. 심리적 안정: 급락장에서도 패닉 방지

분산투자 방법:
- 자산 분산: 주식 60%, 채권 30%, 현금 10%
- 지역 분산: 국내 50%, 해외 50%
- 시간 분산: 매월 정액 적립식 투자 (DCA)
- 섹터 분산: IT, 금융, 헬스케어 등 다양한 업종

초보자 추천 포트폴리오:
- 안정형: 채권 ETF 70% + 주식 ETF 30%
- 균형형: 채권 ETF 50% + 주식 ETF 50%
- 성장형: 채권 ETF 30% + 주식 ETF 70%""",
        "source": "분산투자 전략",
        "id": "diversification_1"
    },
    {
        "content": """초보 투자자를 위한 시작 가이드

투자 시작 전 체크리스트:
1. 비상금 확보: 최소 3~6개월 생활비
2. 부채 정리: 고금리 부채 먼저 상환
3. 투자 목표 설정: 기간, 목표 수익률 명확히

초보자 추천 투자 순서:
1단계: 예적금으로 종잣돈 마련
2단계: ETF로 분산투자 시작
3단계: 개별 주식 소액 투자
4단계: 해외 주식/펀드로 확장

피해야 할 실수:
- 빚내서 투자 (레버리지 투자)
- 한 종목에 올인
- 단기 수익에 집착
- 공포/탐욕에 휩쓸린 매매
- 손실 회복 심리로 추가 매수

월급쟁이 투자 팁:
- 급여일에 자동이체로 투자금 분리
- 매월 같은 금액 적립식 투자
- 연 1회 리밸런싱으로 비중 조절""",
        "source": "초보 투자자 가이드",
        "id": "beginner_guide_1"
    },
    {
        "content": """레버리지 ETF의 위험성

레버리지 ETF란?
기초지수 수익률의 2배, 3배를 추구하는 ETF입니다.
예: KOSPI200이 1% 오르면, 2배 레버리지는 2% 수익

왜 위험한가?

1. 복리 효과의 함정 (Volatility Decay)
- 지수가 10% 상승 후 10% 하락하면 원금 회복
- 2배 레버리지: 20% 상승 → 20% 하락 = -4% 손실
- 횡보장에서 자산이 지속 감소

2. 실제 사례
- 2020년 코로나 폭락 시 일부 레버리지 ETF 90% 이상 하락
- 장기 보유 시 기초지수 대비 수익률 괴리 발생

3. 적합한 투자자
- 단기 트레이딩 목적
- 방향성에 대한 강한 확신
- 손실 감내 능력 있는 투자자

결론: 초보자는 레버리지 ETF 피하세요.
장기투자에는 절대 부적합합니다.""",
        "source": "레버리지 ETF 위험성",
        "id": "leverage_warning_1"
    },
    {
        "content": """복리의 마법과 장기투자

복리란?
원금에 이자가 붙고, 그 이자에 다시 이자가 붙는 것

복리 계산 예시 (연 7% 수익률):
- 10년: 1000만원 → 1967만원 (약 2배)
- 20년: 1000만원 → 3870만원 (약 4배)
- 30년: 1000만원 → 7612만원 (약 7.6배)

72의 법칙:
72 ÷ 수익률 = 원금이 2배 되는 기간
예: 연 7% 수익 → 72÷7 = 약 10년

장기투자가 중요한 이유:
1. 복리 효과 극대화
2. 시장 변동성 상쇄
3. 거래 비용 절감
4. 세금 이연 효과

워렌 버핏의 조언:
"10년 이상 보유할 주식이 아니면 10분도 보유하지 마라"

실천 방법:
- 목표 기간 설정 (최소 5년 이상)
- 정기적 리밸런싱
- 시장 타이밍 포기
- 감정적 매매 금지""",
        "source": "장기투자 가이드",
        "id": "compound_interest_1"
    }
]


def load_sample_data(vectorstore: SimpleVectorStore):
    """샘플 데이터 로드"""
    if vectorstore.count() > 0:
        return

    documents = [d["content"] for d in SAMPLE_FINANCE_DATA]
    metadatas = [{"source": d["source"]} for d in SAMPLE_FINANCE_DATA]
    ids = [d["id"] for d in SAMPLE_FINANCE_DATA]

    vectorstore.add_documents(documents, metadatas, ids)


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.markdown("### 💰 Finance RAG")
    st.caption("금융 문서 기반 AI Q&A")

    st.divider()

    # 네비게이션
    menu = {
        "intro": "🏠 프로젝트 소개",
        "why": "🎯 왜 만들었나",
        "how": "⚙️ 어떻게 동작하나",
        "demo": "💬 Q&A 데모",
        "tech": "🔧 기술 상세"
    }

    for key, label in menu.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key
            st.rerun()

    st.divider()

    # API 키 설정
    st.markdown("### 🔑 설정")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="https://console.groq.com 에서 무료 발급"
    )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("✅ API 키 설정됨")

    # 상태
    st.divider()
    if st.session_state.vectorstore:
        doc_count = st.session_state.vectorstore.count()
        st.metric("📄 문서 수", doc_count)

    st.divider()
    st.caption("Made by 김다운")
    st.caption("[GitHub](https://github.com/araeLaver)")


# ============================================================
# 페이지: 프로젝트 소개
# ============================================================

def render_intro_page():
    st.markdown("""
    <div class="main-header">
        <h1>💰 Finance RAG API</h1>
        <p>금융 문서 기반 AI Q&A 시스템 - LLM 환각 방지</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>RAG</h3>
            <p>검색 증강 생성</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>5개</h3>
            <p>샘플 금융 문서</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Groq</h3>
            <p>Llama 3.1 LLM</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>실시간</h3>
            <p>스트리밍 응답</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📌 이 프로젝트는")

        st.markdown("""
        <div class="info-card">
        <b>금융 분야에 특화된 RAG(Retrieval-Augmented Generation) 시스템</b>입니다.

        사용자의 금융 관련 질문에 대해:
        1. 벡터 DB에서 관련 문서를 검색하고
        2. 검색된 문서를 기반으로 LLM이 답변을 생성합니다

        <b>일반 ChatGPT와의 차이점:</b>
        - 검증된 문서 기반 답변 (환각 방지)
        - 답변의 출처와 신뢰도 제공
        - 금융 도메인에 최적화된 프롬프트
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🛠️ 기술 스택")

        techs = ["FastAPI", "Python", "Groq API", "ChromaDB", "Streamlit", "Docker"]
        tags = " ".join([f'<span class="tech-tag">{t}</span>' for t in techs])
        st.markdown(tags, unsafe_allow_html=True)

        st.write("")
        st.markdown("""
        **왜 이 기술들을 선택했나?**
        - **Groq**: 무료 + 빠른 응답 (Llama 3.1)
        - **ChromaDB**: 경량 벡터 DB, 설치 간편
        - **FastAPI**: 비동기 처리, 자동 문서화
        """)

    st.divider()

    # 데모 시작 버튼
    st.subheader("🚀 바로 체험하기")

    if st.button("Q&A 데모 시작하기", type="primary", use_container_width=True):
        st.session_state.current_page = "demo"
        st.rerun()


# ============================================================
# 페이지: 왜 만들었나
# ============================================================

def render_why_page():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 왜 만들었나</h1>
        <p>문제 인식 → 해결 방안 → 기대 효과</p>
    </div>
    """, unsafe_allow_html=True)

    # 문제 인식
    st.subheader("❌ 문제: LLM의 환각(Hallucination)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
        <h4>환각이란?</h4>
        LLM이 <b>그럴듯하지만 사실이 아닌 정보</b>를 생성하는 현상

        <b>예시:</b>
        - "삼성전자 주가는 현재 8만원입니다" (실제와 다름)
        - "A 펀드의 수익률은 연 15%입니다" (허구의 정보)
        - 존재하지 않는 금융 상품 추천
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="highlight-box">
        <h4>⚠️ 금융 분야에서 특히 위험한 이유</h4>

        1. **실제 금전적 손실** 발생 가능
        2. **법적 책임** 문제 (투자 조언)
        3. **신뢰도 하락** (서비스 폐기)

        → 금융 AI는 **근거 있는 답변**이 필수
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 해결 방안
    st.subheader("✅ 해결: RAG (Retrieval-Augmented Generation)")

    st.markdown("""
    <div class="flow-box">
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     RAG가 환각을 방지하는 원리                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │   [기존 LLM]                        [RAG 적용 LLM]                   │
    │                                                                      │
    │   질문 ──→ LLM ──→ 답변             질문 ──→ 검색 ──→ LLM ──→ 답변   │
    │            │                                  │                      │
    │     (학습된 지식만)                    (검색된 문서 기반)              │
    │            │                                  │                      │
    │     환각 가능성 높음                   환각 가능성 낮음                │
    │                                              +                       │
    │                                         출처 명시 가능                │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="decision-card">
        <h4>1. Retrieval (검색)</h4>
        질문과 관련된 문서를 벡터 DB에서 검색

        - 의미 기반 유사도 검색
        - Top-K개 관련 문서 추출
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="decision-card">
        <h4>2. Augmentation (증강)</h4>
        검색된 문서로 프롬프트 구성

        - 컨텍스트 + 질문 결합
        - "이 문서만 참고하라" 지시
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="decision-card">
        <h4>3. Generation (생성)</h4>
        문서 기반으로만 답변 생성

        - 문서에 없으면 "없다"고 답변
        - 출처와 신뢰도 함께 제공
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 기대 효과
    st.subheader("📈 기대 효과")

    st.markdown("""
    | 측면 | 기존 LLM | RAG 적용 후 |
    |------|----------|-------------|
    | **정확성** | 환각 가능 | 문서 기반 검증 |
    | **신뢰성** | 출처 불명 | 출처 명시 |
    | **최신성** | 학습 데이터 한정 | 문서 업데이트 가능 |
    | **책임** | 불분명 | 근거 추적 가능 |
    """)


# ============================================================
# 페이지: 어떻게 동작하나
# ============================================================

def render_how_page():
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ 어떻게 동작하나</h1>
        <p>시스템 아키텍처와 데이터 흐름</p>
    </div>
    """, unsafe_allow_html=True)

    # 전체 아키텍처
    st.subheader("🏗️ 시스템 아키텍처")

    st.markdown("""
    <div class="flow-box">
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Finance RAG System                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │    ┌─────────────┐                                                       │
    │    │    User     │                                                       │
    │    └──────┬──────┘                                                       │
    │           │ "ETF가 뭔가요?"                                               │
    │           ▼                                                              │
    │    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
    │    │  Streamlit  │─────▶│   FastAPI   │─────▶│ RAG Service │            │
    │    │   (UI/UX)   │      │  (REST API) │      │ (Pipeline)  │            │
    │    └─────────────┘      └─────────────┘      └──────┬──────┘            │
    │                                                      │                   │
    │                         ┌────────────────────────────┼───────────┐       │
    │                         │                            │           │       │
    │                         ▼                            ▼           │       │
    │                  ┌─────────────┐              ┌─────────────┐    │       │
    │                  │  ChromaDB   │              │  Groq API   │    │       │
    │                  │ (Vector DB) │              │   (LLM)     │    │       │
    │                  └─────────────┘              └─────────────┘    │       │
    │                         │                            │           │       │
    │                         │     관련 문서 + 질문        │           │       │
    │                         └────────────────────────────┘           │       │
    │                                        │                          │       │
    │                                        ▼                          │       │
    │                              ┌─────────────────┐                 │       │
    │                              │  답변 + 출처    │◀────────────────┘       │
    │                              │  + 신뢰도      │                          │
    │                              └─────────────────┘                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 상세 흐름
    st.subheader("🔄 상세 처리 흐름")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1️⃣ 문서 등록 흐름")
        st.markdown("""
        ```
        PDF/텍스트 업로드
              │
              ▼
        ┌─────────────┐
        │  Text 추출  │ ← PyPDF, 인코딩 처리
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Chunking  │ ← 500자 단위, 100자 오버랩
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Embedding  │ ← 텍스트 → 벡터 변환
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  ChromaDB   │ ← 벡터 + 메타데이터 저장
        └─────────────┘
        ```
        """)

    with col2:
        st.markdown("#### 2️⃣ 질의 응답 흐름")
        st.markdown("""
        ```
        사용자 질문
              │
              ▼
        ┌─────────────┐
        │ Query 임베딩│ ← 질문 → 벡터 변환
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ 유사도 검색 │ ← Top-3 문서 추출
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ 프롬프트 구성│ ← System + Context + Q
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ LLM 생성    │ ← 스트리밍 응답
        └──────┬──────┘
               │
               ▼
        답변 + 출처 + 신뢰도
        ```
        """)

    st.divider()

    # 핵심 설계 결정
    st.subheader("💡 핵심 설계 결정")

    decisions = [
        {
            "title": "왜 ChromaDB인가?",
            "reason": "경량화, 설치 간편, Python 네이티브. 프로토타입에 최적. 프로덕션은 Pinecone/Weaviate 고려.",
        },
        {
            "title": "왜 청크 사이즈 500자?",
            "reason": "LLM 컨텍스트 제한 고려 + 의미 단위 유지. 너무 작으면 맥락 손실, 너무 크면 검색 정확도 저하.",
        },
        {
            "title": "왜 Top-3 검색?",
            "reason": "정확도와 속도 균형. 1개는 부족, 5개 이상은 노이즈 증가. 3개가 최적점.",
        },
        {
            "title": "왜 Groq API?",
            "reason": "무료 티어 제공, 빠른 응답 속도, Llama 3.1 지원. 클라우드 배포에 적합.",
        }
    ]

    for d in decisions:
        st.markdown(f"""
        <div class="decision-card">
        <b>{d['title']}</b><br>
        {d['reason']}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 페이지: Q&A 데모
# ============================================================

def render_demo_page():
    st.markdown("""
    <div class="main-header">
        <h1>💬 Q&A 데모</h1>
        <p>금융 문서 기반 AI 답변 체험</p>
    </div>
    """, unsafe_allow_html=True)

    # API 키 체크
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ 사이드바에서 Groq API Key를 입력해주세요. [무료 발급](https://console.groq.com)")
        return

    # VectorStore 초기화
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = SimpleVectorStore()
        load_sample_data(st.session_state.vectorstore)
        st.session_state.sample_loaded = True

    vectorstore = st.session_state.vectorstore
    llm = GroqLLM(api_key)

    # 설정
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("검색 문서 수", 1, 5, 3)
    with col2:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # 대화 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 참조 문서"):
                    for src in msg["sources"]:
                        score_pct = int(src["score"] * 100)
                        st.markdown(f"""
                        <div class="source-card">
                            <b>{src['source']}</b> (관련도: {score_pct}%)
                            <br><small>{src['preview']}</small>
                        </div>
                        """, unsafe_allow_html=True)

    # 예시 질문
    if not st.session_state.messages:
        st.markdown("#### 💡 예시 질문")
        examples = ["ETF가 뭔가요?", "분산투자 방법 알려주세요", "레버리지 ETF는 왜 위험한가요?", "복리의 마법이란?"]

        cols = st.columns(4)
        for i, q in enumerate(examples):
            with cols[i]:
                if st.button(q, key=f"ex_{i}"):
                    st.session_state.pending_q = q
                    st.rerun()

    # 질문 처리
    if "pending_q" in st.session_state:
        question = st.session_state.pending_q
        del st.session_state.pending_q
        process_qa(question, vectorstore, llm, top_k)

    if prompt := st.chat_input("금융에 대해 물어보세요..."):
        process_qa(prompt, vectorstore, llm, top_k)


def process_qa(question: str, vectorstore: SimpleVectorStore, llm: GroqLLM, top_k: int):
    """질문 처리"""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 검색
        with st.spinner("🔍 관련 문서 검색 중..."):
            results = vectorstore.search(question, top_k)

        if not results["documents"]:
            st.warning("관련 문서를 찾을 수 없습니다.")
            return

        # 컨텍스트 구성
        context_parts = []
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"], results["metadatas"], results["distances"]
        )):
            source_name = meta.get("source", f"문서 {i+1}")
            context_parts.append(f"[{source_name}]\n{doc}")
            score = max(0, min(1, 1 - dist / 2))
            sources.append({
                "source": source_name,
                "preview": doc[:100] + "..." if len(doc) > 100 else doc,
                "score": score
            })

        context = "\n\n".join(context_parts)

        # 스트리밍 응답
        response_placeholder = st.empty()
        full_response = ""

        for token in llm.generate_stream(context, question):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # 신뢰도
        avg_score = sum(s["score"] for s in sources) / len(sources)
        conf = "🟢 높음" if avg_score > 0.6 else "🟡 보통" if avg_score > 0.4 else "🔴 낮음"
        st.caption(f"신뢰도: {conf}")

        # 출처 표시
        with st.expander("📚 참조 문서", expanded=True):
            for src in sources:
                score_pct = int(src["score"] * 100)
                st.markdown(f"""
                <div class="source-card">
                    <b>{src['source']}</b>
                    <div style="background:#ddd;border-radius:10px;height:8px;margin:5px 0;">
                        <div style="background:linear-gradient(90deg,#667eea,#764ba2);width:{score_pct}%;height:100%;border-radius:10px;"></div>
                    </div>
                    <small>관련도: {score_pct}%</small>
                </div>
                """, unsafe_allow_html=True)

        # 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })


# ============================================================
# 페이지: 기술 상세
# ============================================================

def render_tech_page():
    st.markdown("""
    <div class="main-header">
        <h1>🔧 기술 상세</h1>
        <p>구현 세부사항 및 코드</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 프로젝트 구조", "🔑 핵심 코드", "🧪 테스트"])

    with tab1:
        st.code("""
finance-rag-api/
├── src/
│   ├── api/                    # API Layer
│   │   ├── routes.py           # REST 엔드포인트
│   │   ├── schemas.py          # Pydantic 모델
│   │   └── security.py         # 인증/Rate Limit
│   │
│   ├── rag/                    # RAG Core
│   │   ├── rag_service.py      # RAG 파이프라인
│   │   ├── vectorstore.py      # ChromaDB 래퍼
│   │   ├── llm_provider.py     # LLM 추상화 (Groq/Ollama)
│   │   └── document_loader.py  # 문서 파싱/청킹
│   │
│   └── core/                   # 공통
│       ├── config.py           # 환경설정
│       └── exceptions.py       # 커스텀 예외
│
├── app/
│   └── streamlit_app.py        # 웹 데모
│
├── tests/                      # 35개 테스트
├── Dockerfile
└── docker-compose.yml
        """, language="text")

    with tab2:
        st.markdown("#### 환각 방지 프롬프트")
        st.code('''
SYSTEM_PROMPT = """당신은 금융 전문 상담 AI입니다.

규칙:
1. 문서에 없는 내용은 "해당 정보가 제공된 문서에 없습니다"라고 답하세요
2. 추측하거나 지어내지 마세요
3. 숫자나 수치는 문서 그대로 인용하세요

주의: 이 정보는 투자 권유가 아닙니다."""
        ''', language="python")

        st.markdown("#### 벡터 검색")
        st.code('''
def search(self, query: str, top_k: int = 3):
    results = self.collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 거리 → 유사도 변환 (0~1)
    relevance = 1 - distance / 2
    return results
        ''', language="python")

        st.markdown("#### LLM Provider 추상화")
        st.code('''
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_stream(self, system_prompt, user_prompt):
        pass

class GroqProvider(BaseLLMProvider):
    def generate_stream(self, system_prompt, user_prompt):
        stream = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[...],
            stream=True
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content
        ''', language="python")

    with tab3:
        st.markdown("#### 테스트 현황")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 테스트", "35개")
        with col2:
            st.metric("통과율", "100%")
        with col3:
            st.metric("커버리지", "~85%")

        st.markdown("""
        | 파일 | 테스트 수 | 범위 |
        |------|----------|------|
        | test_api.py | 11개 | REST 엔드포인트 |
        | test_document_loader.py | 16개 | 청킹, PDF 파싱 |
        | test_vectorstore.py | 8개 | 검색, 필터링 |
        """)


# ============================================================
# 라우팅
# ============================================================

page = st.session_state.current_page

if page == "intro":
    render_intro_page()
elif page == "why":
    render_why_page()
elif page == "how":
    render_how_page()
elif page == "demo":
    render_demo_page()
elif page == "tech":
    render_tech_page()
else:
    render_intro_page()

# 푸터
st.divider()
st.markdown("""
<div style="text-align:center;color:#888;padding:1rem;">
    💰 Finance RAG | FastAPI + Groq + ChromaDB + Streamlit<br>
    <small>© 2024 김다운 - AI/ML 포트폴리오</small>
</div>
""", unsafe_allow_html=True)
