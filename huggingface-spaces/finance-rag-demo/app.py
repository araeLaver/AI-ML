# -*- coding: utf-8 -*-
"""
Finance RAG Demo - HuggingFace Spaces Version
Real-time Stock Data + RAG Q&A with HuggingFace Inference API
"""

import streamlit as st
import os
import requests
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json
import time

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Finance RAG Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
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
}

* { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: var(--white); }

header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.main-header {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1px solid #eee;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--black);
    margin-bottom: 0.5rem;
}
.main-subtitle {
    font-size: 1rem;
    color: var(--gray);
}
.badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-left: 10px;
}
.badge-free {
    background: var(--green);
}

.stock-card {
    background: white;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.stock-name { font-size: 0.85rem; color: var(--gray); margin-bottom: 0.25rem; }
.stock-price { font-size: 1.75rem; font-weight: 700; color: var(--black); }
.stock-change { font-size: 0.9rem; font-weight: 500; }
.stock-up { color: var(--green); }
.stock-down { color: var(--red); }

.source-card {
    background: #f8f9fa;
    border-left: 3px solid var(--accent);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
}
.source-title { font-weight: 600; margin-bottom: 0.5rem; }
.source-score { color: var(--green); font-size: 0.8rem; }

.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-title { font-size: 1rem; font-weight: 600; }
.feature-desc { font-size: 0.8rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HuggingFace Inference API
# ============================================================
HF_API_URL = "https://api-inference.huggingface.co/models/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def get_hf_token():
    """Get HuggingFace token from secrets or environment"""
    # Try secrets first (HuggingFace Spaces)
    try:
        return st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    except:
        return os.environ.get("HF_TOKEN", "")

def get_embeddings(texts: List[str], token: str = "") -> List[List[float]]:
    """Get embeddings using HuggingFace Inference API"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = requests.post(
        HF_API_URL + EMBEDDING_MODEL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=30
    )

    if response.status_code == 200:
        return response.json()
    else:
        # Fallback: simple TF-IDF like embedding
        return simple_embeddings(texts)

def simple_embeddings(texts: List[str]) -> List[List[float]]:
    """Simple fallback embeddings using character n-grams"""
    def text_to_vec(text: str, dim: int = 384) -> List[float]:
        text = text.lower()
        vec = [0.0] * dim
        for i, char in enumerate(text):
            idx = (ord(char) * (i + 1)) % dim
            vec[idx] += 1.0
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
    return [text_to_vec(t) for t in texts]

def generate_response(prompt: str, token: str = "", max_tokens: int = 500) -> str:
    """Generate response using HuggingFace Inference API"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Format for Mistral Instruct
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    try:
        response = requests.post(
            HF_API_URL + LLM_MODEL,
            headers=headers,
            json={
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {"wait_for_model": True}
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()

        # If API fails, use template response
        return None
    except Exception as e:
        return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8))

# ============================================================
# Financial Documents (Knowledge Base)
# ============================================================
FINANCIAL_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "삼성전자 2024년 4분기 실적",
        "content": """삼성전자는 2024년 4분기 매출 79조원, 영업이익 8.1조원을 기록했습니다.
반도체 부문은 HBM(고대역폭메모리) 수요 증가로 실적이 크게 개선되었습니다.
HBM3E 양산이 본격화되면서 AI 서버 시장 점유율이 확대되고 있습니다.
2025년에는 HBM4 개발을 완료하고 양산에 돌입할 예정입니다.
메모리 반도체 가격 상승과 함께 수익성이 크게 개선될 전망입니다.""",
        "date": "2024-12-15",
        "source": "삼성전자 IR",
        "category": "실적"
    },
    {
        "id": "doc_2",
        "title": "SK하이닉스 AI 반도체 전망",
        "content": """SK하이닉스는 AI 반도체 시장에서 HBM 점유율 1위를 유지하고 있습니다.
NVIDIA와의 협력을 통해 H100, H200 GPU에 HBM3E를 독점 공급 중입니다.
2024년 HBM 매출은 전년 대비 300% 이상 증가했습니다.
2025년 예상 HBM 매출은 20조원을 상회할 것으로 전망됩니다.
AI 데이터센터 투자 확대로 고대역폭 메모리 수요가 폭발적으로 증가하고 있습니다.""",
        "date": "2024-12-20",
        "source": "SK하이닉스 IR",
        "category": "AI반도체"
    },
    {
        "id": "doc_3",
        "title": "한국은행 기준금리 전망 2025",
        "content": """한국은행은 2024년 11월 기준금리를 3.25%에서 3.0%로 인하했습니다.
물가 안정세가 지속되면서 추가 인하 가능성이 높아지고 있습니다.
2025년 상반기 중 2.75%까지 인하될 것으로 시장은 예상하고 있습니다.
금리 인하는 부동산 시장 회복과 가계 이자 부담 경감에 도움이 될 전망입니다.
다만 환율 변동성과 미국 금리 정책에 따라 속도 조절 가능성도 있습니다.""",
        "date": "2024-11-28",
        "source": "한국은행",
        "category": "금리"
    },
    {
        "id": "doc_4",
        "title": "네이버 AI 사업 현황 및 전략",
        "content": """네이버는 하이퍼클로바X를 기반으로 B2B AI 서비스를 확대하고 있습니다.
클로바 스튜디오 MAU가 100만을 돌파하며 국내 1위 AI 플랫폼으로 자리잡았습니다.
네이버클라우드의 AI 매출 비중이 30%를 넘어섰습니다.
2025년에는 일본, 동남아 시장으로 AI 서비스 진출을 본격화할 계획입니다.
검색, 커머스, 콘텐츠 전 영역에 AI를 적용하여 사용자 경험을 혁신하고 있습니다.""",
        "date": "2024-12-10",
        "source": "네이버 IR",
        "category": "AI사업"
    },
    {
        "id": "doc_5",
        "title": "카카오 구조조정 및 2025 전략",
        "content": """카카오는 비핵심 사업 정리를 통해 수익성 개선에 집중하고 있습니다.
카카오엔터테인먼트와 카카오모빌리티의 IPO를 2025년 추진할 예정입니다.
AI 기술을 활용한 광고 타겟팅 고도화로 광고 매출이 15% 증가했습니다.
2025년 영업이익률 목표는 15%로 설정했습니다.
카카오톡 비즈니스 메시지와 선물하기 서비스가 핵심 수익원으로 성장 중입니다.""",
        "date": "2024-12-05",
        "source": "카카오 IR",
        "category": "전략"
    },
    {
        "id": "doc_6",
        "title": "2025년 글로벌 AI 시장 전망",
        "content": """2025년 글로벌 AI 시장 규모는 5,000억 달러를 넘어설 전망입니다.
생성형 AI가 시장 성장을 주도하며, 기업용 AI 솔루션 수요가 급증하고 있습니다.
주요 성장 분야는 AI 반도체, 클라우드 AI, 엔터프라이즈 AI입니다.
Microsoft, Google, Amazon이 AI 인프라 투자를 대폭 확대하고 있습니다.
한국 기업들은 AI 반도체와 AI 서비스 분야에서 글로벌 경쟁력을 확보하고 있습니다.""",
        "date": "2024-12-18",
        "source": "글로벌 리서치",
        "category": "시장전망"
    },
    {
        "id": "doc_7",
        "title": "테슬라 FSD 및 로보택시 전망",
        "content": """테슬라는 2025년 완전자율주행(FSD) 상용화를 목표로 하고 있습니다.
로보택시 서비스 'Cybercab'을 2025년 하반기 출시 예정입니다.
FSD 구독 매출이 분기당 10억 달러를 돌파했습니다.
자율주행 데이터 축적량이 경쟁사 대비 10배 이상 많습니다.
AI 기반 자율주행 기술이 테슬라의 핵심 가치 동력이 되고 있습니다.""",
        "date": "2024-12-12",
        "source": "테슬라 IR",
        "category": "자율주행"
    },
    {
        "id": "doc_8",
        "title": "비트코인 및 암호화폐 2025 전망",
        "content": """비트코인이 2024년 말 10만 달러를 돌파하며 사상 최고가를 기록했습니다.
비트코인 현물 ETF 승인 이후 기관 투자자 유입이 크게 증가했습니다.
2025년에는 15만 달러까지 상승할 것이라는 전망이 우세합니다.
이더리움 ETF도 승인되면서 암호화폐 시장 전반이 활성화되고 있습니다.
다만 규제 불확실성과 변동성 리스크는 여전히 존재합니다.""",
        "date": "2024-12-22",
        "source": "암호화폐 리서치",
        "category": "암호화폐"
    }
]

# ============================================================
# Vector Store (In-Memory)
# ============================================================
@st.cache_resource
def build_vector_store():
    """Build vector store with document embeddings"""
    texts = [f"{doc['title']} {doc['content']}" for doc in FINANCIAL_DOCUMENTS]
    token = get_hf_token()
    embeddings = get_embeddings(texts, token)
    return embeddings

def search_documents(query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
    """Search documents using vector similarity"""
    # Get query embedding
    token = get_hf_token()
    query_embedding = get_embeddings([query], token)[0]

    # Get document embeddings (cached)
    doc_embeddings = build_vector_store()

    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((FINANCIAL_DOCUMENTS[i], sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# ============================================================
# RAG Pipeline
# ============================================================
def rag_query(question: str) -> Tuple[str, List[Tuple[Dict, float]]]:
    """RAG: Retrieve documents and generate answer"""

    # 1. Retrieve relevant documents
    retrieved = search_documents(question, top_k=3)

    # 2. Build context
    context = "\n\n".join([
        f"[{doc['title']}] ({doc['source']}, {doc['date']})\n{doc['content']}"
        for doc, score in retrieved
    ])

    # 3. Generate answer
    prompt = f"""다음 금융 문서들을 참고하여 질문에 답변해주세요.

### 참고 문서:
{context}

### 질문:
{question}

### 답변:
위 문서들을 바탕으로 정확하고 구체적으로 답변해주세요. 문서에 없는 내용은 추측하지 마세요."""

    token = get_hf_token()
    answer = generate_response(prompt, token)

    # Fallback if API fails
    if not answer:
        answer = generate_template_answer(question, retrieved)

    return answer, retrieved

def generate_template_answer(question: str, retrieved: List[Tuple[Dict, float]]) -> str:
    """Generate template-based answer as fallback"""
    if not retrieved:
        return "관련 문서를 찾을 수 없습니다."

    top_doc, score = retrieved[0]

    answer = f"""**{top_doc['title']}**에서 관련 정보를 찾았습니다.

{top_doc['content']}

---
📊 **관련도**: {score:.1%}
📅 **날짜**: {top_doc['date']}
📌 **출처**: {top_doc['source']}"""

    return answer

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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data() -> Dict[str, StockQuote]:
    """Get real-time stock data using yfinance"""
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
            except Exception as e:
                pass
        if result:
            return result
    except ImportError:
        pass

    # Fallback sample data
    return {
        "삼성전자": StockQuote("005930.KS", "삼성전자", 71500, 1200, 1.71, 12500000),
        "SK하이닉스": StockQuote("000660.KS", "SK하이닉스", 178000, 3500, 2.01, 3200000),
        "NAVER": StockQuote("035420.KS", "NAVER", 185000, -2000, -1.07, 580000),
        "카카오": StockQuote("035720.KS", "카카오", 42000, -500, -1.18, 2100000),
    }

# ============================================================
# Main App
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            Finance RAG Pro
            <span class="badge">AI Powered</span>
            <span class="badge badge-free">Free</span>
        </div>
        <div class="main-subtitle">
            HuggingFace 무료 모델 기반 금융 RAG Q&A 시스템
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("🔍", "벡터 검색", "임베딩 기반 시맨틱 검색"),
        ("🤖", "LLM 답변", "Mistral-7B 모델 사용"),
        ("📊", "실시간 시세", "yfinance 연동"),
        ("💰", "완전 무료", "HuggingFace API"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 RAG Q&A", "📊 실시간 시세", "📚 문서 목록"])

    # ============ TAB 1: RAG Q&A ============
    with tab1:
        st.markdown("### 🤖 금융 AI 어시스턴트")
        st.markdown("금융 문서를 검색하고 AI가 답변을 생성합니다.")

        # Sample questions
        st.markdown("**📝 예시 질문:**")
        sample_qs = [
            "삼성전자 4분기 실적은?",
            "HBM 시장 전망은?",
            "2025년 금리 전망",
            "네이버 AI 사업 현황",
        ]

        cols = st.columns(4)
        selected_q = None
        for col, q in zip(cols, sample_qs):
            if col.button(q, use_container_width=True):
                selected_q = q

        # Query input
        query = st.text_input(
            "질문을 입력하세요",
            value=selected_q if selected_q else "",
            placeholder="예: 삼성전자 실적은 어떤가요?"
        )

        if query:
            with st.spinner("🔍 문서 검색 및 답변 생성 중..."):
                start_time = time.time()
                answer, retrieved = rag_query(query)
                elapsed = time.time() - start_time

            # Answer
            st.markdown("---")
            st.markdown("### 📝 AI 답변")
            st.markdown(answer)
            st.caption(f"⏱️ 응답 시간: {elapsed:.2f}초")

            # Sources
            st.markdown("### 📚 참조 문서")
            for doc, score in retrieved:
                with st.expander(f"📄 {doc['title']} (관련도: {score:.1%})"):
                    st.markdown(f"**카테고리**: {doc['category']}")
                    st.markdown(doc['content'])
                    st.caption(f"출처: {doc['source']} | 날짜: {doc['date']}")

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
        st.markdown(f"총 **{len(FINANCIAL_DOCUMENTS)}개** 문서가 등록되어 있습니다.")

        # Category filter
        categories = list(set(doc['category'] for doc in FINANCIAL_DOCUMENTS))
        selected_cat = st.selectbox("카테고리 필터", ["전체"] + categories)

        for doc in FINANCIAL_DOCUMENTS:
            if selected_cat != "전체" and doc['category'] != selected_cat:
                continue
            with st.expander(f"📄 {doc['title']} [{doc['category']}]"):
                st.markdown(doc['content'])
                st.caption(f"출처: {doc['source']} | 날짜: {doc['date']} | ID: {doc['id']}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem;">
        <p><strong>🛠️ 기술 스택</strong>: Streamlit + HuggingFace Inference API + yfinance</p>
        <p><strong>🤖 모델</strong>: Mistral-7B (LLM) + all-MiniLM-L6-v2 (Embeddings)</p>
        <p>Made with ❤️ by <a href="https://github.com/araeLaver" target="_blank">Kim Dawoon</a> |
        <a href="https://github.com/araeLaver/AI-ML" target="_blank">Full Project on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
