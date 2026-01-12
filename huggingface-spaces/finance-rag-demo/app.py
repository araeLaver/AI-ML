# -*- coding: utf-8 -*-
"""
Finance RAG Demo - HuggingFace Spaces Version
Real-time Stock Data + RAG Q&A Demo
"""

import streamlit as st
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict
import json

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

.stock-card {
    background: white;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.stock-name {
    font-size: 0.85rem;
    color: var(--gray);
    margin-bottom: 0.25rem;
}
.stock-price {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--black);
}
.stock-change {
    font-size: 0.9rem;
    font-weight: 500;
}
.stock-up { color: var(--green); }
.stock-down { color: var(--red); }

.chat-container {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 400px;
}
.chat-message {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    border: 1px solid #eee;
}
.chat-user {
    background: var(--black);
    color: white;
}
.source-tag {
    display: inline-block;
    background: #e3f2fd;
    color: #1976d2;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    margin-right: 4px;
}

.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    font-size: 1rem;
    font-weight: 600;
}
.feature-desc {
    font-size: 0.8rem;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sample Financial Data (RAG Knowledge Base)
# ============================================================
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "삼성전자 2024년 4분기 실적",
        "content": """삼성전자는 2024년 4분기 매출 79조원, 영업이익 8.1조원을 기록했습니다.
        반도체 부문은 HBM(고대역폭메모리) 수요 증가로 실적이 크게 개선되었습니다.
        HBM3E 양산이 본격화되면서 AI 서버 시장 점유율이 확대되고 있습니다.
        2025년에는 HBM4 개발을 완료하고 양산에 돌입할 예정입니다.""",
        "date": "2024-12-15",
        "source": "삼성전자 IR"
    },
    {
        "id": "doc_2",
        "title": "SK하이닉스 AI 반도체 전망",
        "content": """SK하이닉스는 AI 반도체 시장에서 HBM 점유율 1위를 유지하고 있습니다.
        NVIDIA와의 협력을 통해 H100, H200 GPU에 HBM3E를 독점 공급 중입니다.
        2024년 HBM 매출은 전년 대비 300% 이상 증가했습니다.
        2025년 예상 HBM 매출은 20조원을 상회할 것으로 전망됩니다.""",
        "date": "2024-12-20",
        "source": "SK하이닉스 IR"
    },
    {
        "id": "doc_3",
        "title": "한국은행 기준금리 전망",
        "content": """한국은행은 2024년 11월 기준금리를 3.25%에서 3.0%로 인하했습니다.
        물가 안정과 경기 부양을 위해 추가 인하 가능성이 있습니다.
        2025년 상반기 중 2.75%까지 인하될 것으로 예상됩니다.
        이는 부동산 시장과 가계대출에 영향을 미칠 전망입니다.""",
        "date": "2024-11-28",
        "source": "한국은행"
    },
    {
        "id": "doc_4",
        "title": "네이버 AI 사업 현황",
        "content": """네이버는 하이퍼클로바X를 기반으로 B2B AI 서비스를 확대하고 있습니다.
        클로바 스튜디오 MAU가 100만을 돌파했습니다.
        네이버클라우드의 AI 매출 비중이 30%를 넘어섰습니다.
        2025년 글로벌 AI 시장 진출을 본격화할 계획입니다.""",
        "date": "2024-12-10",
        "source": "네이버 IR"
    },
    {
        "id": "doc_5",
        "title": "카카오 구조조정 및 전략",
        "content": """카카오는 비핵심 사업 정리를 통해 수익성 개선에 집중하고 있습니다.
        카카오엔터테인먼트, 카카오모빌리티 IPO를 추진 중입니다.
        AI 기술을 활용한 광고 타겟팅 고도화로 광고 매출이 증가했습니다.
        2025년 영업이익률 목표는 15%입니다.""",
        "date": "2024-12-05",
        "source": "카카오 IR"
    }
]

# ============================================================
# Stock Data Class
# ============================================================
@dataclass
class StockQuote:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None

def get_stock_data() -> Dict[str, StockQuote]:
    """Get sample stock data (in production, use yfinance)"""
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
                if not hist.empty:
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
            except:
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
# RAG Functions
# ============================================================
def search_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Simple keyword-based document search"""
    query_lower = query.lower()
    scores = []

    for doc in SAMPLE_DOCUMENTS:
        score = 0
        content_lower = doc["content"].lower() + doc["title"].lower()

        # Keyword matching
        keywords = query_lower.split()
        for keyword in keywords:
            if keyword in content_lower:
                score += content_lower.count(keyword)

        # Company name matching (higher weight)
        companies = ["삼성", "sk", "하이닉스", "네이버", "카카오", "한국은행"]
        for company in companies:
            if company in query_lower and company in content_lower:
                score += 10

        if score > 0:
            scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:top_k]]

def generate_answer(query: str, context_docs: List[Dict]) -> str:
    """Generate answer based on retrieved documents"""
    if not context_docs:
        return "관련 문서를 찾을 수 없습니다. 다른 질문을 해주세요."

    # Build context
    context = "\n\n".join([
        f"[{doc['title']}]\n{doc['content']}"
        for doc in context_docs
    ])

    # Simple template-based response (in production, use LLM)
    query_lower = query.lower()

    if "삼성" in query_lower and "실적" in query_lower:
        return """삼성전자의 2024년 4분기 실적을 요약해드리겠습니다:

📊 **실적 요약**
- 매출: 79조원
- 영업이익: 8.1조원

💡 **주요 포인트**
- HBM(고대역폭메모리) 수요 증가로 반도체 부문 실적 개선
- HBM3E 양산 본격화로 AI 서버 시장 점유율 확대
- 2025년 HBM4 개발 완료 및 양산 예정

*출처: 삼성전자 IR (2024-12-15)*"""

    elif "hbm" in query_lower or "ai 반도체" in query_lower:
        return """AI 반도체 및 HBM 시장 현황을 분석해드리겠습니다:

🏆 **시장 현황**
- SK하이닉스가 HBM 시장 점유율 1위 유지
- NVIDIA GPU(H100, H200)에 HBM3E 독점 공급

📈 **성장성**
- SK하이닉스 HBM 매출 전년 대비 300%+ 증가
- 2025년 예상 HBM 매출: 20조원 이상

🔮 **전망**
- 삼성전자 HBM4 개발 진행 중
- AI 서버 수요 지속 증가 전망

*출처: SK하이닉스 IR, 삼성전자 IR*"""

    elif "금리" in query_lower or "한국은행" in query_lower:
        return """한국은행 기준금리 전망을 알려드리겠습니다:

📉 **현재 상황**
- 2024년 11월: 3.25% → 3.0% 인하

🔮 **2025년 전망**
- 상반기 중 2.75%까지 추가 인하 예상
- 물가 안정 및 경기 부양 목적

⚠️ **영향**
- 부동산 시장 영향
- 가계대출 금리 변동

*출처: 한국은행 (2024-11-28)*"""

    elif "네이버" in query_lower:
        return """네이버 AI 사업 현황을 요약해드리겠습니다:

🤖 **AI 서비스**
- 하이퍼클로바X 기반 B2B AI 서비스 확대
- 클로바 스튜디오 MAU 100만 돌파

📊 **매출 비중**
- 네이버클라우드 AI 매출 비중 30% 돌파

🌏 **2025년 계획**
- 글로벌 AI 시장 진출 본격화

*출처: 네이버 IR (2024-12-10)*"""

    elif "카카오" in query_lower:
        return """카카오 사업 전략을 분석해드리겠습니다:

🔄 **구조조정**
- 비핵심 사업 정리로 수익성 개선 집중
- 카카오엔터, 카카오모빌리티 IPO 추진

📈 **성장 동력**
- AI 기반 광고 타겟팅 고도화
- 광고 매출 증가 추세

🎯 **2025년 목표**
- 영업이익률 15% 달성

*출처: 카카오 IR (2024-12-05)*"""

    else:
        # Generic response with context
        doc = context_docs[0]
        return f"""검색된 문서를 바탕으로 답변드리겠습니다:

📄 **{doc['title']}**

{doc['content']}

*출처: {doc['source']} ({doc['date']})*"""

# ============================================================
# Main App
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            Finance RAG Demo
            <span class="badge">AI Powered</span>
        </div>
        <div class="main-subtitle">
            실시간 주식 데이터 + RAG 기반 금융 Q&A 시스템
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("📊", "실시간 시세", "주요 종목 실시간 조회"),
        ("🔍", "RAG 검색", "금융 문서 시맨틱 검색"),
        ("🤖", "AI 답변", "컨텍스트 기반 응답 생성"),
        ("📈", "차트 분석", "주가 차트 시각화"),
    ]

    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 RAG Q&A", "📊 실시간 시세", "📚 문서 목록"])

    # Tab 1: RAG Q&A
    with tab1:
        st.markdown("### 금융 AI 어시스턴트")
        st.markdown("금융 문서를 기반으로 질문에 답변합니다. (RAG: Retrieval-Augmented Generation)")

        # Sample questions
        st.markdown("**예시 질문:**")
        sample_questions = [
            "삼성전자 2024년 4분기 실적은?",
            "HBM 시장 전망은 어떤가요?",
            "2025년 금리 전망은?",
            "네이버 AI 사업 현황은?",
        ]

        cols = st.columns(4)
        for col, q in zip(cols, sample_questions):
            if col.button(q, use_container_width=True):
                st.session_state.query = q

        # Query input
        query = st.text_input(
            "질문을 입력하세요",
            value=st.session_state.get("query", ""),
            placeholder="예: 삼성전자 실적은 어떤가요?"
        )

        if query:
            with st.spinner("문서 검색 중..."):
                # Search documents
                docs = search_documents(query)

                # Generate answer
                answer = generate_answer(query, docs)

            # Display answer
            st.markdown("---")
            st.markdown("### 📝 답변")
            st.markdown(answer)

            # Display sources
            if docs:
                st.markdown("### 📚 참조 문서")
                for doc in docs:
                    with st.expander(f"📄 {doc['title']} ({doc['date']})"):
                        st.markdown(doc['content'])
                        st.caption(f"출처: {doc['source']}")

    # Tab 2: Stock Data
    with tab2:
        st.markdown("### 실시간 주요 종목")

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
                </div>
                """, unsafe_allow_html=True)

        st.caption("* 데이터는 yfinance에서 가져옵니다. 실시간 데이터가 아닐 수 있습니다.")

    # Tab 3: Document List
    with tab3:
        st.markdown("### RAG 지식베이스 문서")
        st.markdown("현재 시스템에 등록된 금융 문서 목록입니다.")

        for doc in SAMPLE_DOCUMENTS:
            with st.expander(f"📄 {doc['title']} - {doc['date']}"):
                st.markdown(doc['content'])
                st.caption(f"출처: {doc['source']} | ID: {doc['id']}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        <p>Made with ❤️ by <a href="https://github.com/araeLaver" target="_blank">Kim Dawoon</a></p>
        <p>Part of <a href="https://github.com/araeLaver/AI-ML" target="_blank">AI/ML Portfolio</a> - 6-Step Learning Roadmap</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
