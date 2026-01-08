# Streamlit App: Financial Analysis with ML + LLM
"""
Step 1+2 통합: 금융 데이터 분석 + LLM API 시스템
- ML 기반 이상거래 탐지
- LLM 기반 분석 설명
- 프롬프트 엔지니어링 비교
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import generate_transaction_data
from src.data.preprocessor import TransactionPreprocessor
from src.ml.fraud_detector import FraudDetector
from src.llm.client import create_llm_client
from src.prompts.templates import (
    SYSTEM_PROMPT,
    get_fraud_analysis_prompt,
)
from src.prompts.tools import FINANCIAL_TOOLS, ToolExecutor

# 페이지 설정
st.set_page_config(
    page_title="Financial Analysis - ML + LLM",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .risk-critical { color: #dc2626; font-weight: bold; }
    .risk-high { color: #ea580c; font-weight: bold; }
    .risk-medium { color: #ca8a04; font-weight: bold; }
    .risk-low { color: #16a34a; font-weight: bold; }
    .code-block {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """세션 상태 초기화"""
    if "data" not in st.session_state:
        st.session_state.data = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("### 설정")

        # LLM 제공자 선택
        llm_provider = st.selectbox(
            "LLM Provider",
            ["mock", "openai", "claude"],
            help="API 키 없이 테스트하려면 mock 선택"
        )

        if llm_provider != "mock":
            st.info(f"{llm_provider.upper()} API 키가 .env에 설정되어 있어야 합니다")

        # ML 모델 선택
        ml_algorithm = st.selectbox(
            "ML Algorithm",
            ["random_forest", "gradient_boosting", "isolation_forest"],
        )

        st.markdown("---")
        st.markdown("### 데이터 생성")

        n_samples = st.slider("샘플 수", 100, 5000, 1000, 100)
        fraud_ratio = st.slider("이상거래 비율", 0.01, 0.3, 0.1, 0.01)

        if st.button("데이터 생성 & 모델 학습", type="primary"):
            with st.spinner("데이터 생성 및 모델 학습 중..."):
                generate_and_train(n_samples, fraud_ratio, ml_algorithm, llm_provider)
            st.success("완료!")

        st.markdown("---")
        st.markdown("""
        ### Step 1+2 학습 내용

        **Step 1: Python + AI 기초**
        - NumPy/Pandas 데이터 처리
        - scikit-learn ML 모델

        **Step 2: LLM API**
        - OpenAI/Claude API
        - 프롬프트 엔지니어링
        - Function Calling
        """)

        return llm_provider, ml_algorithm


def generate_and_train(n_samples, fraud_ratio, algorithm, llm_provider):
    """데이터 생성 및 모델 학습"""
    # 데이터 생성
    data = generate_transaction_data(n_samples, fraud_ratio)
    st.session_state.data = data

    # 전처리
    preprocessor = TransactionPreprocessor()
    X, y, feature_names = preprocessor.prepare_ml_data(data)
    st.session_state.preprocessor = preprocessor

    # 모델 학습
    model = FraudDetector(algorithm=algorithm)
    model.fit(X, y, feature_names)
    st.session_state.model = model

    # 예측
    predictions = model.predict(X)
    st.session_state.predictions = predictions

    # LLM 클라이언트
    st.session_state.llm_client = create_llm_client(llm_provider)


def render_overview_tab():
    """개요 탭"""
    st.markdown('<p class="main-header">금융 이상거래 탐지 시스템</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML 모델 + LLM 분석 통합 데모</p>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.info("왼쪽 사이드바에서 '데이터 생성 & 모델 학습' 버튼을 클릭하세요.")

        # 시스템 아키텍처 설명
        st.markdown("### 시스템 구성")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Step 1: ML 파이프라인
            ```
            거래 데이터
                ↓
            전처리 (Pandas/NumPy)
                ↓
            특성 엔지니어링
                ↓
            ML 모델 (scikit-learn)
                ↓
            이상거래 예측
            ```
            """)

        with col2:
            st.markdown("""
            #### Step 2: LLM 파이프라인
            ```
            ML 예측 결과
                ↓
            프롬프트 생성
                ↓
            LLM API 호출
                ↓
            자연어 분석 설명
                ↓
            Function Calling
            ```
            """)
        return

    data = st.session_state.data
    predictions = st.session_state.predictions

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 거래 수", f"{len(data):,}")

    with col2:
        fraud_count = data["is_fraud"].sum()
        st.metric("이상거래 수", f"{fraud_count:,}")

    with col3:
        fraud_rate = fraud_count / len(data) * 100
        st.metric("이상거래 비율", f"{fraud_rate:.1f}%")

    with col4:
        if predictions is not None:
            detected = sum(predictions)
            st.metric("탐지된 이상거래", f"{detected:,}")

    st.markdown("---")

    # 데이터 미리보기
    st.markdown("### 데이터 미리보기")
    st.dataframe(
        data.head(10).style.apply(
            lambda x: ["background-color: #fee2e2" if v else "" for v in data.loc[x.index, "is_fraud"]],
            axis=1
        ),
        use_container_width=True,
    )


def render_ml_tab():
    """ML 분석 탭"""
    st.markdown("### ML 이상거래 탐지")

    if st.session_state.model is None:
        st.warning("먼저 데이터를 생성하고 모델을 학습하세요.")
        return

    model = st.session_state.model
    data = st.session_state.data

    # 모델 성능
    st.markdown("#### 모델 성능 평가")

    X, y, _ = st.session_state.preprocessor.prepare_ml_data(data)
    metrics = model.evaluate(X, y)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.markdown("---")

    # 개별 거래 분석
    st.markdown("#### 개별 거래 분석")

    transaction_idx = st.selectbox(
        "거래 선택",
        range(min(100, len(data))),
        format_func=lambda x: f"거래 {x}: {data.iloc[x]['transaction_id']} - {data.iloc[x]['amount']:,.0f}원"
    )

    if transaction_idx is not None:
        transaction = data.iloc[transaction_idx].to_dict()
        X_single = X[transaction_idx:transaction_idx+1]

        result = model.predict_single(X_single, transaction)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**거래 정보**")
            st.json({
                "거래ID": transaction["transaction_id"],
                "금액": f"{transaction['amount']:,.0f}원",
                "시간": str(transaction.get("timestamp", "N/A")),
                "카테고리": transaction.get("merchant_category", "N/A"),
                "해외거래": "예" if transaction.get("is_international") else "아니오",
            })

        with col2:
            st.markdown("**ML 예측 결과**")

            risk_class = "risk-" + result.risk_level.lower()
            st.markdown(f"""
            - 예측: **{'이상거래' if result.is_fraud else '정상거래'}**
            - 이상거래 확률: **{result.fraud_probability:.1%}**
            - 위험 수준: <span class="{risk_class}">{result.risk_level}</span>
            """, unsafe_allow_html=True)

            if result.explanation:
                st.markdown("**분석 설명**")
                st.info(result.explanation["summary"])

                if result.explanation.get("risk_factors"):
                    st.markdown("**주요 위험 요소**")
                    for rf in result.explanation["risk_factors"]:
                        st.write(f"- {rf['factor']}: {rf['value']} (가중치: {rf['weight']:.2f})")


def render_llm_tab():
    """LLM 분석 탭"""
    st.markdown("### LLM 기반 분석")

    if st.session_state.model is None or st.session_state.llm_client is None:
        st.warning("먼저 데이터를 생성하고 모델을 학습하세요.")
        return

    model = st.session_state.model
    data = st.session_state.data
    llm_client = st.session_state.llm_client
    X, _, _ = st.session_state.preprocessor.prepare_ml_data(data)

    # 프롬프트 타입 선택
    st.markdown("#### 프롬프트 엔지니어링 기법 비교")

    prompt_type = st.radio(
        "프롬프트 타입",
        ["zero_shot", "few_shot", "cot"],
        format_func=lambda x: {
            "zero_shot": "Zero-shot (예시 없이)",
            "few_shot": "Few-shot (예시 포함)",
            "cot": "Chain-of-Thought (단계별 추론)"
        }[x],
        horizontal=True,
    )

    st.markdown("""
    | 기법 | 설명 | 장점 | 단점 |
    |------|------|------|------|
    | **Zero-shot** | 예시 없이 직접 질문 | 토큰 절약, 빠른 응답 | 정확도 낮을 수 있음 |
    | **Few-shot** | 유사 예시 제공 | 일관된 형식, 높은 정확도 | 토큰 소비 많음 |
    | **CoT** | 단계별 추론 유도 | 복잡한 분석 가능 | 응답 시간 길어짐 |
    """)

    st.markdown("---")

    # 거래 선택
    transaction_idx = st.selectbox(
        "분석할 거래 선택",
        range(min(50, len(data))),
        format_func=lambda x: f"거래 {x}: {data.iloc[x]['transaction_id']} - {data.iloc[x]['amount']:,.0f}원",
        key="llm_transaction"
    )

    col1, col2 = st.columns(2)

    with col1:
        transaction = data.iloc[transaction_idx].to_dict()
        X_single = X[transaction_idx:transaction_idx+1]
        ml_result = model.predict_single(X_single, transaction)

        st.markdown("**ML 예측 결과**")
        st.json({
            "is_fraud": ml_result.is_fraud,
            "fraud_probability": f"{ml_result.fraud_probability:.2%}",
            "risk_level": ml_result.risk_level,
        })

    with col2:
        # 프롬프트 생성
        ml_result_dict = {
            "fraud_probability": ml_result.fraud_probability,
            "risk_level": ml_result.risk_level,
            "explanation": ml_result.explanation,
        }
        prompt = get_fraud_analysis_prompt(transaction, ml_result_dict, prompt_type)

        with st.expander("생성된 프롬프트 보기"):
            st.code(prompt, language="markdown")

    # LLM 분석 실행
    if st.button("LLM 분석 실행", type="primary"):
        with st.spinner("LLM이 분석 중..."):
            try:
                response = llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=SYSTEM_PROMPT,
                )

                st.markdown("#### LLM 분석 결과")
                st.markdown(response.content)

                st.markdown("---")
                st.markdown("**토큰 사용량**")
                st.json(response.usage)

            except Exception as e:
                st.error(f"LLM 호출 실패: {e}")


def render_tools_tab():
    """Function Calling 탭"""
    st.markdown("### Function Calling (Tool Use)")

    st.markdown("""
    LLM이 외부 도구(함수)를 호출하여 실시간 정보를 가져오거나 작업을 수행할 수 있습니다.
    """)

    # 도구 목록
    st.markdown("#### 정의된 금융 도구")

    for tool in FINANCIAL_TOOLS:
        func = tool["function"]
        with st.expander(f"**{func['name']}**: {func['description']}"):
            st.json(func["parameters"])

    st.markdown("---")

    # 도구 실행 테스트
    st.markdown("#### 도구 실행 테스트")

    executor = ToolExecutor()

    tool_name = st.selectbox(
        "실행할 도구",
        [t["function"]["name"] for t in FINANCIAL_TOOLS]
    )

    # 도구별 입력 폼
    if tool_name == "analyze_transaction":
        col1, col2 = st.columns(2)
        with col1:
            txn_id = st.text_input("거래 ID", "TXN_TEST_001")
            amount = st.number_input("금액 (원)", 0, 100000000, 5000000)
        with col2:
            is_intl = st.checkbox("해외 거래")
            location = st.text_input("위치", "서울")

        args = {
            "transaction_id": txn_id,
            "amount": amount,
            "is_international": is_intl,
            "location": location,
        }

    elif tool_name == "get_account_history":
        account_id = st.text_input("계좌 번호", "ACC_001")
        days = st.slider("조회 기간 (일)", 1, 90, 30)
        args = {"account_id": account_id, "days": days}

    elif tool_name == "get_risk_score":
        txn_id = st.text_input("거래 ID", "TXN_TEST_001")
        include_history = st.checkbox("과거 이력 포함", True)
        args = {"transaction_id": txn_id, "include_history": include_history}

    elif tool_name == "block_transaction":
        txn_id = st.text_input("거래 ID", "TXN_TEST_001")
        reason = st.text_input("차단 사유", "의심 거래")
        args = {"transaction_id": txn_id, "reason": reason}

    elif tool_name == "send_alert":
        col1, col2 = st.columns(2)
        with col1:
            alert_type = st.selectbox("알림 유형", ["fraud_suspected", "high_risk", "unusual_pattern"])
            txn_id = st.text_input("관련 거래 ID", "TXN_TEST_001")
        with col2:
            priority = st.selectbox("우선순위", ["low", "medium", "high", "critical"])
            message = st.text_input("메시지", "이상 거래 감지")
        args = {
            "alert_type": alert_type,
            "transaction_id": txn_id,
            "message": message,
            "priority": priority,
        }

    if st.button("도구 실행"):
        result = executor.execute(tool_name, args)
        st.markdown("**실행 결과**")
        st.json(result)


def render_code_tab():
    """코드 예시 탭"""
    st.markdown("### 핵심 코드 예시")

    tab1, tab2, tab3 = st.tabs(["데이터 처리", "ML 모델", "LLM API"])

    with tab1:
        st.markdown("#### NumPy/Pandas 데이터 전처리")
        st.code('''
from src.data.generator import generate_transaction_data
from src.data.preprocessor import TransactionPreprocessor

# 샘플 데이터 생성
data = generate_transaction_data(n_samples=1000, fraud_ratio=0.1)

# 전처리
preprocessor = TransactionPreprocessor()
X, y, feature_names = preprocessor.prepare_ml_data(data)

print(f"Features: {feature_names}")
print(f"Shape: {X.shape}")
        ''', language="python")

    with tab2:
        st.markdown("#### scikit-learn 이상거래 탐지")
        st.code('''
from src.ml.fraud_detector import FraudDetector

# 모델 학습
model = FraudDetector(algorithm="random_forest")
model.fit(X, y, feature_names)

# 예측
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# 평가
metrics = model.evaluate(X, y)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
        ''', language="python")

    with tab3:
        st.markdown("#### LLM API 호출")
        st.code('''
from src.llm.client import create_llm_client
from src.prompts.templates import get_fraud_analysis_prompt, SYSTEM_PROMPT

# 클라이언트 생성 (mock/openai/claude)
client = create_llm_client("openai")

# 프롬프트 생성 (zero_shot/few_shot/cot)
prompt = get_fraud_analysis_prompt(
    transaction={"amount": 5000000, "is_international": True},
    ml_result={"fraud_probability": 0.85, "risk_level": "high"},
    prompt_type="cot"
)

# API 호출
response = client.chat(
    messages=[{"role": "user", "content": prompt}],
    system_prompt=SYSTEM_PROMPT
)

print(response.content)
        ''', language="python")


def main():
    """메인 함수"""
    init_session_state()

    llm_provider, ml_algorithm = render_sidebar()

    # 탭 구성
    tabs = st.tabs([
        "개요",
        "ML 분석",
        "LLM 분석",
        "Function Calling",
        "코드 예시"
    ])

    with tabs[0]:
        render_overview_tab()

    with tabs[1]:
        render_ml_tab()

    with tabs[2]:
        render_llm_tab()

    with tabs[3]:
        render_tools_tab()

    with tabs[4]:
        render_code_tab()


if __name__ == "__main__":
    main()
