"""
Fraud Detection Demo - Streamlit UI
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# 페이지 설정
st.set_page_config(
    page_title="Fraud Detection MLOps",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 스타일
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .risk-critical { color: #dc3545; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-minimal { color: #6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health():
    """API 상태 확인"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def get_model_info():
    """모델 정보 조회"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict_single(data: dict):
    """단일 예측"""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def predict_batch(transactions: list):
    """배치 예측"""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"transactions": transactions},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("## 🔍 Fraud Detection")
        st.markdown("### MLOps Pipeline Demo")

        st.divider()

        # API 상태
        is_healthy, health_info = check_api_health()
        if is_healthy:
            st.success("✅ API 연결됨")
            st.caption(f"Model: {health_info.get('model_version', 'N/A')}")
        else:
            st.error("❌ API 연결 실패")
            st.caption("로컬 모드로 실행")

        st.divider()

        # 모델 정보
        model_info = get_model_info()
        if model_info:
            st.markdown("### 모델 정보")
            st.metric("임계값", f"{model_info['threshold']:.2f}")
            st.caption(f"버전: {model_info['model_version']}")

        st.divider()

        # 네비게이션
        page = st.radio(
            "메뉴",
            ["🎯 실시간 예측", "📊 배치 분석", "📈 모니터링"],
            label_visibility="collapsed",
        )

        return page


def render_single_prediction():
    """단일 예측 페이지"""
    st.markdown("## 🎯 실시간 거래 분석")
    st.caption("개별 거래의 이상 여부를 실시간으로 분석합니다.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 거래 정보 입력")

        amount = st.number_input("거래 금액 (원)", min_value=0.0, value=150000.0, step=10000.0)
        time_hour = st.slider("거래 시간", 0, 23, 14)
        location_distance = st.number_input("평소 위치와의 거리 (km)", min_value=0.0, value=5.0)
        previous_avg = st.number_input("이전 평균 거래 금액", min_value=0.0, value=100000.0)

        with st.expander("추가 정보 (선택)"):
            day_of_week = st.selectbox(
                "요일", ["월", "화", "수", "목", "금", "토", "일"], index=2
            )
            merchant_cat = st.selectbox(
                "가맹점 카테고리",
                ["일반상점", "온라인쇼핑", "음식점", "주유소", "편의점", "백화점", "해외"],
                index=0,
            )
            tx_count_1h = st.number_input("최근 1시간 거래 횟수", min_value=0, value=1)
            tx_count_24h = st.number_input("최근 24시간 거래 횟수", min_value=0, value=5)
            is_weekend = 1 if day_of_week in ["토", "일"] else 0
            is_night = 1 if time_hour < 6 or time_hour >= 22 else 0
            device_change = st.checkbox("디바이스 변경")

        if st.button("🔍 분석하기", type="primary", use_container_width=True):
            transaction_data = {
                "amount": amount,
                "time_hour": time_hour,
                "location_distance": location_distance,
                "previous_avg_amount": previous_avg,
                "day_of_week": ["월", "화", "수", "목", "금", "토", "일"].index(day_of_week),
                "merchant_category": ["일반상점", "온라인쇼핑", "음식점", "주유소", "편의점", "백화점", "해외"].index(merchant_cat),
                "transaction_count_1h": tx_count_1h,
                "transaction_count_24h": tx_count_24h,
                "is_weekend": is_weekend,
                "is_night": is_night,
                "device_change": 1 if device_change else 0,
            }

            with st.spinner("분석 중..."):
                result = predict_single(transaction_data)

            st.session_state["prediction_result"] = result

    with col2:
        st.markdown("### 분석 결과")

        if "prediction_result" in st.session_state:
            result = st.session_state["prediction_result"]

            if "error" in result:
                st.error(f"오류: {result['error']}")
            else:
                # 위험 수준에 따른 색상
                risk_colors = {
                    "CRITICAL": "#dc3545",
                    "HIGH": "#fd7e14",
                    "MEDIUM": "#ffc107",
                    "LOW": "#28a745",
                    "MINIMAL": "#6c757d",
                }

                risk_level = result["risk_level"]
                probability = result["probability"]

                # 게이지 차트
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=probability * 100,
                        title={"text": "이상 거래 확률"},
                        delta={"reference": 50},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": risk_colors[risk_level]},
                            "steps": [
                                {"range": [0, 20], "color": "#e8f5e9"},
                                {"range": [20, 40], "color": "#fff3e0"},
                                {"range": [40, 60], "color": "#fff8e1"},
                                {"range": [60, 80], "color": "#ffecb3"},
                                {"range": [80, 100], "color": "#ffcdd2"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": result["threshold"] * 100,
                            },
                        },
                    )
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # 결과 메트릭
                col_a, col_b = st.columns(2)
                with col_a:
                    if result["is_fraud"]:
                        st.error(f"⚠️ 이상 거래 의심")
                    else:
                        st.success(f"✅ 정상 거래")

                with col_b:
                    st.metric("위험 수준", risk_level)

                st.caption(f"응답 시간: {result['latency_ms']:.2f}ms")
        else:
            st.info("거래 정보를 입력하고 '분석하기' 버튼을 클릭하세요.")


def render_batch_analysis():
    """배치 분석 페이지"""
    st.markdown("## 📊 배치 거래 분석")
    st.caption("여러 거래를 한 번에 분석합니다.")

    # 샘플 데이터 생성
    if st.button("📥 샘플 데이터 생성"):
        np.random.seed(42)
        n = 20

        sample_data = []
        for i in range(n):
            is_suspicious = np.random.random() < 0.15  # 15%는 의심 거래

            if is_suspicious:
                sample_data.append({
                    "amount": float(np.random.uniform(500000, 2000000)),
                    "time_hour": int(np.random.choice([2, 3, 4, 23])),
                    "location_distance": float(np.random.uniform(100, 500)),
                    "previous_avg_amount": float(np.random.uniform(50000, 100000)),
                    "transaction_count_1h": int(np.random.randint(5, 15)),
                    "transaction_count_24h": int(np.random.randint(15, 30)),
                    "device_change": 1,
                })
            else:
                sample_data.append({
                    "amount": float(np.random.uniform(10000, 200000)),
                    "time_hour": int(np.random.randint(9, 21)),
                    "location_distance": float(np.random.uniform(0, 20)),
                    "previous_avg_amount": float(np.random.uniform(80000, 150000)),
                    "transaction_count_1h": int(np.random.randint(0, 3)),
                    "transaction_count_24h": int(np.random.randint(3, 10)),
                    "device_change": 0,
                })

        st.session_state["batch_data"] = sample_data

    if "batch_data" in st.session_state:
        df = pd.DataFrame(st.session_state["batch_data"])
        st.dataframe(df, use_container_width=True)

        if st.button("🔍 배치 분석 실행", type="primary"):
            with st.spinner("분석 중..."):
                result = predict_batch(st.session_state["batch_data"])

            if "error" in result:
                st.error(f"오류: {result['error']}")
            else:
                st.session_state["batch_result"] = result

        if "batch_result" in st.session_state:
            result = st.session_state["batch_result"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 건수", result["total_count"])
            with col2:
                st.metric("이상 거래", result["fraud_count"])
            with col3:
                fraud_rate = result["fraud_count"] / result["total_count"] * 100
                st.metric("이상 비율", f"{fraud_rate:.1f}%")

            # 결과 테이블
            predictions_df = pd.DataFrame(result["predictions"])
            st.dataframe(
                predictions_df.style.apply(
                    lambda x: [
                        "background-color: #ffcdd2" if v else ""
                        for v in x == True
                    ],
                    subset=["is_fraud"],
                ),
                use_container_width=True,
            )

            # 확률 분포
            fig = px.histogram(
                predictions_df,
                x="probability",
                nbins=20,
                title="예측 확률 분포",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_monitoring():
    """모니터링 페이지"""
    st.markdown("## 📈 모니터링 대시보드")
    st.caption("모델 성능 및 시스템 상태를 모니터링합니다.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 시스템 상태")

        is_healthy, health_info = check_api_health()

        if is_healthy:
            st.success("API 서버: 정상")
            st.json(health_info)
        else:
            st.error("API 서버: 연결 실패")

    with col2:
        st.markdown("### 모델 정보")

        model_info = get_model_info()
        if model_info:
            st.json(model_info)
        else:
            st.warning("모델 정보를 가져올 수 없습니다.")

    st.divider()

    st.markdown("### 외부 모니터링 링크")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.link_button(
            "🔍 MLflow",
            "http://localhost:5000",
            use_container_width=True,
        )

    with col_b:
        st.link_button(
            "📊 Prometheus",
            "http://localhost:9090",
            use_container_width=True,
        )

    with col_c:
        st.link_button(
            "📈 Grafana",
            "http://localhost:3000",
            use_container_width=True,
        )


def main():
    page = render_sidebar()

    if page == "🎯 실시간 예측":
        render_single_prediction()
    elif page == "📊 배치 분석":
        render_batch_analysis()
    elif page == "📈 모니터링":
        render_monitoring()


if __name__ == "__main__":
    main()
