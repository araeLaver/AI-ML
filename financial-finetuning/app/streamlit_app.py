# Streamlit Demo UI
"""
Financial LLM Fine-tuning Demo
"""

import streamlit as st
import requests
import json
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import FINANCIAL_INSTRUCTIONS, FinancialInstructionDataset

# 페이지 설정
st.set_page_config(
    page_title="Financial LLM Fine-tuning",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .instruction-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .category-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .output-box {
        background: #f0f7f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # 헤더
    st.markdown('<p class="main-header">💰 Financial LLM Fine-tuning</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">금융 도메인 특화 LLM 파인튜닝 데모 - LoRA/QLoRA 기반</p>',
        unsafe_allow_html=True
    )

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 데이터셋",
        "🎯 학습 설정",
        "🤖 추론 테스트",
        "📈 학습 모니터링"
    ])

    # 탭 1: 데이터셋
    with tab1:
        render_dataset_tab()

    # 탭 2: 학습 설정
    with tab2:
        render_training_config_tab()

    # 탭 3: 추론 테스트
    with tab3:
        render_inference_tab()

    # 탭 4: 학습 모니터링
    with tab4:
        render_monitoring_tab()


def render_dataset_tab():
    """데이터셋 탭 렌더링"""
    st.subheader("📊 금융 도메인 학습 데이터셋")

    # 데이터셋 통계
    try:
        dataset = FinancialInstructionDataset()
        stats = dataset.get_statistics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("전체 샘플", stats["total_samples"])
        with col2:
            st.metric("학습 샘플", stats["train_samples"])
        with col3:
            st.metric("평가 샘플", stats["eval_samples"])
        with col4:
            st.metric("평균 길이", f"{stats['avg_text_length']:.0f}자")

        # 카테고리별 분포
        st.subheader("카테고리별 분포")
        categories = stats["categories"]

        # 차트 데이터
        import pandas as pd
        df_cat = pd.DataFrame([
            {"카테고리": k, "샘플 수": v}
            for k, v in categories.items()
        ])
        st.bar_chart(df_cat.set_index("카테고리"))

    except Exception as e:
        st.info(f"데이터셋 로드 중 오류: {e}")
        st.info("datasets 라이브러리가 필요합니다.")

    # 샘플 데이터 표시
    st.subheader("샘플 데이터")

    # 카테고리 필터
    all_categories = list(set(item.get("category", "general") for item in FINANCIAL_INSTRUCTIONS))
    selected_category = st.selectbox("카테고리 선택", ["전체"] + all_categories)

    # 필터링된 데이터
    filtered_data = FINANCIAL_INSTRUCTIONS
    if selected_category != "전체":
        filtered_data = [
            item for item in FINANCIAL_INSTRUCTIONS
            if item.get("category") == selected_category
        ]

    # 데이터 표시
    for i, item in enumerate(filtered_data[:5]):  # 최대 5개만 표시
        with st.expander(f"샘플 {i+1}: {item['instruction'][:50]}...", expanded=(i == 0)):
            st.markdown(f"""
            <div class="instruction-card">
                <span class="category-badge">{item.get('category', 'general')}</span>
                <h4>지시사항</h4>
                <p>{item['instruction']}</p>
            </div>
            """, unsafe_allow_html=True)

            if item.get("input"):
                st.markdown("**입력:**")
                st.code(item["input"], language=None)

            st.markdown("**응답:**")
            st.markdown(f"""
            <div class="output-box">
                {item['output'][:500]}{'...' if len(item['output']) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)


def render_training_config_tab():
    """학습 설정 탭 렌더링"""
    st.subheader("🎯 LoRA/QLoRA 학습 설정")

    # 설정 파일 로드
    config_path = project_root / "configs" / "training_config.yaml"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 모델 설정")

        model_name = st.selectbox(
            "베이스 모델",
            [
                "beomi/Llama-3-Open-Ko-8B",
                "beomi/llama-2-ko-7b",
                "Qwen/Qwen2.5-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
            ],
            index=0,
        )

        max_seq_length = st.slider("최대 시퀀스 길이", 512, 4096, 2048, 256)

        st.markdown("### LoRA 설정")
        lora_r = st.slider("LoRA Rank (r)", 4, 64, 16, 4)
        lora_alpha = st.slider("LoRA Alpha", 8, 128, 32, 8)
        lora_dropout = st.slider("LoRA Dropout", 0.0, 0.2, 0.05, 0.01)

    with col2:
        st.markdown("### 양자화 설정")

        use_quantization = st.checkbox("QLoRA 사용 (4-bit)", value=True)

        if use_quantization:
            quant_type = st.selectbox("양자화 타입", ["nf4", "fp4"], index=0)
            use_double_quant = st.checkbox("Double Quantization", value=True)

        st.markdown("### 학습 설정")
        num_epochs = st.slider("에포크 수", 1, 10, 3)
        batch_size = st.slider("배치 크기", 1, 16, 4)
        learning_rate = st.select_slider(
            "학습률",
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        gradient_accumulation = st.slider("Gradient Accumulation", 1, 16, 4)

    # 설정 요약
    st.markdown("---")
    st.subheader("📋 설정 요약")

    effective_batch = batch_size * gradient_accumulation
    st.info(f"실효 배치 크기: {effective_batch} (batch_size × gradient_accumulation)")

    # 설정 JSON 표시
    config_dict = {
        "model": {
            "name": model_name,
            "max_seq_length": max_seq_length,
        },
        "lora": {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        "quantization": {
            "enabled": use_quantization,
            "load_in_4bit": use_quantization,
        },
        "training": {
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation,
        },
    }

    with st.expander("설정 JSON 보기"):
        st.json(config_dict)

    # 학습 시작 버튼
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("🚀 학습 시작", type="primary", use_container_width=True):
            st.warning("실제 학습은 GPU가 필요합니다. CLI에서 실행해주세요:")
            st.code("python -m src.training.train_lora --config configs/training_config.yaml")


def render_inference_tab():
    """추론 테스트 탭 렌더링"""
    st.subheader("🤖 추론 테스트")

    # API 서버 상태 확인
    api_url = st.text_input("API 서버 URL", value="http://localhost:8000")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("서버 상태 확인"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded"):
                        st.success("✅ 모델 로드됨")
                    else:
                        st.warning("⚠️ 모델 미로드")
                else:
                    st.error("❌ 서버 오류")
            except requests.exceptions.ConnectionError:
                st.error("❌ 서버 연결 실패")
            except Exception as e:
                st.error(f"❌ 오류: {e}")

    st.markdown("---")

    # 추론 유형 선택
    inference_type = st.radio(
        "추론 유형",
        ["일반 질의", "이상거래 탐지", "투자 분석", "상품 설명"],
        horizontal=True,
    )

    # 입력 영역
    if inference_type == "일반 질의":
        instruction = st.text_area(
            "지시사항",
            value="금리 인상이 주식 시장에 미치는 영향을 설명해주세요.",
            height=100,
        )
        input_text = st.text_area("입력 (선택)", value="", height=100)

    elif inference_type == "이상거래 탐지":
        instruction = "다음 금융 거래 정보를 분석하여 이상 거래 여부를 판단해주세요."
        st.text_area("지시사항 (고정)", value=instruction, height=80, disabled=True)
        input_text = st.text_area(
            "거래 정보",
            value="계좌번호: 123-456-789\n거래금액: 5억원\n거래시간: 새벽 3시 15분\n거래 유형: 해외 송금\n수취인: 해외 법인",
            height=150,
        )

    elif inference_type == "투자 분석":
        instruction = "금융 전문가로서 다음 투자 관련 질문에 답변해주세요."
        st.text_area("지시사항 (고정)", value=instruction, height=80, disabled=True)
        input_text = st.text_area(
            "투자 질문",
            value="현재 금리 인상 기조에서 채권 투자 전략은 어떻게 세워야 할까요?",
            height=100,
        )

    else:  # 상품 설명
        instruction = "다음 금융 상품에 대해 일반 고객이 이해하기 쉽게 설명해주세요."
        st.text_area("지시사항 (고정)", value=instruction, height=80, disabled=True)
        input_text = st.text_area(
            "상품 정보",
            value="상품명: KB국민 적립식 펀드\n유형: 주식혼합형\n투자기간: 3년\n목표수익률: 연 7%",
            height=150,
        )

    # 생성 파라미터
    with st.expander("생성 파라미터"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
        with col2:
            top_k = st.slider("Top K", 1, 100, 50)
            max_tokens = st.slider("Max Tokens", 64, 2048, 512, 64)
        with col3:
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)

    # 생성 버튼
    if st.button("🎯 응답 생성", type="primary"):
        with st.spinner("응답 생성 중..."):
            try:
                response = requests.post(
                    f"{api_url}/generate",
                    json={
                        "instruction": instruction,
                        "input_text": input_text,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "max_new_tokens": max_tokens,
                        "repetition_penalty": repetition_penalty,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### 📝 생성된 응답")
                    st.markdown(f"""
                    <div class="output-box">
                        {data['response']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"API 오류: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("API 서버에 연결할 수 없습니다. 서버를 먼저 실행해주세요.")
                st.code("python api/server.py")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    # 데모 응답 (API 없이)
    st.markdown("---")
    st.subheader("📌 샘플 응답 (데모)")

    demo_responses = {
        "일반 질의": """금리 인상은 주식 시장에 다양한 영향을 미칩니다:

1. **부정적 영향**
   - 기업 차입비용 증가로 수익성 악화
   - 할인율 상승으로 주식 가치 하락
   - 채권 등 대체 투자처로 자금 이동

2. **섹터별 차별화**
   - 금융주: 순이자마진 개선으로 긍정적
   - 성장주: 미래 현금흐름 할인에 민감하여 부정적
   - 배당주: 상대적으로 안정적

3. **투자 전략**
   - 가치주 중심 포트폴리오 구성
   - 현금 흐름이 안정적인 기업 선호
   - 분산 투자로 리스크 관리""",

        "이상거래 탐지": """**위험 수준: 높음 (High Risk)**

**이상 징후 분석:**
1. 거래 시간 이상: 새벽 3시 15분은 일반적인 금융 거래 시간대가 아님
2. 대규모 거래: 5억원은 고액 거래로 추가 검증 필요
3. 해외 송금: 자금세탁 위험이 있는 거래 유형
4. 수취인 확인: 해외 법인의 실체 확인 필요

**권장 조치:**
- 거래 보류 및 고객 신원 재확인
- 자금출처 증빙 요청
- 의심거래보고(STR) 검토
- AML 담당부서 에스컬레이션""",
    }

    demo_key = inference_type if inference_type in demo_responses else "일반 질의"
    st.markdown(f"""
    <div class="output-box">
        {demo_responses[demo_key]}
    </div>
    """, unsafe_allow_html=True)


def render_monitoring_tab():
    """학습 모니터링 탭 렌더링"""
    st.subheader("📈 학습 모니터링")

    st.info("실제 학습 진행 시 TensorBoard 또는 Weights & Biases에서 상세 모니터링이 가능합니다.")

    # 모의 학습 메트릭
    st.markdown("### 학습 진행 상황 (샘플)")

    import pandas as pd
    import numpy as np

    # 모의 데이터 생성
    np.random.seed(42)
    epochs = list(range(1, 11))
    train_loss = [2.5 * np.exp(-0.3 * e) + 0.2 + np.random.normal(0, 0.05) for e in epochs]
    eval_loss = [2.7 * np.exp(-0.25 * e) + 0.25 + np.random.normal(0, 0.08) for e in epochs]

    df_metrics = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss": train_loss,
        "Eval Loss": eval_loss,
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loss 곡선")
        st.line_chart(df_metrics.set_index("Epoch")[["Train Loss", "Eval Loss"]])

    with col2:
        st.markdown("#### 학습 상태")
        progress = 0.7  # 70% 완료
        st.progress(progress)
        st.write(f"진행률: {progress * 100:.0f}%")

        st.metric("현재 에포크", "7 / 10")
        st.metric("현재 Loss", f"{train_loss[6]:.4f}")
        st.metric("Best Eval Loss", f"{min(eval_loss[:7]):.4f}")

    # 리소스 사용량
    st.markdown("### 리소스 사용량 (샘플)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("GPU 메모리", "12.5 GB / 24 GB")
        st.progress(0.52)
    with col2:
        st.metric("GPU 사용률", "85%")
        st.progress(0.85)
    with col3:
        st.metric("학습 속도", "2.3 it/s")

    # TensorBoard 연동
    st.markdown("---")
    st.markdown("### TensorBoard 연동")
    st.code("""
# TensorBoard 실행
tensorboard --logdir outputs/runs

# 브라우저에서 접속
# http://localhost:6006
    """)


if __name__ == "__main__":
    main()
