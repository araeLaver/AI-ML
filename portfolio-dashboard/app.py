# AI/ML Portfolio Dashboard
"""
통합 포트폴리오 대시보드 - 클린 디자인
참조: Microsoft Streamlit UI Template, Awesome Streamlit Themes
"""

import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="AI/ML Portfolio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 클린 CSS 스타일 (라이트 테마, 가독성 중심)
st.markdown("""
<style>
    /* 기본 폰트 및 배경 */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* 헤더 스타일 */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* 통계 배지 */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }

    .stat-badge {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        text-align: center;
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2563eb;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* 섹션 타이틀 */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563eb;
        display: inline-block;
    }

    /* 프로젝트 카드 */
    .project-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s, border-color 0.2s;
    }

    .project-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #2563eb;
    }

    .step-label {
        display: inline-block;
        background: #2563eb;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }

    .project-name {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .project-desc {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* 기술 태그 */
    .tech-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .tech-tag {
        background: #f1f5f9;
        color: #475569;
        font-size: 0.8rem;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }

    /* 기능 리스트 */
    .feature-list {
        margin: 0;
        padding-left: 1.25rem;
        color: #475569;
    }

    .feature-list li {
        margin-bottom: 0.35rem;
        font-size: 0.9rem;
    }

    /* 기술 스택 그리드 */
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .tech-category {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }

    .tech-category-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    .tech-category-items {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* 포트 정보 */
    .port-info {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: #166534;
        margin-top: 0.75rem;
    }

    /* 타임라인 */
    .timeline-container {
        position: relative;
        padding-left: 2rem;
    }

    .timeline-item {
        position: relative;
        padding-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 2px solid #e2e8f0;
    }

    .timeline-item:last-child {
        border-left: 2px solid transparent;
    }

    .timeline-dot {
        position: absolute;
        left: -0.5rem;
        top: 0.25rem;
        width: 10px;
        height: 10px;
        background: #2563eb;
        border-radius: 50%;
        border: 2px solid white;
        box-shadow: 0 0 0 2px #2563eb;
    }

    .timeline-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
    }

    .timeline-desc {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    /* 코드 블록 */
    .code-block {
        background: #1e293b;
        color: #e2e8f0;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        line-height: 1.5;
    }

    /* 푸터 */
    .footer {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
    }

    /* 반응형 */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.75rem; }
        .stats-container { gap: 1rem; }
        .main .block-container { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)


# 프로젝트 데이터
PROJECTS = [
    {
        "step": "Step 1+2",
        "name": "Financial Analysis",
        "folder": "financial-analysis",
        "port": 8503,
        "desc": "ML 기반 금융 이상거래 탐지와 LLM API 분석을 통합한 시스템. NumPy/Pandas로 데이터를 전처리하고, scikit-learn 모델로 이상거래를 탐지한 후, OpenAI/Claude API로 분석 결과를 자연어로 설명합니다.",
        "features": [
            "NumPy/Pandas 데이터 전처리",
            "scikit-learn 이상거래 탐지 (RF, GB, IF)",
            "OpenAI/Claude API 통합",
            "Zero-shot, Few-shot, CoT 프롬프트",
            "Function Calling 도구 연동",
        ],
        "tech": ["Python", "NumPy", "Pandas", "scikit-learn", "OpenAI", "Claude"],
    },
    {
        "step": "Step 3",
        "name": "Finance RAG API",
        "folder": "finance-rag-api",
        "port": 8501,
        "desc": "금융 문서를 벡터화하여 저장하고, 사용자 질문에 관련 문서를 검색한 후 LLM으로 답변을 생성하는 RAG 시스템. ChromaDB 벡터 저장소와 FastAPI를 사용합니다.",
        "features": [
            "PDF/텍스트 문서 청킹 및 임베딩",
            "ChromaDB 벡터 저장소",
            "하이브리드 검색 (시맨틱 + 키워드)",
            "FastAPI REST API",
            "스트리밍 응답 지원",
        ],
        "tech": ["Python", "LangChain", "ChromaDB", "OpenAI", "FastAPI"],
    },
    {
        "step": "Step 4",
        "name": "Code Review Agent",
        "folder": "code-review-agent",
        "port": 8500,
        "desc": "GitHub PR을 자동으로 분석하여 코드 품질, 보안 취약점, 개선 사항을 리뷰하는 AI 에이전트. LangGraph 기반 멀티 에이전트 워크플로우를 구현합니다.",
        "features": [
            "GitHub PR 자동 분석",
            "코드 품질 평가",
            "보안 취약점 탐지",
            "개선 제안 생성",
            "LangGraph 멀티 에이전트",
        ],
        "tech": ["Python", "LangGraph", "OpenAI", "GitHub API"],
    },
    {
        "step": "Step 5",
        "name": "MLOps Pipeline",
        "folder": "mlops-pipeline",
        "port": 8504,
        "desc": "금융 ML 모델의 학습, 배포, 모니터링을 자동화하는 파이프라인. MLflow로 실험을 추적하고, 모델 성능 저하와 데이터 드리프트를 모니터링합니다.",
        "features": [
            "자동화된 학습 파이프라인",
            "MLflow 실험 추적",
            "모델 버전 관리",
            "A/B 테스팅",
            "드리프트 감지 및 알림",
        ],
        "tech": ["Python", "MLflow", "DVC", "Docker", "FastAPI"],
    },
    {
        "step": "Step 6",
        "name": "Financial Fine-tuning",
        "folder": "financial-finetuning",
        "port": 8502,
        "desc": "금융 도메인에 특화된 LLM을 효율적으로 파인튜닝하는 시스템. LoRA/QLoRA 기법으로 메모리 사용량을 줄이면서 도메인 성능을 향상시킵니다.",
        "features": [
            "금융 도메인 데이터셋 생성",
            "LoRA/QLoRA 효율적 학습",
            "4-bit 양자화 (bitsandbytes)",
            "모델 병합 및 배포",
            "BLEU, ROUGE 성능 평가",
        ],
        "tech": ["Python", "Transformers", "PEFT", "bitsandbytes"],
    },
]

TECH_STACK = {
    "Data Processing": "NumPy, Pandas, scikit-learn",
    "LLM APIs": "OpenAI, Claude, Transformers",
    "RAG & Vector DB": "LangChain, ChromaDB, FAISS",
    "AI Agent": "LangGraph, Function Calling",
    "MLOps": "MLflow, DVC, Docker",
    "Web & API": "FastAPI, Streamlit",
    "Fine-tuning": "PEFT, LoRA, QLoRA, bitsandbytes",
    "Language": "Python 3.11+",
}


def render_hero():
    """Hero 섹션"""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">AI/ML Engineering Portfolio</div>
        <div class="hero-subtitle">금융 도메인 특화 AI 솔루션 개발 프로젝트</div>
        <div class="stats-container">
            <div class="stat-badge">
                <div class="stat-number">5</div>
                <div class="stat-label">Projects</div>
            </div>
            <div class="stat-badge">
                <div class="stat-number">6</div>
                <div class="stat-label">Steps</div>
            </div>
            <div class="stat-badge">
                <div class="stat-number">15+</div>
                <div class="stat-label">Technologies</div>
            </div>
            <div class="stat-badge">
                <div class="stat-number">100%</div>
                <div class="stat-label">Complete</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_projects():
    """프로젝트 섹션"""
    st.markdown('<div class="section-title">Projects</div>', unsafe_allow_html=True)

    for project in PROJECTS:
        col1, col2 = st.columns([3, 2])

        with col1:
            tech_tags = "".join([f'<span class="tech-tag">{t}</span>' for t in project["tech"]])

            st.markdown(f"""
            <div class="project-card">
                <span class="step-label">{project['step']}</span>
                <div class="project-name">{project['name']}</div>
                <div class="project-desc">{project['desc']}</div>
                <div class="tech-tags">{tech_tags}</div>
                <div class="port-info">localhost:{project['port']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("**Features**")
            for f in project["features"]:
                st.markdown(f"- {f}")


def render_tech_stack():
    """기술 스택 섹션"""
    st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    items = list(TECH_STACK.items())

    for i, (category, techs) in enumerate(items):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="tech-category">
                <div class="tech-category-title">{category}</div>
                <div class="tech-category-items">{techs}</div>
            </div>
            """, unsafe_allow_html=True)


def render_timeline():
    """학습 타임라인"""
    st.markdown('<div class="section-title">Learning Path</div>', unsafe_allow_html=True)

    timeline = [
        ("Step 1-2", "Python + AI 기초, LLM API", "데이터 처리, ML 모델, 프롬프트 엔지니어링 학습"),
        ("Step 3", "RAG 시스템", "벡터 DB, 임베딩, 검색 증강 생성 구현"),
        ("Step 4", "AI Agent", "LangGraph 기반 자동화 에이전트 개발"),
        ("Step 5", "MLOps", "모델 운영, 파이프라인, 모니터링 구축"),
        ("Step 6", "Fine-tuning", "LoRA/QLoRA 효율적 도메인 특화 학습"),
    ]

    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
    for step, title, desc in timeline:
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-title">{step}: {title}</div>
            <div class="timeline-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_quick_start():
    """빠른 시작"""
    st.markdown('<div class="section-title">Quick Start</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="code-block">
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 개별 프로젝트 실행
streamlit run financial-analysis/app/streamlit_app.py --server.port 8503
streamlit run finance-rag-api/app/streamlit_app.py --server.port 8501
streamlit run code-review-agent/app/streamlit_app.py --server.port 8500
streamlit run mlops-pipeline/app/streamlit_app.py --server.port 8504
streamlit run financial-finetuning/app/streamlit_app.py --server.port 8502
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """푸터"""
    st.markdown("""
    <div class="footer">
        Built with Streamlit | AI/ML Engineering Portfolio
    </div>
    """, unsafe_allow_html=True)


def main():
    render_hero()
    render_projects()
    render_tech_stack()
    render_timeline()
    render_quick_start()
    render_footer()


if __name__ == "__main__":
    main()
