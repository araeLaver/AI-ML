# Financial Analysis: ML + LLM

Step 1+2 통합 프로젝트: 금융 이상거래 탐지 시스템

## 학습 목표

### Step 1: Python + AI 기초
- NumPy/Pandas를 활용한 금융 데이터 처리
- scikit-learn을 활용한 ML 이상거래 탐지
- 특성 엔지니어링 및 모델 평가

### Step 2: LLM API + 프롬프트 엔지니어링
- OpenAI/Claude API 활용
- Zero-shot, Few-shot, Chain-of-Thought 프롬프트
- Function Calling (Tool Use)

## 프로젝트 구조

```
financial-analysis/
├── app/
│   └── streamlit_app.py    # 통합 UI
├── src/
│   ├── data/
│   │   ├── generator.py     # 샘플 데이터 생성
│   │   └── preprocessor.py  # 전처리 (NumPy/Pandas)
│   ├── ml/
│   │   └── fraud_detector.py  # ML 모델 (scikit-learn)
│   ├── llm/
│   │   └── client.py        # LLM API 클라이언트
│   └── prompts/
│       ├── templates.py     # 프롬프트 템플릿
│       └── tools.py         # Function Calling 도구
├── requirements.txt
└── README.md
```

## 설치

```bash
cd financial-analysis
pip install -r requirements.txt
```

## 실행

```bash
# Streamlit UI 실행
streamlit run app/streamlit_app.py
```

## 주요 기능

### 1. 데이터 처리 (NumPy/Pandas)

```python
from src.data.generator import generate_transaction_data
from src.data.preprocessor import TransactionPreprocessor

# 샘플 데이터 생성
data = generate_transaction_data(n_samples=1000, fraud_ratio=0.1)

# 전처리 및 특성 추출
preprocessor = TransactionPreprocessor()
X, y, feature_names = preprocessor.prepare_ml_data(data)
```

### 2. ML 이상거래 탐지 (scikit-learn)

```python
from src.ml.fraud_detector import FraudDetector

# 모델 학습
model = FraudDetector(algorithm="random_forest")
model.fit(X, y, feature_names)

# 예측 및 평가
predictions = model.predict(X)
metrics = model.evaluate(X, y)
```

### 3. LLM API 호출

```python
from src.llm.client import create_llm_client

# 클라이언트 생성 (mock/openai/claude)
client = create_llm_client("openai")

# 채팅 요청
response = client.chat([
    {"role": "user", "content": "이 거래를 분석해주세요."}
])
```

### 4. 프롬프트 엔지니어링

```python
from src.prompts.templates import get_fraud_analysis_prompt

# Zero-shot: 예시 없이 직접 분석
prompt = get_fraud_analysis_prompt(transaction, ml_result, "zero_shot")

# Few-shot: 유사 예시 포함
prompt = get_fraud_analysis_prompt(transaction, ml_result, "few_shot")

# Chain-of-Thought: 단계별 추론
prompt = get_fraud_analysis_prompt(transaction, ml_result, "cot")
```

### 5. Function Calling

```python
from src.prompts.tools import FINANCIAL_TOOLS, ToolExecutor

# LLM에 도구 전달
response = client.chat(messages, tools=FINANCIAL_TOOLS)

# 도구 실행
executor = ToolExecutor()
result = executor.execute("analyze_transaction", {"transaction_id": "TXN001", "amount": 5000000})
```

## 환경 변수

LLM API를 사용하려면 `.env` 파일에 API 키를 설정하세요:

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

API 키 없이 테스트하려면 `mock` 프로바이더를 사용하세요.

## 기술 스택

| 분야 | 기술 |
|------|------|
| 데이터 처리 | NumPy, Pandas |
| 머신러닝 | scikit-learn |
| LLM API | OpenAI, Anthropic Claude |
| UI | Streamlit |

## 프롬프트 기법 비교

| 기법 | 설명 | 토큰 | 정확도 |
|------|------|------|--------|
| Zero-shot | 예시 없이 직접 질문 | 적음 | 보통 |
| Few-shot | 유사 예시 2-3개 포함 | 많음 | 높음 |
| CoT | 단계별 추론 유도 | 많음 | 높음 |
