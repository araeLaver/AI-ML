# Financial LLM Fine-tuning

금융 도메인 특화 LLM Fine-tuning 프로젝트 - LoRA/QLoRA 기반

## 프로젝트 구조

```
financial-finetuning/
├── api/
│   └── server.py           # FastAPI 추론 서버
├── app/
│   └── streamlit_app.py    # Streamlit 데모 UI
├── configs/
│   └── training_config.yaml # 학습 설정
├── src/
│   ├── data/
│   │   ├── financial_instructions.py  # 금융 도메인 학습 데이터
│   │   └── prepare_dataset.py         # 데이터셋 준비
│   ├── inference/
│   │   └── inference_engine.py        # 추론 엔진
│   └── training/
│       └── train_lora.py              # LoRA 학습 파이프라인
├── tests/
│   ├── test_data.py
│   ├── test_training.py
│   └── test_inference.py
├── requirements.txt
└── README.md
```

## 주요 기능

### 1. 금융 도메인 데이터셋 (100+ 샘플)

6개 카테고리로 구성된 고품질 한국어 금융 Instruction 데이터셋:

| 카테고리 | 샘플 수 | 설명 |
|---------|--------|------|
| 이상거래 탐지 (Fraud Detection) | 15 | 보이스피싱, 카드 도용, 계좌 이상 탐지 |
| 투자 분석 (Investment Analysis) | 20 | 주식/ETF 분석, 포트폴리오 구성, 섹터 분석 |
| 금융 상품 설명 (Product Explanation) | 16 | 예금, 펀드, ELS, 보험 상품 설명 |
| 리스크 평가 (Risk Assessment) | 15 | VaR, 신용리스크, 유동성 리스크 분석 |
| 시장 분석 (Market Analysis) | 17 | 금리, 환율, 경기 사이클 분석 |
| 금융 용어 설명 (Term Explanation) | 17 | PER, ROE, 듀레이션 등 용어 해설 |

### 2. LoRA/QLoRA Fine-tuning
- Parameter-Efficient Fine-Tuning (PEFT)
- 4-bit 양자화 (QLoRA)
- Hugging Face Transformers 통합
- TRL SFTTrainer 활용
- Early Stopping 및 체크포인트 저장
- 학습 로그 및 메트릭 추적

### 3. 추론 API
- FastAPI 기반 REST API
- 스트리밍 응답 지원
- 금융 특화 엔드포인트 (이상거래, 투자분석, 상품설명)
- Pydantic 기반 요청/응답 검증
- Thread-safe 추론 엔진
- LRU 캐싱 및 에러 핸들링

### 4. 데모 UI
- Streamlit 기반 인터랙티브 UI
- 데이터셋 탐색 및 통계
- 실시간 추론 테스트
- 세션 상태 관리 및 결과 내보내기

## 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 데이터셋 준비

```python
from src.data import FinancialInstructionDataset

# 데이터셋 생성
dataset = FinancialInstructionDataset()

# 통계 확인
stats = dataset.get_statistics()
print(f"Total samples: {stats['total_samples']}")

# 학습/평가 분할
train_ds = dataset.get_train_dataset()
eval_ds = dataset.get_eval_dataset()
```

### 모델 학습

```bash
# 설정 파일로 학습 실행
python -m src.training.train_lora --config configs/training_config.yaml

# 드라이런 (설정 확인만)
python -m src.training.train_lora --config configs/training_config.yaml --dry-run
```

```python
from src.training import FinancialLoRATrainer

# 트레이너 초기화
trainer = FinancialLoRATrainer(config_path="configs/training_config.yaml")

# 설정 및 학습
trainer.setup()
trainer.train()
trainer.save_model("./outputs/financial-lora")
```

### 추론

```python
from src.inference import FinancialLLMInference

# 추론 엔진 초기화
inference = FinancialLLMInference(
    base_model="beomi/Llama-3-Open-Ko-8B",
    adapter_path="./outputs/lora_adapter"
)
inference.load()

# 텍스트 생성
response = inference.generate(
    instruction="삼성전자 주식의 투자 전망을 분석해주세요.",
    temperature=0.7,
)
print(response)

# 스트리밍 생성
for token in inference.generate_stream("금리 인상의 영향을 설명해주세요."):
    print(token, end="", flush=True)
```

### API 서버 실행

```bash
# 서버 시작
python api/server.py

# 또는 uvicorn으로 실행
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

API 엔드포인트:
- `GET /health` - 헬스 체크
- `GET /model/info` - 모델 정보
- `POST /model/load` - 모델 로드
- `POST /generate` - 텍스트 생성
- `POST /generate/stream` - 스트리밍 생성
- `POST /financial/fraud-detection` - 이상거래 탐지
- `POST /financial/investment-analysis` - 투자 분석
- `POST /financial/product-explanation` - 상품 설명

### Streamlit UI 실행

```bash
streamlit run app/streamlit_app.py
```

## 학습 설정

`configs/training_config.yaml`에서 설정 조정:

```yaml
# 모델 설정
model:
  name: "beomi/Llama-3-Open-Ko-8B"
  max_seq_length: 2048

# LoRA 설정
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

# 양자화 설정 (QLoRA)
quantization:
  enabled: true
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"

# 학습 설정
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
```

## 지원 모델

- `beomi/Llama-3-Open-Ko-8B` (권장)
- `beomi/llama-2-ko-7b`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`

## Docker 실행

```bash
# GPU 환경 - Streamlit 데모
docker-compose up streamlit

# CPU 환경 - Streamlit 데모 (개발/테스트용)
docker-compose --profile cpu up streamlit-cpu

# API 서버 (GPU 필요)
docker-compose up api

# 학습 실행 (GPU 필요)
docker-compose --profile train up train

# 테스트 실행
docker-compose --profile test run test
```

## 테스트

```bash
# 전체 테스트
pytest tests/ -v

# 특정 테스트
pytest tests/test_data.py -v
pytest tests/test_training.py -v
pytest tests/test_inference.py -v
```

## 요구사항

- Python 3.9+
- CUDA 11.7+ (GPU 학습 시)
- 16GB+ VRAM (QLoRA 사용 시 12GB+)

## 라이선스

MIT License
