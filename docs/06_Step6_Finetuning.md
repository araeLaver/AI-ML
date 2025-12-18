# Step 6: Fine-tuning + 고급 최적화 (2-3개월)

## 목표
> 모델 커스터마이징 + 성능/비용 최적화

## 언제 Fine-tuning이 필요한가?

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Fine-tuning vs RAG vs Prompt                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Prompt Engineering:                                                │
│    - 빠른 적용, 비용 낮음                                           │
│    - 간단한 작업에 적합                                             │
│                                                                     │
│  RAG:                                                               │
│    - 도메인 지식 주입                                               │
│    - 실시간 데이터 필요 시                                          │
│    - 대부분의 기업 use case에 적합                                  │
│                                                                     │
│  Fine-tuning:                                                       │
│    - 특정 스타일/형식 학습 필요                                     │
│    - 대량 요청으로 비용 최적화 필요                                 │
│    - 매우 특화된 도메인 (의료, 법률)                                │
│                                                                     │
│  ※ 대부분 RAG로 해결 가능. Fine-tuning은 마지막 수단               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 학습 순서

### Week 1-2: Hugging Face 기초

#### Transformers 라이브러리

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 로드
model_name = "beomi/llama-2-ko-7b"  # 한국어 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 텍스트 생성
def generate(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 사용
response = generate("금융 이상 거래 탐지 방법은")
print(response)
```

#### Datasets 라이브러리

```python
from datasets import Dataset, load_dataset

# 커스텀 데이터셋 생성
data = {
    "instruction": [
        "다음 거래가 이상 거래인지 판단해주세요.",
        "금융 사기 유형을 설명해주세요."
    ],
    "input": [
        "금액: 5000만원, 시간: 새벽 3시, 위치: 해외",
        ""
    ],
    "output": [
        "이 거래는 이상 거래로 판단됩니다. 이유: 1) 평소 대비 매우 높은 금액...",
        "금융 사기의 주요 유형은 다음과 같습니다: 1) 피싱..."
    ]
}

dataset = Dataset.from_dict(data)

# 학습/검증 분할
dataset = dataset.train_test_split(test_size=0.1)

# Hugging Face Hub 업로드
dataset.push_to_hub("username/financial-instruction-dataset")
```

---

### Week 3-4: PEFT/LoRA (효율적 파인튜닝)

#### LoRA란?

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LoRA 개념                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  전체 파인튜닝:  모든 파라미터 업데이트 (수십억 개)                  │
│                 → GPU 메모리 많이 필요, 비용 높음                   │
│                                                                     │
│  LoRA:  작은 어댑터만 학습 (수백만 개)                              │
│         → GPU 메모리 적게 사용, 비용 낮음                           │
│         → 원본 모델 가중치는 동결                                   │
│                                                                     │
│  ┌─────────────┐                                                   │
│  │ 원본 모델    │ (동결)                                           │
│  │  W (frozen) │                                                   │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ↓                                                           │
│  ┌─────────────┐                                                   │
│  │ LoRA 어댑터 │ (학습)                                            │
│  │  A × B      │ (rank << dim)                                     │
│  └─────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### LoRA 구현

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "beomi/llama-2-ko-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,                      # rank (작을수록 경량)
    lora_alpha=32,             # 스케일링 팩터
    target_modules=[           # 적용할 모듈
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# PEFT 모델 생성
peft_model = get_peft_model(model, lora_config)

# 학습 가능 파라미터 확인
peft_model.print_trainable_parameters()
# 출력: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

#### QLoRA (양자화 + LoRA)

```python
from transformers import BitsAndBytesConfig

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 4-bit 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "beomi/llama-2-ko-7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# QLoRA 적용
peft_model = get_peft_model(model, lora_config)
```

---

### Week 5-6: SFT (Supervised Fine-Tuning)

#### 데이터 포맷팅

```python
def format_instruction(sample):
    """Instruction 형식으로 포맷팅"""
    if sample['input']:
        return f"""### 지시사항:
{sample['instruction']}

### 입력:
{sample['input']}

### 응답:
{sample['output']}"""
    else:
        return f"""### 지시사항:
{sample['instruction']}

### 응답:
{sample['output']}"""

# 데이터셋 변환
dataset = dataset.map(lambda x: {"text": format_instruction(x)})
```

#### SFTTrainer로 학습

```python
from trl import SFTTrainer, SFTConfig

# 학습 설정
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    max_seq_length=2048,
)

# 트레이너 생성
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./fine-tuned-model")

# Hugging Face Hub 업로드
trainer.push_to_hub("username/financial-llm-lora")
```

---

### Week 7-8: 모델 경량화

#### 양자화 (Quantization)

```python
from transformers import AutoModelForCausalLM
import torch

# 8-bit 양자화
model_8bit = AutoModelForCausalLM.from_pretrained(
    "model-path",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit 양자화 (더 경량)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "model-path",
    load_in_4bit=True,
    device_map="auto"
)

# GPTQ 양자화 (추론 최적화)
from auto_gptq import AutoGPTQForCausalLM

model_gptq = AutoGPTQForCausalLM.from_quantized(
    "model-path-gptq",
    device="cuda:0",
    use_safetensors=True
)
```

#### GGUF 변환 (로컬 실행용)

```bash
# llama.cpp로 변환
python convert.py ./model --outfile ./model.gguf --outtype f16

# 양자화
./quantize ./model.gguf ./model-q4_k_m.gguf q4_k_m
```

```python
# Python에서 GGUF 사용 (llama-cpp-python)
from llama_cpp import Llama

llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=8
)

output = llm("금융 이상 거래 탐지 방법은", max_tokens=200)
print(output['choices'][0]['text'])
```

---

### Week 9-10: 성능 최적화

#### vLLM (고성능 추론)

```python
from vllm import LLM, SamplingParams

# vLLM 엔진 로드
llm = LLM(
    model="beomi/llama-2-ko-7b",
    tensor_parallel_size=1,  # GPU 수
    dtype="float16"
)

# 배치 추론
prompts = [
    "금융 이상 거래 탐지 방법은",
    "신용 점수 계산 방법은",
    "리스크 관리의 핵심은"
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=200
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

#### TensorRT-LLM (최고 성능)

```python
# TensorRT 엔진 빌드 (CLI)
# trtllm-build --checkpoint_dir ./model --output_dir ./engine

from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./engine")

outputs = runner.generate(
    batch_input_ids=input_ids,
    max_new_tokens=200,
    temperature=0.7
)
```

#### 비용 최적화 전략

| 전략 | 효과 | 적용 시점 |
|------|------|----------|
| 모델 캐싱 | 지연시간 50% 감소 | 반복 요청 |
| 배치 처리 | 처리량 5-10배 증가 | 대량 요청 |
| 4-bit 양자화 | 메모리 75% 절감 | 리소스 제한 |
| vLLM | 처리량 2-5배 증가 | 프로덕션 |
| 캐시 프롬프트 | 토큰 50% 절감 | 유사 요청 |

---

## 오픈소스 LLM 가이드

| 모델 | 크기 | 특징 | 용도 |
|------|------|------|------|
| Llama 3.1 | 8B/70B/405B | Meta, 범용 | 일반 |
| Mistral | 7B | 효율적, 빠름 | 경량 배포 |
| Qwen 2.5 | 7B/72B | 다국어, 코딩 | 한국어 OK |
| DeepSeek | 7B/67B | 코딩 특화 | 개발 도구 |

### 한국어 특화 모델

| 모델 | 기반 | 특징 |
|------|------|------|
| beomi/llama-2-ko-7b | Llama 2 | 한국어 사전학습 |
| KULLM3 | Llama 3.1 | 고려대 개발 |
| Bllossom | Llama | KAIST 개발 |

---

## 실습 프로젝트

### 프로젝트: 금융 도메인 LLM 파인튜닝

```
financial_llm/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리된 데이터
│   └── prepare_data.py   # 데이터 준비 스크립트
├── training/
│   ├── config.yaml       # 학습 설정
│   ├── train_lora.py     # LoRA 학습
│   └── evaluate.py       # 평가
├── inference/
│   ├── api.py            # 추론 API
│   └── optimize.py       # 최적화 (vLLM)
├── notebooks/
│   └── analysis.ipynb    # 결과 분석
└── requirements.txt
```

---

## 체크리스트

### Hugging Face
- [ ] Transformers 기본 사용
- [ ] Datasets 로드 및 처리
- [ ] Hub 업로드/다운로드

### PEFT/LoRA
- [ ] LoRA 개념 이해
- [ ] LoRA 설정 및 적용
- [ ] QLoRA 사용

### Fine-tuning
- [ ] 데이터 포맷팅
- [ ] SFTTrainer 사용
- [ ] 학습 모니터링

### 최적화
- [ ] 양자화 (4-bit, 8-bit)
- [ ] GGUF 변환
- [ ] vLLM 사용

---

## 학습 완료 후

6단계를 모두 완료하면 **AI Platform Engineer / MLOps Engineer**로서의 역량을 갖추게 됩니다.

**최종 스킬셋**:
1. Python + AI/ML 기초
2. LLM API + 프롬프트 엔지니어링
3. RAG 시스템 설계/구축
4. AI Agent 개발
5. MLOps (배포/모니터링)
6. Fine-tuning + 최적화

**목표 연봉**: 1억+ (국내), $150K+ (글로벌)
