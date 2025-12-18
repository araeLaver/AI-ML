# Step 2: LLM API + 프롬프트 엔지니어링 (1-2개월)

## 목표
> 주요 LLM API 활용 + 효과적인 프롬프트 작성

## 왜 이 단계가 중요한가?
- 현업에서 가장 빠르게 적용 가능한 AI 스킬
- API만 알아도 대부분의 AI 애플리케이션 구축 가능
- 김다운님의 REST API 경험이 직접 활용됨

---

## 학습 순서

### Week 1-2: LLM API 기초

#### OpenAI API 시작하기

**설정**:
```bash
pip install openai python-dotenv
```

```python
# .env 파일
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

**기본 사용법**:
```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 기본 텍스트 생성
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "당신은 금융 전문가입니다."},
        {"role": "user", "content": "주식 투자의 기본 원칙을 알려주세요."}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

#### 주요 파라미터 이해

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `model` | 사용할 모델 | gpt-4o, gpt-4o-mini |
| `temperature` | 창의성 조절 (0-2) | 분석: 0.1-0.3, 창작: 0.7-1.0 |
| `max_tokens` | 최대 출력 토큰 | 용도에 따라 조절 |
| `top_p` | 확률 분포 제한 | 0.9-0.95 |
| `presence_penalty` | 새 주제 유도 | -2.0 ~ 2.0 |
| `frequency_penalty` | 반복 방지 | -2.0 ~ 2.0 |

#### Anthropic Claude API

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "금융 데이터 분석 코드를 작성해주세요."}
    ]
)

print(message.content[0].text)
```

#### Google Gemini API

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("금융 리포트를 요약해주세요.")
print(response.text)
```

---

### Week 3-4: 프롬프트 엔지니어링

#### 1. Zero-shot 프롬프팅
```python
# 예시 없이 바로 요청
prompt = """
다음 고객 리뷰의 감정을 분석하세요.
리뷰: "이 서비스는 정말 끔찍합니다. 다시는 이용하지 않겠습니다."
감정 (긍정/부정/중립):
"""
```

#### 2. Few-shot 프롬프팅
```python
# 예시를 제공하여 패턴 학습
prompt = """
다음 예시처럼 고객 리뷰의 감정을 분석하세요.

예시 1:
리뷰: "정말 좋은 경험이었습니다!"
감정: 긍정

예시 2:
리뷰: "보통이에요. 특별히 좋지도 나쁘지도 않아요."
감정: 중립

예시 3:
리뷰: "최악의 서비스입니다."
감정: 부정

분석할 리뷰: "배송은 빨랐지만 제품 품질이 기대 이하였습니다."
감정:
"""
```

#### 3. Chain-of-Thought (CoT)
```python
# 단계별 사고 과정 유도
prompt = """
다음 금융 거래가 이상 거래인지 분석하세요.

거래 정보:
- 금액: 5,000만원
- 시간: 새벽 3시
- 위치: 평소 거래지역에서 500km 떨어진 곳
- 이전 최대 거래금액: 100만원

단계별로 분석해주세요:
1. 금액 분석:
2. 시간 분석:
3. 위치 분석:
4. 이전 패턴과 비교:
5. 최종 판단:
"""
```

#### 4. 시스템 프롬프트 설계

```python
# 금융 분석 봇 시스템 프롬프트
system_prompt = """
당신은 10년 경력의 금융 데이터 분석가입니다.

## 역할
- 금융 데이터를 분석하고 인사이트를 제공합니다
- 이상 거래를 탐지하고 설명합니다
- 리스크를 평가하고 대응 방안을 제시합니다

## 응답 규칙
1. 항상 데이터 기반으로 분석합니다
2. 불확실한 정보는 명확히 밝힙니다
3. 수치를 포함하여 구체적으로 답변합니다
4. 전문 용어는 쉽게 설명을 덧붙입니다

## 출력 형식
분석 결과는 다음 형식으로 제공합니다:
- 요약: (1-2문장)
- 상세 분석: (bullet points)
- 권장 조치: (구체적인 행동)
- 신뢰도: (높음/중간/낮음)
"""
```

---

### Week 5-6: Function Calling (Tool Use)

**김다운님의 API 경험이 직접 활용되는 영역**

#### OpenAI Function Calling

```python
import json

# 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "주식의 현재 가격을 조회합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "주식 심볼 (예: AAPL, GOOGL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_transaction",
            "description": "거래 데이터를 분석하여 이상 여부를 판단합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "거래 금액"},
                    "time": {"type": "string", "description": "거래 시간"},
                    "location": {"type": "string", "description": "거래 위치"}
                },
                "required": ["amount", "time"]
            }
        }
    }
]

# API 호출
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "삼성전자 주가를 알려주세요"}],
    tools=tools,
    tool_choice="auto"
)

# 함수 호출 처리
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    # 실제 함수 실행
    if function_name == "get_stock_price":
        result = get_stock_price(arguments["symbol"])

    # 결과를 다시 LLM에 전달
    messages.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
```

#### 실습: 금융 데이터 조회 챗봇

```python
# 실제 API 연동 예시
def get_exchange_rate(currency_pair: str) -> dict:
    """환율 정보 조회"""
    # 실제로는 환율 API 호출
    return {"pair": currency_pair, "rate": 1350.50, "change": "+0.5%"}

def get_account_balance(account_id: str) -> dict:
    """계좌 잔액 조회"""
    # 실제로는 은행 API 호출
    return {"account_id": account_id, "balance": 1000000, "currency": "KRW"}

# Function Calling으로 자연어 → API 호출 자동화
```

---

### Week 7-8: 멀티모달 + 스트리밍

#### 이미지 분석 (Vision)

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 금융 문서 이미지 분석
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 금융 보고서에서 핵심 수치를 추출해주세요."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image('report.png')}"
                    }
                }
            ]
        }
    ]
)
```

#### 스트리밍 응답

```python
# 실시간 응답 출력 (UX 향상)
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "긴 분석 보고서를 작성해주세요."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 실습 프로젝트

### 프로젝트: 금융 문서 분석 챗봇

**목표**: LLM API를 활용한 금융 문서 질의응답 시스템

```python
# 프로젝트 구조
financial_chatbot/
├── app/
│   ├── __init__.py
│   ├── llm_client.py      # LLM API 래퍼
│   ├── tools.py           # Function Calling 도구
│   ├── prompts.py         # 프롬프트 템플릿
│   └── chat.py            # 챗봇 로직
├── tests/
│   └── test_chat.py
├── .env
├── requirements.txt
└── main.py
```

**핵심 기능**:
1. 금융 보고서 요약
2. 특정 수치 추출 (Function Calling)
3. 이상 거래 분석 (CoT)
4. 멀티턴 대화 지원

---

## 학습 자료

### 공식 문서 (필수)
| 자료 | 링크 |
|------|------|
| OpenAI Cookbook | github.com/openai/openai-cookbook |
| Anthropic Prompt Guide | docs.anthropic.com/claude/docs/intro-to-prompting |
| Google AI for Developers | ai.google.dev |

### 강의
| 자료 | 비용 | 특징 |
|------|------|------|
| DeepLearning.AI ChatGPT Prompt Engineering | 무료 | Andrew Ng, Isa Fulford |
| OpenAI API 튜토리얼 | 무료 | 공식 YouTube |

---

## 체크리스트

### LLM API 기초
- [ ] OpenAI API 키 발급 및 설정
- [ ] Anthropic Claude API 키 발급
- [ ] 기본 API 호출 코드 작성
- [ ] 파라미터 (temperature, max_tokens 등) 이해

### 프롬프트 엔지니어링
- [ ] Zero-shot 프롬프팅
- [ ] Few-shot 프롬프팅
- [ ] Chain-of-Thought 프롬프팅
- [ ] 시스템 프롬프트 설계

### Function Calling
- [ ] 함수 스키마 정의
- [ ] 도구 호출 및 결과 처리
- [ ] 여러 도구 조합

### 고급 기능
- [ ] 멀티모달 (이미지 분석)
- [ ] 스트리밍 응답
- [ ] 에러 핸들링 및 재시도 로직

---

## 비용 관리 팁

```python
# 토큰 사용량 추적
from tiktoken import encoding_for_model

def count_tokens(text, model="gpt-4o"):
    encoding = encoding_for_model(model)
    return len(encoding.encode(text))

# 비용 계산 (2024년 12월 기준)
# gpt-4o: 입력 $2.5/1M, 출력 $10/1M
# gpt-4o-mini: 입력 $0.15/1M, 출력 $0.6/1M
```

**초기 예산**: $5-10/월로 충분히 학습 가능

---

## 다음 단계
Step 2 완료 후 → **Step 3: RAG 시스템** 으로 진행 (가장 중요!)
