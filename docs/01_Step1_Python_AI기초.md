# Step 1: Python + AI 기초 (1-2개월)

## 목표
> Python 숙달 + AI/ML 기본 개념 이해

## 왜 Python인가?
- AI/ML 생태계의 90% 이상이 Python 기반
- NumPy, Pandas, TensorFlow, PyTorch 모두 Python
- JavaScript 개발자라면 문법 유사성으로 빠른 학습 가능

---

## 학습 순서

### Week 1-2: Python 기초 (빠르게 통과)

**김다운님은 JavaScript 경험이 있으므로 기초는 빠르게 넘어갑니다**

| 일차 | 학습 내용 | 실습 |
|------|----------|------|
| Day 1-2 | 변수, 자료형, 조건문, 반복문 | 점프 투 파이썬 1-4장 |
| Day 3-4 | 함수, 클래스, 모듈 | 점프 투 파이썬 5-6장 |
| Day 5-7 | 파일 I/O, 예외 처리 | 간단한 데이터 처리 스크립트 작성 |

**학습 자료**:
- [무료] 점프 투 파이썬: https://wikidocs.net/book/1

**실습 과제**:
```python
# 과제 1: JSON 파일 읽어서 데이터 정제하기
# - 헥토데이터 경험을 Python으로 재현
# - requests로 API 호출 → JSON 파싱 → 정제 → 저장
```

---

### Week 3-4: Python 심화

**이 부분이 핵심! AI 개발에 필수적인 고급 기능들**

#### 1. 데코레이터 (Decorator)
```python
# ML에서 많이 사용: 로깅, 타이밍, 캐싱
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 실행 시간: {time.time() - start:.2f}초")
        return result
    return wrapper

@timer
def train_model():
    # 모델 학습 코드
    pass
```

#### 2. 제너레이터 (Generator)
```python
# 대용량 데이터 처리 시 메모리 효율
def data_loader(file_path, batch_size=32):
    """ML 학습용 배치 데이터 제너레이터"""
    batch = []
    with open(file_path) as f:
        for line in f:
            batch.append(process(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

#### 3. 비동기 프로그래밍 (asyncio)
```python
# LLM API 동시 호출 시 필수
import asyncio
import aiohttp

async def call_llm_api(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json={"prompt": prompt}) as resp:
            return await resp.json()

async def batch_inference(prompts):
    tasks = [call_llm_api(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

#### 4. Type Hints
```python
# 코드 가독성 + IDE 지원
from typing import List, Dict, Optional

def process_embeddings(
    texts: List[str],
    model: str = "text-embedding-ada-002"
) -> List[List[float]]:
    """텍스트를 임베딩 벡터로 변환"""
    pass
```

---

### Week 5-6: 데이터 처리 (NumPy, Pandas)

**헥토데이터 경험을 Python으로 전환하는 핵심**

#### NumPy 필수 개념
```python
import numpy as np

# 벡터 연산 (임베딩 처리에 필수)
embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

# 코사인 유사도 계산 (RAG 검색의 핵심)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 행렬 연산
attention_scores = np.matmul(query, key.T) / np.sqrt(dim)
```

#### Pandas 필수 개념
```python
import pandas as pd

# 데이터 로딩 및 정제
df = pd.read_csv("data.csv")
df = df.dropna()  # 결측치 제거
df = df[df['value'] > 0]  # 필터링

# 그룹별 통계 (ML 피처 엔지니어링)
features = df.groupby('category').agg({
    'value': ['mean', 'std', 'count']
})

# 데이터 변환
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
```

**실습 과제**:
```python
# 과제 2: 금융 거래 데이터 전처리
# - CSV 파일 로딩
# - 결측치 처리
# - 이상치 탐지 (표준편차 기반)
# - 피처 정규화
# - 학습/테스트 데이터 분할
```

---

### Week 7-8: ML 기초 개념

#### 지도학습 (Supervised Learning)
| 유형 | 설명 | 예시 |
|------|------|------|
| 분류 (Classification) | 카테고리 예측 | 스팸 메일 분류, 이상 거래 탐지 |
| 회귀 (Regression) | 연속값 예측 | 주가 예측, 신용 점수 |

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 평가
predictions = model.predict(X_test)
print(f"정확도: {accuracy_score(y_test, predictions)}")
```

#### 비지도학습 (Unsupervised Learning)
| 유형 | 설명 | 예시 |
|------|------|------|
| 클러스터링 | 그룹화 | 고객 세그먼트, 문서 분류 |
| 차원 축소 | 피처 압축 | 임베딩 시각화, 노이즈 제거 |

#### 딥러닝 기본 원리
```
입력층 → 은닉층(들) → 출력층
   ↓         ↓          ↓
 피처     특징 추출    예측값
```

---

### Week 7-8: LLM 기본 이해

#### 트랜스포머 아키텍처 (개념만 이해)
```
┌─────────────────────────────────────────┐
│           Transformer 구조               │
├─────────────────────────────────────────┤
│  Input → Embedding → Positional Encoding │
│              ↓                           │
│     Multi-Head Self-Attention            │
│              ↓                           │
│        Feed Forward Network              │
│              ↓                           │
│           Output                         │
└─────────────────────────────────────────┘
```

**핵심 개념**:
- **토큰화 (Tokenization)**: 텍스트를 숫자로 변환
- **임베딩 (Embedding)**: 토큰을 벡터로 변환
- **어텐션 (Attention)**: 중요한 부분에 집중
- **컨텍스트 윈도우**: 한 번에 처리할 수 있는 토큰 수

**GPT vs BERT**:
| 모델 | 특징 | 용도 |
|------|------|------|
| GPT | 다음 토큰 예측 (자기회귀) | 텍스트 생성, 챗봇 |
| BERT | 양방향 컨텍스트 이해 | 분류, 질의응답, 임베딩 |

---

## 학습 자료 총정리

### 무료 자료
| 자료 | 링크 | 용도 |
|------|------|------|
| 점프 투 파이썬 | wikidocs.net/book/1 | Python 기초 |
| Andrew Ng ML Course | Coursera (청강) | ML 기초 이론 |
| 3Blue1Brown 신경망 | YouTube | 딥러닝 직관적 이해 |

### 유료 자료
| 자료 | 가격 | 용도 |
|------|------|------|
| 패스트캠퍼스 AI/ML | ~30만원 | 종합 학습 |
| LLM 실전 AI 애플리케이션 개발 (한빛미디어) | ~3만원 | LLM 실무 |

---

## 체크리스트

### Python 기초
- [ ] 변수, 자료형, 조건문, 반복문
- [ ] 함수, 클래스, 모듈
- [ ] 파일 I/O, 예외 처리
- [ ] 가상환경 (venv, conda) 사용

### Python 심화
- [ ] 데코레이터 이해 및 활용
- [ ] 제너레이터 이해 및 활용
- [ ] asyncio 비동기 프로그래밍
- [ ] Type Hints 사용

### 데이터 처리
- [ ] NumPy 기본 연산
- [ ] Pandas DataFrame 조작
- [ ] 데이터 정제 및 전처리
- [ ] 데이터 시각화 (matplotlib/seaborn)

### ML 기초
- [ ] 지도학습 개념 이해
- [ ] 비지도학습 개념 이해
- [ ] scikit-learn 기본 사용
- [ ] 모델 평가 지표 이해

### LLM 기본
- [ ] 트랜스포머 아키텍처 개념
- [ ] 토큰화, 임베딩 개념
- [ ] GPT, BERT 차이점 이해

---

## 실습 프로젝트

### 미니 프로젝트: 금융 이상 거래 탐지 (ML 기초)

**목표**: scikit-learn으로 간단한 이상 탐지 모델 구현

```python
# 프로젝트 구조
financial_anomaly_detection/
├── data/
│   └── transactions.csv
├── notebooks/
│   └── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   └── preprocess.py
│   └── model.py
│   └── evaluate.py
└── requirements.txt
```

**구현 순서**:
1. 데이터 탐색 (EDA)
2. 데이터 전처리 (정규화, 피처 엔지니어링)
3. 모델 학습 (RandomForest, IsolationForest)
4. 모델 평가 (정확도, F1-score, ROC-AUC)
5. 결과 시각화

---

## 다음 단계
Step 1 완료 후 → **Step 2: LLM API + 프롬프트 엔지니어링** 으로 진행
