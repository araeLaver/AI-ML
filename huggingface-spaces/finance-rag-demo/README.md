---
title: Finance RAG Demo
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: mit
---

# Finance RAG Pro

**Premium Portfolio-Level Financial AI Dashboard**

Production-Grade RAG 시스템 | 하이브리드 검색 + Groq LLM + Interactive Charts

## Live Demo

👉 **[https://huggingface.co/spaces/downkim/finance-rag-demo](https://huggingface.co/spaces/downkim/finance-rag-demo)**

## Features

### Core RAG Features
| 기능 | 설명 |
|:---|:---|
| **Hybrid Search** | Vector + BM25 + RRF 결합 검색 |
| **Groq LLM** | 빠른 응답 (2-3초) |
| **50+ 샘플 문서** | DART 스타일 금융 리포트 |
| **문서 업로드** | PDF/TXT 파일 지원 |
| **Re-ranking** | 키워드 기반 재정렬 |

### Premium UI Features
| 기능 | 설명 |
|:---|:---|
| **Dark Theme** | GitHub-style 프리미엄 다크 테마 |
| **Chat History** | 대화 히스토리 저장/조회/삭제 |
| **Export** | PDF/CSV 다운로드 (fpdf2) |
| **Plotly Charts** | 인터랙티브 캔들스틱 차트 |
| **실시간 시세** | yfinance 연동 + 볼륨 차트 |

## Tech Stack

```
Frontend:  Streamlit + Premium Dark Theme
Charts:    Plotly (Candlestick + Volume)
Search:    Hybrid (Vector + BM25 + RRF)
LLM:       Groq API (llama-3.1-8b-instant)
Embedding: HuggingFace Inference API
Export:    fpdf2 (PDF) + CSV
Data:      yfinance (실시간 주가)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Query     │  │   Hybrid    │  │    Groq LLM     │ │
│  │   Input     │─▶│   Search    │─▶│    Response     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                          │                              │
│         ┌────────────────┼────────────────┐            │
│         ▼                ▼                ▼            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Vector    │  │    BM25     │  │  Re-ranker  │    │
│  │   Store     │  │   Search    │  │  (Keyword)  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
finance-rag-demo/
├── app.py                 # Streamlit 메인 앱 (Premium Dark Theme)
├── config.py              # 설정 관리 (UIConfig 포함)
├── requirements.txt       # 의존성
├── rag/                   # RAG 핵심 모듈
│   ├── llm_provider.py   # Groq API 연동
│   ├── vectorstore.py    # In-Memory Vector Store
│   ├── bm25.py           # BM25 키워드 검색
│   ├── hybrid_search.py  # RRF 하이브리드 검색
│   └── reranker.py       # Re-ranking
├── data/                  # 데이터 관리
│   ├── sample_docs.py    # 50+ 샘플 문서
│   └── document_loader.py # PDF/TXT 업로드
└── utils/
    ├── tokenizer.py       # 2-gram 토크나이저
    ├── session_manager.py # 대화 히스토리 관리
    └── export_utils.py    # PDF/CSV 내보내기
```

## Sample Questions

- 삼성전자 4분기 실적은?
- HBM 시장 전망은?
- 2025년 금리 전망
- 네이버 AI 사업 현황
- 비트코인 전망

## Environment Variables

HuggingFace Spaces Secrets에 설정:

| Key | Required | Description |
|:---|:---|:---|
| `GROQ_API_KEY` | Yes | Groq API 키 (무료) |
| `HF_TOKEN` | No | HuggingFace 토큰 (임베딩 API) |

## Local Development

```bash
# 클론
git clone https://huggingface.co/spaces/downkim/finance-rag-demo

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export GROQ_API_KEY=your_key

# 실행
streamlit run app.py
```

## Performance

| 지표 | 수치 |
|:---|:---|
| 응답 시간 | 2-3초 |
| 샘플 문서 | 50+ |
| 메모리 사용 | ~900MB |

## Author

**Kim Dawoon** - Backend Developer (9 years) → AI/ML Engineer

- [GitHub](https://github.com/araeLaver)
- [Portfolio](https://github.com/araeLaver/AI-ML)
