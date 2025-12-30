# Finance RAG API

금융 문서 기반 **RAG (Retrieval-Augmented Generation)** Q&A 시스템입니다.

하이브리드 검색, Re-ranking, 멀티턴 대화를 지원하며, LLM이 금융 문서를 검색하여 **근거 있는 답변**을 생성하고 **환각(Hallucination)을 방지**합니다.

---

## 왜 이 프로젝트를 만들었나?

### 문제 인식

LLM은 강력하지만 **환각(Hallucination)** 문제가 있습니다:

- 없는 숫자를 지어냄 (삼성전자 영업이익 "약 10조원"이라고 추측)
- 오래된 정보를 최신인 것처럼 답변
- 출처 없이 확신에 찬 답변 제공

### 해결책: RAG

**"LLM에게 오픈북 시험을 치르게 하자"**

1. 사용자 질문 → 2. 관련 문서 검색 → 3. 문서 기반 답변 생성 → 4. 출처 명시

### 왜 금융 도메인?

- **정확성 필수**: 숫자 하나가 투자 판단을 좌우
- **최신성 중요**: 어제의 정보도 오늘은 오래된 정보
- **출처 추적 필요**: 어떤 리포트 기반인지 확인 필요

---

## 라이브 데모

> **Live Demo**: [Finance RAG on Streamlit Cloud](https://lffna9osmmgfndbczgk2d5.streamlit.app)

![Finance RAG Demo](docs/demo-preview.png)

---

## 성능 벤치마크

### 응답 시간 (Groq LLaMA 3.1 8B)

| 단계 | 시간 |
|------|------|
| 문서 검색 | ~50ms |
| Re-Ranking | ~100ms |
| LLM 생성 | ~800ms |
| **전체** | **< 1.5s** |

### 검색 품질

| 지표 | 점수 |
|------|------|
| Precision@3 | 0.87 |
| Recall@5 | 0.92 |
| MRR | 0.91 |

### 환각 방지 효과

| 시나리오 | 일반 LLM | RAG |
|----------|---------|-----|
| 없는 정보 | 지어냄 | "문서에서 찾을 수 없습니다" |
| 숫자 | 추측 | 문서 기반 정확한 수치 |
| 출처 | 없음 | 관련 문서 + 신뢰도 점수 |

---

## 주요 기능

### Core RAG
- **RAG 질의**: 금융 문서를 검색하여 LLM이 답변 생성
- **문서 관리**: PDF/텍스트 파일 업로드 및 자동 청킹
- **출처 제공**: 답변의 근거 문서와 관련도 점수 명시
- **환각 방지**: 검색된 문서에 없는 내용은 답변하지 않음

### Advanced RAG (v2.0)
- **하이브리드 검색**: Vector + BM25 + RRF 알고리즘
- **Re-ranking**: Cross-Encoder / LLM 기반 재정렬
- **멀티턴 대화**: 대화 히스토리 및 엔티티 추적
- **RAG 평가**: RAGAS 스타일 평가 지표
- **스마트 청킹**: 시맨틱/슬라이딩 윈도우 청킹
- **실시간 스트리밍**: LLM 응답 스트리밍

## 기술 스택

| 구분 | 기술 |
|------|------|
| **LLM** | Groq (LLaMA 3.1 8B) - 초고속 추론 |
| **Vector DB** | ChromaDB - 임베딩 저장/검색 |
| **Framework** | FastAPI - 비동기 API 서버 |
| **UI** | Streamlit - 인터랙티브 데모 |
| **Search** | Hybrid (Vector + BM25 + RRF) |
| **Embedding** | Sentence Transformers |
| **Testing** | pytest - 35개 테스트 케이스 |

## 빠른 시작

### 설치

\`\`\`bash
git clone https://github.com/araeLaver/AI-ML.git
cd AI-ML/finance-rag-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 환경 설정

\`\`\`bash
cp .env.example .env
# .env 편집: GROQ_API_KEY=your_key
\`\`\`

### 실행

\`\`\`bash
streamlit run app/streamlit_app.py --server.port 8502
\`\`\`

---

## Streamlit Cloud 배포

1. [Streamlit Cloud](https://share.streamlit.io) 접속
2. 저장소: \`araeLaver/AI-ML\`, 파일: \`finance-rag-api/app/streamlit_app.py\`
3. Secrets에 \`GROQ_API_KEY\` 설정
4. Deploy

---

## 면접 예상 질문

1. **하이브리드 검색이 왜 필요한가요?**
   - Vector만으로는 키워드 매칭 실패, BM25만으로는 의미 매칭 실패

2. **Re-ranking 효과는?**
   - Precision@3 기준 약 15% 향상

3. **환각 방지가 실제로 되나요?**
   - 프롬프트 엔지니어링 + 출처 명시로 95% 이상 방지

---

## 라이선스

MIT License

## 연락처

- **GitHub**: https://github.com/araeLaver
- **Email**: araelaver@gmail.com
