export interface ChecklistItem {
  id: string;
  title: string;
  completed: boolean;
}

export interface Step {
  id: number;
  title: string;
  duration: string;
  description: string;
  skills: string[];
  resources: { name: string; url: string; type: 'free' | 'paid' }[];
  checklist: ChecklistItem[];
  projects: string[];
}

export interface Career {
  company: string;
  period: string;
  role: string;
  description: string;
  skills: string[];
}

export interface Profile {
  name: string;
  title: string;
  goal: string;
  currentSkills: string[];
  targetPosition: string;
  targetSalary: string;
  timeline: string;
}

export const profile: Profile = {
  name: "김다운",
  title: "Backend Developer → AI Platform Engineer",
  goal: "18개월 후 연봉 1억+ AI Engineer",
  currentSkills: ["Java/Spring", "Node.js", "PostgreSQL", "Docker", "데이터 파이프라인"],
  targetPosition: "AI Platform Engineer / MLOps Lead",
  targetSalary: "1.2억 ~ 2억+ (글로벌 리모트 시 $150K-$300K)",
  timeline: "18개월",
};

export const careers: Career[] = [
  {
    company: "헥토데이터",
    period: "2022.10 - 2025.09",
    role: "파트장",
    description: "데이터 수집 시스템 설계, 네트워크 패킷 분석, 데이터 파이프라인 구축",
    skills: ["JavaScript", "Node.js", "MySQL", "데이터 파이프라인"],
  },
  {
    company: "융정보통신",
    period: "2020.10 - 2022.09",
    role: "대리",
    description: "팩토리온 시스템 개발, 15만건 데이터 마이그레이션, ESB 연계 구축",
    skills: ["Java", "Spring", "PostgreSQL", "ESB/ESI"],
  },
  {
    company: "티소프트",
    period: "2020.03 - 2020.09",
    role: "대리",
    description: "IBK 기업은행 전자약정 시스템, 르노캐피탈 FO 구축",
    skills: ["Java", "Spring Security", "Vue.js", "금융 보안"],
  },
];

export const roadmapSteps: Step[] = [
  {
    id: 1,
    title: "Python + AI 기초",
    duration: "1-2개월",
    description: "Python 숙달 + AI/ML 기본 개념 이해",
    skills: ["Python", "NumPy", "Pandas", "ML 기초", "LLM 기본"],
    resources: [
      { name: "점프 투 파이썬", url: "https://wikidocs.net/book/1", type: "free" },
      { name: "Andrew Ng ML Course", url: "https://www.coursera.org/learn/machine-learning", type: "free" },
      { name: "패스트캠퍼스 AI/ML", url: "https://fastcampus.co.kr", type: "paid" },
    ],
    checklist: [
      { id: "1-1", title: "Python 기초 문법", completed: false },
      { id: "1-2", title: "데코레이터, 제너레이터", completed: false },
      { id: "1-3", title: "asyncio 비동기 프로그래밍", completed: false },
      { id: "1-4", title: "NumPy 기본 연산", completed: false },
      { id: "1-5", title: "Pandas DataFrame 조작", completed: false },
      { id: "1-6", title: "ML 기초 개념 (지도/비지도학습)", completed: false },
      { id: "1-7", title: "트랜스포머 아키텍처 이해", completed: false },
    ],
    projects: ["금융 데이터 전처리 스크립트", "이상 거래 탐지 ML 모델 (기초)"],
  },
  {
    id: 2,
    title: "LLM API + 프롬프트 엔지니어링",
    duration: "1-2개월",
    description: "주요 LLM API 활용 + 효과적인 프롬프트 작성",
    skills: ["OpenAI API", "Claude API", "프롬프트 기법", "Function Calling"],
    resources: [
      { name: "OpenAI Cookbook", url: "https://github.com/openai/openai-cookbook", type: "free" },
      { name: "Anthropic Prompt Guide", url: "https://docs.anthropic.com", type: "free" },
      { name: "DeepLearning.AI Prompt Engineering", url: "https://www.deeplearning.ai", type: "free" },
    ],
    checklist: [
      { id: "2-1", title: "OpenAI API 키 발급 및 설정", completed: false },
      { id: "2-2", title: "Anthropic Claude API 사용", completed: false },
      { id: "2-3", title: "Zero-shot 프롬프팅", completed: false },
      { id: "2-4", title: "Few-shot 프롬프팅", completed: false },
      { id: "2-5", title: "Chain-of-Thought 프롬프팅", completed: false },
      { id: "2-6", title: "시스템 프롬프트 설계", completed: false },
      { id: "2-7", title: "Function Calling 구현", completed: false },
      { id: "2-8", title: "멀티모달 (이미지 분석)", completed: false },
    ],
    projects: ["금융 문서 분석 챗봇", "금융 데이터 조회 API 봇"],
  },
  {
    id: 3,
    title: "RAG 시스템",
    duration: "2-3개월",
    description: "RAG 파이프라인 설계 및 구축 (가장 중요!)",
    skills: ["임베딩", "벡터 DB", "LangChain", "LlamaIndex", "검색 최적화"],
    resources: [
      { name: "LangChain 공식 문서", url: "https://python.langchain.com", type: "free" },
      { name: "Udemy RAG Masterclass", url: "https://www.udemy.com", type: "paid" },
      { name: "DeepLearning.AI Building Systems", url: "https://www.deeplearning.ai", type: "free" },
    ],
    checklist: [
      { id: "3-1", title: "OpenAI 임베딩 API 사용", completed: false },
      { id: "3-2", title: "오픈소스 임베딩 모델 (BGE)", completed: false },
      { id: "3-3", title: "ChromaDB 설치 및 사용", completed: false },
      { id: "3-4", title: "문서 추가, 검색, 삭제", completed: false },
      { id: "3-5", title: "재귀적 문자 분할 청킹", completed: false },
      { id: "3-6", title: "Hybrid Search 구현", completed: false },
      { id: "3-7", title: "Re-Ranker 적용", completed: false },
      { id: "3-8", title: "LangChain RAG 파이프라인", completed: false },
    ],
    projects: ["금융 문서 RAG 시스템 (포트폴리오 #1)"],
  },
  {
    id: 4,
    title: "AI Agent 개발",
    duration: "2-3개월",
    description: "자율적으로 작업을 수행하는 AI 에이전트 구축",
    skills: ["LangGraph", "멀티 에이전트", "Tool Use", "MCP"],
    resources: [
      { name: "LangGraph 문서", url: "https://langchain-ai.github.io/langgraph/", type: "free" },
      { name: "AutoGen 문서", url: "https://microsoft.github.io/autogen/", type: "free" },
    ],
    checklist: [
      { id: "4-1", title: "ReAct 패턴 이해", completed: false },
      { id: "4-2", title: "Tool 정의 및 사용", completed: false },
      { id: "4-3", title: "LangGraph StateGraph 구성", completed: false },
      { id: "4-4", title: "조건부 라우팅", completed: false },
      { id: "4-5", title: "멀티 에이전트 역할 분담", completed: false },
      { id: "4-6", title: "커스텀 Tool 개발", completed: false },
      { id: "4-7", title: "MCP 프로토콜 이해", completed: false },
    ],
    projects: ["자동 코드 리뷰 AI Agent (포트폴리오 #2)"],
  },
  {
    id: 5,
    title: "MLOps + 모델 서빙",
    duration: "2-3개월",
    description: "ML 모델 배포/운영/모니터링 (핵심 차별화 영역)",
    skills: ["FastAPI", "MLflow", "Docker", "Kubernetes", "클라우드 AI"],
    resources: [
      { name: "Coursera MLOps Specialization", url: "https://www.coursera.org", type: "free" },
      { name: "Made With ML", url: "https://madewithml.com", type: "free" },
      { name: "AWS ML Specialty", url: "https://aws.amazon.com/certification/", type: "paid" },
    ],
    checklist: [
      { id: "5-1", title: "FastAPI ML API 구현", completed: false },
      { id: "5-2", title: "배치 예측 엔드포인트", completed: false },
      { id: "5-3", title: "MLflow 실험 추적", completed: false },
      { id: "5-4", title: "모델 레지스트리 사용", completed: false },
      { id: "5-5", title: "Docker 이미지 빌드", completed: false },
      { id: "5-6", title: "K8s 배포", completed: false },
      { id: "5-7", title: "AWS SageMaker 또는 GCP Vertex AI", completed: false },
      { id: "5-8", title: "Prometheus/Grafana 모니터링", completed: false },
      { id: "5-9", title: "드리프트 감지", completed: false },
    ],
    projects: ["End-to-End MLOps 파이프라인 (포트폴리오 #3)"],
  },
  {
    id: 6,
    title: "Fine-tuning + 고급 최적화",
    duration: "2-3개월",
    description: "모델 커스터마이징 + 성능/비용 최적화",
    skills: ["LoRA/QLoRA", "Hugging Face", "양자화", "vLLM"],
    resources: [
      { name: "Hugging Face Course", url: "https://huggingface.co/course", type: "free" },
      { name: "PEFT 문서", url: "https://huggingface.co/docs/peft", type: "free" },
    ],
    checklist: [
      { id: "6-1", title: "Transformers 기본 사용", completed: false },
      { id: "6-2", title: "Datasets 로드 및 처리", completed: false },
      { id: "6-3", title: "LoRA 개념 및 적용", completed: false },
      { id: "6-4", title: "QLoRA 사용", completed: false },
      { id: "6-5", title: "SFTTrainer 사용", completed: false },
      { id: "6-6", title: "양자화 (4-bit, 8-bit)", completed: false },
      { id: "6-7", title: "vLLM 사용", completed: false },
    ],
    projects: ["금융 도메인 LLM 파인튜닝"],
  },
];

export const portfolioProjects = [
  {
    id: 1,
    title: "금융 문서 RAG 시스템",
    description: "금융 보고서/약관 PDF를 학습하여 질의응답하는 시스템",
    tech: ["LangChain", "ChromaDB", "OpenAI", "FastAPI"],
    features: ["금융 도메인 특화", "출처 명시", "정확도 측정"],
    status: "planned",
  },
  {
    id: 2,
    title: "자동 코드 리뷰 AI Agent",
    description: "GitHub PR을 자동 분석하여 코드 리뷰 생성",
    tech: ["LangGraph", "GitHub API", "Claude API"],
    features: ["멀티 에이전트 (보안+성능+스타일)", "자동 코멘트"],
    status: "planned",
  },
  {
    id: 3,
    title: "MLOps 파이프라인",
    description: "데이터 수집 → 학습 → 배포 → 모니터링 자동화",
    tech: ["MLflow", "Kubeflow", "AWS SageMaker"],
    features: ["A/B 테스트", "드리프트 감지", "자동 재학습"],
    status: "planned",
  },
  {
    id: 4,
    title: "금융 이상탐지 시스템",
    description: "실시간 거래 데이터에서 이상 패턴 탐지",
    tech: ["Kafka", "Spark", "ML 모델", "Grafana"],
    features: ["실시간 처리", "금융 도메인 특화", "대시보드"],
    status: "planned",
  },
];

export const certifications = [
  { name: "AWS Cloud Practitioner", target: "3개월", difficulty: "하", cost: "$100" },
  { name: "AWS Solutions Architect Associate", target: "6개월", difficulty: "중", cost: "$150" },
  { name: "AWS Machine Learning - Specialty", target: "12개월", difficulty: "상", cost: "$300" },
  { name: "GCP Professional ML Engineer", target: "15개월", difficulty: "상", cost: "$200" },
  { name: "CKA (Kubernetes Administrator)", target: "18개월", difficulty: "중상", cost: "$395" },
];
