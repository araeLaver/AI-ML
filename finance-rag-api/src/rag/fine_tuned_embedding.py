# -*- coding: utf-8 -*-
"""
Fine-tuned Embedding 모듈

금융 도메인에 특화된 임베딩 모델 학습 및 사용을 지원합니다.
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class FineTuneConfig:
    """Fine-tuning 설정

    Attributes:
        base_model: 기본 모델 이름
        output_path: 출력 경로
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        warmup_steps: 워밍업 스텝 수
        evaluation_steps: 평가 주기
        use_amp: Mixed Precision 사용 여부
    """
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    output_path: str = "./models/finance-embedding"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 500
    use_amp: bool = True

    # 금융 특화 모델 옵션
    FINANCE_MODELS = {
        "ko-sroberta": {
            "name": "jhgan/ko-sroberta-multitask",
            "description": "한국어 특화 문장 임베딩 모델",
            "dimensions": 768,
        },
        "multilingual-e5": {
            "name": "intfloat/multilingual-e5-base",
            "description": "다국어 E5 임베딩 모델",
            "dimensions": 768,
        },
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "description": "다국어 BGE 임베딩 모델",
            "dimensions": 1024,
        },
    }


@dataclass
class TrainingPair:
    """학습 쌍 데이터

    Attributes:
        query: 쿼리 텍스트
        positive: 관련 문서
        negative: 비관련 문서 (선택)
        score: 관련도 점수 (0.0~1.0)
    """
    query: str
    positive: str
    negative: Optional[str] = None
    score: float = 1.0


@dataclass
class TrainingDataset:
    """학습 데이터셋

    Attributes:
        pairs: 학습 쌍 목록
        name: 데이터셋 이름
        description: 설명
    """
    pairs: list[TrainingPair] = field(default_factory=list)
    name: str = "finance-dataset"
    description: str = ""

    def add_pair(
        self,
        query: str,
        positive: str,
        negative: Optional[str] = None,
        score: float = 1.0,
    ) -> None:
        """학습 쌍 추가"""
        self.pairs.append(TrainingPair(
            query=query,
            positive=positive,
            negative=negative,
            score=score,
        ))

    def __len__(self) -> int:
        return len(self.pairs)

    def to_json(self, path: str) -> None:
        """JSON 파일로 저장"""
        data = {
            "name": self.name,
            "description": self.description,
            "pairs": [
                {
                    "query": p.query,
                    "positive": p.positive,
                    "negative": p.negative,
                    "score": p.score,
                }
                for p in self.pairs
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingDataset":
        """JSON 파일에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
        )

        for pair_data in data.get("pairs", []):
            dataset.add_pair(
                query=pair_data["query"],
                positive=pair_data["positive"],
                negative=pair_data.get("negative"),
                score=pair_data.get("score", 1.0),
            )

        return dataset


class FinanceDatasetGenerator:
    """금융 도메인 학습 데이터셋 생성기"""

    # 샘플 쿼리-문서 쌍
    SAMPLE_PAIRS = [
        # 실적 관련
        {
            "query": "삼성전자 영업이익",
            "positive": "삼성전자는 2024년 1분기 영업이익 6조 6,000억원을 기록했다.",
            "negative": "LG전자는 2024년 1분기 매출액이 전년 대비 증가했다.",
        },
        {
            "query": "SK하이닉스 반도체 실적",
            "positive": "SK하이닉스의 HBM 메모리 매출이 급증하며 분기 최대 실적을 달성했다.",
            "negative": "현대차 전기차 판매량이 전년 대비 30% 증가했다.",
        },
        {
            "query": "네이버 분기 매출",
            "positive": "네이버의 분기 매출액이 2조 5,000억원을 돌파하며 신기록을 세웠다.",
            "negative": "카카오페이 결제 서비스가 일시 중단되었다.",
        },
        # 주가 관련
        {
            "query": "코스피 시가총액 순위",
            "positive": "삼성전자, SK하이닉스, LG에너지솔루션이 코스피 시가총액 상위 3위를 차지한다.",
            "negative": "미국 나스닥 지수가 사상 최고치를 경신했다.",
        },
        {
            "query": "현대차 주가 전망",
            "positive": "현대차 주가는 전기차 판매 호조에 힘입어 상승세를 이어갈 전망이다.",
            "negative": "기아 SUV 신모델이 북미 시장에서 인기를 끌고 있다.",
        },
        # 공시 관련
        {
            "query": "삼성전자 배당금",
            "positive": "삼성전자는 주당 배당금 361원을 결정했다고 공시했다.",
            "negative": "삼성전자 반도체 생산라인 투자 계획을 발표했다.",
        },
        {
            "query": "LG화학 유상증자",
            "positive": "LG화학이 배터리 사업 확대를 위해 2조원 규모 유상증자를 결정했다.",
            "negative": "LG화학 석유화학 부문 구조조정을 검토 중이다.",
        },
        # 산업 관련
        {
            "query": "HBM 메모리 시장",
            "positive": "AI 반도체 수요 증가로 HBM 메모리 시장이 급성장하고 있다.",
            "negative": "DRAM 가격이 하락세를 보이고 있다.",
        },
        {
            "query": "전기차 배터리 점유율",
            "positive": "LG에너지솔루션은 글로벌 전기차 배터리 시장 점유율 2위를 기록했다.",
            "negative": "테슬라 모델3 가격이 인하되었다.",
        },
        # 경제 지표
        {
            "query": "기준금리 인상",
            "positive": "한국은행이 기준금리를 0.25%p 인상하여 3.5%로 결정했다.",
            "negative": "미국 연준이 금리 동결을 시사했다.",
        },
    ]

    def __init__(self):
        """초기화"""
        pass

    def generate_dataset(
        self,
        num_pairs: int = 100,
        include_negatives: bool = True,
    ) -> TrainingDataset:
        """학습 데이터셋 생성

        Args:
            num_pairs: 생성할 쌍 수
            include_negatives: 네거티브 샘플 포함 여부

        Returns:
            학습 데이터셋
        """
        dataset = TrainingDataset(
            name="finance-embedding-dataset",
            description="금융 도메인 임베딩 학습용 데이터셋",
        )

        # 샘플 데이터 반복 사용 (실제로는 더 많은 데이터 필요)
        for i in range(num_pairs):
            sample = self.SAMPLE_PAIRS[i % len(self.SAMPLE_PAIRS)]

            dataset.add_pair(
                query=sample["query"],
                positive=sample["positive"],
                negative=sample["negative"] if include_negatives else None,
            )

        return dataset

    def generate_from_documents(
        self,
        documents: list[dict[str, Any]],
        queries_per_doc: int = 3,
    ) -> TrainingDataset:
        """문서에서 학습 데이터셋 생성

        Args:
            documents: 문서 목록 [{"content": "...", "metadata": {...}}]
            queries_per_doc: 문서당 생성할 쿼리 수

        Returns:
            학습 데이터셋
        """
        dataset = TrainingDataset(
            name="finance-doc-dataset",
            description="공시 문서 기반 학습 데이터셋",
        )

        # 간단한 쿼리 생성 (실제로는 LLM 사용)
        for doc in documents:
            content = doc.get("content", "")[:500]
            metadata = doc.get("metadata", {})

            # 기업명 + 핵심 키워드로 쿼리 생성
            corp_name = metadata.get("corp_name", "")

            if corp_name:
                queries = [
                    f"{corp_name} 실적",
                    f"{corp_name} 공시",
                    f"{corp_name} 재무",
                ]

                for query in queries[:queries_per_doc]:
                    dataset.add_pair(
                        query=query,
                        positive=content,
                    )

        return dataset


class FinanceEmbeddingTrainer:
    """금융 임베딩 모델 학습기"""

    def __init__(
        self,
        config: Optional[FineTuneConfig] = None,
    ):
        """
        Args:
            config: Fine-tuning 설정
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers가 필요합니다")

        self.config = config or FineTuneConfig()
        self._model: Optional[SentenceTransformer] = None

    def load_base_model(self) -> SentenceTransformer:
        """기본 모델 로드"""
        self._model = SentenceTransformer(self.config.base_model)
        return self._model

    def prepare_training_data(
        self,
        dataset: TrainingDataset,
    ) -> DataLoader:
        """학습 데이터 준비

        Args:
            dataset: 학습 데이터셋

        Returns:
            DataLoader
        """
        examples = []

        for pair in dataset.pairs:
            if pair.negative:
                # Triplet: (query, positive, negative)
                examples.append(InputExample(
                    texts=[pair.query, pair.positive, pair.negative],
                ))
            else:
                # Pair: (query, positive) with score
                examples.append(InputExample(
                    texts=[pair.query, pair.positive],
                    label=pair.score,
                ))

        return DataLoader(
            examples,
            shuffle=True,
            batch_size=self.config.batch_size,
        )

    def train(
        self,
        dataset: TrainingDataset,
        eval_dataset: Optional[TrainingDataset] = None,
    ) -> str:
        """모델 학습

        Args:
            dataset: 학습 데이터셋
            eval_dataset: 평가 데이터셋

        Returns:
            저장된 모델 경로
        """
        if self._model is None:
            self.load_base_model()

        train_dataloader = self.prepare_training_data(dataset)

        # 손실 함수 선택
        if dataset.pairs and dataset.pairs[0].negative:
            # Triplet Loss
            train_loss = losses.TripletLoss(model=self._model)
        else:
            # Cosine Similarity Loss
            train_loss = losses.CosineSimilarityLoss(model=self._model)

        # 평가기 설정
        evaluator = None
        if eval_dataset:
            evaluator = self._create_evaluator(eval_dataset)

        # 학습
        self._model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            warmup_steps=self.config.warmup_steps,
            evaluator=evaluator,
            evaluation_steps=self.config.evaluation_steps,
            output_path=self.config.output_path,
            use_amp=self.config.use_amp,
        )

        return self.config.output_path

    def _create_evaluator(
        self,
        dataset: TrainingDataset,
    ) -> InformationRetrievalEvaluator:
        """평가기 생성"""
        queries = {}
        corpus = {}
        relevant_docs = {}

        for i, pair in enumerate(dataset.pairs):
            query_id = f"q{i}"
            doc_id = f"d{i}"

            queries[query_id] = pair.query
            corpus[doc_id] = pair.positive
            relevant_docs[query_id] = {doc_id}

        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="finance-eval",
        )


class FinanceEmbeddingModel:
    """금융 특화 임베딩 모델

    Fine-tuned 모델 또는 기본 모델을 사용하여 임베딩을 생성합니다.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pretrained: str = "ko-sroberta",
    ):
        """
        Args:
            model_path: Fine-tuned 모델 경로
            use_pretrained: 사전학습 모델 키 (FineTuneConfig.FINANCE_MODELS)
        """
        self._model: Optional[Any] = None
        self.model_path = model_path
        self.use_pretrained = use_pretrained

        # 모델 정보
        if model_path and Path(model_path).exists():
            self._model_name = model_path
        elif use_pretrained in FineTuneConfig.FINANCE_MODELS:
            self._model_name = FineTuneConfig.FINANCE_MODELS[use_pretrained]["name"]
        else:
            self._model_name = FineTuneConfig().base_model

    @property
    def model(self) -> Any:
        """모델 로드 (lazy loading)"""
        if self._model is None:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._model = SentenceTransformer(self._model_name)
            else:
                # 폴백: 간단한 임베딩
                self._model = SimpleEmbedding()
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """텍스트 임베딩 생성

        Args:
            texts: 텍스트 목록
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            임베딩 벡터 (N x D)
        """
        if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self.model, 'encode'):
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )
        else:
            return self.model.encode(texts)

    def similarity(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """쿼리-문서 유사도 계산

        Args:
            query: 쿼리 텍스트
            documents: 문서 목록

        Returns:
            유사도 점수 목록
        """
        query_embedding = self.encode([query])[0]
        doc_embeddings = self.encode(documents)

        # 코사인 유사도 계산
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(float(similarity))

        return similarities


class SimpleEmbedding:
    """간단한 임베딩 (sentence-transformers 없을 때 폴백)"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """간단한 해시 기반 임베딩"""
        embeddings = []
        for text in texts:
            # 해시 기반 임의 임베딩 (재현성 있음)
            random.seed(hash(text) % (2**32))
            embedding = [random.gauss(0, 1) for _ in range(self.dimension)]
            embeddings.append(embedding)
        return np.array(embeddings)


# 편의 함수
def load_finance_embedding(
    model_path: Optional[str] = None,
) -> FinanceEmbeddingModel:
    """금융 임베딩 모델 로드

    Args:
        model_path: 모델 경로 (None이면 기본 모델)

    Returns:
        임베딩 모델
    """
    return FinanceEmbeddingModel(model_path=model_path)


def generate_training_data(
    num_pairs: int = 100,
) -> TrainingDataset:
    """학습 데이터 생성

    Args:
        num_pairs: 생성할 쌍 수

    Returns:
        학습 데이터셋
    """
    generator = FinanceDatasetGenerator()
    return generator.generate_dataset(num_pairs=num_pairs)
