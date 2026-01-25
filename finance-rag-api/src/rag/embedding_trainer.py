# -*- coding: utf-8 -*-
"""
금융 도메인 임베딩 Fine-tuning 모듈

[목표]
- 금융 용어 간 의미적 유사도 개선
- PER ↔ 주가수익비율, 삼전 ↔ 삼성전자 등 동의어 관계 학습
- DART 공시 문서 기반 도메인 특화

[학습 방법]
- Contrastive Learning (대조 학습)
- Multiple Negatives Ranking Loss
- LoRA Fine-tuning (효율적 학습)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """학습 데이터 예시"""
    query: str
    positive: str  # 관련 문서
    negative: Optional[str] = None  # 비관련 문서 (hard negative)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """학습 설정"""
    # 모델 설정
    base_model: str = "intfloat/multilingual-e5-base"
    output_dir: str = "models/finance-embedding-v1"

    # 학습 파라미터
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1

    # LoRA 설정
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # 데이터 설정
    max_seq_length: int = 512
    train_split: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "max_seq_length": self.max_seq_length,
        }


class TrainingDataGenerator:
    """
    학습 데이터 생성기

    [데이터 소스]
    1. 금융 동의어 사전 - 동의어 쌍 생성
    2. DART 공시 문서 - 제목-본문 쌍 생성
    3. 평가 데이터셋 - 쿼리-문서 쌍 활용
    """

    def __init__(self, data_dir: str = "data/embedding_training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_synonym_pairs(self) -> List[TrainingExample]:
        """동의어 사전에서 학습 데이터 생성"""
        from .financial_dictionary import FINANCIAL_SYNONYMS

        examples = []

        for canonical, synonyms in FINANCIAL_SYNONYMS.items():
            # 표준 용어와 각 동의어 쌍 생성
            for syn in synonyms:
                # 정방향: 약어 → 풀네임
                examples.append(TrainingExample(
                    query=canonical,
                    positive=syn,
                    metadata={"type": "synonym", "direction": "forward"}
                ))

                # 역방향: 풀네임 → 약어
                examples.append(TrainingExample(
                    query=syn,
                    positive=canonical,
                    metadata={"type": "synonym", "direction": "reverse"}
                ))

            # 동의어 간 쌍 생성
            for i, syn1 in enumerate(synonyms):
                for syn2 in synonyms[i+1:]:
                    examples.append(TrainingExample(
                        query=syn1,
                        positive=syn2,
                        metadata={"type": "synonym", "direction": "cross"}
                    ))

        logger.info(f"Generated {len(examples)} synonym pairs")
        return examples

    def generate_query_document_pairs(
        self,
        documents_dir: str = "data/documents"
    ) -> List[TrainingExample]:
        """문서에서 쿼리-문서 쌍 생성"""
        examples = []
        docs_path = Path(documents_dir)

        if not docs_path.exists():
            logger.warning(f"Documents directory not found: {docs_path}")
            return examples

        for doc_file in docs_path.glob("*.txt"):
            content = doc_file.read_text(encoding="utf-8")

            # 파일명에서 메타데이터 추출
            filename = doc_file.stem
            parts = filename.split("_")

            if len(parts) >= 2:
                company = parts[0]
                report_type = parts[1] if len(parts) > 1 else "report"

                # 쿼리 생성
                queries = [
                    f"{company} 실적",
                    f"{company} 영업이익",
                    f"{company} {report_type}",
                ]

                for query in queries:
                    # 문서 요약 (첫 500자)
                    doc_summary = content[:500].strip()

                    examples.append(TrainingExample(
                        query=query,
                        positive=doc_summary,
                        metadata={
                            "type": "query_document",
                            "source": filename,
                            "company": company,
                        }
                    ))

        logger.info(f"Generated {len(examples)} query-document pairs")
        return examples

    def generate_financial_context_pairs(self) -> List[TrainingExample]:
        """금융 문맥 쌍 생성 (수동 큐레이션)"""
        # 금융 도메인 특화 쿼리-문서 쌍
        curated_pairs = [
            # 재무 지표 관련
            ("PER이 낮은 기업", "주가수익비율이 업종 평균 대비 저평가된 종목으로 투자 매력도가 높습니다."),
            ("ROE가 높은 회사", "자기자본이익률이 높은 기업은 주주 자본을 효율적으로 활용하고 있습니다."),
            ("영업이익률 개선", "영업이익률이 전분기 대비 상승하며 수익성이 개선되었습니다."),

            # 산업 관련
            ("HBM 수요 전망", "고대역폭메모리 시장은 AI 서버 수요 증가로 급성장하고 있습니다."),
            ("2차전지 시장", "전기차 배터리 시장은 글로벌 EV 보급 확대로 성장세가 지속됩니다."),
            ("반도체 업황", "메모리 반도체 가격이 반등하며 업황 개선 기대감이 높아지고 있습니다."),

            # 기업 관련
            ("삼성전자 실적", "삼성전자는 반도체 부문 흑자 전환으로 영업이익이 크게 개선되었습니다."),
            ("SK하이닉스 HBM", "SK하이닉스는 HBM3E 양산을 통해 AI 반도체 시장을 선도하고 있습니다."),
            ("현대차 전기차", "현대자동차는 아이오닉 시리즈 판매 호조로 전기차 시장 점유율을 확대하고 있습니다."),

            # 시장 관련
            ("코스피 전망", "국내 증시는 외국인 순매수와 기관 매수세로 상승 흐름을 보이고 있습니다."),
            ("금리 인하", "한국은행의 기준금리 인하 기대감이 주식시장에 긍정적으로 작용하고 있습니다."),
            ("배당주 추천", "고배당 종목은 안정적인 현금흐름과 함께 배당수익을 제공합니다."),
        ]

        examples = []
        for query, positive in curated_pairs:
            examples.append(TrainingExample(
                query=query,
                positive=positive,
                metadata={"type": "curated", "domain": "finance"}
            ))

        logger.info(f"Generated {len(examples)} curated pairs")
        return examples

    def generate_hard_negatives(
        self,
        examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """Hard Negative 샘플 추가"""
        import random

        # positive 문서 목록
        all_positives = [ex.positive for ex in examples]

        enhanced_examples = []
        for ex in examples:
            # 다른 문서를 negative로 사용
            negatives = [p for p in all_positives if p != ex.positive]
            if negatives:
                negative = random.choice(negatives)
                enhanced_examples.append(TrainingExample(
                    query=ex.query,
                    positive=ex.positive,
                    negative=negative,
                    metadata=ex.metadata
                ))
            else:
                enhanced_examples.append(ex)

        return enhanced_examples

    def save_training_data(
        self,
        examples: List[TrainingExample],
        filename: str = "training_data.jsonl"
    ) -> str:
        """학습 데이터 저장"""
        output_path = self.data_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                data = {
                    "query": ex.query,
                    "positive": ex.positive,
                    "negative": ex.negative,
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return str(output_path)

    def load_training_data(self, filename: str = "training_data.jsonl") -> List[TrainingExample]:
        """학습 데이터 로드"""
        input_path = self.data_dir / filename
        examples = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(TrainingExample(
                    query=data["query"],
                    positive=data["positive"],
                    negative=data.get("negative"),
                    metadata=data.get("metadata", {}),
                ))

        logger.info(f"Loaded {len(examples)} examples from {input_path}")
        return examples

    def generate_all(self, include_hard_negatives: bool = True) -> List[TrainingExample]:
        """모든 소스에서 학습 데이터 생성"""
        all_examples = []

        # 1. 동의어 쌍
        all_examples.extend(self.generate_synonym_pairs())

        # 2. 쿼리-문서 쌍
        all_examples.extend(self.generate_query_document_pairs())

        # 3. 큐레이션된 금융 문맥 쌍
        all_examples.extend(self.generate_financial_context_pairs())

        # 4. Hard Negative 추가
        if include_hard_negatives:
            all_examples = self.generate_hard_negatives(all_examples)

        logger.info(f"Total training examples: {len(all_examples)}")
        return all_examples


class FinancialEmbeddingTrainer:
    """
    금융 도메인 임베딩 학습기

    [지원 모델]
    - intfloat/multilingual-e5-base (다국어)
    - BAAI/bge-base-en-v1.5 (영어 특화)
    - jhgan/ko-sbert-sts (한국어 특화)

    [학습 방식]
    - Multiple Negatives Ranking Loss
    - LoRA Fine-tuning (선택)
    """

    SUPPORTED_MODELS = [
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-small",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "jhgan/ko-sbert-sts",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.training_history = []

    def _check_dependencies(self) -> bool:
        """필수 라이브러리 확인"""
        try:
            import sentence_transformers
            import torch
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install with: pip install sentence-transformers torch")
            return False

    def load_base_model(self):
        """베이스 모델 로드"""
        if not self._check_dependencies():
            raise ImportError("Required dependencies not installed")

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading base model: {self.config.base_model}")
        self.model = SentenceTransformer(self.config.base_model)

        if self.config.use_lora:
            self._apply_lora()

        return self.model

    def _apply_lora(self):
        """LoRA 적용 (선택적)"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            logger.info("Applying LoRA configuration")

            # LoRA 설정
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )

            # 모델에 LoRA 적용
            self.model._first_module().auto_model = get_peft_model(
                self.model._first_module().auto_model,
                lora_config
            )

            logger.info("LoRA applied successfully")

        except ImportError:
            logger.warning("PEFT not installed, skipping LoRA. Install with: pip install peft")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA: {e}")

    def prepare_training_data(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[Any, Any]:
        """학습 데이터 준비"""
        from sentence_transformers import InputExample
        import random

        random.shuffle(examples)

        # Train/Eval 분할
        split_idx = int(len(examples) * self.config.train_split)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]

        # InputExample 형식으로 변환
        train_data = []
        for ex in train_examples:
            if ex.negative:
                train_data.append(InputExample(
                    texts=[ex.query, ex.positive, ex.negative]
                ))
            else:
                train_data.append(InputExample(
                    texts=[ex.query, ex.positive]
                ))

        eval_data = []
        for ex in eval_examples:
            eval_data.append(InputExample(
                texts=[ex.query, ex.positive]
            ))

        logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
        return train_data, eval_data

    def train(
        self,
        examples: List[TrainingExample],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """모델 학습"""
        from sentence_transformers import losses, evaluation
        from torch.utils.data import DataLoader

        if self.model is None:
            self.load_base_model()

        # 데이터 준비
        train_data, eval_data = self.prepare_training_data(examples)

        # DataLoader 생성
        train_dataloader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        # Loss 함수
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        # 평가자 설정
        evaluator = None
        if eval_data:
            # 간단한 평가용 쿼리-문서 쌍
            queries = [ex.texts[0] for ex in eval_data[:100]]
            docs = [ex.texts[1] for ex in eval_data[:100]]

            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                sentences1=queries,
                sentences2=docs,
                scores=[1.0] * len(queries),  # 모두 positive pair
            )

        # 학습 실행
        logger.info("Starting training...")
        warmup_steps = int(len(train_dataloader) * self.config.epochs * self.config.warmup_ratio)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=len(train_dataloader) // 2,
            output_path=self.config.output_dir,
            show_progress_bar=show_progress,
        )

        # 학습 기록 저장
        training_info = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "num_examples": len(examples),
            "train_size": len(train_data),
            "eval_size": len(eval_data),
        }
        self.training_history.append(training_info)

        # 설정 저장
        self._save_training_info(training_info)

        logger.info(f"Training complete. Model saved to: {self.config.output_dir}")
        return training_info

    def _save_training_info(self, info: Dict[str, Any]):
        """학습 정보 저장"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        info_path = output_dir / "training_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def evaluate(
        self,
        test_queries: List[str],
        test_documents: List[str],
        relevance_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """모델 평가"""
        from sentence_transformers import evaluation

        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        if relevance_scores is None:
            relevance_scores = [1.0] * len(test_queries)

        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1=test_queries,
            sentences2=test_documents,
            scores=relevance_scores,
        )

        score = evaluator(self.model)

        return {
            "embedding_similarity": score,
            "num_samples": len(test_queries),
        }

    @classmethod
    def load_trained_model(cls, model_path: str) -> "FinancialEmbeddingTrainer":
        """학습된 모델 로드"""
        from sentence_transformers import SentenceTransformer

        trainer = cls()
        trainer.model = SentenceTransformer(model_path)

        # 학습 정보 로드
        info_path = Path(model_path) / "training_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                trainer.training_history.append(info)

        logger.info(f"Loaded model from: {model_path}")
        return trainer


# =============================================================================
# 편의 함수
# =============================================================================

def prepare_training_data(
    output_dir: str = "data/embedding_training",
    include_hard_negatives: bool = True
) -> str:
    """학습 데이터 준비 및 저장"""
    generator = TrainingDataGenerator(output_dir)
    examples = generator.generate_all(include_hard_negatives)
    output_path = generator.save_training_data(examples)
    return output_path


def train_financial_embedding(
    data_path: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> str:
    """금융 임베딩 학습 실행"""
    config = config or TrainingConfig(**kwargs)
    trainer = FinancialEmbeddingTrainer(config)

    # 데이터 로드
    if data_path:
        generator = TrainingDataGenerator()
        examples = generator.load_training_data(data_path)
    else:
        generator = TrainingDataGenerator()
        examples = generator.generate_all()

    # 학습 실행
    trainer.train(examples)

    return config.output_dir


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train financial embedding model")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-base", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output-dir", default="models/finance-embedding-v1", help="Output directory")

    args = parser.parse_args()

    if args.prepare_data:
        output = prepare_training_data()
        print(f"Training data saved to: {output}")

    if args.train:
        config = TrainingConfig(
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
        output = train_financial_embedding(config=config)
        print(f"Model saved to: {output}")
