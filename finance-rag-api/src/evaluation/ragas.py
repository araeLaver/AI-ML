# -*- coding: utf-8 -*-
"""
RAGAS (Retrieval Augmented Generation Assessment) 평가기

[기능]
- Faithfulness: 답변의 사실적 일관성
- Answer Relevancy: 질문에 대한 답변 관련성
- Context Precision: 검색된 컨텍스트 정밀도
- Context Recall: 컨텍스트 재현율
"""

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RAGASMetrics:
    """RAGAS 메트릭 결과"""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # 추가 메트릭
    harmfulness: float = 0.0  # 0이 좋음
    context_utilization: float = 0.0

    # 메타데이터
    evaluated_at: float = field(default_factory=time.time)
    evaluation_time_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        """종합 점수 (가중 평균)"""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "context_precision": 0.2,
            "context_recall": 0.2,
        }
        score = (
            self.faithfulness * weights["faithfulness"] +
            self.answer_relevancy * weights["answer_relevancy"] +
            self.context_precision * weights["context_precision"] +
            self.context_recall * weights["context_recall"]
        )
        # 유해성 페널티
        score *= (1 - self.harmfulness * 0.5)
        return round(score, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "harmfulness": round(self.harmfulness, 4),
            "context_utilization": round(self.context_utilization, 4),
            "overall_score": self.overall_score,
            "evaluated_at": self.evaluated_at,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }


@dataclass
class EvaluationSample:
    """평가 샘플"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """평가기 기본 클래스"""

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func or self._default_llm

    def _default_llm(self, prompt: str) -> str:
        """기본 LLM (시뮬레이션)"""
        return f"[Evaluation response for prompt length: {len(prompt)}]"

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> float:
        """평가 실행"""
        pass


class FaithfulnessEvaluator(BaseEvaluator):
    """
    충실도 평가기

    답변이 주어진 컨텍스트에서만 도출되었는지 평가
    (환각 검출)
    """

    def evaluate(self, sample: EvaluationSample) -> float:
        """충실도 평가 (0-1)"""
        if not sample.contexts or not sample.answer:
            return 0.0

        # 1. 답변에서 주장(claims) 추출
        claims = self._extract_claims(sample.answer)
        if not claims:
            return 1.0  # 주장이 없으면 충실함

        # 2. 각 주장이 컨텍스트에 의해 뒷받침되는지 확인
        supported_claims = 0
        context_text = " ".join(sample.contexts)

        for claim in claims:
            if self._is_claim_supported(claim, context_text):
                supported_claims += 1

        return supported_claims / len(claims)

    def _extract_claims(self, answer: str) -> List[str]:
        """답변에서 주장 추출"""
        # 간단한 문장 분리 기반 추출
        sentences = re.split(r'[.!?]', answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims

    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """주장이 컨텍스트에 의해 뒷받침되는지 확인"""
        # 간단한 키워드 매칭 (실제로는 LLM 사용)
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())

        # 핵심 단어의 50% 이상이 컨텍스트에 존재
        common_words = claim_words & context_words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "에", "를", "이", "가", "은", "는"}
        meaningful_claim_words = claim_words - stopwords

        if not meaningful_claim_words:
            return True

        overlap_ratio = len(common_words & meaningful_claim_words) / len(meaningful_claim_words)
        return overlap_ratio >= 0.3


class AnswerRelevancyEvaluator(BaseEvaluator):
    """
    답변 관련성 평가기

    답변이 질문에 얼마나 관련되는지 평가
    """

    def evaluate(self, sample: EvaluationSample) -> float:
        """답변 관련성 평가 (0-1)"""
        if not sample.question or not sample.answer:
            return 0.0

        # 1. 답변에서 역질문 생성 시뮬레이션
        # (실제로는 LLM으로 "이 답변에 대한 질문을 생성하세요" 프롬프트)
        generated_questions = self._generate_questions(sample.answer)

        # 2. 생성된 질문과 원래 질문의 유사도 계산
        similarities = [
            self._compute_similarity(sample.question, gen_q)
            for gen_q in generated_questions
        ]

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _generate_questions(self, answer: str) -> List[str]:
        """답변에서 질문 생성 (시뮬레이션)"""
        # 간단한 규칙 기반 (실제로는 LLM 사용)
        questions = []

        # 핵심 단어 추출
        words = answer.split()[:5]
        if words:
            questions.append(f"What about {' '.join(words)}?")
            questions.append(f"{' '.join(words)}에 대해 알려주세요")

        return questions if questions else ["질문"]

    def _compute_similarity(self, q1: str, q2: str) -> float:
        """두 질문의 유사도 계산"""
        # 간단한 자카드 유사도
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class ContextPrecisionEvaluator(BaseEvaluator):
    """
    컨텍스트 정밀도 평가기

    검색된 컨텍스트 중 관련 있는 것의 비율
    """

    def evaluate(self, sample: EvaluationSample) -> float:
        """컨텍스트 정밀도 평가 (0-1)"""
        if not sample.contexts:
            return 0.0

        # 각 컨텍스트가 질문에 관련되는지 확인
        relevant_count = 0

        for context in sample.contexts:
            if self._is_context_relevant(context, sample.question, sample.answer):
                relevant_count += 1

        return relevant_count / len(sample.contexts)

    def _is_context_relevant(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> bool:
        """컨텍스트가 관련있는지 확인"""
        # 질문 또는 답변의 핵심 단어가 컨텍스트에 있는지
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        # 불용어 제거
        stopwords = {"the", "a", "an", "is", "are", "에", "를", "이", "가", "은", "는", "을", "의"}
        question_words -= stopwords
        answer_words -= stopwords

        # 질문 또는 답변과 20% 이상 겹치면 관련 있음
        q_overlap = len(question_words & context_words) / len(question_words) if question_words else 0
        a_overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0

        return q_overlap >= 0.2 or a_overlap >= 0.2


class ContextRecallEvaluator(BaseEvaluator):
    """
    컨텍스트 재현율 평가기

    정답을 도출하는데 필요한 정보가 컨텍스트에 있는지
    (ground truth 필요)
    """

    def evaluate(self, sample: EvaluationSample) -> float:
        """컨텍스트 재현율 평가 (0-1)"""
        if not sample.contexts or not sample.ground_truth:
            return 0.0

        # Ground truth에서 핵심 정보 추출
        gt_claims = self._extract_claims(sample.ground_truth)
        if not gt_claims:
            return 1.0

        # 각 핵심 정보가 컨텍스트에 포함되어 있는지
        context_text = " ".join(sample.contexts)
        recalled_claims = 0

        for claim in gt_claims:
            if self._is_claim_in_context(claim, context_text):
                recalled_claims += 1

        return recalled_claims / len(gt_claims)

    def _extract_claims(self, text: str) -> List[str]:
        """텍스트에서 주요 주장 추출"""
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _is_claim_in_context(self, claim: str, context: str) -> bool:
        """주장이 컨텍스트에 있는지"""
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())

        stopwords = {"the", "a", "an", "is", "에", "를", "이", "가"}
        claim_words -= stopwords

        if not claim_words:
            return True

        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap >= 0.4


class HarmfulnessEvaluator(BaseEvaluator):
    """
    유해성 평가기

    답변에 유해한 내용이 있는지 평가
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.harmful_patterns = [
            r"죽",
            r"자살",
            r"폭력",
            r"차별",
            r"혐오",
            r"kill",
            r"suicide",
            r"hate",
        ]

    def evaluate(self, sample: EvaluationSample) -> float:
        """유해성 평가 (0-1, 0이 좋음)"""
        if not sample.answer:
            return 0.0

        answer_lower = sample.answer.lower()
        harmful_count = 0

        for pattern in self.harmful_patterns:
            if re.search(pattern, answer_lower):
                harmful_count += 1

        # 유해 패턴 수에 따른 점수 (시그모이드 형태)
        if harmful_count == 0:
            return 0.0
        elif harmful_count == 1:
            return 0.3
        elif harmful_count == 2:
            return 0.6
        else:
            return 0.9


class ContextUtilizationEvaluator(BaseEvaluator):
    """
    컨텍스트 활용도 평가기

    주어진 컨텍스트를 얼마나 활용했는지
    """

    def evaluate(self, sample: EvaluationSample) -> float:
        """컨텍스트 활용도 평가 (0-1)"""
        if not sample.contexts or not sample.answer:
            return 0.0

        answer_words = set(sample.answer.lower().split())
        utilized_contexts = 0

        for context in sample.contexts:
            context_words = set(context.lower().split())
            # 컨텍스트의 핵심 단어가 답변에 사용되었는지
            overlap = len(answer_words & context_words)
            if overlap >= 3:  # 최소 3개 단어 겹침
                utilized_contexts += 1

        return utilized_contexts / len(sample.contexts)


class RAGASEvaluator:
    """
    RAGAS 통합 평가기

    모든 RAGAS 메트릭을 한번에 평가
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func

        # 개별 평가기 초기화
        self.faithfulness = FaithfulnessEvaluator(llm_func=llm_func)
        self.answer_relevancy = AnswerRelevancyEvaluator(llm_func=llm_func)
        self.context_precision = ContextPrecisionEvaluator(llm_func=llm_func)
        self.context_recall = ContextRecallEvaluator(llm_func=llm_func)
        self.harmfulness = HarmfulnessEvaluator(llm_func=llm_func)
        self.context_utilization = ContextUtilizationEvaluator(llm_func=llm_func)

    def evaluate(self, sample: EvaluationSample) -> RAGASMetrics:
        """전체 RAGAS 메트릭 평가"""
        start_time = time.time()

        metrics = RAGASMetrics(
            faithfulness=self.faithfulness.evaluate(sample),
            answer_relevancy=self.answer_relevancy.evaluate(sample),
            context_precision=self.context_precision.evaluate(sample),
            context_recall=self.context_recall.evaluate(sample),
            harmfulness=self.harmfulness.evaluate(sample),
            context_utilization=self.context_utilization.evaluate(sample),
        )

        metrics.evaluation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"RAGAS evaluation completed: overall={metrics.overall_score:.4f}, "
            f"faithfulness={metrics.faithfulness:.4f}, "
            f"relevancy={metrics.answer_relevancy:.4f}"
        )

        return metrics

    def evaluate_batch(
        self,
        samples: List[EvaluationSample],
    ) -> List[RAGASMetrics]:
        """배치 평가"""
        results = []
        for sample in samples:
            result = self.evaluate(sample)
            results.append(result)
        return results

    def aggregate_metrics(
        self,
        metrics_list: List[RAGASMetrics],
    ) -> Dict[str, Any]:
        """메트릭 집계"""
        if not metrics_list:
            return {}

        n = len(metrics_list)

        return {
            "count": n,
            "faithfulness": {
                "mean": sum(m.faithfulness for m in metrics_list) / n,
                "min": min(m.faithfulness for m in metrics_list),
                "max": max(m.faithfulness for m in metrics_list),
            },
            "answer_relevancy": {
                "mean": sum(m.answer_relevancy for m in metrics_list) / n,
                "min": min(m.answer_relevancy for m in metrics_list),
                "max": max(m.answer_relevancy for m in metrics_list),
            },
            "context_precision": {
                "mean": sum(m.context_precision for m in metrics_list) / n,
                "min": min(m.context_precision for m in metrics_list),
                "max": max(m.context_precision for m in metrics_list),
            },
            "context_recall": {
                "mean": sum(m.context_recall for m in metrics_list) / n,
                "min": min(m.context_recall for m in metrics_list),
                "max": max(m.context_recall for m in metrics_list),
            },
            "overall_score": {
                "mean": sum(m.overall_score for m in metrics_list) / n,
                "min": min(m.overall_score for m in metrics_list),
                "max": max(m.overall_score for m in metrics_list),
            },
            "harmfulness": {
                "mean": sum(m.harmfulness for m in metrics_list) / n,
            },
        }
