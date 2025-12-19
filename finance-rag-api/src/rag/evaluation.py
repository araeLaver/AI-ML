# -*- coding: utf-8 -*-
"""
RAG 평가 지표 모듈

[설계 의도]
- RAGAS 스타일 평가 지표 구현
- RAG 시스템 품질 정량화
- 면접에서 "평가는 어떻게 했나요?" 대응

[주요 지표]
1. Faithfulness: 답변이 컨텍스트에 기반하는지
2. Answer Relevancy: 답변이 질문과 관련있는지
3. Context Precision: 검색된 문서가 관련있는지
4. Context Recall: 필요한 정보가 검색되었는지
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """평가 지표 유형"""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    metric: MetricType
    score: float  # 0.0 ~ 1.0
    details: Dict[str, Any]
    explanation: str


@dataclass
class RAGEvaluationInput:
    """RAG 평가 입력 데이터"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None  # 정답 (있는 경우)


class RAGEvaluator:
    """
    RAG 시스템 평가기

    [RAGAS 지표 설명]
    1. Faithfulness (충실도)
       - 답변의 각 문장이 컨텍스트에서 유추 가능한지
       - 환각(Hallucination) 탐지

    2. Answer Relevancy (답변 관련성)
       - 답변이 질문에 적절히 대응하는지
       - 질문과 답변의 의미적 유사도

    3. Context Precision (컨텍스트 정밀도)
       - 검색된 문서들 중 관련 있는 비율
       - 불필요한 문서가 많으면 낮음

    4. Context Recall (컨텍스트 재현율)
       - 정답에 필요한 정보가 컨텍스트에 있는지
       - ground_truth 필요
    """

    def __init__(self, llm_provider=None):
        """
        Args:
            llm_provider: LLM 제공자 (정교한 평가 시 사용)
        """
        self.llm = llm_provider

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> EvaluationResult:
        """
        충실도 평가

        답변의 각 문장이 컨텍스트에서 지지되는지 확인
        """
        # 답변을 문장으로 분리
        sentences = self._split_sentences(answer)

        if not sentences:
            return EvaluationResult(
                metric=MetricType.FAITHFULNESS,
                score=0.0,
                details={"sentences": [], "supported": []},
                explanation="답변에 평가할 문장이 없습니다."
            )

        # 전체 컨텍스트 텍스트
        context_text = ' '.join(contexts).lower()

        # 각 문장의 지지 여부 확인
        supported = []
        for sentence in sentences:
            # 키워드 기반 간단한 지지 확인
            # (실제로는 LLM이나 NLI 모델 사용)
            keywords = self._extract_keywords(sentence)
            support_score = sum(1 for kw in keywords if kw.lower() in context_text) / max(len(keywords), 1)
            supported.append(support_score > 0.3)

        faithfulness_score = sum(supported) / len(supported)

        return EvaluationResult(
            metric=MetricType.FAITHFULNESS,
            score=faithfulness_score,
            details={
                "sentences": sentences,
                "supported": supported,
                "supported_count": sum(supported),
                "total_count": len(sentences)
            },
            explanation=f"{len(sentences)}개 문장 중 {sum(supported)}개가 컨텍스트에서 지지됨"
        )

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> EvaluationResult:
        """
        답변 관련성 평가

        질문과 답변의 키워드 오버랩 기반 평가
        (실제로는 임베딩 유사도 사용)
        """
        question_keywords = set(self._extract_keywords(question))
        answer_keywords = set(self._extract_keywords(answer))

        if not question_keywords:
            return EvaluationResult(
                metric=MetricType.ANSWER_RELEVANCY,
                score=0.0,
                details={},
                explanation="질문에서 키워드를 추출할 수 없습니다."
            )

        # 질문 키워드가 답변에 얼마나 반영되었는지
        overlap = question_keywords & answer_keywords
        relevancy_score = len(overlap) / len(question_keywords)

        # 답변 길이 보정 (너무 짧으면 감점)
        length_penalty = min(1.0, len(answer) / 50)
        final_score = relevancy_score * 0.7 + length_penalty * 0.3

        return EvaluationResult(
            metric=MetricType.ANSWER_RELEVANCY,
            score=min(1.0, final_score),
            details={
                "question_keywords": list(question_keywords),
                "answer_keywords": list(answer_keywords),
                "overlap": list(overlap),
                "answer_length": len(answer)
            },
            explanation=f"질문 키워드 {len(question_keywords)}개 중 {len(overlap)}개가 답변에 반영됨"
        )

    def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str]
    ) -> EvaluationResult:
        """
        컨텍스트 정밀도 평가

        검색된 문서들이 질문과 관련있는 비율
        """
        if not contexts:
            return EvaluationResult(
                metric=MetricType.CONTEXT_PRECISION,
                score=0.0,
                details={"context_scores": []},
                explanation="검색된 컨텍스트가 없습니다."
            )

        question_keywords = set(self._extract_keywords(question))

        # 각 컨텍스트의 관련도 계산
        context_scores = []
        for ctx in contexts:
            ctx_keywords = set(self._extract_keywords(ctx))
            if question_keywords:
                overlap = question_keywords & ctx_keywords
                score = len(overlap) / len(question_keywords)
            else:
                score = 0.0
            context_scores.append(score)

        # 관련 있는 컨텍스트 비율 (임계값 0.2)
        relevant_count = sum(1 for s in context_scores if s > 0.2)
        precision = relevant_count / len(contexts)

        return EvaluationResult(
            metric=MetricType.CONTEXT_PRECISION,
            score=precision,
            details={
                "context_scores": context_scores,
                "relevant_count": relevant_count,
                "total_count": len(contexts)
            },
            explanation=f"{len(contexts)}개 컨텍스트 중 {relevant_count}개가 관련 있음"
        )

    def evaluate_context_recall(
        self,
        ground_truth: str,
        contexts: List[str]
    ) -> EvaluationResult:
        """
        컨텍스트 재현율 평가

        정답에 필요한 정보가 컨텍스트에 있는지
        """
        if not ground_truth:
            return EvaluationResult(
                metric=MetricType.CONTEXT_RECALL,
                score=0.0,
                details={},
                explanation="정답(ground_truth)이 제공되지 않았습니다."
            )

        truth_keywords = set(self._extract_keywords(ground_truth))
        context_text = ' '.join(contexts)
        context_keywords = set(self._extract_keywords(context_text))

        if not truth_keywords:
            return EvaluationResult(
                metric=MetricType.CONTEXT_RECALL,
                score=0.0,
                details={},
                explanation="정답에서 키워드를 추출할 수 없습니다."
            )

        # 정답 키워드가 컨텍스트에 얼마나 있는지
        recall_overlap = truth_keywords & context_keywords
        recall_score = len(recall_overlap) / len(truth_keywords)

        return EvaluationResult(
            metric=MetricType.CONTEXT_RECALL,
            score=recall_score,
            details={
                "truth_keywords": list(truth_keywords),
                "found_keywords": list(recall_overlap),
                "missing_keywords": list(truth_keywords - recall_overlap)
            },
            explanation=f"정답 키워드 {len(truth_keywords)}개 중 {len(recall_overlap)}개가 컨텍스트에 있음"
        )

    def evaluate_all(
        self,
        evaluation_input: RAGEvaluationInput
    ) -> Dict[str, EvaluationResult]:
        """전체 평가 수행"""
        results = {}

        # Faithfulness
        results["faithfulness"] = self.evaluate_faithfulness(
            evaluation_input.answer,
            evaluation_input.contexts
        )

        # Answer Relevancy
        results["answer_relevancy"] = self.evaluate_answer_relevancy(
            evaluation_input.question,
            evaluation_input.answer
        )

        # Context Precision
        results["context_precision"] = self.evaluate_context_precision(
            evaluation_input.question,
            evaluation_input.contexts
        )

        # Context Recall (ground_truth가 있을 때만)
        if evaluation_input.ground_truth:
            results["context_recall"] = self.evaluate_context_recall(
                evaluation_input.ground_truth,
                evaluation_input.contexts
            )

        return results

    def get_summary_score(
        self,
        results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """평가 결과 요약"""
        scores = {name: r.score for name, r in results.items()}
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        return {
            "individual_scores": scores,
            "average_score": round(avg_score, 4),
            "grade": self._score_to_grade(avg_score),
            "recommendations": self._get_recommendations(results)
        }

    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        # 한국어 + 영어 문장 분리
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출 (간단한 버전)"""
        # 불용어
        stopwords = {
            '있다', '없다', '하다', '되다', '이다', '있는', '하는', '되는',
            '그리고', '하지만', '또한', '그러나', '따라서', '때문에',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            '를', '을', '가', '이', '에', '의', '로', '으로', '와', '과',
        }

        # 토큰화
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text.lower())

        # 불용어 제거 및 길이 필터
        keywords = [
            t for t in tokens
            if t not in stopwords and len(t) >= 2
        ]

        return keywords

    def _score_to_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.9:
            return "A+ (우수)"
        elif score >= 0.8:
            return "A (양호)"
        elif score >= 0.7:
            return "B (보통)"
        elif score >= 0.6:
            return "C (개선필요)"
        else:
            return "D (미흡)"

    def _get_recommendations(
        self,
        results: Dict[str, EvaluationResult]
    ) -> List[str]:
        """개선 권고사항"""
        recommendations = []

        for name, result in results.items():
            if result.score < 0.6:
                if name == "faithfulness":
                    recommendations.append(
                        "충실도 낮음: 프롬프트에 '문서에 없는 내용은 답하지 마세요' 강조"
                    )
                elif name == "answer_relevancy":
                    recommendations.append(
                        "답변 관련성 낮음: 질문을 더 잘 이해하도록 프롬프트 개선"
                    )
                elif name == "context_precision":
                    recommendations.append(
                        "컨텍스트 정밀도 낮음: 검색 top_k 줄이거나 re-ranking 적용"
                    )
                elif name == "context_recall":
                    recommendations.append(
                        "컨텍스트 재현율 낮음: 청킹 전략 변경 또는 top_k 증가"
                    )

        if not recommendations:
            recommendations.append("전반적으로 양호합니다.")

        return recommendations


# 테스트 케이스 예시
EVALUATION_TEST_CASES = [
    {
        "name": "기본 테스트",
        "input": RAGEvaluationInput(
            question="삼성전자 3분기 영업이익은?",
            answer="삼성전자의 2024년 3분기 영업이익은 9조 1,834억원으로, 전년동기대비 274.5% 증가했습니다.",
            contexts=[
                "삼성전자 2024년 3분기 실적: 영업이익 9조 1,834억원 (전년동기대비 +274.5%)",
                "삼성전자 반도체 부문 실적 개선"
            ],
            ground_truth="삼성전자 3분기 영업이익은 9조 1,834억원입니다."
        )
    },
    {
        "name": "환각 테스트",
        "input": RAGEvaluationInput(
            question="삼성전자 3분기 영업이익은?",
            answer="삼성전자의 3분기 영업이익은 15조원이며, 애플을 인수할 계획입니다.",
            contexts=[
                "삼성전자 2024년 3분기 실적: 영업이익 9조 1,834억원"
            ],
            ground_truth="삼성전자 3분기 영업이익은 9조 1,834억원입니다."
        )
    }
]


def run_test_evaluation():
    """테스트 평가 실행"""
    evaluator = RAGEvaluator()
    results = []

    for test_case in EVALUATION_TEST_CASES:
        eval_results = evaluator.evaluate_all(test_case["input"])
        summary = evaluator.get_summary_score(eval_results)
        results.append({
            "name": test_case["name"],
            "summary": summary
        })

    return results
