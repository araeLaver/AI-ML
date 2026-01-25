# -*- coding: utf-8 -*-
"""
Human Evaluation 모듈

[기능]
- 어노테이션 작업 관리
- 평가자 간 신뢰도 (Inter-rater reliability)
- 블라인드 평가
"""

import logging
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EvaluationCriteria(Enum):
    """평가 기준"""
    RELEVANCE = "relevance"  # 관련성
    ACCURACY = "accuracy"  # 정확성
    FLUENCY = "fluency"  # 유창성
    COHERENCE = "coherence"  # 일관성
    COMPLETENESS = "completeness"  # 완전성
    HELPFULNESS = "helpfulness"  # 유용성
    HARMLESSNESS = "harmlessness"  # 무해성


@dataclass
class AnnotationTask:
    """어노테이션 작업"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    criteria: List[EvaluationCriteria] = field(default_factory=list)

    # 작업 상태
    status: str = "pending"  # pending, assigned, completed
    assigned_to: Optional[str] = None
    assigned_at: Optional[float] = None

    # 블라인드 평가용
    blind_id: Optional[str] = None  # 익명화된 ID
    model_source: Optional[str] = None  # 실제 모델 (평가자에게 숨김)

    # 메타데이터
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer[:200] if self.answer else None,
            "criteria": [c.value for c in self.criteria],
            "status": self.status,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
        }


@dataclass
class AnnotationResult:
    """어노테이션 결과"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    annotator_id: str = ""

    # 평가 점수 (1-5 척도)
    scores: Dict[str, int] = field(default_factory=dict)

    # 자유 형식 피드백
    comments: str = ""
    issues: List[str] = field(default_factory=list)  # 발견된 문제

    # 메타데이터
    time_spent_seconds: float = 0.0
    completed_at: float = field(default_factory=time.time)
    confidence: float = 1.0  # 평가자 확신도

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "annotator_id": self.annotator_id,
            "scores": self.scores,
            "comments": self.comments[:100] if self.comments else None,
            "issues": self.issues,
            "time_spent_seconds": round(self.time_spent_seconds, 2),
            "confidence": round(self.confidence, 2),
        }


class InterRaterReliability:
    """
    평가자 간 신뢰도 계산

    - Cohen's Kappa: 2명의 평가자
    - Fleiss' Kappa: 3명 이상
    - Krippendorff's Alpha: 일반적 신뢰도
    """

    @staticmethod
    def cohens_kappa(
        rater1_scores: List[int],
        rater2_scores: List[int],
    ) -> float:
        """
        Cohen's Kappa 계산

        두 평가자 간의 일치도
        """
        if len(rater1_scores) != len(rater2_scores):
            raise ValueError("두 평가자의 점수 수가 다릅니다")

        n = len(rater1_scores)
        if n == 0:
            return 0.0

        # 범주 수집
        categories = sorted(set(rater1_scores) | set(rater2_scores))
        k = len(categories)
        cat_to_idx = {c: i for i, c in enumerate(categories)}

        # 혼동 행렬 생성
        matrix = [[0] * k for _ in range(k)]
        for r1, r2 in zip(rater1_scores, rater2_scores):
            matrix[cat_to_idx[r1]][cat_to_idx[r2]] += 1

        # 관찰된 일치도 (Po)
        po = sum(matrix[i][i] for i in range(k)) / n

        # 기대 일치도 (Pe)
        row_sums = [sum(row) / n for row in matrix]
        col_sums = [sum(matrix[i][j] for i in range(k)) / n for j in range(k)]
        pe = sum(row_sums[i] * col_sums[i] for i in range(k))

        # Kappa 계산
        if pe == 1.0:
            return 1.0
        kappa = (po - pe) / (1 - pe)
        return round(kappa, 4)

    @staticmethod
    def fleiss_kappa(
        ratings: List[List[int]],
        n_categories: int,
    ) -> float:
        """
        Fleiss' Kappa 계산

        여러 평가자 간의 일치도
        ratings: 각 항목별 평가자들의 점수 리스트
        """
        n_items = len(ratings)
        if n_items == 0:
            return 0.0

        n_raters = len(ratings[0]) if ratings else 0

        # 각 항목-범주별 빈도 계산
        frequencies = []
        for item_ratings in ratings:
            freq = [0] * n_categories
            for r in item_ratings:
                if 0 <= r < n_categories:
                    freq[r] += 1
            frequencies.append(freq)

        # 각 범주의 전체 비율
        p_j = []
        total = n_items * n_raters
        for j in range(n_categories):
            count = sum(freq[j] for freq in frequencies)
            p_j.append(count / total if total > 0 else 0)

        # 각 항목의 일치도
        p_i = []
        for freq in frequencies:
            if n_raters <= 1:
                p_i.append(1.0)
            else:
                sum_squared = sum(f * f for f in freq)
                p = (sum_squared - n_raters) / (n_raters * (n_raters - 1))
                p_i.append(p)

        # P_bar: 평균 항목 일치도
        p_bar = sum(p_i) / n_items if n_items > 0 else 0

        # P_e: 기대 일치도
        p_e = sum(p * p for p in p_j)

        # Kappa 계산
        if p_e == 1.0:
            return 1.0
        kappa = (p_bar - p_e) / (1 - p_e)
        return round(kappa, 4)

    @staticmethod
    def krippendorff_alpha(
        data: List[List[Optional[int]]],
        level: str = "ordinal",
    ) -> float:
        """
        Krippendorff's Alpha 계산

        결측값 허용, 다양한 데이터 레벨 지원
        data: 평가자 x 항목 매트릭스 (None=결측)
        level: nominal, ordinal, interval, ratio
        """
        # 유효한 쌍 수집
        n_items = len(data[0]) if data else 0
        n_raters = len(data)

        # 값 쌍 수집
        values_by_item = []
        for j in range(n_items):
            values = [data[i][j] for i in range(n_raters) if data[i][j] is not None]
            if len(values) >= 2:
                values_by_item.append(values)

        if not values_by_item:
            return 0.0

        # 관찰된 불일치
        observed_disagreement = 0
        total_pairs = 0

        for values in values_by_item:
            n = len(values)
            for i in range(n):
                for k in range(i + 1, n):
                    diff = InterRaterReliability._difference(
                        values[i], values[k], level
                    )
                    observed_disagreement += diff
                    total_pairs += 1

        if total_pairs == 0:
            return 1.0

        do = observed_disagreement / total_pairs

        # 기대 불일치
        all_values = []
        for values in values_by_item:
            all_values.extend(values)

        expected_disagreement = 0
        n_all = len(all_values)
        total_expected_pairs = 0

        for i in range(n_all):
            for k in range(i + 1, n_all):
                diff = InterRaterReliability._difference(
                    all_values[i], all_values[k], level
                )
                expected_disagreement += diff
                total_expected_pairs += 1

        if total_expected_pairs == 0:
            return 1.0

        de = expected_disagreement / total_expected_pairs

        # Alpha 계산
        if de == 0:
            return 1.0
        alpha = 1 - (do / de)
        return round(alpha, 4)

    @staticmethod
    def _difference(v1: int, v2: int, level: str) -> float:
        """두 값의 차이 계산"""
        if level == "nominal":
            return 0 if v1 == v2 else 1
        elif level in ["ordinal", "interval", "ratio"]:
            return (v1 - v2) ** 2
        return abs(v1 - v2)

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Kappa 값 해석"""
        if kappa < 0:
            return "Poor (less than chance)"
        elif kappa < 0.2:
            return "Slight"
        elif kappa < 0.4:
            return "Fair"
        elif kappa < 0.6:
            return "Moderate"
        elif kappa < 0.8:
            return "Substantial"
        else:
            return "Almost Perfect"


class BlindEvaluator:
    """
    블라인드 평가 관리

    모델 출처를 숨기고 공정한 평가 수행
    """

    def __init__(self, seed: Optional[int] = None):
        self._blind_mapping: Dict[str, str] = {}  # blind_id -> original_id
        self._reverse_mapping: Dict[str, str] = {}  # original_id -> blind_id
        self._rng = random.Random(seed)

    def blind_task(self, task: AnnotationTask) -> AnnotationTask:
        """작업을 블라인드 처리"""
        # 새 블라인드 ID 생성
        blind_id = f"EVAL_{self._rng.randint(10000, 99999)}"

        # 매핑 저장
        self._blind_mapping[blind_id] = task.id
        self._reverse_mapping[task.id] = blind_id

        # 블라인드 처리된 복사본 생성
        task.blind_id = blind_id

        return task

    def blind_batch(
        self,
        tasks: List[AnnotationTask],
        shuffle: bool = True,
    ) -> List[AnnotationTask]:
        """배치 블라인드 처리"""
        blinded = [self.blind_task(task) for task in tasks]

        if shuffle:
            self._rng.shuffle(blinded)

        return blinded

    def reveal(self, blind_id: str) -> Optional[str]:
        """블라인드 ID로 원본 ID 조회"""
        return self._blind_mapping.get(blind_id)

    def reveal_results(
        self,
        results: List[AnnotationResult],
    ) -> List[Tuple[AnnotationResult, str]]:
        """결과에서 원본 ID 복원"""
        revealed = []
        for result in results:
            # blind_id로 된 task_id에서 원본 찾기
            original_id = self._blind_mapping.get(result.task_id)
            if original_id is None:
                # task_id가 이미 원본일 수도 있음
                original_id = result.task_id
            revealed.append((result, original_id))
        return revealed


class HumanEvaluator:
    """
    Human Evaluation 관리자

    어노테이션 작업 생성, 할당, 결과 수집
    """

    def __init__(
        self,
        criteria: Optional[List[EvaluationCriteria]] = None,
        min_annotators_per_task: int = 2,
    ):
        self.criteria = criteria or [
            EvaluationCriteria.RELEVANCE,
            EvaluationCriteria.ACCURACY,
            EvaluationCriteria.HELPFULNESS,
        ]
        self.min_annotators = min_annotators_per_task

        self._tasks: Dict[str, AnnotationTask] = {}
        self._results: Dict[str, List[AnnotationResult]] = defaultdict(list)
        self._annotators: Set[str] = set()
        self._blind_evaluator = BlindEvaluator()

    def create_task(
        self,
        question: str,
        answer: str,
        contexts: Optional[List[str]] = None,
        model_source: Optional[str] = None,
        priority: int = 0,
    ) -> AnnotationTask:
        """어노테이션 작업 생성"""
        task = AnnotationTask(
            question=question,
            answer=answer,
            contexts=contexts or [],
            criteria=self.criteria,
            model_source=model_source,
            priority=priority,
        )
        self._tasks[task.id] = task
        logger.info(f"Created annotation task: {task.id}")
        return task

    def create_comparison_task(
        self,
        question: str,
        answers: Dict[str, str],  # model_name -> answer
        contexts: Optional[List[str]] = None,
    ) -> List[AnnotationTask]:
        """비교 평가 작업 생성 (A/B 테스트)"""
        tasks = []
        for model_name, answer in answers.items():
            task = self.create_task(
                question=question,
                answer=answer,
                contexts=contexts,
                model_source=model_name,
            )
            # 블라인드 처리
            self._blind_evaluator.blind_task(task)
            tasks.append(task)
        return tasks

    def assign_task(
        self,
        task_id: str,
        annotator_id: str,
    ) -> bool:
        """작업 할당"""
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.assigned_to = annotator_id
        task.assigned_at = time.time()
        task.status = "assigned"
        self._annotators.add(annotator_id)

        logger.info(f"Task {task_id} assigned to {annotator_id}")
        return True

    def submit_result(
        self,
        task_id: str,
        annotator_id: str,
        scores: Dict[str, int],
        comments: str = "",
        issues: Optional[List[str]] = None,
        time_spent_seconds: float = 0.0,
        confidence: float = 1.0,
    ) -> AnnotationResult:
        """평가 결과 제출"""
        result = AnnotationResult(
            task_id=task_id,
            annotator_id=annotator_id,
            scores=scores,
            comments=comments,
            issues=issues or [],
            time_spent_seconds=time_spent_seconds,
            confidence=confidence,
        )

        self._results[task_id].append(result)

        # 작업 상태 업데이트
        task = self._tasks.get(task_id)
        if task and len(self._results[task_id]) >= self.min_annotators:
            task.status = "completed"

        logger.info(f"Result submitted for task {task_id} by {annotator_id}")
        return result

    def get_pending_tasks(
        self,
        annotator_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[AnnotationTask]:
        """대기 중인 작업 조회"""
        pending = []
        for task in self._tasks.values():
            if task.status == "pending":
                pending.append(task)
            elif task.status == "assigned" and task.assigned_to == annotator_id:
                pending.append(task)

        # 우선순위 정렬
        pending.sort(key=lambda t: (-t.priority, t.created_at))
        return pending[:limit]

    def get_task_results(self, task_id: str) -> List[AnnotationResult]:
        """작업 결과 조회"""
        return self._results.get(task_id, [])

    def compute_inter_rater_reliability(
        self,
        task_ids: Optional[List[str]] = None,
        criterion: Optional[EvaluationCriteria] = None,
    ) -> Dict[str, Any]:
        """평가자 간 신뢰도 계산"""
        if task_ids is None:
            task_ids = list(self._tasks.keys())

        if criterion is None:
            criterion = EvaluationCriteria.RELEVANCE

        # 각 작업별 점수 수집
        all_scores: List[List[int]] = []

        for task_id in task_ids:
            results = self._results.get(task_id, [])
            if len(results) >= 2:
                scores = [
                    r.scores.get(criterion.value, 0)
                    for r in results
                ]
                all_scores.append(scores)

        if not all_scores:
            return {"error": "No valid results for IRR calculation"}

        # 2명의 평가자인 경우 Cohen's Kappa
        if all(len(scores) == 2 for scores in all_scores):
            rater1 = [scores[0] for scores in all_scores]
            rater2 = [scores[1] for scores in all_scores]
            kappa = InterRaterReliability.cohens_kappa(rater1, rater2)

            return {
                "method": "cohens_kappa",
                "kappa": kappa,
                "interpretation": InterRaterReliability.interpret_kappa(kappa),
                "n_items": len(all_scores),
            }

        # 여러 평가자인 경우 Fleiss' Kappa
        max_score = 5  # 1-5 척도 가정
        fleiss_kappa = InterRaterReliability.fleiss_kappa(
            all_scores, n_categories=max_score
        )

        return {
            "method": "fleiss_kappa",
            "kappa": fleiss_kappa,
            "interpretation": InterRaterReliability.interpret_kappa(fleiss_kappa),
            "n_items": len(all_scores),
        }

    def aggregate_scores(
        self,
        task_id: str,
    ) -> Dict[str, float]:
        """작업별 점수 집계"""
        results = self._results.get(task_id, [])
        if not results:
            return {}

        # 기준별 점수 집계
        aggregated = {}
        for criterion in self.criteria:
            scores = [
                r.scores.get(criterion.value, 0)
                for r in results
                if criterion.value in r.scores
            ]
            if scores:
                aggregated[criterion.value] = sum(scores) / len(scores)

        return aggregated

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """전체 평가 요약"""
        total_tasks = len(self._tasks)
        completed_tasks = sum(
            1 for t in self._tasks.values()
            if t.status == "completed"
        )

        # 기준별 평균 점수
        criterion_scores = defaultdict(list)
        for task_id in self._tasks:
            scores = self.aggregate_scores(task_id)
            for criterion, score in scores.items():
                criterion_scores[criterion].append(score)

        avg_scores = {
            criterion: sum(scores) / len(scores) if scores else 0
            for criterion, scores in criterion_scores.items()
        }

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "total_annotators": len(self._annotators),
            "total_annotations": sum(len(r) for r in self._results.values()),
            "average_scores": avg_scores,
        }
