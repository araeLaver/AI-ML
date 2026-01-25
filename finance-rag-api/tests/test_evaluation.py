# -*- coding: utf-8 -*-
"""
평가 시스템 테스트
"""

import time

import pytest

from src.evaluation.ragas import (
    RAGASEvaluator,
    RAGASMetrics,
    EvaluationSample,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    HarmfulnessEvaluator,
)
from src.evaluation.human_eval import (
    HumanEvaluator,
    AnnotationTask,
    AnnotationResult,
    EvaluationCriteria,
    InterRaterReliability,
    BlindEvaluator,
)
from src.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationRun,
    EvaluationReport,
    BatchEvaluator,
    ComparisonAnalyzer,
    EvaluationStatus,
)
from src.evaluation.metrics import (
    MetricAggregator,
    ConfidenceInterval,
    StatisticalTest,
    EffectSize,
)
from src.evaluation.sampling import (
    RandomSampler,
    StratifiedSampler,
    DiversitySampler,
    TemporalSampler,
    ComposedSampler,
)


# =============================================================================
# RAGAS Tests
# =============================================================================

class TestEvaluationSample:
    """평가 샘플 테스트"""

    def test_create_sample(self):
        """샘플 생성 테스트"""
        sample = EvaluationSample(
            question="삼성전자 주가는?",
            answer="삼성전자 주가는 7만원입니다.",
            contexts=["삼성전자의 현재 주가는 약 7만원 수준입니다."],
        )

        assert sample.question == "삼성전자 주가는?"
        assert len(sample.contexts) == 1


class TestFaithfulnessEvaluator:
    """충실도 평가기 테스트"""

    def test_faithful_answer(self):
        """충실한 답변 테스트"""
        evaluator = FaithfulnessEvaluator()

        sample = EvaluationSample(
            question="삼성전자 매출은?",
            answer="삼성전자의 매출은 300조원입니다.",
            contexts=["삼성전자의 2023년 매출은 약 300조원입니다."],
        )

        score = evaluator.evaluate(sample)
        assert score > 0.5

    def test_unfaithful_answer(self):
        """비충실한 답변 테스트"""
        evaluator = FaithfulnessEvaluator()

        sample = EvaluationSample(
            question="삼성전자 매출은?",
            answer="삼성전자의 매출은 500조원이며 세계 1위입니다.",
            contexts=["삼성전자의 2023년 매출은 약 200조원입니다."],
        )

        score = evaluator.evaluate(sample)
        # 컨텍스트에 없는 정보가 있으므로 점수가 낮을 수 있음
        assert 0 <= score <= 1


class TestAnswerRelevancyEvaluator:
    """답변 관련성 평가기 테스트"""

    def test_relevant_answer(self):
        """관련있는 답변 테스트"""
        evaluator = AnswerRelevancyEvaluator()

        sample = EvaluationSample(
            question="삼성전자 주가는 얼마입니까?",
            answer="삼성전자의 현재 주가는 7만원입니다.",
        )

        score = evaluator.evaluate(sample)
        assert 0 <= score <= 1


class TestContextPrecisionEvaluator:
    """컨텍스트 정밀도 평가기 테스트"""

    def test_high_precision(self):
        """높은 정밀도 테스트"""
        evaluator = ContextPrecisionEvaluator()

        sample = EvaluationSample(
            question="삼성전자 매출",
            answer="삼성전자 매출은 300조원입니다.",
            contexts=[
                "삼성전자의 매출은 300조원입니다.",
                "삼성전자의 영업이익은 40조원입니다.",
            ],
        )

        score = evaluator.evaluate(sample)
        assert score > 0


class TestContextRecallEvaluator:
    """컨텍스트 재현율 평가기 테스트"""

    def test_recall_with_ground_truth(self):
        """Ground truth 기반 재현율 테스트"""
        evaluator = ContextRecallEvaluator()

        sample = EvaluationSample(
            question="삼성전자 실적",
            answer="삼성전자의 매출은 300조원입니다.",
            contexts=["삼성전자의 연간 매출은 약 300조원 수준입니다."],
            ground_truth="삼성전자의 매출은 300조원입니다.",
        )

        score = evaluator.evaluate(sample)
        assert score > 0.5


class TestHarmfulnessEvaluator:
    """유해성 평가기 테스트"""

    def test_safe_answer(self):
        """안전한 답변 테스트"""
        evaluator = HarmfulnessEvaluator()

        sample = EvaluationSample(
            answer="삼성전자는 한국의 대표적인 IT 기업입니다.",
        )

        score = evaluator.evaluate(sample)
        assert score == 0.0  # 유해하지 않음

    def test_harmful_content(self):
        """유해 콘텐츠 테스트"""
        evaluator = HarmfulnessEvaluator()

        sample = EvaluationSample(
            answer="이것은 폭력적인 내용입니다.",
        )

        score = evaluator.evaluate(sample)
        assert score > 0


class TestRAGASEvaluator:
    """RAGAS 통합 평가기 테스트"""

    def test_evaluate_sample(self):
        """샘플 평가 테스트"""
        evaluator = RAGASEvaluator()

        sample = EvaluationSample(
            question="삼성전자 주가는?",
            answer="삼성전자의 현재 주가는 7만원입니다.",
            contexts=["삼성전자 주가는 약 7만원입니다."],
            ground_truth="삼성전자 주가는 7만원입니다.",
        )

        metrics = evaluator.evaluate(sample)

        assert isinstance(metrics, RAGASMetrics)
        assert 0 <= metrics.overall_score <= 1
        assert metrics.evaluation_time_ms > 0

    def test_evaluate_batch(self):
        """배치 평가 테스트"""
        evaluator = RAGASEvaluator()

        samples = [
            EvaluationSample(
                question=f"질문 {i}",
                answer=f"답변 {i}",
                contexts=[f"컨텍스트 {i}"],
            )
            for i in range(5)
        ]

        results = evaluator.evaluate_batch(samples)
        assert len(results) == 5

    def test_aggregate_metrics(self):
        """메트릭 집계 테스트"""
        evaluator = RAGASEvaluator()

        metrics_list = [
            RAGASMetrics(faithfulness=0.8, answer_relevancy=0.7),
            RAGASMetrics(faithfulness=0.9, answer_relevancy=0.8),
        ]

        aggregated = evaluator.aggregate_metrics(metrics_list)

        assert abs(aggregated["faithfulness"]["mean"] - 0.85) < 0.0001  # 부동소수점 비교
        assert aggregated["count"] == 2


# =============================================================================
# Human Evaluation Tests
# =============================================================================

class TestInterRaterReliability:
    """평가자 간 신뢰도 테스트"""

    def test_cohens_kappa_perfect_agreement(self):
        """완벽한 일치 Kappa 테스트"""
        rater1 = [1, 2, 3, 4, 5]
        rater2 = [1, 2, 3, 4, 5]

        kappa = InterRaterReliability.cohens_kappa(rater1, rater2)
        assert kappa == 1.0

    def test_cohens_kappa_no_agreement(self):
        """불일치 Kappa 테스트"""
        rater1 = [1, 1, 1, 1, 1]
        rater2 = [2, 2, 2, 2, 2]

        kappa = InterRaterReliability.cohens_kappa(rater1, rater2)
        assert kappa < 0.5

    def test_fleiss_kappa(self):
        """Fleiss Kappa 테스트"""
        # 3명의 평가자가 5개 항목을 평가
        ratings = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 0],
        ]

        kappa = InterRaterReliability.fleiss_kappa(ratings, n_categories=2)
        assert -1 <= kappa <= 1

    def test_interpret_kappa(self):
        """Kappa 해석 테스트"""
        assert "Slight" in InterRaterReliability.interpret_kappa(0.1)
        assert "Moderate" in InterRaterReliability.interpret_kappa(0.5)
        assert "Almost Perfect" in InterRaterReliability.interpret_kappa(0.9)


class TestBlindEvaluator:
    """블라인드 평가기 테스트"""

    def test_blind_task(self):
        """작업 블라인드 처리 테스트"""
        blind_eval = BlindEvaluator(seed=42)

        task = AnnotationTask(
            question="테스트 질문",
            answer="테스트 답변",
            model_source="GPT-4",
        )

        blind_eval.blind_task(task)

        assert task.blind_id is not None
        assert task.blind_id.startswith("EVAL_")

    def test_reveal(self):
        """블라인드 해제 테스트"""
        blind_eval = BlindEvaluator(seed=42)

        task = AnnotationTask(id="original_123")
        blind_eval.blind_task(task)

        revealed = blind_eval.reveal(task.blind_id)
        assert revealed == "original_123"


class TestHumanEvaluator:
    """Human 평가기 테스트"""

    def test_create_task(self):
        """작업 생성 테스트"""
        evaluator = HumanEvaluator()

        task = evaluator.create_task(
            question="테스트 질문",
            answer="테스트 답변",
        )

        assert task.status == "pending"
        assert len(task.criteria) > 0

    def test_assign_and_submit(self):
        """할당 및 제출 테스트"""
        evaluator = HumanEvaluator()

        task = evaluator.create_task(
            question="질문",
            answer="답변",
        )

        # 할당
        evaluator.assign_task(task.id, "annotator_1")

        # 결과 제출
        result = evaluator.submit_result(
            task_id=task.id,
            annotator_id="annotator_1",
            scores={"relevance": 4, "accuracy": 5},
        )

        assert result.annotator_id == "annotator_1"
        assert result.scores["relevance"] == 4

    def test_get_pending_tasks(self):
        """대기 작업 조회 테스트"""
        evaluator = HumanEvaluator()

        for i in range(5):
            evaluator.create_task(
                question=f"질문 {i}",
                answer=f"답변 {i}",
                priority=i,
            )

        pending = evaluator.get_pending_tasks(limit=3)
        assert len(pending) == 3
        # 우선순위 높은 것이 먼저
        assert pending[0].priority >= pending[1].priority

    def test_aggregate_scores(self):
        """점수 집계 테스트"""
        evaluator = HumanEvaluator()

        task = evaluator.create_task("질문", "답변")

        # 두 평가자의 결과
        evaluator.submit_result(task.id, "a1", {"relevance": 4})
        evaluator.submit_result(task.id, "a2", {"relevance": 5})

        aggregated = evaluator.aggregate_scores(task.id)
        assert aggregated["relevance"] == 4.5


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestBatchEvaluator:
    """배치 평가기 테스트"""

    def test_batch_evaluate(self):
        """배치 평가 테스트"""
        batch_eval = BatchEvaluator(batch_size=2)

        samples = [
            EvaluationSample(
                question=f"Q{i}",
                answer=f"A{i}",
                contexts=[f"C{i}"],
            )
            for i in range(5)
        ]

        results = batch_eval.evaluate(samples)
        assert len(results) == 5


class TestComparisonAnalyzer:
    """비교 분석기 테스트"""

    def test_compare_runs(self):
        """두 실행 비교 테스트"""
        analyzer = ComparisonAnalyzer()

        run_a = EvaluationRun(name="Model A")
        run_a.ragas_results = [
            RAGASMetrics(faithfulness=0.7, answer_relevancy=0.6),
        ]

        run_b = EvaluationRun(name="Model B")
        run_b.ragas_results = [
            RAGASMetrics(faithfulness=0.8, answer_relevancy=0.7),
        ]

        comparison = analyzer.compare_runs(run_a, run_b)

        assert "metrics_comparison" in comparison
        assert comparison["winner"] == run_b.id

    def test_rank_runs(self):
        """실행 순위 매기기 테스트"""
        analyzer = ComparisonAnalyzer()

        runs = []
        for i, score in enumerate([0.7, 0.9, 0.8]):
            run = EvaluationRun(name=f"Run {i}")
            run.ragas_results = [
                RAGASMetrics(
                    faithfulness=score,
                    answer_relevancy=score,
                    context_precision=score,
                    context_recall=score,
                )
            ]
            runs.append(run)

        ranked = analyzer.rank_runs(runs)
        assert ranked[0][1] > ranked[1][1]  # 첫 번째가 가장 높은 점수


class TestEvaluationPipeline:
    """평가 파이프라인 테스트"""

    def test_create_and_execute_run(self):
        """실행 생성 및 실행 테스트"""
        pipeline = EvaluationPipeline()

        run = pipeline.create_run(
            name="Test Run",
            model_name="test-model",
        )

        samples = [
            EvaluationSample(
                question="Q",
                answer="A",
                contexts=["C"],
            )
        ]

        executed = pipeline.execute_run(run, samples)

        assert executed.status == EvaluationStatus.COMPLETED
        assert len(executed.ragas_results) == 1

    def test_generate_report(self):
        """리포트 생성 테스트"""
        pipeline = EvaluationPipeline()

        run = pipeline.create_run(name="Test")
        samples = [
            EvaluationSample(question="Q", answer="A", contexts=["C"])
        ]
        pipeline.execute_run(run, samples)

        report = pipeline.generate_report(run)

        assert report.run_id == run.id
        assert "faithfulness" in report.ragas_summary

    def test_report_to_markdown(self):
        """마크다운 리포트 테스트"""
        pipeline = EvaluationPipeline()

        run = pipeline.create_run(name="Test")
        samples = [
            EvaluationSample(question="Q", answer="A", contexts=["C"])
        ]
        pipeline.execute_run(run, samples)

        report = pipeline.generate_report(run)
        markdown = report.to_markdown()

        assert "# Evaluation Report" in markdown


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetricAggregator:
    """메트릭 집계기 테스트"""

    def test_mean(self):
        """평균 테스트"""
        values = [1, 2, 3, 4, 5]
        assert MetricAggregator.mean(values) == 3.0

    def test_std_dev(self):
        """표준편차 테스트"""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        std = MetricAggregator.std_dev(values)
        assert abs(std - 2.138) < 0.01

    def test_median(self):
        """중앙값 테스트"""
        values = [1, 3, 5, 7, 9]
        assert MetricAggregator.median(values) == 5

        values = [1, 2, 3, 4]
        assert MetricAggregator.median(values) == 2.5

    def test_confidence_interval(self):
        """신뢰 구간 테스트"""
        values = [10, 12, 14, 16, 18]
        ci = MetricAggregator.confidence_interval(values, 0.95)

        assert ci.mean == 14.0
        assert ci.lower < ci.mean < ci.upper


class TestStatisticalTest:
    """통계 검정 테스트"""

    def test_t_test(self):
        """t-검정 테스트"""
        group_a = [1, 2, 3, 4, 5]
        group_b = [2, 3, 4, 5, 6]

        result = StatisticalTest.t_test(group_a, group_b)

        assert "t_statistic" in result
        assert "mean_difference" in result

    def test_paired_t_test(self):
        """대응 t-검정 테스트"""
        before = [10, 12, 14, 16, 18]
        after = [11, 13, 15, 17, 19]

        result = StatisticalTest.t_test(before, after, paired=True)

        assert result["mean_difference"] == 1.0


class TestEffectSize:
    """효과 크기 테스트"""

    def test_cohens_d(self):
        """Cohen's d 테스트"""
        group_a = [1, 2, 3, 4, 5]
        group_b = [3, 4, 5, 6, 7]

        result = EffectSize.cohens_d(group_a, group_b)

        assert "d" in result
        assert "interpretation" in result

    def test_hedges_g(self):
        """Hedges' g 테스트"""
        group_a = [1, 2, 3]
        group_b = [4, 5, 6]

        result = EffectSize.hedges_g(group_a, group_b)

        assert "g" in result
        assert "correction_factor" in result


# =============================================================================
# Sampling Tests
# =============================================================================

class TestRandomSampler:
    """무작위 샘플러 테스트"""

    def test_random_sample(self):
        """무작위 샘플링 테스트"""
        sampler = RandomSampler(seed=42)

        samples = [
            EvaluationSample(question=f"Q{i}")
            for i in range(10)
        ]

        result = sampler.sample(samples, 5)
        assert len(result) == 5


class TestStratifiedSampler:
    """계층화 샘플러 테스트"""

    def test_stratified_sample(self):
        """계층화 샘플링 테스트"""
        sampler = StratifiedSampler(
            stratify_by=lambda s: "short" if len(s.answer) < 10 else "long",
            seed=42,
        )

        samples = [
            EvaluationSample(question="Q1", answer="short"),
            EvaluationSample(question="Q2", answer="short"),
            EvaluationSample(question="Q3", answer="this is a long answer"),
            EvaluationSample(question="Q4", answer="this is also long"),
        ]

        result = sampler.sample(samples, 4)
        assert len(result) == 4

    def test_by_question_type(self):
        """질문 유형별 계층화 테스트"""
        sample = EvaluationSample(question="What is the price?")
        assert StratifiedSampler.by_question_type(sample) == "what"


class TestDiversitySampler:
    """다양성 샘플러 테스트"""

    def test_diversity_sample(self):
        """다양성 샘플링 테스트"""
        sampler = DiversitySampler(seed=42)

        samples = [
            EvaluationSample(question="삼성전자 주가"),
            EvaluationSample(question="삼성전자 매출"),
            EvaluationSample(question="애플 주가"),
            EvaluationSample(question="테슬라 실적"),
        ]

        result = sampler.sample(samples, 3)
        assert len(result) == 3


class TestTemporalSampler:
    """시간 기반 샘플러 테스트"""

    def test_temporal_sample(self):
        """시간 기반 샘플링 테스트"""
        sampler = TemporalSampler(
            timestamp_func=lambda s: s.metadata.get("timestamp", 0),
            seed=42,
        )

        samples = []
        for i in range(10):
            s = EvaluationSample(question=f"Q{i}")
            s.metadata["timestamp"] = time.time() - i * 3600
            samples.append(s)

        result = sampler.sample(samples, 5, recent_bias=0.8)
        assert len(result) == 5


class TestComposedSampler:
    """복합 샘플러 테스트"""

    def test_composed_sample(self):
        """복합 샘플링 테스트"""
        random_sampler = RandomSampler(seed=42)
        diversity_sampler = DiversitySampler(seed=42)

        composed = ComposedSampler([
            (random_sampler, 0.5),
            (diversity_sampler, 0.5),
        ])

        samples = [
            EvaluationSample(question=f"Question {i}")
            for i in range(10)
        ]

        result = composed.sample(samples, 6)
        assert len(result) <= 6
