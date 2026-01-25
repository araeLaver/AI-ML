# -*- coding: utf-8 -*-
"""
피드백 모듈 테스트
"""

import os
import tempfile
import time

import pytest

from src.feedback.collector import (
    FeedbackCollector,
    FeedbackType,
    FeedbackData,
    FeedbackSentiment,
    ImplicitFeedbackTracker,
    ExplicitFeedbackHandler,
)
from src.feedback.storage import (
    FeedbackStorage,
    InMemoryFeedbackStorage,
    SQLFeedbackStorage,
    FeedbackQuery,
)
from src.feedback.analyzer import (
    FeedbackAnalyzer,
    QualityMetrics,
    TrendAnalyzer,
    SentimentAnalyzer,
)
from src.feedback.learning import (
    LearningPipeline,
    RetrainTrigger,
    DataSelector,
    IncrementalLearner,
    TrainingJob,
    TriggerType,
)
from src.feedback.ab_testing import (
    ABTestManager,
    Experiment,
    Variant,
    StatisticalAnalyzer,
    ExperimentStatus,
)
from src.feedback.active_learning import (
    ActiveLearner,
    UncertaintySampler,
    DiversitySampler,
    UnlabeledSample,
    SamplingStrategy,
)


# =============================================================================
# Collector Tests
# =============================================================================

class TestFeedbackCollector:
    """피드백 수집기 테스트"""

    def test_collect_thumbs_up(self):
        """좋아요 수집 테스트"""
        collector = FeedbackCollector()
        feedback = collector.collect_thumbs_up(
            query_id="q1",
            query="삼성전자 실적",
            response="삼성전자 실적은...",
            user_id="user1",
        )

        assert feedback.feedback_type == FeedbackType.THUMBS_UP
        assert feedback.sentiment == FeedbackSentiment.POSITIVE
        assert feedback.value == 1

    def test_collect_thumbs_down(self):
        """싫어요 수집 테스트"""
        collector = FeedbackCollector()
        feedback = collector.collect_thumbs_down(
            query_id="q2",
            query="애플 주가",
            response="애플 주가는...",
            reason="정보가 오래됨",
        )

        assert feedback.feedback_type == FeedbackType.THUMBS_DOWN
        assert feedback.sentiment == FeedbackSentiment.NEGATIVE
        assert feedback.metadata.get("reason") == "정보가 오래됨"

    def test_collect_rating(self):
        """평점 수집 테스트"""
        collector = FeedbackCollector()

        # 높은 평점
        feedback = collector.collect_rating(
            query_id="q3",
            query="테스트",
            response="응답",
            rating=5,
        )
        assert feedback.sentiment == FeedbackSentiment.POSITIVE

        # 낮은 평점
        feedback = collector.collect_rating(
            query_id="q4",
            query="테스트",
            response="응답",
            rating=2,
        )
        assert feedback.sentiment == FeedbackSentiment.NEGATIVE

    def test_collect_correction(self):
        """수정 제안 수집 테스트"""
        collector = FeedbackCollector()
        feedback = collector.collect_correction(
            query_id="q5",
            query="삼성전자 CEO",
            original_response="이재용",
            corrected_response="이재용 (부회장)",
        )

        assert feedback.feedback_type == FeedbackType.CORRECTION
        assert feedback.value == "이재용 (부회장)"

    def test_callback(self):
        """콜백 테스트"""
        received = []

        def on_feedback(fb):
            received.append(fb)

        collector = FeedbackCollector(on_feedback=on_feedback)
        collector.collect_thumbs_up("q1", "test", "response")

        assert len(received) == 1


class TestImplicitFeedbackTracker:
    """암시적 피드백 추적기 테스트"""

    def test_track_dwell_time(self):
        """체류 시간 추적 테스트"""
        collector = FeedbackCollector()
        tracker = ImplicitFeedbackTracker(collector)

        tracker.track_query_start("q1")
        time.sleep(0.1)
        feedback = tracker.track_dwell_time("q1", "query", "response")

        assert feedback is not None
        assert feedback.feedback_type == FeedbackType.DWELL_TIME
        assert feedback.value >= 0.1

    def test_track_click(self):
        """클릭 추적 테스트"""
        collector = FeedbackCollector()
        tracker = ImplicitFeedbackTracker(collector)

        feedback = tracker.track_click("q1", "query", "response", "link1")

        assert feedback.feedback_type == FeedbackType.CLICK
        assert feedback.sentiment == FeedbackSentiment.POSITIVE

    def test_track_follow_up(self):
        """후속 질문 추적 테스트"""
        collector = FeedbackCollector()
        tracker = ImplicitFeedbackTracker(collector)

        # 유사한 질문 (부정적 신호)
        feedback = tracker.track_follow_up(
            "q1",
            "삼성전자 실적",
            "삼성전자 실적은...",
            "삼성전자 2024년 실적",  # 유사
        )

        assert feedback.feedback_type == FeedbackType.FOLLOW_UP
        assert feedback.metadata.get("similarity", 0) > 0


class TestExplicitFeedbackHandler:
    """명시적 피드백 핸들러 테스트"""

    def test_handle_reaction(self):
        """반응 처리 테스트"""
        collector = FeedbackCollector()
        handler = ExplicitFeedbackHandler(collector)

        fb1 = handler.handle_reaction("q1", "query", "response", "up")
        assert fb1.feedback_type == FeedbackType.THUMBS_UP

        fb2 = handler.handle_reaction("q2", "query", "response", "down")
        assert fb2.feedback_type == FeedbackType.THUMBS_DOWN

    def test_handle_rating_with_comment(self):
        """평점 + 코멘트 처리 테스트"""
        collector = FeedbackCollector()
        handler = ExplicitFeedbackHandler(collector)

        feedbacks = handler.handle_rating(
            "q1", "query", "response",
            rating=4,
            comment="좋은 답변입니다",
        )

        assert len(feedbacks) == 2
        assert feedbacks[0].feedback_type == FeedbackType.RATING
        assert feedbacks[1].feedback_type == FeedbackType.COMMENT


# =============================================================================
# Storage Tests
# =============================================================================

class TestInMemoryStorage:
    """인메모리 저장소 테스트"""

    def test_save_and_get(self):
        """저장 및 조회 테스트"""
        storage = InMemoryFeedbackStorage()

        feedback = FeedbackData(
            query_id="q1",
            query="test",
            response="response",
        )
        storage.save(feedback)

        retrieved = storage.get(feedback.id)
        assert retrieved is not None
        assert retrieved.query == "test"

    def test_query(self):
        """쿼리 테스트"""
        storage = InMemoryFeedbackStorage()

        # 다양한 피드백 저장
        for i in range(10):
            fb = FeedbackData(
                query_id=f"q{i}",
                query=f"query {i}",
                response=f"response {i}",
                sentiment=FeedbackSentiment.POSITIVE if i % 2 == 0 else FeedbackSentiment.NEGATIVE,
            )
            storage.save(fb)

        # 긍정 피드백만 조회
        positive = storage.query(FeedbackQuery(
            sentiments=[FeedbackSentiment.POSITIVE],
        ))
        assert len(positive) == 5

    def test_max_size(self):
        """최대 크기 테스트"""
        storage = InMemoryFeedbackStorage(max_size=5)

        for i in range(10):
            fb = FeedbackData(query_id=f"q{i}", query=f"q{i}", response="r")
            storage.save(fb)

        assert storage.count() == 5


class TestSQLStorage:
    """SQL 저장소 테스트"""

    def test_save_and_get(self):
        """저장 및 조회 테스트"""
        db_path = tempfile.mktemp(suffix=".db")

        try:
            storage = SQLFeedbackStorage(db_path)

            feedback = FeedbackData(
                query_id="q1",
                query="test query",
                response="test response",
                feedback_type=FeedbackType.RATING,
                value=5,
            )
            storage.save(feedback)

            retrieved = storage.get(feedback.id)
            assert retrieved is not None
            assert retrieved.query == "test query"
            assert retrieved.value == 5

            # 명시적으로 연결 해제를 위해 del
            del storage
        finally:
            try:
                os.unlink(db_path)
            except (OSError, PermissionError):
                pass  # Windows에서 파일 잠금 이슈 무시

    def test_statistics(self):
        """통계 테스트"""
        db_path = tempfile.mktemp(suffix=".db")

        try:
            storage = SQLFeedbackStorage(db_path)

            # 피드백 추가
            for i in range(10):
                fb = FeedbackData(
                    query_id=f"q{i}",
                    query=f"query {i}",
                    response="response",
                    feedback_type=FeedbackType.RATING if i % 2 == 0 else FeedbackType.THUMBS_UP,
                    value=4 if i % 2 == 0 else 1,
                    sentiment=FeedbackSentiment.POSITIVE,
                )
                storage.save(fb)

            stats = storage.get_statistics()
            assert stats["total"] == 10
            assert "by_type" in stats

            del storage
        finally:
            try:
                os.unlink(db_path)
            except (OSError, PermissionError):
                pass  # Windows에서 파일 잠금 이슈 무시


# =============================================================================
# Analyzer Tests
# =============================================================================

class TestFeedbackAnalyzer:
    """피드백 분석기 테스트"""

    def test_calculate_metrics(self):
        """메트릭 계산 테스트"""
        storage = InMemoryFeedbackStorage()

        # 피드백 추가
        for i in range(100):
            fb = FeedbackData(
                query_id=f"q{i}",
                query=f"query {i}",
                response="response",
                feedback_type=FeedbackType.RATING,
                value=4 if i < 70 else 2,  # 70% 긍정
                sentiment=FeedbackSentiment.POSITIVE if i < 70 else FeedbackSentiment.NEGATIVE,
            )
            storage.save(fb)

        analyzer = FeedbackAnalyzer(storage)
        metrics = analyzer.calculate_metrics()

        assert metrics.total_feedback == 100
        assert metrics.positive_count == 70
        assert metrics.satisfaction_rate == 0.7

    def test_get_top_issues(self):
        """상위 이슈 테스트"""
        storage = InMemoryFeedbackStorage()

        # 같은 쿼리에 대한 부정 피드백
        for i in range(5):
            fb = FeedbackData(
                query_id=f"q{i}",
                query="문제 쿼리",
                response="잘못된 응답",
                sentiment=FeedbackSentiment.NEGATIVE,
            )
            storage.save(fb)

        analyzer = FeedbackAnalyzer(storage)
        issues = analyzer.get_top_issues()

        assert len(issues) >= 1
        assert issues[0]["negative_count"] == 5


class TestSentimentAnalyzer:
    """감정 분석기 테스트"""

    def test_analyze_positive(self):
        """긍정 분석 테스트"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("정말 좋은 답변입니다! 만족합니다.")

        assert result["sentiment"] == FeedbackSentiment.POSITIVE

    def test_analyze_negative(self):
        """부정 분석 테스트"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("별로입니다. 틀린 정보가 있어요.")

        assert result["sentiment"] == FeedbackSentiment.NEGATIVE


# =============================================================================
# Learning Tests
# =============================================================================

class TestRetrainTrigger:
    """재훈련 트리거 테스트"""

    def test_should_retrain_low_satisfaction(self):
        """낮은 만족도 재훈련 테스트"""
        storage = InMemoryFeedbackStorage()

        # 부정 피드백 추가
        for i in range(200):
            fb = FeedbackData(
                query_id=f"q{i}",
                query=f"query {i}",
                response="response",
                sentiment=FeedbackSentiment.NEGATIVE if i < 100 else FeedbackSentiment.POSITIVE,
            )
            storage.save(fb)

        trigger = RetrainTrigger(
            storage,
            satisfaction_threshold=0.7,
            min_feedback_count=100,
        )
        should_retrain, reason = trigger.should_retrain()

        assert should_retrain is True
        assert "만족도" in reason


class TestDataSelector:
    """데이터 선택기 테스트"""

    def test_select_training_data(self):
        """훈련 데이터 선택 테스트"""
        storage = InMemoryFeedbackStorage()

        # 수정 제안
        fb1 = FeedbackData(
            query_id="q1",
            query="query1",
            response="original",
            value="corrected",
            feedback_type=FeedbackType.CORRECTION,
        )
        storage.save(fb1)

        # 긍정 피드백
        fb2 = FeedbackData(
            query_id="q2",
            query="query2",
            response="good response",
            feedback_type=FeedbackType.THUMBS_UP,
            sentiment=FeedbackSentiment.POSITIVE,
        )
        storage.save(fb2)

        selector = DataSelector(storage)
        data = selector.select_training_data(max_samples=10)

        assert len(data) >= 2
        assert any(d["type"] == "correction" for d in data)
        assert any(d["type"] == "positive" for d in data)


class TestLearningPipeline:
    """학습 파이프라인 테스트"""

    def test_run_training_job(self):
        """훈련 작업 실행 테스트"""
        storage = InMemoryFeedbackStorage()

        # 피드백 추가
        for i in range(50):
            fb = FeedbackData(
                query_id=f"q{i}",
                query=f"query {i}",
                response="response",
                feedback_type=FeedbackType.THUMBS_UP,
                sentiment=FeedbackSentiment.POSITIVE,
            )
            storage.save(fb)

        pipeline = LearningPipeline(storage)
        job = pipeline.run_training_job(TriggerType.MANUAL)

        assert job.status == "completed"
        assert job.data_count > 0


# =============================================================================
# A/B Testing Tests
# =============================================================================

class TestStatisticalAnalyzer:
    """통계 분석기 테스트"""

    def test_z_score(self):
        """Z-score 계산 테스트"""
        analyzer = StatisticalAnalyzer()

        # 같은 비율
        z = analyzer.calculate_z_score(0.5, 0.5, 100, 100)
        assert z == 0.0

        # 다른 비율
        z = analyzer.calculate_z_score(0.4, 0.6, 1000, 1000)
        assert z != 0.0

    def test_compare_variants(self):
        """변형 비교 테스트"""
        analyzer = StatisticalAnalyzer()

        control = Variant(name="control")
        control.impressions = 1000
        control.conversions = 100

        treatment = Variant(name="treatment")
        treatment.impressions = 1000
        treatment.conversions = 150

        result = analyzer.compare_variants(control, treatment, "conversion_rate")

        assert result["control"]["value"] == 0.1
        assert result["treatment"]["value"] == 0.15
        assert "p_value" in result


class TestABTestManager:
    """A/B 테스트 관리자 테스트"""

    def test_create_experiment(self):
        """실험 생성 테스트"""
        manager = ABTestManager()

        exp = manager.create_experiment(
            name="Test Experiment",
            variants=[
                {"name": "control", "weight": 1.0},
                {"name": "treatment", "weight": 1.0},
            ],
        )

        assert exp.name == "Test Experiment"
        assert len(exp.variants) == 2
        assert exp.status == ExperimentStatus.DRAFT

    def test_get_variant(self):
        """변형 할당 테스트"""
        manager = ABTestManager()

        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "A", "weight": 1.0},
                {"name": "B", "weight": 1.0},
            ],
        )
        manager.start_experiment(exp.id)

        # 여러 사용자에게 변형 할당
        variants = [manager.get_variant(exp.id, f"user{i}") for i in range(100)]
        variant_names = [v.name for v in variants if v]

        # 두 변형 모두 할당됨
        assert "A" in variant_names
        assert "B" in variant_names

    def test_sticky_assignment(self):
        """고정 할당 테스트"""
        manager = ABTestManager(sticky_assignment=True)

        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "A", "weight": 1.0},
                {"name": "B", "weight": 1.0},
            ],
        )
        manager.start_experiment(exp.id)

        # 같은 사용자는 같은 변형
        variant1 = manager.get_variant(exp.id, "user1")
        variant2 = manager.get_variant(exp.id, "user1")

        assert variant1.id == variant2.id


# =============================================================================
# Active Learning Tests
# =============================================================================

class TestUncertaintySampler:
    """불확실성 샘플러 테스트"""

    def test_select_uncertain(self):
        """불확실한 샘플 선택 테스트"""
        sampler = UncertaintySampler(threshold=0.5)

        samples = [
            UnlabeledSample(query="q1", response="r1", confidence=0.9),
            UnlabeledSample(query="q2", response="r2", confidence=0.3),
            UnlabeledSample(query="q3", response="r3", confidence=0.1),
            UnlabeledSample(query="q4", response="r4", confidence=0.7),
        ]

        selected = sampler.select(samples, 2)

        assert len(selected) == 2
        # 가장 불확실한 샘플이 선택됨
        assert selected[0].confidence <= selected[1].confidence


class TestDiversitySampler:
    """다양성 샘플러 테스트"""

    def test_select_diverse(self):
        """다양한 샘플 선택 테스트"""
        sampler = DiversitySampler()

        samples = [
            UnlabeledSample(query="삼성전자 실적", response="r1", confidence=0.5),
            UnlabeledSample(query="삼성전자 주가", response="r2", confidence=0.5),
            UnlabeledSample(query="애플 실적", response="r3", confidence=0.5),
            UnlabeledSample(query="테슬라 주가", response="r4", confidence=0.5),
        ]

        selected = sampler.select(samples, 3)

        assert len(selected) == 3


class TestActiveLearner:
    """능동 학습기 테스트"""

    def test_add_and_select(self):
        """샘플 추가 및 선택 테스트"""
        learner = ActiveLearner(strategy=SamplingStrategy.UNCERTAINTY)

        # 샘플 추가
        for i in range(10):
            learner.add_sample(
                query=f"query {i}",
                response=f"response {i}",
                confidence=0.1 * i,
            )

        # 선택
        selected = learner.select_for_labeling(3)

        assert len(selected) == 3
        # 가장 불확실한 샘플이 선택됨
        assert all(s.confidence < 0.5 for s in selected)

    def test_submit_label(self):
        """레이블 제출 테스트"""
        learner = ActiveLearner()

        sample = learner.add_sample(
            query="test query",
            response="test response",
            confidence=0.3,
        )

        success = learner.submit_label(
            sample.id,
            label="corrected response",
            labeler_id="annotator1",
        )

        assert success is True

        training_data = learner.get_training_data()
        assert len(training_data) == 1
        assert training_data[0]["response"] == "corrected response"

    def test_statistics(self):
        """통계 테스트"""
        learner = ActiveLearner()

        for i in range(5):
            learner.add_sample(f"q{i}", f"r{i}", 0.5)

        stats = learner.get_statistics()

        assert stats["unlabeled_pool_size"] == 5
        assert stats["labeled_count"] == 0
