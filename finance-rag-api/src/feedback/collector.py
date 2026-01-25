# -*- coding: utf-8 -*-
"""
피드백 수집 모듈

[기능]
- 명시적 피드백 (좋아요/싫어요, 평점, 코멘트)
- 암시적 피드백 (클릭, 체류 시간, 재질의)
- 피드백 검증 및 정규화
"""

import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """피드백 유형"""
    # 명시적 피드백
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5점
    COMMENT = "comment"
    CORRECTION = "correction"  # 답변 수정 제안

    # 암시적 피드백
    CLICK = "click"
    DWELL_TIME = "dwell_time"
    COPY = "copy"
    SHARE = "share"
    FOLLOW_UP = "follow_up"  # 후속 질문
    ABANDON = "abandon"  # 중도 이탈
    RETRY = "retry"  # 재시도


class FeedbackSentiment(Enum):
    """피드백 감정"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FeedbackData:
    """피드백 데이터"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP
    query_id: str = ""
    query: str = ""
    response: str = ""
    value: Any = None  # 피드백 값 (점수, 텍스트 등)
    sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "feedback_type": self.feedback_type.value,
            "query_id": self.query_id,
            "query": self.query,
            "response": self.response[:500] if self.response else "",
            "value": self.value,
            "sentiment": self.sentiment.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackData":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            feedback_type=FeedbackType(data.get("feedback_type", "thumbs_up")),
            query_id=data.get("query_id", ""),
            query=data.get("query", ""),
            response=data.get("response", ""),
            value=data.get("value"),
            sentiment=FeedbackSentiment(data.get("sentiment", "neutral")),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class FeedbackCollector:
    """
    피드백 수집기

    명시적/암시적 피드백을 수집하고 처리
    """

    def __init__(
        self,
        storage=None,
        validators: List[Callable[[FeedbackData], bool]] = None,
        on_feedback: Callable[[FeedbackData], None] = None,
    ):
        self.storage = storage
        self.validators = validators or []
        self.on_feedback = on_feedback
        self._feedback_buffer: List[FeedbackData] = []
        self._buffer_size = 100

    def collect(self, feedback: FeedbackData) -> bool:
        """피드백 수집"""
        # 검증
        if not self._validate(feedback):
            logger.warning(f"Invalid feedback: {feedback.id}")
            return False

        # 감정 분류
        if feedback.sentiment == FeedbackSentiment.NEUTRAL:
            feedback.sentiment = self._classify_sentiment(feedback)

        # 저장
        if self.storage:
            self.storage.save(feedback)
        else:
            self._feedback_buffer.append(feedback)
            if len(self._feedback_buffer) > self._buffer_size:
                self._feedback_buffer = self._feedback_buffer[-self._buffer_size:]

        # 콜백
        if self.on_feedback:
            try:
                self.on_feedback(feedback)
            except Exception as e:
                logger.error(f"Feedback callback error: {e}")

        logger.info(f"Collected feedback: {feedback.feedback_type.value} for query {feedback.query_id}")
        return True

    def collect_thumbs_up(
        self,
        query_id: str,
        query: str,
        response: str,
        user_id: Optional[str] = None,
        **metadata,
    ) -> FeedbackData:
        """좋아요 수집"""
        feedback = FeedbackData(
            feedback_type=FeedbackType.THUMBS_UP,
            query_id=query_id,
            query=query,
            response=response,
            value=1,
            sentiment=FeedbackSentiment.POSITIVE,
            user_id=user_id,
            metadata=metadata,
        )
        self.collect(feedback)
        return feedback

    def collect_thumbs_down(
        self,
        query_id: str,
        query: str,
        response: str,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata,
    ) -> FeedbackData:
        """싫어요 수집"""
        if reason:
            metadata["reason"] = reason

        feedback = FeedbackData(
            feedback_type=FeedbackType.THUMBS_DOWN,
            query_id=query_id,
            query=query,
            response=response,
            value=-1,
            sentiment=FeedbackSentiment.NEGATIVE,
            user_id=user_id,
            metadata=metadata,
        )
        self.collect(feedback)
        return feedback

    def collect_rating(
        self,
        query_id: str,
        query: str,
        response: str,
        rating: int,
        user_id: Optional[str] = None,
        **metadata,
    ) -> FeedbackData:
        """평점 수집 (1-5)"""
        rating = max(1, min(5, rating))  # 1-5 범위로 제한

        sentiment = FeedbackSentiment.NEUTRAL
        if rating >= 4:
            sentiment = FeedbackSentiment.POSITIVE
        elif rating <= 2:
            sentiment = FeedbackSentiment.NEGATIVE

        feedback = FeedbackData(
            feedback_type=FeedbackType.RATING,
            query_id=query_id,
            query=query,
            response=response,
            value=rating,
            sentiment=sentiment,
            user_id=user_id,
            metadata=metadata,
        )
        self.collect(feedback)
        return feedback

    def collect_comment(
        self,
        query_id: str,
        query: str,
        response: str,
        comment: str,
        user_id: Optional[str] = None,
        **metadata,
    ) -> FeedbackData:
        """코멘트 수집"""
        feedback = FeedbackData(
            feedback_type=FeedbackType.COMMENT,
            query_id=query_id,
            query=query,
            response=response,
            value=comment,
            user_id=user_id,
            metadata=metadata,
        )
        self.collect(feedback)
        return feedback

    def collect_correction(
        self,
        query_id: str,
        query: str,
        original_response: str,
        corrected_response: str,
        user_id: Optional[str] = None,
        **metadata,
    ) -> FeedbackData:
        """수정 제안 수집"""
        feedback = FeedbackData(
            feedback_type=FeedbackType.CORRECTION,
            query_id=query_id,
            query=query,
            response=original_response,
            value=corrected_response,
            sentiment=FeedbackSentiment.NEGATIVE,  # 수정이 필요했으므로
            user_id=user_id,
            metadata={**metadata, "original_response": original_response},
        )
        self.collect(feedback)
        return feedback

    def _validate(self, feedback: FeedbackData) -> bool:
        """피드백 검증"""
        # 기본 검증
        if not feedback.query_id or not feedback.query:
            return False

        # 커스텀 검증
        for validator in self.validators:
            if not validator(feedback):
                return False

        return True

    def _classify_sentiment(self, feedback: FeedbackData) -> FeedbackSentiment:
        """피드백 감정 분류"""
        if feedback.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.COPY, FeedbackType.SHARE]:
            return FeedbackSentiment.POSITIVE
        elif feedback.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.ABANDON]:
            return FeedbackSentiment.NEGATIVE
        elif feedback.feedback_type == FeedbackType.RATING and feedback.value:
            if feedback.value >= 4:
                return FeedbackSentiment.POSITIVE
            elif feedback.value <= 2:
                return FeedbackSentiment.NEGATIVE

        return FeedbackSentiment.NEUTRAL

    def get_recent(self, limit: int = 100) -> List[FeedbackData]:
        """최근 피드백 조회"""
        if self.storage:
            return self.storage.get_recent(limit)
        return self._feedback_buffer[-limit:]


class ImplicitFeedbackTracker:
    """
    암시적 피드백 추적기

    사용자 행동 기반 피드백 수집
    """

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._query_starts: Dict[str, float] = {}

    def start_session(self, session_id: str, user_id: Optional[str] = None) -> None:
        """세션 시작"""
        self._sessions[session_id] = {
            "user_id": user_id,
            "start_time": time.time(),
            "queries": [],
            "interactions": [],
        }

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """세션 종료 및 요약"""
        if session_id not in self._sessions:
            return {}

        session = self._sessions.pop(session_id)
        session["end_time"] = time.time()
        session["duration"] = session["end_time"] - session["start_time"]

        return session

    def track_query_start(self, query_id: str) -> None:
        """쿼리 시작 추적"""
        self._query_starts[query_id] = time.time()

    def track_dwell_time(
        self,
        query_id: str,
        query: str,
        response: str,
        session_id: Optional[str] = None,
    ) -> Optional[FeedbackData]:
        """체류 시간 추적"""
        start_time = self._query_starts.pop(query_id, None)
        if not start_time:
            return None

        dwell_time = time.time() - start_time

        # 짧은 체류 시간은 부정적 신호
        sentiment = FeedbackSentiment.NEUTRAL
        if dwell_time < 2.0:  # 2초 미만
            sentiment = FeedbackSentiment.NEGATIVE
        elif dwell_time > 10.0:  # 10초 이상
            sentiment = FeedbackSentiment.POSITIVE

        feedback = FeedbackData(
            feedback_type=FeedbackType.DWELL_TIME,
            query_id=query_id,
            query=query,
            response=response,
            value=dwell_time,
            sentiment=sentiment,
            session_id=session_id,
        )

        self.collector.collect(feedback)
        return feedback

    def track_click(
        self,
        query_id: str,
        query: str,
        response: str,
        clicked_element: str,
        session_id: Optional[str] = None,
    ) -> FeedbackData:
        """클릭 추적"""
        feedback = FeedbackData(
            feedback_type=FeedbackType.CLICK,
            query_id=query_id,
            query=query,
            response=response,
            value=clicked_element,
            sentiment=FeedbackSentiment.POSITIVE,
            session_id=session_id,
        )

        self.collector.collect(feedback)
        return feedback

    def track_copy(
        self,
        query_id: str,
        query: str,
        response: str,
        copied_text: str,
        session_id: Optional[str] = None,
    ) -> FeedbackData:
        """복사 추적"""
        feedback = FeedbackData(
            feedback_type=FeedbackType.COPY,
            query_id=query_id,
            query=query,
            response=response,
            value=copied_text[:200],
            sentiment=FeedbackSentiment.POSITIVE,
            session_id=session_id,
        )

        self.collector.collect(feedback)
        return feedback

    def track_follow_up(
        self,
        original_query_id: str,
        original_query: str,
        original_response: str,
        follow_up_query: str,
        session_id: Optional[str] = None,
    ) -> FeedbackData:
        """후속 질문 추적"""
        # 후속 질문이 유사하면 부정적 (재질의 필요)
        # 다른 주제면 중립적
        similarity = self._calculate_similarity(original_query, follow_up_query)

        sentiment = FeedbackSentiment.NEUTRAL
        if similarity > 0.7:  # 유사한 질문 = 불충분한 답변
            sentiment = FeedbackSentiment.NEGATIVE

        feedback = FeedbackData(
            feedback_type=FeedbackType.FOLLOW_UP,
            query_id=original_query_id,
            query=original_query,
            response=original_response,
            value=follow_up_query,
            sentiment=sentiment,
            session_id=session_id,
            metadata={"similarity": similarity},
        )

        self.collector.collect(feedback)
        return feedback

    def track_abandon(
        self,
        query_id: str,
        query: str,
        partial_response: str,
        session_id: Optional[str] = None,
    ) -> FeedbackData:
        """중도 이탈 추적"""
        feedback = FeedbackData(
            feedback_type=FeedbackType.ABANDON,
            query_id=query_id,
            query=query,
            response=partial_response,
            value=None,
            sentiment=FeedbackSentiment.NEGATIVE,
            session_id=session_id,
        )

        self.collector.collect(feedback)
        return feedback

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """간단한 텍스트 유사도 계산"""
        # 단어 집합 기반 Jaccard 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class ExplicitFeedbackHandler:
    """
    명시적 피드백 핸들러

    UI에서 직접 수집되는 피드백 처리
    """

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector

    def handle_reaction(
        self,
        query_id: str,
        query: str,
        response: str,
        reaction: str,
        user_id: Optional[str] = None,
        **metadata,
    ) -> FeedbackData:
        """반응 처리 (좋아요/싫어요)"""
        if reaction.lower() in ["up", "like", "thumbs_up", "👍"]:
            return self.collector.collect_thumbs_up(
                query_id, query, response, user_id, **metadata
            )
        elif reaction.lower() in ["down", "dislike", "thumbs_down", "👎"]:
            return self.collector.collect_thumbs_down(
                query_id, query, response, user_id=user_id, **metadata
            )
        else:
            raise ValueError(f"Unknown reaction: {reaction}")

    def handle_rating(
        self,
        query_id: str,
        query: str,
        response: str,
        rating: int,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[FeedbackData]:
        """평점 및 코멘트 처리"""
        feedbacks = []

        # 평점
        rating_feedback = self.collector.collect_rating(
            query_id, query, response, rating, user_id
        )
        feedbacks.append(rating_feedback)

        # 코멘트 (선택적)
        if comment:
            comment_feedback = self.collector.collect_comment(
                query_id, query, response, comment, user_id
            )
            feedbacks.append(comment_feedback)

        return feedbacks

    def handle_correction(
        self,
        query_id: str,
        query: str,
        original_response: str,
        corrected_response: str,
        correction_type: str = "general",
        user_id: Optional[str] = None,
    ) -> FeedbackData:
        """수정 제안 처리"""
        return self.collector.collect_correction(
            query_id=query_id,
            query=query,
            original_response=original_response,
            corrected_response=corrected_response,
            user_id=user_id,
            correction_type=correction_type,
        )

    def handle_survey(
        self,
        query_id: str,
        query: str,
        response: str,
        survey_responses: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> FeedbackData:
        """설문 응답 처리"""
        # 설문 응답을 종합하여 피드백 생성
        overall_score = survey_responses.get("overall_score", 3)

        feedback = FeedbackData(
            feedback_type=FeedbackType.RATING,
            query_id=query_id,
            query=query,
            response=response,
            value=overall_score,
            sentiment=self._score_to_sentiment(overall_score),
            user_id=user_id,
            metadata={"survey_responses": survey_responses},
        )

        self.collector.collect(feedback)
        return feedback

    def _score_to_sentiment(self, score: float) -> FeedbackSentiment:
        """점수를 감정으로 변환"""
        if score >= 4:
            return FeedbackSentiment.POSITIVE
        elif score <= 2:
            return FeedbackSentiment.NEGATIVE
        return FeedbackSentiment.NEUTRAL
