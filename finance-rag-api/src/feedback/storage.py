# -*- coding: utf-8 -*-
"""
피드백 저장소 모듈

[기능]
- 피드백 저장 및 조회
- 다양한 백엔드 지원 (메모리, SQL, NoSQL)
- 쿼리 및 필터링
"""

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .collector import FeedbackData, FeedbackSentiment, FeedbackType

logger = logging.getLogger(__name__)


@dataclass
class FeedbackQuery:
    """피드백 쿼리"""
    feedback_types: Optional[List[FeedbackType]] = None
    sentiments: Optional[List[FeedbackSentiment]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_contains: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    limit: int = 100
    offset: int = 0

    def matches(self, feedback: FeedbackData) -> bool:
        """피드백이 쿼리 조건에 맞는지 확인"""
        if self.feedback_types and feedback.feedback_type not in self.feedback_types:
            return False
        if self.sentiments and feedback.sentiment not in self.sentiments:
            return False
        if self.user_id and feedback.user_id != self.user_id:
            return False
        if self.session_id and feedback.session_id != self.session_id:
            return False
        if self.query_contains and self.query_contains.lower() not in feedback.query.lower():
            return False
        if self.start_time and feedback.timestamp < self.start_time:
            return False
        if self.end_time and feedback.timestamp > self.end_time:
            return False
        return True


class FeedbackStorage(ABC):
    """피드백 저장소 인터페이스"""

    @abstractmethod
    def save(self, feedback: FeedbackData) -> bool:
        """피드백 저장"""
        pass

    @abstractmethod
    def get(self, feedback_id: str) -> Optional[FeedbackData]:
        """ID로 피드백 조회"""
        pass

    @abstractmethod
    def query(self, query: FeedbackQuery) -> List[FeedbackData]:
        """쿼리로 피드백 조회"""
        pass

    @abstractmethod
    def get_recent(self, limit: int = 100) -> List[FeedbackData]:
        """최근 피드백 조회"""
        pass

    @abstractmethod
    def count(self, query: Optional[FeedbackQuery] = None) -> int:
        """피드백 개수"""
        pass

    @abstractmethod
    def delete(self, feedback_id: str) -> bool:
        """피드백 삭제"""
        pass

    def get_by_query_id(self, query_id: str) -> List[FeedbackData]:
        """특정 쿼리에 대한 피드백 조회"""
        query = FeedbackQuery(limit=1000)
        all_feedback = self.query(query)
        return [f for f in all_feedback if f.query_id == query_id]

    def get_statistics(self) -> Dict[str, Any]:
        """피드백 통계"""
        all_feedback = self.query(FeedbackQuery(limit=10000))

        stats = {
            "total": len(all_feedback),
            "by_type": {},
            "by_sentiment": {},
            "avg_rating": None,
        }

        ratings = []
        for fb in all_feedback:
            # 유형별 카운트
            type_key = fb.feedback_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

            # 감정별 카운트
            sentiment_key = fb.sentiment.value
            stats["by_sentiment"][sentiment_key] = stats["by_sentiment"].get(sentiment_key, 0) + 1

            # 평점 수집
            if fb.feedback_type == FeedbackType.RATING and fb.value:
                ratings.append(fb.value)

        # 평균 평점
        if ratings:
            stats["avg_rating"] = round(sum(ratings) / len(ratings), 2)

        return stats


class InMemoryFeedbackStorage(FeedbackStorage):
    """
    인메모리 피드백 저장소

    개발/테스트용
    """

    def __init__(self, max_size: int = 10000):
        self._storage: Dict[str, FeedbackData] = {}
        self._ordered_ids: List[str] = []
        self._max_size = max_size
        self._lock = threading.RLock()

    def save(self, feedback: FeedbackData) -> bool:
        """피드백 저장"""
        with self._lock:
            # 최대 크기 초과 시 오래된 항목 삭제
            while len(self._ordered_ids) >= self._max_size:
                old_id = self._ordered_ids.pop(0)
                self._storage.pop(old_id, None)

            self._storage[feedback.id] = feedback
            self._ordered_ids.append(feedback.id)

        return True

    def get(self, feedback_id: str) -> Optional[FeedbackData]:
        """ID로 피드백 조회"""
        return self._storage.get(feedback_id)

    def query(self, query: FeedbackQuery) -> List[FeedbackData]:
        """쿼리로 피드백 조회"""
        with self._lock:
            results = []
            for feedback in self._storage.values():
                if query.matches(feedback):
                    results.append(feedback)

            # 시간 역순 정렬
            results.sort(key=lambda x: x.timestamp, reverse=True)

            # 페이지네이션
            return results[query.offset:query.offset + query.limit]

    def get_recent(self, limit: int = 100) -> List[FeedbackData]:
        """최근 피드백 조회"""
        with self._lock:
            recent_ids = self._ordered_ids[-limit:]
            results = [self._storage[fid] for fid in reversed(recent_ids) if fid in self._storage]
            return results

    def count(self, query: Optional[FeedbackQuery] = None) -> int:
        """피드백 개수"""
        if query is None:
            return len(self._storage)

        return len([f for f in self._storage.values() if query.matches(f)])

    def delete(self, feedback_id: str) -> bool:
        """피드백 삭제"""
        with self._lock:
            if feedback_id in self._storage:
                del self._storage[feedback_id]
                if feedback_id in self._ordered_ids:
                    self._ordered_ids.remove(feedback_id)
                return True
        return False

    def clear(self) -> None:
        """모든 피드백 삭제"""
        with self._lock:
            self._storage.clear()
            self._ordered_ids.clear()


class SQLFeedbackStorage(FeedbackStorage):
    """
    SQLite 피드백 저장소

    영구 저장용
    """

    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    feedback_type TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    value TEXT,
                    sentiment TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback(sentiment)")

    def save(self, feedback: FeedbackData) -> bool:
        """피드백 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feedback
                    (id, feedback_type, query_id, query, response, value, sentiment,
                     user_id, session_id, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.id,
                        feedback.feedback_type.value,
                        feedback.query_id,
                        feedback.query,
                        feedback.response,
                        json.dumps(feedback.value) if feedback.value is not None else None,
                        feedback.sentiment.value,
                        feedback.user_id,
                        feedback.session_id,
                        feedback.timestamp,
                        json.dumps(feedback.metadata),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False

    def get(self, feedback_id: str) -> Optional[FeedbackData]:
        """ID로 피드백 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_feedback(row)
        return None

    def query(self, query: FeedbackQuery) -> List[FeedbackData]:
        """쿼리로 피드백 조회"""
        conditions = []
        params = []

        if query.feedback_types:
            placeholders = ",".join("?" * len(query.feedback_types))
            conditions.append(f"feedback_type IN ({placeholders})")
            params.extend([ft.value for ft in query.feedback_types])

        if query.sentiments:
            placeholders = ",".join("?" * len(query.sentiments))
            conditions.append(f"sentiment IN ({placeholders})")
            params.extend([s.value for s in query.sentiments])

        if query.user_id:
            conditions.append("user_id = ?")
            params.append(query.user_id)

        if query.session_id:
            conditions.append("session_id = ?")
            params.append(query.session_id)

        if query.query_contains:
            conditions.append("query LIKE ?")
            params.append(f"%{query.query_contains}%")

        if query.start_time:
            conditions.append("timestamp >= ?")
            params.append(query.start_time)

        if query.end_time:
            conditions.append("timestamp <= ?")
            params.append(query.end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM feedback
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([query.limit, query.offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            return [self._row_to_feedback(row) for row in cursor.fetchall()]

    def get_recent(self, limit: int = 100) -> List[FeedbackData]:
        """최근 피드백 조회"""
        return self.query(FeedbackQuery(limit=limit))

    def count(self, query: Optional[FeedbackQuery] = None) -> int:
        """피드백 개수"""
        if query is None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM feedback")
                return cursor.fetchone()[0]

        # 쿼리 조건 적용
        results = self.query(FeedbackQuery(
            feedback_types=query.feedback_types,
            sentiments=query.sentiments,
            user_id=query.user_id,
            session_id=query.session_id,
            query_contains=query.query_contains,
            start_time=query.start_time,
            end_time=query.end_time,
            limit=100000,
        ))
        return len(results)

    def delete(self, feedback_id: str) -> bool:
        """피드백 삭제"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete feedback: {e}")
            return False

    def _row_to_feedback(self, row: sqlite3.Row) -> FeedbackData:
        """Row를 FeedbackData로 변환"""
        value = row["value"]
        if value:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return FeedbackData(
            id=row["id"],
            feedback_type=FeedbackType(row["feedback_type"]),
            query_id=row["query_id"],
            query=row["query"],
            response=row["response"] or "",
            value=value,
            sentiment=FeedbackSentiment(row["sentiment"]),
            user_id=row["user_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            metadata=metadata,
        )

    def vacuum(self) -> None:
        """데이터베이스 최적화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

    def export_to_json(self, filepath: str) -> int:
        """JSON으로 내보내기"""
        all_feedback = self.query(FeedbackQuery(limit=100000))
        data = [f.to_dict() for f in all_feedback]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return len(data)

    def import_from_json(self, filepath: str) -> int:
        """JSON에서 가져오기"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for item in data:
            feedback = FeedbackData.from_dict(item)
            if self.save(feedback):
                count += 1

        return count
