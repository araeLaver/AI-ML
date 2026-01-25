# -*- coding: utf-8 -*-
"""
DART API 실시간 연동 모듈

일별/시간별 자동 업데이트 스케줄러 및 차등 동기화 기능을 제공합니다.
"""

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False


class SyncStatus(Enum):
    """동기화 상태"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SCHEDULED = "scheduled"


@dataclass
class SyncConfig:
    """동기화 설정

    Attributes:
        interval_hours: 동기화 주기 (시간)
        daily_time: 일별 동기화 시간 (HH:MM)
        max_disclosures_per_sync: 1회 동기화 최대 건수
        report_types: 수집할 공시 유형
        lookback_days: 과거 조회 일수
        enable_webhooks: 웹훅 알림 활성화
        webhook_urls: 웹훅 URL 목록
    """
    interval_hours: int = 1
    daily_time: str = "09:00"
    max_disclosures_per_sync: int = 100
    report_types: list[str] = field(default_factory=lambda: ["A", "B", "I"])
    lookback_days: int = 1
    enable_webhooks: bool = True
    webhook_urls: list[str] = field(default_factory=list)

    # 동기화 대상 기업 (빈 리스트면 전체)
    target_corp_codes: list[str] = field(default_factory=list)

    # 저장 경로
    data_dir: str = "data/dart/sync"
    state_file: str = "sync_state.json"


@dataclass
class SyncResult:
    """동기화 결과

    Attributes:
        status: 동기화 상태
        started_at: 시작 시간
        completed_at: 완료 시간
        new_disclosures: 새로 수집된 공시 수
        updated_disclosures: 업데이트된 공시 수
        errors: 오류 목록
        disclosures: 수집된 공시 ID 목록
    """
    status: SyncStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    new_disclosures: int = 0
    updated_disclosures: int = 0
    errors: list[str] = field(default_factory=list)
    disclosures: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """동기화 소요 시간 (초)"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "new_disclosures": self.new_disclosures,
            "updated_disclosures": self.updated_disclosures,
            "errors": self.errors,
            "disclosure_count": len(self.disclosures),
        }


class DARTSyncState:
    """동기화 상태 관리"""

    def __init__(self, state_file: Path):
        """
        Args:
            state_file: 상태 저장 파일 경로
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """상태 파일 로드"""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "last_sync": None,
            "last_success": None,
            "known_disclosures": {},  # rcept_no -> hash
            "sync_history": [],
        }

    def _save_state(self) -> None:
        """상태 저장"""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    @property
    def last_sync(self) -> Optional[datetime]:
        """마지막 동기화 시간"""
        if self._state.get("last_sync"):
            return datetime.fromisoformat(self._state["last_sync"])
        return None

    @property
    def last_success(self) -> Optional[datetime]:
        """마지막 성공 시간"""
        if self._state.get("last_success"):
            return datetime.fromisoformat(self._state["last_success"])
        return None

    def is_known(self, rcept_no: str, content_hash: str) -> bool:
        """이미 알려진 공시인지 확인

        Args:
            rcept_no: 접수번호
            content_hash: 내용 해시

        Returns:
            이미 동일한 내용으로 저장되어 있으면 True
        """
        return self._state["known_disclosures"].get(rcept_no) == content_hash

    def mark_synced(self, rcept_no: str, content_hash: str) -> None:
        """동기화 완료 마킹"""
        self._state["known_disclosures"][rcept_no] = content_hash

    def record_sync(self, result: SyncResult) -> None:
        """동기화 결과 기록"""
        self._state["last_sync"] = result.started_at.isoformat()
        if result.status == SyncStatus.SUCCESS:
            self._state["last_success"] = result.completed_at.isoformat() if result.completed_at else None

        # 히스토리 추가 (최근 100건 유지)
        self._state["sync_history"].append(result.to_dict())
        self._state["sync_history"] = self._state["sync_history"][-100:]

        self._save_state()

    def get_history(self, limit: int = 10) -> list[dict]:
        """동기화 히스토리 조회"""
        return self._state["sync_history"][-limit:][::-1]


class DARTSyncScheduler:
    """DART 실시간 동기화 스케줄러

    일별/시간별 자동 업데이트 및 차등 동기화를 관리합니다.
    """

    def __init__(
        self,
        config: Optional[SyncConfig] = None,
        dart_collector: Optional[Any] = None,
    ):
        """
        Args:
            config: 동기화 설정
            dart_collector: DARTCollector 인스턴스 (없으면 새로 생성)
        """
        self.config = config or SyncConfig()
        self._collector = dart_collector

        # 상태 관리
        state_path = Path(self.config.data_dir) / self.config.state_file
        self._state = DARTSyncState(state_path)

        # 스케줄러 상태
        self._running = False
        self._current_status = SyncStatus.IDLE
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 콜백
        self._on_new_disclosure: list[Callable] = []
        self._on_sync_complete: list[Callable] = []

    @property
    def collector(self):
        """DARTCollector 인스턴스 (lazy loading)"""
        if self._collector is None:
            try:
                from ..data.dart_collector import DARTCollector
                self._collector = DARTCollector()
            except Exception:
                # 테스트 환경에서 API 키 없을 수 있음
                self._collector = None
        return self._collector

    @property
    def status(self) -> SyncStatus:
        """현재 동기화 상태"""
        return self._current_status

    @property
    def last_sync(self) -> Optional[datetime]:
        """마지막 동기화 시간"""
        return self._state.last_sync

    @property
    def last_success(self) -> Optional[datetime]:
        """마지막 성공 시간"""
        return self._state.last_success

    def on_new_disclosure(self, callback: Callable) -> None:
        """새 공시 콜백 등록"""
        self._on_new_disclosure.append(callback)

    def on_sync_complete(self, callback: Callable) -> None:
        """동기화 완료 콜백 등록"""
        self._on_sync_complete.append(callback)

    def _compute_hash(self, content: str) -> str:
        """내용 해시 계산"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def sync_now(self) -> SyncResult:
        """즉시 동기화 실행

        Returns:
            동기화 결과
        """
        result = SyncResult(
            status=SyncStatus.RUNNING,
            started_at=datetime.now(),
        )
        self._current_status = SyncStatus.RUNNING

        try:
            if self.collector is None:
                result.status = SyncStatus.FAILED
                result.errors.append("DARTCollector not available")
                result.completed_at = datetime.now()
                return result

            # 조회 기간 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days)

            # 공시 수집
            all_disclosures = []
            for report_type in self.config.report_types:
                try:
                    disclosures = self.collector.search_disclosures(
                        start_date=start_date.strftime("%Y%m%d"),
                        end_date=end_date.strftime("%Y%m%d"),
                        report_type=report_type,
                        page_count=self.config.max_disclosures_per_sync,
                    )

                    # 대상 기업 필터링
                    if self.config.target_corp_codes:
                        disclosures = [
                            d for d in disclosures
                            if d.corp_code in self.config.target_corp_codes
                        ]

                    all_disclosures.extend(disclosures)
                except Exception as e:
                    result.errors.append(f"Error fetching {report_type}: {str(e)}")

            # 차등 동기화 - 새로운/변경된 것만 처리
            for disc in all_disclosures:
                # 내용 해시 계산 (내용 없으면 메타데이터로)
                content = disc.content or f"{disc.report_nm}_{disc.rcept_dt}"
                content_hash = self._compute_hash(content)

                if self._state.is_known(disc.rcept_no, content_hash):
                    continue

                # 원문 다운로드 (아직 없으면)
                if not disc.content:
                    try:
                        disc.content = self.collector.download_document(disc.rcept_no)
                        if disc.content:
                            content_hash = self._compute_hash(disc.content)
                    except Exception as e:
                        result.errors.append(f"Download error {disc.rcept_no}: {str(e)}")

                # 새 공시/업데이트 처리
                is_new = disc.rcept_no not in self._state._state["known_disclosures"]
                if is_new:
                    result.new_disclosures += 1
                else:
                    result.updated_disclosures += 1

                result.disclosures.append(disc.rcept_no)
                self._state.mark_synced(disc.rcept_no, content_hash)

                # 콜백 호출
                for callback in self._on_new_disclosure:
                    try:
                        callback(disc)
                    except Exception:
                        pass

            result.status = SyncStatus.SUCCESS
            result.completed_at = datetime.now()

        except Exception as e:
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now()

        finally:
            self._current_status = result.status
            self._state.record_sync(result)

            # 완료 콜백
            for callback in self._on_sync_complete:
                try:
                    callback(result)
                except Exception:
                    pass

        return result

    def start(self) -> None:
        """스케줄러 시작"""
        if self._running:
            return

        if not SCHEDULE_AVAILABLE:
            # schedule 라이브러리 없으면 수동 모드
            self._running = True
            self._current_status = SyncStatus.SCHEDULED
            return

        self._running = True
        self._stop_event.clear()
        self._current_status = SyncStatus.SCHEDULED

        # 스케줄 설정
        schedule.clear()

        # 시간별 동기화
        if self.config.interval_hours > 0:
            schedule.every(self.config.interval_hours).hours.do(self.sync_now)

        # 일별 동기화
        if self.config.daily_time:
            schedule.every().day.at(self.config.daily_time).do(self.sync_now)

        # 백그라운드 스레드 시작
        def run_scheduler():
            while not self._stop_event.is_set():
                schedule.run_pending()
                self._stop_event.wait(timeout=60)

        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()

    def stop(self) -> None:
        """스케줄러 중지"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            self._scheduler_thread = None

        if SCHEDULE_AVAILABLE:
            schedule.clear()

        self._current_status = SyncStatus.IDLE

    def get_status(self) -> dict[str, Any]:
        """현재 상태 조회"""
        return {
            "running": self._running,
            "status": self._current_status.value,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "config": {
                "interval_hours": self.config.interval_hours,
                "daily_time": self.config.daily_time,
                "report_types": self.config.report_types,
                "lookback_days": self.config.lookback_days,
            },
        }

    def get_history(self, limit: int = 10) -> list[dict]:
        """동기화 히스토리 조회"""
        return self._state.get_history(limit)


class DARTWebhook:
    """DART 공시 웹훅 관리자

    새 공시 발생 시 외부 시스템에 알림을 보냅니다.
    """

    def __init__(self, urls: Optional[list[str]] = None):
        """
        Args:
            urls: 웹훅 URL 목록
        """
        self.urls = urls or []
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """웹훅 활성화 여부"""
        return self._enabled and len(self.urls) > 0

    def enable(self) -> None:
        """웹훅 활성화"""
        self._enabled = True

    def disable(self) -> None:
        """웹훅 비활성화"""
        self._enabled = False

    def add_url(self, url: str) -> None:
        """웹훅 URL 추가"""
        if url not in self.urls:
            self.urls.append(url)

    def remove_url(self, url: str) -> None:
        """웹훅 URL 제거"""
        if url in self.urls:
            self.urls.remove(url)

    async def notify_async(self, disclosure: Any) -> dict[str, bool]:
        """비동기 웹훅 알림 전송

        Args:
            disclosure: 공시 정보

        Returns:
            URL별 전송 성공 여부
        """
        if not self.enabled:
            return {}

        results = {}

        try:
            import aiohttp
        except ImportError:
            # aiohttp 없으면 동기 방식으로 폴백
            return self.notify_sync(disclosure)

        payload = self._build_payload(disclosure)

        async with aiohttp.ClientSession() as session:
            for url in self.urls:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        results[url] = response.status < 400
                except Exception:
                    results[url] = False

        return results

    def notify_sync(self, disclosure: Any) -> dict[str, bool]:
        """동기 웹훅 알림 전송"""
        if not self.enabled:
            return {}

        import requests

        payload = self._build_payload(disclosure)
        results = {}

        for url in self.urls:
            try:
                response = requests.post(url, json=payload, timeout=10)
                results[url] = response.status_code < 400
            except Exception:
                results[url] = False

        return results

    def _build_payload(self, disclosure: Any) -> dict[str, Any]:
        """웹훅 페이로드 생성"""
        if hasattr(disclosure, "__dict__"):
            # Disclosure 객체
            return {
                "event": "new_disclosure",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "rcept_no": getattr(disclosure, "rcept_no", ""),
                    "corp_name": getattr(disclosure, "corp_name", ""),
                    "corp_code": getattr(disclosure, "corp_code", ""),
                    "report_nm": getattr(disclosure, "report_nm", ""),
                    "rcept_dt": getattr(disclosure, "rcept_dt", ""),
                },
            }
        return {
            "event": "new_disclosure",
            "timestamp": datetime.now().isoformat(),
            "data": disclosure,
        }


# 전역 스케줄러 인스턴스
_scheduler: Optional[DARTSyncScheduler] = None


def get_scheduler() -> DARTSyncScheduler:
    """전역 스케줄러 인스턴스 반환"""
    global _scheduler
    if _scheduler is None:
        _scheduler = DARTSyncScheduler()
    return _scheduler


def start_sync_scheduler(config: Optional[SyncConfig] = None) -> DARTSyncScheduler:
    """동기화 스케줄러 시작

    Args:
        config: 동기화 설정

    Returns:
        스케줄러 인스턴스
    """
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()

    _scheduler = DARTSyncScheduler(config)
    _scheduler.start()
    return _scheduler


def stop_sync_scheduler() -> None:
    """동기화 스케줄러 중지"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()


def get_sync_status() -> dict[str, Any]:
    """동기화 상태 조회"""
    scheduler = get_scheduler()
    return scheduler.get_status()


def trigger_manual_sync() -> SyncResult:
    """수동 동기화 트리거"""
    scheduler = get_scheduler()
    return scheduler.sync_now()
