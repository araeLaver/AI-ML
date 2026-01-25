# -*- coding: utf-8 -*-
"""
알림 관리 모듈

[기능]
- 알림 규칙 정의
- 알림 발송 (Slack, Email, Webhook)
- 알림 상태 관리
- 알림 이력

[사용 예시]
>>> alert_manager = AlertManager()
>>> alert_manager.add_rule(AlertRule(
...     name="high_latency",
...     condition=lambda m: m.get("latency_ms", 0) > 1000,
...     severity=AlertSeverity.WARNING,
... ))
>>> alert_manager.check_and_alert({"latency_ms": 1500})
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """알림 심각도"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """알림 상태"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """알림 규칙"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    cooldown_seconds: int = 300  # 재알림 대기 시간
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Alert:
    """알림"""
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """알림 지속 시간"""
        end = self.resolved_at or time.time()
        return end - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "labels": self.labels,
        }


class AlertNotifier(ABC):
    """알림 발송자 인터페이스"""

    @abstractmethod
    def send(self, alert: Alert):
        """알림 발송"""
        pass


class LogNotifier(AlertNotifier):
    """로그 알림 발송자"""

    def send(self, alert: Alert):
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(
            level,
            f"[ALERT] {alert.rule_name}: {alert.message}"
        )


class SlackNotifier(AlertNotifier):
    """Slack 알림 발송자"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alert: Alert):
        """Slack 웹훅으로 알림 발송"""
        try:
            import requests

            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000",
            }.get(alert.severity, "#808080")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {"title": k, "value": str(v), "short": True}
                            for k, v in alert.labels.items()
                        ],
                        "ts": int(alert.timestamp),
                    }
                ]
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class WebhookNotifier(AlertNotifier):
    """일반 웹훅 알림 발송자"""

    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {}

    def send(self, alert: Alert):
        """웹훅으로 알림 발송"""
        try:
            import requests

            response = requests.post(
                self.url,
                json=alert.to_dict(),
                headers=self.headers,
                timeout=10,
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


class AlertManager:
    """
    알림 관리자

    [특징]
    - 규칙 기반 알림
    - 다중 발송 채널
    - 쿨다운 관리
    - 알림 이력
    """

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._last_fired: Dict[str, float] = {}
        self._notifiers: List[AlertNotifier] = [LogNotifier()]
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule):
        """알림 규칙 추가"""
        with self._lock:
            self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str):
        """알림 규칙 제거"""
        with self._lock:
            self._rules.pop(name, None)

    def add_notifier(self, notifier: AlertNotifier):
        """알림 발송자 추가"""
        self._notifiers.append(notifier)

    def check_and_alert(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        메트릭 확인 및 알림 발생

        Args:
            metrics: 확인할 메트릭

        Returns:
            발생한 알림 리스트
        """
        fired_alerts = []

        with self._lock:
            for name, rule in self._rules.items():
                if not rule.enabled:
                    continue

                try:
                    should_fire = rule.condition(metrics)

                    if should_fire:
                        # 쿨다운 확인
                        last = self._last_fired.get(name, 0)
                        if time.time() - last < rule.cooldown_seconds:
                            continue

                        # 알림 생성
                        alert = Alert(
                            rule_name=name,
                            severity=rule.severity,
                            status=AlertStatus.FIRING,
                            message=rule.description or f"Alert triggered: {name}",
                            labels=rule.labels,
                            annotations={"metrics": metrics},
                        )

                        self._active_alerts[name] = alert
                        self._alerts.append(alert)
                        self._last_fired[name] = time.time()
                        fired_alerts.append(alert)

                        # 알림 발송
                        self._notify(alert)

                    else:
                        # 해결된 알림
                        if name in self._active_alerts:
                            active = self._active_alerts.pop(name)
                            active.status = AlertStatus.RESOLVED
                            active.resolved_at = time.time()

                            # 해결 알림 발송
                            self._notify(active)

                except Exception as e:
                    logger.error(f"Error checking rule {name}: {e}")

        return fired_alerts

    def _notify(self, alert: Alert):
        """알림 발송"""
        for notifier in self._notifiers:
            try:
                notifier.send(alert)
            except Exception as e:
                logger.error(f"Notifier error: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 100,
        severity: AlertSeverity = None,
    ) -> List[Alert]:
        """알림 이력 조회"""
        with self._lock:
            alerts = list(self._alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """알림 통계"""
        with self._lock:
            total = len(self._alerts)
            active = len(self._active_alerts)

            by_severity = {}
            for alert in self._alerts:
                sev = alert.severity.value
                by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_alerts": total,
            "active_alerts": active,
            "by_severity": by_severity,
            "rules_count": len(self._rules),
        }

    def acknowledge(self, rule_name: str) -> bool:
        """알림 확인"""
        with self._lock:
            if rule_name in self._active_alerts:
                self._active_alerts[rule_name].status = AlertStatus.RESOLVED
                return True
        return False


# =============================================================================
# 사전 정의된 알림 규칙
# =============================================================================

def create_default_rules() -> List[AlertRule]:
    """기본 알림 규칙 생성"""
    return [
        AlertRule(
            name="high_query_latency",
            condition=lambda m: m.get("query_latency_ms", 0) > 5000,
            severity=AlertSeverity.WARNING,
            description="Query latency exceeded 5 seconds",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="very_high_query_latency",
            condition=lambda m: m.get("query_latency_ms", 0) > 10000,
            severity=AlertSeverity.ERROR,
            description="Query latency exceeded 10 seconds",
            cooldown_seconds=60,
        ),
        AlertRule(
            name="high_error_rate",
            condition=lambda m: m.get("error_rate", 0) > 0.05,
            severity=AlertSeverity.ERROR,
            description="Error rate exceeded 5%",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="high_memory_usage",
            condition=lambda m: m.get("memory_usage_percent", 0) > 90,
            severity=AlertSeverity.WARNING,
            description="Memory usage exceeded 90%",
            cooldown_seconds=600,
        ),
        AlertRule(
            name="critical_memory_usage",
            condition=lambda m: m.get("memory_usage_percent", 0) > 95,
            severity=AlertSeverity.CRITICAL,
            description="Memory usage exceeded 95%",
            cooldown_seconds=60,
        ),
        AlertRule(
            name="llm_api_error",
            condition=lambda m: m.get("llm_error_count", 0) > 10,
            severity=AlertSeverity.ERROR,
            description="LLM API errors exceeded 10 in period",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="vectordb_unavailable",
            condition=lambda m: not m.get("vectordb_healthy", True),
            severity=AlertSeverity.CRITICAL,
            description="Vector database is unavailable",
            cooldown_seconds=60,
        ),
        AlertRule(
            name="low_cache_hit_rate",
            condition=lambda m: m.get("cache_hit_rate", 1) < 0.3,
            severity=AlertSeverity.INFO,
            description="Cache hit rate below 30%",
            cooldown_seconds=1800,
        ),
    ]


def setup_default_alerts(manager: AlertManager):
    """기본 알림 규칙 설정"""
    for rule in create_default_rules():
        manager.add_rule(rule)
    logger.info(f"Set up {len(create_default_rules())} default alert rules")
