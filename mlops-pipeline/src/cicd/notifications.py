"""
배포 알림 시스템
- Slack, Email, Webhook 채널 지원
- 배포 이벤트별 알림 (시작, 성공, 실패, 롤백)
- 모델 검증 결과 알림
"""

import json
import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional imports
try:
    from slack_sdk import WebClient as SlackClient

    HAS_SLACK = True
except ImportError:
    SlackClient = None
    HAS_SLACK = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False


class NotificationChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class DeploymentNotification:
    channel: NotificationChannel
    event_type: str
    environment: str
    status: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationManager:
    """알림 매니저 - 배포 이벤트 알림 전송"""

    def __init__(
        self,
        slack_token: Optional[str] = None,
        slack_channel: str = "#deployments",
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None,
        channels: Optional[List[NotificationChannel]] = None,
    ):
        self.slack_token = slack_token
        self.slack_channel = slack_channel
        self.webhook_url = webhook_url
        self.email_config = email_config or {}
        self.channels = channels or [NotificationChannel.SLACK]
        self._history: List[DeploymentNotification] = []

    def notify_deployment_start(
        self, environment: str, image: str, strategy: str, **kwargs: Any
    ) -> List[DeploymentNotification]:
        """배포 시작 알림"""
        message = f"🚀 Deployment started\nEnv: {environment}\nImage: {image}\nStrategy: {strategy}"
        return self._send_all(
            event_type="deployment_start",
            environment=environment,
            status="started",
            message=message,
            metadata={"image": image, "strategy": strategy, **kwargs},
        )

    def notify_deployment_success(
        self, environment: str, image: str, duration_s: float = 0, **kwargs: Any
    ) -> List[DeploymentNotification]:
        """배포 성공 알림"""
        message = f"✅ Deployment successful\nEnv: {environment}\nImage: {image}\nDuration: {duration_s:.1f}s"
        return self._send_all(
            event_type="deployment_success",
            environment=environment,
            status="success",
            message=message,
            metadata={"image": image, "duration_s": duration_s, **kwargs},
        )

    def notify_deployment_failure(
        self, environment: str, error: str, **kwargs: Any
    ) -> List[DeploymentNotification]:
        """배포 실패 알림"""
        message = f"❌ Deployment failed\nEnv: {environment}\nError: {error}"
        return self._send_all(
            event_type="deployment_failure",
            environment=environment,
            status="failed",
            message=message,
            metadata={"error": error, **kwargs},
        )

    def notify_deployment_rollback(
        self, environment: str, reason: str, **kwargs: Any
    ) -> List[DeploymentNotification]:
        """롤백 알림"""
        message = f"⏪ Deployment rolled back\nEnv: {environment}\nReason: {reason}"
        return self._send_all(
            event_type="deployment_rollback",
            environment=environment,
            status="rolled_back",
            message=message,
            metadata={"reason": reason, **kwargs},
        )

    def notify_model_validation_result(
        self, passed: bool, gates: Dict[str, Any], **kwargs: Any
    ) -> List[DeploymentNotification]:
        """모델 검증 결과 알림"""
        status_icon = "✅" if passed else "❌"
        gate_summary = ", ".join(
            f"{k}: {'pass' if v.get('passed') else 'fail'}" for k, v in gates.items()
        )
        message = f"{status_icon} Model validation {'passed' if passed else 'failed'}\nGates: {gate_summary}"
        return self._send_all(
            event_type="model_validation",
            environment="ci",
            status="passed" if passed else "failed",
            message=message,
            metadata={"gates": gates, **kwargs},
        )

    def _send_all(
        self,
        event_type: str,
        environment: str,
        status: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DeploymentNotification]:
        """모든 설정된 채널로 알림 전송"""
        notifications = []
        for channel in self.channels:
            notification = DeploymentNotification(
                channel=channel,
                event_type=event_type,
                environment=environment,
                status=status,
                message=message,
                metadata=metadata or {},
            )
            try:
                if channel == NotificationChannel.SLACK:
                    self._send_slack(notification)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email(notification)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook(notification)
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")

            self._history.append(notification)
            notifications.append(notification)

        return notifications

    def _send_slack(self, notification: DeploymentNotification) -> None:
        """Slack 알림 전송"""
        if HAS_SLACK and self.slack_token:
            client = SlackClient(token=self.slack_token)
            client.chat_postMessage(
                channel=self.slack_channel,
                text=notification.message,
            )
            logger.info(f"Slack notification sent to {self.slack_channel}")
        else:
            logger.info(f"[Slack-Sim] {notification.message}")

    def _send_email(self, notification: DeploymentNotification) -> None:
        """Email 알림 전송"""
        smtp_host = self.email_config.get("smtp_host")
        sender = self.email_config.get("sender")
        recipients = self.email_config.get("recipients", "")

        if smtp_host and sender:
            msg = MIMEText(notification.message)
            msg["Subject"] = f"[{notification.status.upper()}] {notification.event_type} - {notification.environment}"
            msg["From"] = sender
            msg["To"] = recipients

            with smtplib.SMTP(smtp_host, int(self.email_config.get("smtp_port", 587))) as server:
                server.starttls()
                password = self.email_config.get("password", "")
                if password:
                    server.login(sender, password)
                server.sendmail(sender, recipients.split(","), msg.as_string())
            logger.info(f"Email notification sent to {recipients}")
        else:
            logger.info(f"[Email-Sim] {notification.event_type}: {notification.status}")

    def _send_webhook(self, notification: DeploymentNotification) -> None:
        """Webhook 알림 전송"""
        payload = {
            "event_type": notification.event_type,
            "environment": notification.environment,
            "status": notification.status,
            "message": notification.message,
            "timestamp": notification.timestamp.isoformat(),
            "metadata": notification.metadata,
        }

        if HAS_REQUESTS and self.webhook_url:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"Webhook notification sent to {self.webhook_url}")
        else:
            logger.info(f"[Webhook-Sim] {json.dumps(payload, default=str)}")

    def get_notification_history(
        self, limit: int = 50, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """알림 이력 조회"""
        history = self._history
        if event_type:
            history = [n for n in history if n.event_type == event_type]

        return [
            {
                "channel": n.channel.value,
                "event_type": n.event_type,
                "environment": n.environment,
                "status": n.status,
                "message": n.message,
                "timestamp": n.timestamp.isoformat(),
            }
            for n in history[-limit:]
        ]
