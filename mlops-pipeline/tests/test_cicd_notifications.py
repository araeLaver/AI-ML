"""CI/CD Notifications 테스트"""

import pytest
from unittest.mock import MagicMock, patch

from src.cicd.notifications import (
    NotificationManager,
    NotificationChannel,
    DeploymentNotification,
)


class TestDeploymentNotification:
    def test_notification_defaults(self):
        n = DeploymentNotification(
            channel=NotificationChannel.SLACK,
            event_type="deploy",
            environment="staging",
            status="success",
            message="test",
        )
        assert n.channel == NotificationChannel.SLACK
        assert n.metadata == {}
        assert n.timestamp is not None


class TestNotificationManager:
    def setup_method(self):
        self.manager = NotificationManager(
            channels=[NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
        )

    def test_notify_deployment_start(self):
        results = self.manager.notify_deployment_start(
            environment="staging", image="img:v1", strategy="canary"
        )
        assert len(results) == 2
        assert results[0].event_type == "deployment_start"
        assert results[0].status == "started"

    def test_notify_deployment_success(self):
        results = self.manager.notify_deployment_success(
            environment="production", image="img:v1", duration_s=45.5
        )
        assert len(results) == 2
        assert results[0].status == "success"
        assert "45.5s" in results[0].message

    def test_notify_deployment_failure(self):
        results = self.manager.notify_deployment_failure(
            environment="staging", error="OOM killed"
        )
        assert len(results) == 2
        assert results[0].status == "failed"
        assert "OOM killed" in results[0].message

    def test_notify_deployment_rollback(self):
        results = self.manager.notify_deployment_rollback(
            environment="production", reason="health check failed"
        )
        assert len(results) == 2
        assert results[0].status == "rolled_back"

    def test_notify_model_validation_passed(self):
        gates = {"performance": {"passed": True}, "size": {"passed": True}}
        results = self.manager.notify_model_validation_result(passed=True, gates=gates)
        assert results[0].status == "passed"
        assert "passed" in results[0].message

    def test_notify_model_validation_failed(self):
        gates = {"performance": {"passed": False}}
        results = self.manager.notify_model_validation_result(passed=False, gates=gates)
        assert results[0].status == "failed"

    def test_get_notification_history(self):
        self.manager.notify_deployment_start("staging", "img:v1", "canary")
        self.manager.notify_deployment_success("staging", "img:v1", 30.0)
        history = self.manager.get_notification_history()
        assert len(history) == 4  # 2 channels x 2 events

    def test_get_notification_history_filtered(self):
        self.manager.notify_deployment_start("staging", "img:v1", "canary")
        self.manager.notify_deployment_success("staging", "img:v1", 30.0)
        history = self.manager.get_notification_history(event_type="deployment_start")
        assert len(history) == 2

    def test_single_channel(self):
        mgr = NotificationManager(channels=[NotificationChannel.EMAIL])
        results = mgr.notify_deployment_start("staging", "img:v1", "canary")
        assert len(results) == 1
        assert results[0].channel == NotificationChannel.EMAIL

    def test_notification_metadata(self):
        results = self.manager.notify_deployment_start(
            environment="prod", image="img:v2", strategy="blue_green", version="2.0"
        )
        assert results[0].metadata["version"] == "2.0"
        assert results[0].metadata["image"] == "img:v2"

    def test_webhook_channel_sim(self):
        mgr = NotificationManager(channels=[NotificationChannel.WEBHOOK])
        results = mgr.notify_deployment_failure("staging", "crash")
        assert len(results) == 1
        assert results[0].channel == NotificationChannel.WEBHOOK

    def test_history_limit(self):
        mgr = NotificationManager(channels=[NotificationChannel.SLACK])
        for i in range(60):
            mgr.notify_deployment_start("staging", f"img:v{i}", "canary")
        history = mgr.get_notification_history(limit=10)
        assert len(history) == 10

    def test_notification_channel_enum(self):
        assert NotificationChannel.SLACK.value == "slack"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.WEBHOOK.value == "webhook"

    def test_empty_history(self):
        history = self.manager.get_notification_history()
        assert history == []
