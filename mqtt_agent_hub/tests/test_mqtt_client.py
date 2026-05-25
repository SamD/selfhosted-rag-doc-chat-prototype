from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.models import GlobalMessage, TaskRequest

from ..mqtt.client import AgentHubClient
from ..mqtt.registry import AgentRegistry


class TestHubClient:
    @pytest.fixture
    def registry(self) -> AgentRegistry:
        return AgentRegistry()

    @pytest.fixture
    def client(self, registry: AgentRegistry) -> AgentHubClient:
        return AgentHubClient(registry)

    def test_validate_token_valid(self, client: AgentHubClient) -> None:
        assert client._validate_token({"token": "wrong"}) is False

    def test_handle_discovery_invalid_token(self, client: AgentHubClient) -> None:
        payload = {"agent_id": "agent-1", "name": "Bad Agent", "token": "wrong"}
        client._handle_discovery(payload)

    def test_handle_telemetry_invalid_token(self, client: AgentHubClient) -> None:
        payload = {"agent_id": "agent-1", "cpu": 50.0, "token": "wrong"}
        client._handle_telemetry(payload)

    def test_publish_task(self, client: AgentHubClient) -> None:
        client._client = MagicMock()
        task = TaskRequest(command="test-command", params={"key": "val"})
        client.publish_task("agent-1", task)

        client._client.publish.assert_called_once()
        call_args = client._client.publish.call_args
        assert call_args[0][0] == "agent/task/agent-1"
        assert call_args[1]["qos"] == 1

    def test_publish_global(self, client: AgentHubClient) -> None:
        client._client = MagicMock()
        msg = GlobalMessage(content="Hello all")
        client.publish_global(msg)

        client._client.publish.assert_called_once()
        call_args = client._client.publish.call_args
        assert call_args[0][0] == "agent/global"
        assert call_args[1]["qos"] == 1
