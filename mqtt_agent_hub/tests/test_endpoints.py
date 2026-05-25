from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ..api.endpoints import router
from ..mqtt.registry import AgentRegistry


def create_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.registry = AgentRegistry()
    app.state.mqtt_client = None
    return app


class TestEndpoints:
    @pytest.fixture
    def app(self) -> FastAPI:
        return create_test_app()

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_list_agents_empty(self, client: TestClient) -> None:
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_agent_not_found(self, client: TestClient) -> None:
        response = client.get("/api/v1/agents/nonexistent")
        assert response.status_code == 404

    def test_send_task_mqtt_not_ready(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/task/agent-1",
            json={"command": "test", "params": {}},
        )
        assert response.status_code == 503

    def test_send_task_to_agent(self, app: FastAPI) -> None:
        mock_client = MagicMock()
        app.state.mqtt_client = mock_client
        test_client = TestClient(app)

        response = test_client.post(
            "/api/v1/task/agent-1",
            json={"command": "test", "params": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "dispatched"
        assert data["target"] == "agent-1"
        mock_client.publish_task.assert_called_once()

    def test_send_global(self, app: FastAPI) -> None:
        mock_client = MagicMock()
        app.state.mqtt_client = mock_client
        test_client = TestClient(app)

        response = test_client.post(
            "/api/v1/global",
            json={"content": "Hello all agents"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "broadcast"
        mock_client.publish_global.assert_called_once()
