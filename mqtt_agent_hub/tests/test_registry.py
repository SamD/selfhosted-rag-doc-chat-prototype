from __future__ import annotations

import time

import pytest

from shared.models import AgentDiscovery, TaskResult, TelemetryReport

from ..mqtt.registry import AgentRegistry


class TestAgentRegistry:
    @pytest.fixture
    def registry(self) -> AgentRegistry:
        return AgentRegistry()

    @pytest.mark.asyncio
    async def test_register_agent(self, registry: AgentRegistry) -> None:
        discovery = AgentDiscovery(
            agent_id="agent-1",
            name="Test Agent",
            capabilities=["shell"],
            status="online",
            uptime=100.0,
        )
        await registry.register(discovery)

        agent = await registry.get_agent("agent-1")
        assert agent is not None
        assert agent["agent_id"] == "agent-1"
        assert agent["name"] == "Test Agent"
        assert agent["status"] == "online"
        assert agent["capabilities"] == ["shell"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, registry: AgentRegistry) -> None:
        agent = await registry.get_agent("nonexistent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_list_agents(self, registry: AgentRegistry) -> None:
        await registry.register(AgentDiscovery(agent_id="agent-1", name="A1", status="online"))
        await registry.register(AgentDiscovery(agent_id="agent-2", name="A2", status="online"))

        agents = await registry.list_agents()
        assert len(agents) == 2
        ids = {a["agent_id"] for a in agents}
        assert ids == {"agent-1", "agent-2"}

    @pytest.mark.asyncio
    async def test_mark_offline(self, registry: AgentRegistry) -> None:
        await registry.register(AgentDiscovery(agent_id="agent-1", name="A1", status="online"))
        await registry.mark_offline("agent-1")

        agent = await registry.get_agent("agent-1")
        assert agent is not None
        assert agent["status"] == "offline"

    @pytest.mark.asyncio
    async def test_record_telemetry(self, registry: AgentRegistry) -> None:
        await registry.register(AgentDiscovery(agent_id="agent-1", name="A1", status="online"))

        report = TelemetryReport(
            agent_id="agent-1",
            cpu=42.5,
            memory=65.0,
            disk=30.0,
            uptime=200.0,
            timestamp=time.time(),
        )
        await registry.record_telemetry(report)

        agent = await registry.get_agent("agent-1")
        assert agent is not None
        assert agent["latest_telemetry"] is not None
        assert agent["latest_telemetry"]["cpu"] == 42.5
        assert agent["latest_telemetry"]["memory"] == 65.0

    @pytest.mark.asyncio
    async def test_record_task_result(self, registry: AgentRegistry) -> None:
        await registry.register(AgentDiscovery(agent_id="agent-1", name="A1", status="online"))

        result = TaskResult(
            task_id="task-123",
            agent_id="agent-1",
            status="success",
            output="done",
        )
        await registry.record_task_result(result)

        agent = await registry.get_agent("agent-1")
        assert agent is not None
        assert len(agent["recent_tasks"]) == 1
        assert agent["recent_tasks"][0]["task_id"] == "task-123"
        assert agent["recent_tasks"][0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_prune_stale_marks_offline(self, registry: AgentRegistry) -> None:
        await registry.register(AgentDiscovery(agent_id="agent-1", name="A1", status="online"))
        registry._last_seen["agent-1"] = time.time() - 120

        stale = await registry.prune_stale()
        assert "agent-1" in stale

        agent = await registry.get_agent("agent-1")
        assert agent is not None
        assert agent["status"] == "offline"

    @pytest.mark.asyncio
    async def test_telemetry_buffer_limit(self, registry: AgentRegistry) -> None:
        await registry.register(AgentDiscovery(agent_id="agent-1", name="A1", status="online"))

        for i in range(150):
            await registry.record_telemetry(
                TelemetryReport(agent_id="agent-1", cpu=float(i), timestamp=time.time())
            )

        assert len(registry._telemetry["agent-1"]) == 100
        assert registry._telemetry["agent-1"][0].cpu == 50.0
        assert registry._telemetry["agent-1"][-1].cpu == 149.0
