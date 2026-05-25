from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from shared.defaults import AGENT_HEARTBEAT_TIMEOUT, TELEMETRY_BUFFER_SIZE
from shared.models import AgentDiscovery, TaskResult, TelemetryReport


class AgentRegistry:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._agents: dict[str, AgentDiscovery] = {}
        self._telemetry: dict[str, list[TelemetryReport]] = defaultdict(list)
        self._task_history: dict[str, list[TaskResult]] = defaultdict(list)
        self._last_seen: dict[str, float] = {}

    async def register(self, discovery: AgentDiscovery) -> None:
        async with self._lock:
            self._agents[discovery.agent_id] = discovery
            self._last_seen[discovery.agent_id] = time.time()

    async def mark_offline(self, agent_id: str) -> None:
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].status = "offline"

    async def record_telemetry(self, report: TelemetryReport) -> None:
        async with self._lock:
            history = self._telemetry[report.agent_id]
            history.append(report)
            if len(history) > TELEMETRY_BUFFER_SIZE:
                self._telemetry[report.agent_id] = history[-TELEMETRY_BUFFER_SIZE:]
            self._last_seen[report.agent_id] = time.time()

    async def record_task_result(self, result: TaskResult) -> None:
        async with self._lock:
            self._task_history[result.agent_id].append(result)

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._get_agent_unlocked(agent_id)

    def _get_agent_unlocked(self, agent_id: str) -> dict[str, Any] | None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return None
        telemetry = self._telemetry.get(agent_id, [])
        latest = telemetry[-1] if telemetry else None
        tasks = self._task_history.get(agent_id, [])
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "capabilities": agent.capabilities,
            "status": agent.status,
            "uptime": agent.uptime,
            "last_seen": self._last_seen.get(agent_id, 0),
            "latest_telemetry": latest.model_dump() if latest else None,
            "recent_tasks": [t.model_dump() for t in tasks[-10:]],
            "is_stale": self._is_stale(agent_id),
            "hostname": agent.hostname,
            "ip_addresses": agent.ip_addresses,
            "mac_address": agent.mac_address,
            "os_info": agent.os_info,
            "cpu_info": agent.cpu_info,
            "memory_total_gb": agent.memory_total_gb,
            "disk_total_gb": agent.disk_total_gb,
            "python_version": agent.python_version,
            "agent_version": agent.agent_version,
        }

    async def list_agents(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [self._get_agent_unlocked(aid) for aid in self._agents]

    async def get_all_agents_raw(self) -> dict[str, AgentDiscovery]:
        async with self._lock:
            return dict(self._agents)

    async def prune_stale(self) -> list[str]:
        stale = []
        async with self._lock:
            now = time.time()
            for agent_id, last in list(self._last_seen.items()):
                if now - last > AGENT_HEARTBEAT_TIMEOUT:
                    if agent_id in self._agents:
                        self._agents[agent_id].status = "offline"
                    stale.append(agent_id)
        return stale

    def _is_stale(self, agent_id: str) -> bool:
        last = self._last_seen.get(agent_id, 0)
        return time.time() - last > AGENT_HEARTBEAT_TIMEOUT
