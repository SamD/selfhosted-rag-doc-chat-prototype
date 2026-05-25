from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class AgentDiscovery(BaseModel):
    agent_id: str
    name: str
    capabilities: list[str] = Field(default_factory=list)
    status: str = "online"
    uptime: float = 0.0
    token: str = ""
    hostname: str = ""
    ip_addresses: dict[str, str] = Field(default_factory=dict)
    mac_address: str = ""
    os_info: str = ""
    cpu_info: dict[str, Any] = Field(default_factory=dict)
    memory_total_gb: float = 0.0
    disk_total_gb: float = 0.0
    python_version: str = ""
    agent_version: str = "NOT_SET"


class TelemetryReport(BaseModel):
    agent_id: str
    cpu: float = 0.0
    memory: float = 0.0
    disk: float = 0.0
    uptime: float = 0.0
    token: str = ""
    timestamp: float = Field(default_factory=time.time)


class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
    origin: str = "hub"


class TaskResult(BaseModel):
    task_id: str
    agent_id: str
    status: str  # "success" | "failed" | "running"
    output: str = ""
    error: str = ""


class GlobalMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    content: str
    origin: str = "hub"
