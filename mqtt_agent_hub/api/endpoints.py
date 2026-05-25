from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from shared.models import GlobalMessage, TaskRequest

from ..mqtt.client import AgentHubClient

router = APIRouter(prefix="/api/v1")


def get_mqtt_client(request: Request) -> AgentHubClient:
    client = request.app.state.mqtt_client
    if client is None:
        raise HTTPException(status_code=503, detail="MQTT hub not ready")
    return client


def get_registry(request: Request):
    return request.app.state.registry


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}


@router.get("/agents")
async def list_agents(request: Request) -> list[dict[str, Any]]:
    registry = get_registry(request)
    return await registry.list_agents()


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, request: Request) -> dict[str, Any]:
    registry = get_registry(request)
    agent = await registry.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/task")
async def send_task(task: TaskRequest, request: Request) -> dict[str, Any]:
    client = get_mqtt_client(request)
    agent_id = task.origin
    if not agent_id or agent_id == "hub":
        raise HTTPException(status_code=400, detail="task.origin must be a target agent_id")

    # origin doubles as target agent_id for the UI convention
    target = task.origin
    task.origin = "hub"
    client.publish_task(target, task)
    return {"status": "dispatched", "task_id": task.task_id, "target": target}


@router.post("/task/{agent_id}")
async def send_task_to_agent(agent_id: str, task: TaskRequest, request: Request) -> dict[str, Any]:
    client = get_mqtt_client(request)
    task.origin = "hub"
    client.publish_task(agent_id, task)
    return {"status": "dispatched", "task_id": task.task_id, "target": agent_id}


@router.post("/global")
async def send_global(message: GlobalMessage, request: Request) -> dict[str, Any]:
    client = get_mqtt_client(request)
    message.origin = "hub"
    client.publish_global(message)
    return {"status": "broadcast", "message_id": message.message_id}
