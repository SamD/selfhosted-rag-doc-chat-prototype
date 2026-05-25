from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import paho.mqtt.client as mqtt

from shared.models import AgentDiscovery, GlobalMessage, TaskRequest, TaskResult, TelemetryReport
from shared.topics import (
    DISCOVERY_TOPIC,
    GLOBAL_TOPIC,
    TASK_RESULT_TOPIC,
    TASK_TOPIC,
    TELEMETRY_TOPIC,
)

from ..config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_HUB_TOKEN
from .registry import AgentRegistry

log = logging.getLogger("mqtt.client")


class AgentHubClient:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._running = False
        self._prune_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        self._client.username_pw_set(username="hub", password=MQTT_HUB_TOKEN)
        self._client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=30)
        self._client.loop_start()
        self._running = True
        self._prune_task = asyncio.create_task(self._prune_loop())
        log.info("MQTT client connected to %s:%s", MQTT_BROKER_HOST, MQTT_BROKER_PORT)

    async def disconnect(self) -> None:
        self._running = False
        if self._prune_task:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
        self._client.loop_stop()
        self._client.disconnect()

    async def _prune_loop(self) -> None:
        while self._running:
            await asyncio.sleep(15)
            try:
                stale = await self.registry.prune_stale()
                if stale:
                    log.info("Pruned stale agents: %s", stale)
            except Exception:
                log.exception("Prune loop error")

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: dict[str, Any], reason_code: Any, properties: Any | None = None) -> None:
        if hasattr(reason_code, 'is_failure') and not reason_code.is_failure:
            log.info("Connected to MQTT broker")
            client.subscribe(DISCOVERY_TOPIC, qos=1)
            client.subscribe(TELEMETRY_TOPIC, qos=0)
            client.subscribe(TASK_RESULT_TOPIC, qos=1)
        else:
            log.error("MQTT connection failed with code: %s", reason_code)

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, flags: dict[str, Any], reason_code: Any, properties: Any | None = None) -> None:
        log.warning("MQTT disconnected: %s", reason_code)

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        topic = msg.topic
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            log.warning("Invalid JSON on topic %s", topic)
            return

        if topic == DISCOVERY_TOPIC:
            self._handle_discovery(payload)
        elif topic == TELEMETRY_TOPIC:
            self._handle_telemetry(payload)
        elif topic.startswith(f"{TASK_TOPIC}/") and topic.endswith("/result"):
            self._handle_task_result(payload)

    def _validate_token(self, payload: dict[str, Any]) -> bool:
        return payload.get("token") == MQTT_HUB_TOKEN

    def _handle_discovery(self, payload: dict[str, Any]) -> None:
        if not self._validate_token(payload):
            log.warning("Invalid token on discovery from %s", payload.get("agent_id"))
            return
        try:
            discovery = AgentDiscovery(**payload)
            asyncio.run_coroutine_threadsafe(
                self.registry.register(discovery), asyncio.get_event_loop()
            )
            log.info("Agent discovered: %s (%s)", discovery.agent_id, discovery.name)
        except Exception:
            log.exception("Failed to parse discovery payload")

    def _handle_telemetry(self, payload: dict[str, Any]) -> None:
        if not self._validate_token(payload):
            return
        try:
            payload.setdefault("timestamp", time.time())
            report = TelemetryReport(**payload)
            asyncio.run_coroutine_threadsafe(
                self.registry.record_telemetry(report), asyncio.get_event_loop()
            )
        except Exception:
            log.exception("Failed to parse telemetry payload")

    def _handle_task_result(self, payload: dict[str, Any]) -> None:
        try:
            result = TaskResult(**payload)
            asyncio.run_coroutine_threadsafe(
                self.registry.record_task_result(result), asyncio.get_event_loop()
            )
            log.info("Task result from %s: %s=%s", result.agent_id, result.task_id, result.status)
        except Exception:
            log.exception("Failed to parse task result payload")

    def publish_task(self, agent_id: str, task: TaskRequest) -> None:
        topic = f"{TASK_TOPIC}/{agent_id}"
        self._client.publish(topic, task.model_dump_json(), qos=1)

    def publish_global(self, message: GlobalMessage) -> None:
        self._client.publish(GLOBAL_TOPIC, message.model_dump_json(), qos=1)
