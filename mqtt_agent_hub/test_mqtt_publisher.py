#!/usr/bin/env python3
"""
Reference MQTT agent - publishes discovery, sends telemetry, listens for tasks.
The shared/ directory must be deployed alongside this script for imports.
Run: python test_mqtt_publisher.py [--agent-id myagent] [--name "My Agent"]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import sys
import time
import uuid
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_parent, "shared")):
    sys.path.insert(0, _parent)
elif os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared")):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paho.mqtt.client as mqtt  # noqa: E402
import psutil  # noqa: E402

from shared.defaults import (  # noqa: E402
    DEFAULT_AGENT_VERSION,
    DEFAULT_MQTT_BROKER_HOST,
    DEFAULT_MQTT_BROKER_PORT,
    DEFAULT_MQTT_HUB_TOKEN,
)
from shared.env_names import (  # noqa: E402
    ENV_AGENT_VERSION,
    ENV_MQTT_BROKER_HOST,
    ENV_MQTT_BROKER_PORT,
    ENV_MQTT_HUB_TOKEN,
)
from shared.topics import (  # noqa: E402
    DISCOVERY_TOPIC,
    GLOBAL_TOPIC,
    TASK_TOPIC,
    TELEMETRY_TOPIC,
)

MQTT_HOST = os.getenv(ENV_MQTT_BROKER_HOST, DEFAULT_MQTT_BROKER_HOST)
MQTT_PORT = int(os.getenv(ENV_MQTT_BROKER_PORT, str(DEFAULT_MQTT_BROKER_PORT)))
MQTT_TOKEN = os.getenv(ENV_MQTT_HUB_TOKEN, DEFAULT_MQTT_HUB_TOKEN)

_client: mqtt.Client | None = None
_agent_id: str
_name: str


def gather_system_identity() -> dict[str, Any]:
    try:
        hostname = socket.gethostname().split(".")[0]
    except Exception:
        hostname = "unknown"
    ip_addresses: dict[str, str] = {}
    mac_address = ""
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            if iface == "lo":
                continue
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip_addresses[iface] = addr.address
                elif addr.family == psutil.AF_LINK and not mac_address:
                    mac_address = addr.address
    except Exception:
        pass
    if not mac_address:
        mac_address = ":".join(f"{(uuid.getnode() >> (8 * i)) & 0xff:02x}" for i in reversed(range(6)))
    try:
        cpu_info = {
            "model": platform.processor() or "unknown",
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        }
    except Exception:
        cpu_info = {"model": "unknown", "cores_physical": 0, "cores_logical": 0}
    mem_gb = 0.0
    disk_gb = 0.0
    try:
        mem_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass
    try:
        disk_gb = round(psutil.disk_usage("/").total / (1024**3), 1)
    except Exception:
        pass
    try:
        os_info = platform.platform()
    except Exception:
        os_info = f"Linux {platform.release()}"
    return {
        "hostname": hostname,
        "ip_addresses": ip_addresses,
        "mac_address": mac_address,
        "os_info": os_info,
        "cpu_info": cpu_info,
        "memory_total_gb": mem_gb,
        "disk_total_gb": disk_gb,
        "python_version": sys.version.split()[0],
        "agent_version": os.getenv(ENV_AGENT_VERSION, DEFAULT_AGENT_VERSION),
    }


def on_connect(client: mqtt.Client, userdata: Any, flags: dict[str, Any], reason_code: int, properties: Any | None = None) -> None:
    print(f"[agent:{_agent_id}] Connected to MQTT broker")

    system_info = gather_system_identity()
    discovery = {
        "agent_id": _agent_id,
        "name": _name,
        "capabilities": ["shell", "file", "monitor"],
        "status": "online",
        "uptime": 0.0,
        "token": MQTT_TOKEN,
        **system_info,
    }
    client.publish(DISCOVERY_TOPIC, json.dumps(discovery), qos=1, retain=True)
    print(f"[agent:{_agent_id}] Published discovery (retained)")

    client.subscribe(f"{TASK_TOPIC}/{_agent_id}", qos=1)
    client.subscribe(GLOBAL_TOPIC, qos=1)


def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    try:
        payload = json.loads(msg.payload)
    except json.JSONDecodeError:
        print(f"[agent:{_agent_id}] Invalid message on {msg.topic}")
        return

    if msg.topic.startswith(f"{TASK_TOPIC}/"):
        print(f"[agent:{_agent_id}] TASK RECEIVED: {payload.get('command')} (task_id={payload.get('task_id')})")
        result = {
            "task_id": payload.get("task_id"),
            "agent_id": _agent_id,
            "status": "success",
            "output": f"Executed: {payload.get('command')}",
            "error": "",
        }
        client.publish(f"{TASK_TOPIC}/{_agent_id}/result", json.dumps(result), qos=1)
        print(f"[agent:{_agent_id}] Task result published")

    elif msg.topic == GLOBAL_TOPIC:
        print(f"[agent:{_agent_id}] GLOBAL: {payload.get('content')}")


def publish_telemetry(client: mqtt.Client) -> None:
    telemetry = {
        "agent_id": _agent_id,
        "cpu": psutil.cpu_percent(interval=0.1),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": time.time(),
        "token": MQTT_TOKEN,
    }
    client.publish(TELEMETRY_TOPIC, json.dumps(telemetry), qos=0)
    print(f"[agent:{_agent_id}] Telemetry: CPU={telemetry['cpu']:.1f}% MEM={telemetry['memory']:.1f}%", end="\r")


def shutdown(signum: int, frame: Any) -> None:
    print(f"\n[agent:{_agent_id}] Shutting down...")
    if _client:
        system_info = gather_system_identity()
        discovery_offline = {
            "agent_id": _agent_id,
            "name": _name,
            "capabilities": [],
            "status": "offline",
            "uptime": 0.0,
            "token": MQTT_TOKEN,
            "hostname": system_info["hostname"],
        }
        _client.publish(DISCOVERY_TOPIC, json.dumps(discovery_offline), qos=1, retain=True)
        _client.disconnect()
        _client.loop_stop()
    exit(0)


def main() -> None:
    global _agent_id, _name, _client

    parser = argparse.ArgumentParser(description="MQTT Agent Hub reference agent")
    parser.add_argument("--agent-id", default=f"agent-{uuid.uuid4().hex[:6]}")
    parser.add_argument("--name", default="Reference Agent")
    args = parser.parse_args()

    _agent_id = args.agent_id
    _name = args.name

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    _client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    _client.on_connect = on_connect
    _client.on_message = on_message
    _client.username_pw_set(username="hub", password=MQTT_TOKEN)

    will_payload = json.dumps({
        "agent_id": _agent_id,
        "name": _name,
        "capabilities": [],
        "status": "offline",
        "uptime": 0.0,
        "token": MQTT_TOKEN,
    })
    _client.will_set(DISCOVERY_TOPIC, payload=will_payload, qos=1, retain=True)

    _client.connect(MQTT_HOST, MQTT_PORT, keepalive=15)
    _client.loop_start()

    while True:
        time.sleep(15)
        publish_telemetry(_client)


if __name__ == "__main__":
    main()
