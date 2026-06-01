#!/usr/bin/env python3
"""
Edge SRE Agent — deploys on Debian minipcs to monitor, report, and optionally
analyze system health via a local LLM.

Run:
  python edge_agent.py [--agent-id bee1] [--name "Bee Monitor"]

The shared/ directory must be deployed alongside this script for imports.

Environment Variables:
  MQTT_BROKER_HOST   — Mosquitto broker IP/hostname (default: localhost)
  MQTT_BROKER_PORT   — MQTT port (default: 1883)
  MQTT_HUB_TOKEN     — Pre-shared authentication token
  AGENT_VERSION      — Version string reported in discovery (default: NOT_SET)
  LLM_PATH           — Path to a GGUF model file (optional; enables SRE analysis mode)

Without LLM_PATH the agent runs in bare monitoring mode (telemetry only).
With LLM_PATH it periodically invokes the LLM to analyze system state and
report findings using function-calling tools defined in sre_prompt.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import signal
import socket
import subprocess
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
from sre_prompt import AVAILABLE_TOOLS, SRE_SYSTEM_PROMPT  # noqa: E402

from shared.defaults import (  # noqa: E402
    DEFAULT_AGENT_VERSION,
    DEFAULT_MQTT_BROKER_HOST,
    DEFAULT_MQTT_BROKER_PORT,
    DEFAULT_MQTT_HUB_TOKEN,
)
from shared.env_names import (  # noqa: E402
    ENV_AGENT_VERSION,
    ENV_LLM_PATH,
    ENV_MQTT_BROKER_HOST,
    ENV_MQTT_BROKER_PORT,
    ENV_MQTT_HUB_TOKEN,
    ENV_TELEMETRY_INTERVAL,
)
from shared.topics import (  # noqa: E402
    DISCOVERY_TOPIC,
    GLOBAL_TOPIC,
    TASK_TOPIC,
    TELEMETRY_TOPIC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] agent: %(message)s",
)
log = logging.getLogger("edge_agent")

MQTT_HOST = os.getenv(ENV_MQTT_BROKER_HOST, DEFAULT_MQTT_BROKER_HOST)
MQTT_PORT = int(os.getenv(ENV_MQTT_BROKER_PORT, str(DEFAULT_MQTT_BROKER_PORT)))
MQTT_TOKEN = os.getenv(ENV_MQTT_HUB_TOKEN, DEFAULT_MQTT_HUB_TOKEN)
LLM_PATH = os.getenv(ENV_LLM_PATH, "")
TELEMETRY_INTERVAL = int(os.getenv(ENV_TELEMETRY_INTERVAL, "300"))

_client: mqtt.Client | None = None
_llm: Any = None
_agent_id: str = ""
_name: str = ""
_start_time: float = 0.0


def generate_agent_id(override: str | None = None) -> str:
    if override:
        return override
    try:
        hostname = socket.gethostname().split(".")[0]
    except Exception:
        hostname = "unknown"
    try:
        with open("/etc/machine-id") as fh:
            machine_id = fh.read().strip()
    except Exception:
        machine_id = uuid.uuid4().hex
    short_hash = hashlib.md5(machine_id.encode()).hexdigest()[:6]
    return f"{hostname}-{short_hash}"


def load_llm() -> Any | None:
    if not LLM_PATH or not os.path.isfile(LLM_PATH):
        return None
    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=LLM_PATH,
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
        log.info("LLM loaded: %s", os.path.basename(LLM_PATH))
        return llm
    except ImportError:
        log.warning("llama-cpp-python not installed. Running in bare mode.")
        return None
    except Exception as exc:
        log.warning("Failed to load LLM (%s). Running in bare mode.", exc)
        return None


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

    cpu_info: dict[str, Any] = {}
    try:
        cpu_info["model"] = platform.processor() or "unknown"
        cpu_info["cores_physical"] = psutil.cpu_count(logical=False)
        cpu_info["cores_logical"] = psutil.cpu_count(logical=True)
    except Exception:
        cpu_info = {"model": "unknown", "cores_physical": 0, "cores_logical": 0}

    memory_total_gb = 0.0
    disk_total_gb = 0.0
    try:
        memory_total_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass
    try:
        disk_total_gb = round(psutil.disk_usage("/").total / (1024**3), 1)
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
        "memory_total_gb": memory_total_gb,
        "disk_total_gb": disk_total_gb,
        "python_version": sys.version.split()[0],
        "agent_version": os.getenv(ENV_AGENT_VERSION, DEFAULT_AGENT_VERSION),
    }


def on_connect(
    client: mqtt.Client,
    userdata: Any,
    flags: dict[str, Any],
    reason_code: Any,
    properties: Any | None = None,
) -> None:
    log.info("Connected to MQTT broker at %s:%s", MQTT_HOST, MQTT_PORT)

    system_info = gather_system_identity()
    discovery = {
        "agent_id": _agent_id,
        "name": _name,
        "capabilities": get_capabilities(),
        "status": "online",
        "uptime": 0.0,
        "token": MQTT_TOKEN,
        **system_info,
    }
    client.publish(DISCOVERY_TOPIC, json.dumps(discovery), qos=1, retain=True)
    log.info("Published discovery (retained)")

    client.subscribe(f"{TASK_TOPIC}/{_agent_id}", qos=1)
    client.subscribe(GLOBAL_TOPIC, qos=1)


def on_message(
    client: mqtt.Client,
    userdata: Any,
    msg: mqtt.MQTTMessage,
) -> None:
    try:
        payload = json.loads(msg.payload)
    except json.JSONDecodeError:
        log.warning("Invalid message on %s", msg.topic)
        return

    if msg.topic.startswith(f"{TASK_TOPIC}/"):
        task_id = payload.get("task_id", "unknown")
        command = payload.get("command", "")
        log.info("TASK: %s (task_id=%s)", command, task_id)
        result = execute_task(command)
        result_payload = {
            "task_id": task_id,
            "agent_id": _agent_id,
            "status": result["status"],
            "output": result["output"],
            "error": result["error"],
        }
        client.publish(
            f"{TASK_TOPIC}/{_agent_id}/result",
            json.dumps(result_payload),
            qos=1,
        )

    elif msg.topic == GLOBAL_TOPIC:
        log.info("GLOBAL: %s", payload.get("content"))


def get_capabilities() -> list[str]:
    caps = ["shell", "monitor"]
    if _llm is not None:
        caps.append("sre-analysis")
    return caps


def gather_metrics() -> dict[str, Any]:

    cpu_pct = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    load = os.getloadavg()
    uptime_seconds = time.time() - _start_time

    disk_partitions = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disk_partitions.append({
                "mount": part.mountpoint,
                "total_gb": round(usage.total / (1024**3), 1),
                "used_gb": round(usage.used / (1024**3), 1),
                "percent": usage.percent,
            })
        except PermissionError:
            pass

    return {
        "agent_id": _agent_id,
        "cpu": cpu_pct,
        "memory": mem.percent,
        "memory_available_gb": round(mem.available / (1024**3), 1),
        "memory_total_gb": round(mem.total / (1024**3), 1),
        "disk": disk.percent,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "load_1m": round(load[0], 2),
        "load_5m": round(load[1], 2),
        "load_15m": round(load[2], 2),
        "uptime": uptime_seconds,
        "disk_partitions": disk_partitions,
        "token": MQTT_TOKEN,
        "timestamp": time.time(),
    }


def execute_tool_call(name: str, args: dict[str, Any]) -> str:
    if name == "get_system_metrics":
        metrics = gather_metrics()
        return json.dumps(metrics)

    elif name == "list_top_processes":
        count = args.get("count", 5)
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda p: p["cpu_percent"] or 0, reverse=True)
        return json.dumps(procs[:count])

    elif name == "list_top_processes_by_memory":
        count = args.get("count", 5)
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda p: p["memory_percent"] or 0, reverse=True)
        return json.dumps(procs[:count])

    elif name == "check_disk_health":
        results = []
        skip_fs = {"squashfs", "tmpfs", "devtmpfs", "overlay", "fuse.gvfsd-fuse"}
        for part in psutil.disk_partitions():
            if part.fstype in skip_fs or part.mountpoint.startswith("/snap"):
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
                if usage.percent >= 80:
                    results.append({
                        "mount": part.mountpoint,
                        "percent": usage.percent,
                        "free_gb": round(usage.free / (1024**3), 1),
                        "status": "WARNING" if usage.percent < 95 else "CRITICAL",
                    })
            except PermissionError:
                pass
        return json.dumps(results)

    elif name == "find_stale_temp_files":
        days = args.get("days", 7)
        cutoff = time.time() - (days * 86400)
        stale = []
        for tmpdir in ["/tmp", "/var/tmp"]:
            if not os.path.isdir(tmpdir):
                continue
            count = 0
            total_size = 0
            try:
                for entry in os.scandir(tmpdir):
                    try:
                        stat = entry.stat()
                        if stat.st_mtime < cutoff:
                            count += 1
                            total_size += stat.st_size
                    except OSError:
                        pass
            except PermissionError:
                pass
            if count > 0:
                stale.append({
                    "directory": tmpdir,
                    "stale_count": count,
                    "total_size_mb": round(total_size / (1024**2), 1),
                    "threshold_days": days,
                })
        return json.dumps(stale)

    elif name == "check_suspicious_processes":
        suspicious = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "exe", "cwd"]):
            try:
                info = proc.info
                reasons = []
                exe_path = info.get("exe") or ""
                cwd = info.get("cwd") or ""
                cpu = info.get("cpu_percent") or 0

                if "/tmp" in exe_path or "/tmp" in cwd:
                    reasons.append("running_from_tmp")
                if cpu > 50:
                    reasons.append(f"high_cpu:{cpu:.1f}%")

                if reasons:
                    suspicious.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu_percent": cpu,
                        "reasons": reasons,
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return json.dumps(suspicious[:10])

    elif name == "check_network_connections":
        try:
            output = subprocess.check_output(
                ["ss", "-tlnp"],
                text=True,
                timeout=10,
                stderr=subprocess.DEVNULL,
            )
            return output
        except Exception as exc:
            return f"Error: {exc}"

    elif name == "report_finding":
        finding = {
            "type": "sre_finding",
            "severity": args.get("severity", "INFO"),
            "title": args.get("title", ""),
            "detail": args.get("detail", ""),
            "agent_id": _agent_id,
            "timestamp": time.time(),
        }
        if _client:
            _client.publish(TELEMETRY_TOPIC, json.dumps(finding), qos=0)
        return json.dumps({"status": "reported"})

    elif name == "execute_command":
        command = args.get("command", "")
        timeout = args.get("timeout", 30)
        if not command:
            return json.dumps({"error": "No command provided"})
        try:
            output = subprocess.check_output(
                command,
                shell=True,
                text=True,
                timeout=timeout,
                stderr=subprocess.STDOUT,
            )
            return json.dumps({"status": "success", "output": output})
        except subprocess.CalledProcessError as exc:
            return json.dumps({"status": "failed", "output": exc.output or "", "error": str(exc)})
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "failed", "output": "", "error": "Command timed out"})

    elif name == "kill_process":
        pid = args.get("pid")
        sig = args.get("signal", "SIGTERM")
        signal_map = {"SIGTERM": signal.SIGTERM, "SIGKILL": signal.SIGKILL,
                      "SIGSTOP": signal.SIGSTOP, "SIGCONT": signal.SIGCONT}
        sig_num = signal_map.get(sig, signal.SIGTERM)
        try:
            os.kill(pid, sig_num)
            return json.dumps({"status": "success", "signal": sig, "pid": pid})
        except ProcessLookupError:
            return json.dumps({"status": "failed", "error": f"Process {pid} not found"})
        except PermissionError:
            return json.dumps({"status": "failed", "error": f"Permission denied to signal process {pid}"})
        except Exception as exc:
            return json.dumps({"status": "failed", "error": str(exc)})

    elif name == "cleanup_temp_files":
        directory = args.get("directory", "/tmp")
        days = args.get("days", 7)
        pattern = args.get("pattern", None)
        import fnmatch
        cutoff = time.time() - (days * 86400)
        if not os.path.isdir(directory):
            return json.dumps({"error": f"Directory not found: {directory}"})
        deleted = 0
        freed_bytes = 0
        try:
            for entry in os.scandir(directory):
                try:
                    if pattern and not fnmatch.fnmatch(entry.name, pattern):
                        continue
                    stat = entry.stat()
                    if stat.st_mtime < cutoff:
                        if entry.is_file() or entry.is_symlink():
                            size = stat.st_size
                            os.unlink(entry.path)
                            deleted += 1
                            freed_bytes += size
                except OSError:
                    pass
        except PermissionError:
            return json.dumps({"status": "partial", "error": "Permission denied",
                               "deleted": deleted, "freed_bytes": freed_bytes})
        return json.dumps({"status": "success", "deleted": deleted,
                           "freed_bytes": freed_bytes, "freed_mb": round(freed_bytes / (1024**2), 1)})

    elif name == "restart_service":
        service_name = args.get("service_name", "")
        if not service_name:
            return json.dumps({"error": "No service name provided"})
        try:
            output = subprocess.check_output(
                ["systemctl", "restart", service_name],
                text=True,
                timeout=60,
                stderr=subprocess.STDOUT,
            )
            return json.dumps({"status": "success", "service": service_name, "output": output})
        except subprocess.CalledProcessError as exc:
            return json.dumps({"status": "failed", "service": service_name,
                               "output": exc.output or "", "error": str(exc)})
        except FileNotFoundError:
            return json.dumps({"error": "systemctl not found — not a systemd system?"})
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "failed", "output": "", "error": "Service restart timed out"})

    return json.dumps({"error": f"Unknown tool: {name}"})


def run_sre_analysis(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if _llm is None:
        return []

    system_state = f"""Current system state:
CPU: {metrics['cpu']:.1f}% | Load: {metrics['load_1m']}/{metrics['load_5m']}/{metrics['load_15m']}
Memory: {metrics['memory']:.1f}% used ({metrics['memory_available_gb']} GB available of {metrics['memory_total_gb']} GB)
Disk: {metrics['disk']:.1f}% used ({metrics['disk_free_gb']} GB free)
Uptime: {metrics['uptime']:.0f}s

Analyze this state. If everything is normal, respond briefly.
If you find anomalies, use the available tools to investigate and report findings."""
    if metrics.get("disk_partitions"):
        system_state += "\nDisk partitions:\n"
        for dp in metrics["disk_partitions"]:
            flag = " ⚠️" if dp["percent"] >= 80 else ""
            system_state += f"  {dp['mount']}: {dp['percent']}% used ({dp['used_gb']}/{dp['total_gb']} GB){flag}\n"

    tools_desc = "AVAILABLE TOOLS:\n" + json.dumps(AVAILABLE_TOOLS, indent=2)

    messages = [
        {"role": "system", "content": SRE_SYSTEM_PROMPT + "\n\n" + tools_desc},
        {"role": "user", "content": system_state},
    ]

    try:
        response = _llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
    except Exception as exc:
        log.warning("LLM invocation failed: %s", exc)
        return []

    content = ""
    if response and "choices" in response:
        content = response["choices"][0].get("message", {}).get("content", "")
    if not content:
        return []

    log.info("LLM: %s", content[:200])

    # Parse <tool_call> blocks from the response
    findings = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
        try:
            tool_block = json.loads(match.group(1))
            tool_name = tool_block.get("name", "")
            tool_args = tool_block.get("arguments", {})
            log.info("Tool call: %s(%s)", tool_name, json.dumps(tool_args))
            result = execute_tool_call(tool_name, tool_args)
            findings.append({"tool": tool_name, "args": tool_args, "result": result})
        except json.JSONDecodeError:
            log.warning("Invalid tool_call JSON: %s", match.group(1)[:100])

    # If the LLM text contains severity keywords, treat the response as a finding
    content_no_tool = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
    if content_no_tool:
        severity = "INFO"
        for kw in ["CRITICAL", "critical", "Critical"]:
            if kw in content_no_tool:
                severity = "CRITICAL"
                break
        for kw in ["WARNING", "warning", "Warning"]:
            if kw in content_no_tool:
                severity = "WARNING"
                break
        finding = {
            "type": "sre_analysis",
            "severity": severity,
            "content": content_no_tool[:500],
            "tool_results": findings,
            "agent_id": _agent_id,
            "timestamp": time.time(),
        }
        if _client:
            _client.publish(TELEMETRY_TOPIC, json.dumps(finding), qos=0)

    return findings


def execute_task(command: str) -> dict[str, str]:
    try:
        output = subprocess.check_output(
            command,
            shell=True,
            text=True,
            timeout=30,
            stderr=subprocess.STDOUT,
        )
        return {"status": "success", "output": output, "error": ""}
    except subprocess.CalledProcessError as exc:
        return {"status": "failed", "output": exc.output or "", "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "output": "", "error": "Command timed out"}


def publish_telemetry() -> None:
    if _client is None:
        return
    metrics = gather_metrics()
    _client.publish(TELEMETRY_TOPIC, json.dumps(metrics), qos=0)


def shutdown(signum: int, frame: Any) -> None:
    log.info("Shutting down")
    if _client:
        system_info = gather_system_identity()
        offline = {
            "agent_id": _agent_id,
            "name": _name,
            "capabilities": [],
            "status": "offline",
            "uptime": 0.0,
            "token": MQTT_TOKEN,
            "hostname": system_info["hostname"],
        }
        _client.publish(DISCOVERY_TOPIC, json.dumps(offline), qos=1, retain=True)
        _client.disconnect()
        _client.loop_stop()
    exit(0)


def main() -> None:
    global _agent_id, _name, _client, _llm, _start_time

    parser = argparse.ArgumentParser(description="MQTT Edge SRE Agent")
    parser.add_argument("--agent-id", default=None, help="Override auto-generated agent ID")
    parser.add_argument("--name", default="Edge SRE Agent", help="Human-readable name")
    parser.add_argument("--sre-interval", type=int, default=300,
                        help="Seconds between LLM analysis runs (default 300)")
    args = parser.parse_args()

    _agent_id = generate_agent_id(args.agent_id)
    _name = args.name
    _start_time = time.time()
    sre_interval = args.sre_interval

    _llm = load_llm()

    log.info("Edge SRE Agent starting")
    log.info("Agent ID: %s", _agent_id)
    log.info("LLM mode: %s", "enabled" if _llm else "bare (telemetry only)")

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

    _client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    _client.loop_start()

    telemetry_counter = 0
    while True:
        time.sleep(TELEMETRY_INTERVAL)
        telemetry_counter += 1

        publish_telemetry()

        if _llm is not None and telemetry_counter % (sre_interval // TELEMETRY_INTERVAL) == 0:
            try:
                metrics = gather_metrics()
                run_sre_analysis(metrics)
            except Exception:
                log.exception("SRE analysis error")


if __name__ == "__main__":
    main()
