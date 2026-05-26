**[< MQTT Hub](../mqtt_agent_hub/README.md) | [Overview](overview.md) | [Operations](operations.md)**

# Edge Agent Deployment Guide

Deploy a lightweight MQTT SRE agent on a standalone Debian Linux minipc or edge device. The agent registers with the private MQTT Agent Hub, reports telemetry (CPU, memory, disk, load), executes tasks, and optionally runs a local LLM for autonomous system reliability analysis.

**Two modes of operation:**
- **Bare mode** (default): MQTT telemetry + task execution — no LLM required
- **LLM mode** (`LLM_PATH` set): Same as bare, plus periodic LLM-driven analysis using function-calling tools. The agent persona is a read-only SRE that investigates anomalies and reports findings

---

## Prerequisites

| Requirement | Bare mode | LLM mode | Notes |
|-------------|-----------|----------|-------|
| OS | Debian 11/12 or Ubuntu 20.04+ | Same | Any Debian derivative |
| Python | 3.8+ | 3.11+ | 3.11+ for llama-cpp-python |
| Pip | `python3-pip` | Same | Package installation |
| RAM | 128 MB | 1.5 GB | LLM needs ~800 MB for Q5_K_M GGUF |
| Disk (agent) | 20 MB | 50 MB | Agent script + Python packages |
| Disk (model) | 0 | 800 MB | Q5_K_M quantized GGUF file |
| Build tools | None | cmake, gcc, g++ | Required to compile llama-cpp-python |
| Network | Outbound TCP to hub on port 1883 | Same | MQTT connection |

---

## Network Requirements

The edge device must be able to reach the hub host on the LAN. Default ports:

| Source | Destination | Port | Protocol | Purpose |
|--------|-------------|------|----------|---------|
| Edge agent | Hub host | 1883 | TCP (MQTT) | Agent registration, telemetry, task delivery |
| Edge agent (optional) | Hub host | 8100 | TCP (HTTP) | Verification via REST API |
| Browser (dashboard) | Hub host | 9001 | TCP (WS) | Dashboard WebSocket |
| Browser (dashboard) | Hub host | 8100 | TCP (HTTP) | Dashboard REST fallback |

Verify connectivity from the edge device:

```bash
# Check MQTT port is reachable
nc -zv <HUB_IP> 1883

# Check hub API port (optional)
nc -zv <HUB_IP> 8100
```

Open the port on the hub host if blocked:

```bash
sudo ufw allow 1883/tcp
```

---

## Installation — Bare Mode

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

### 2. Install Python packages

```bash
pip install paho-mqtt psutil
```

### 3. Deploy the agent scripts

Copy both files from the hub host to the edge device:

```bash
scp mqtt_agent_hub/edge_agent.py user@<EDGE_IP>:/opt/mqtt-agent/agent.py
scp mqtt_agent_hub/sre_prompt.py user@<EDGE_IP>:/opt/mqtt-agent/sre_prompt.py
scp -r shared/ user@<EDGE_IP>:/opt/mqtt-agent/shared/
```

---

## Installation — LLM Mode (SRE Analysis)

### Additional packages

```bash
sudo apt install -y cmake gcc g++ python3-dev
pip install llama-cpp-python
```

### Download the model

Download `LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf` (recommended edge model — 1.2B params, function-calling tuned, runs on minipc):

```bash
# Requires huggingface-cli
pip install huggingface_hub
huggingface-cli download NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF \
    LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf \
    --local-dir /opt/mqtt-agent/models
```

Or via `wget`:

```bash
mkdir -p /opt/mqtt-agent/models
wget -P /opt/mqtt-agent/models \
    https://huggingface.co/NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF/resolve/main/LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf
```

The model uses ChatML format (`` tokens) with 4096 context and is specialized for tool-calling via `<tool_call>` JSON blocks.

---

## Configuration

The agent reads environment variables. Create the env file:

```bash
sudo mkdir -p /opt/mqtt-agent
sudo tee /etc/default/mqtt-agent << 'EOF'
# MQTT Agent Configuration
MQTT_BROKER_HOST=192.168.1.100       # ← change to your hub's IP
MQTT_BROKER_PORT=1883
MQTT_HUB_TOKEN=MySecretToken123      # ← must match ingest-svc.env on hub

# Optional: enable LLM-based SRE analysis
# LLM_PATH=/opt/mqtt-agent/models/LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf
EOF
sudo chmod 600 /etc/default/mqtt-agent
```

### Environment Variable Contract

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MQTT_BROKER_HOST` | **Yes** | `localhost` | IP or hostname of the host running Mosquitto |
| `MQTT_BROKER_PORT` | No | `1883` | MQTT broker port |
| `MQTT_HUB_TOKEN` | **Yes** | `changeme` | Pre-shared secret — must match `MQTT_HUB_TOKEN` in `doc-ingest-chat/ingest-svc.env` on the hub |
| `LLM_PATH` | No | (none) | Path to a GGUF model file. When set, enables periodic SRE analysis via the local LLM. When unset, runs in bare monitoring mode |

---

## Agent Identity

The agent ID is auto-generated on first launch from the hostname and a hash of `/etc/machine-id`. This produces a stable, unique identifier across restarts:

```
Format: {hostname}-{machine_id_hash_6_chars}
Example: bee1-a3f2c1
```

You can override with `--agent-id`:

```bash
python3 /opt/mqtt-agent/agent.py --agent-id my-custom-id
```

| Argument | Purpose | Default |
|----------|---------|---------|
| `--agent-id` | Unique identifier. Override the auto-generated ID. | `{hostname}-{machine_hash}` |
| `--name` | Human-readable label shown in the dashboard. | `Edge SRE Agent` |
| `--sre-interval` | Seconds between LLM analysis runs (LLM mode only). | `300` (5 minutes) |

### Capabilities (auto-detected)

The agent reports capabilities based on its configuration:
- `shell` — always available (executes shell tasks via subprocess)
- `monitor` — always available (CPU, memory, disk, load telemetry)
- `sre-analysis` — added when `LLM_PATH` is set (LLM-based system analysis)

These appear in the dashboard and help operators understand which agents can handle which tasks.

---

## SRE System Prompt (LLM Mode)

When running in LLM mode, the agent loads a read-only SRE persona (`sre_prompt.py`):

> "You are a meticulous system reliability engineer deployed on this host. Monitor CPU, memory, disk usage, and process health continuously. Watch for excessive resource consumption, suspicious behavior, stale resources, and configuration drift. You are NOT permitted to modify files, kill processes, restart services, or change system settings. Report findings first — await instructions before taking action."

The LLM has access to eight function-calling tools:

| Tool | Description |
|------|-------------|
| `get_system_metrics` | CPU, memory, disk, load, uptime |
| `list_top_processes` | Top N processes by CPU |
| `list_top_processes_by_memory` | Top N processes by memory |
| `check_disk_health` | Partitions exceeding 80% usage |
| `find_stale_temp_files` | Temp files older than N days |
| `check_suspicious_processes` | Processes from /tmp, high CPU, unusual names |
| `check_network_connections` | Active listeners and connections |
| `report_finding` | Publish a finding to the telemetry channel |

The analysis cycle works as follows:

```
Every 5 minutes (configurable):
1. Gather system metrics (CPU, memory, disk, load, processes)
2. Build ChatML prompt: system prompt + tools + current state
3. Invoke LLM → parse response for <tool_call> blocks
4. Execute each function call with real system data
5. Feed results back to LLM for final analysis
6. If LLM reports findings, publish to agent/telemetry with type: "sre_finding"
```

---

## Registration Flow (Automatic)

When the agent starts, it performs these steps automatically:

```
1. Generate agent ID from hostname + machine-id hash (e.g. "bee1-a3f2c1")

2. Connect to Mosquitto
   ├── Auth: username="hub", password=$MQTT_HUB_TOKEN
   └── LWT: Sets "Last Will" → agent/discovery with status:"offline"
           (Mosquitto publishes this automatically if the agent disconnects)

3. Publish discovery → agent/discovery (QoS 1, retained)
   Payload includes agent identity + system information:
   {
     "agent_id": "bee1-a3f2c1",
     "name": "Edge SRE Agent",
     "capabilities": ["shell", "monitor"],
     "status": "online",
     "hostname": "bee1",
     "ip_addresses": {"eth0": "192.168.1.5"},
     "mac_address": "b8:27:eb:12:34:56",
     "os_info": "Linux-6.8.0-45-generic-x86_64",
     "cpu_info": {"model": "Intel N100", "cores_physical": 4, "cores_logical": 4},
     "memory_total_gb": 15.6,
     "disk_total_gb": 238.5,
     "python_version": "3.11.2",
     "agent_version": "0.1.0",
     ...
   }

4. Subscribe to own task channel → agent/task/bee1-a3f2c1
   Subscribe to global channel → agent/global

5. Enter main loop
   ├── Every 15s → publish telemetry to agent/telemetry
   ├── Every 300s (if LLM loaded) → run SRE analysis cycle
   └── Forever → listen for tasks on agent/task/bee1-a3f2c1
```

The hub picks up the discovery message immediately. The agent appears in the dashboard within seconds.

---

## Launch & Background

### Manual launch — bare mode

```bash
export $(cat /etc/default/mqtt-agent | xargs)
python3 /opt/mqtt-agent/agent.py
```

### Manual launch — LLM mode

```bash
export $(cat /etc/default/mqtt-agent | xargs)
export LLM_PATH=/opt/mqtt-agent/models/LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf
python3 /opt/mqtt-agent/agent.py --sre-interval 300
```

Press `Ctrl+C` to stop. The agent publishes a final `status: "offline"` message before exiting.

### systemd service

```bash
sudo tee /etc/systemd/system/mqtt-agent.service << 'EOF'
[Unit]
Description=MQTT SRE Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=nobody
EnvironmentFile=/etc/default/mqtt-agent
ExecStart=/usr/bin/python3 /opt/mqtt-agent/agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mqtt-agent
```

Check status:

```bash
sudo systemctl status mqtt-agent
```

---

## Verification

Two ways to confirm the agent is registered and reporting.

### 1. Hub REST API

```bash
curl -s http://<HUB_IP>:8100/api/v1/agents | python3 -m json.tool
```

Look for your agent in the response:

```json
{
  "agent_id": "bee1-a3f2c1",
  "name": "Edge SRE Agent",
  "capabilities": ["shell", "monitor"],
  "status": "online",
  "uptime": 360.0,
  "last_seen": 1779500000.0,
  "hostname": "bee1",
  "ip_addresses": {"eth0": "192.168.1.5"},
  "mac_address": "b8:27:eb:12:34:56",
  "os_info": "Linux-6.8.0-45-generic-x86_64",
  "cpu_info": {"model": "Intel N100", "cores_physical": 4, "cores_logical": 4},
  "memory_total_gb": 15.6,
  "disk_total_gb": 238.5,
  "python_version": "3.11.2",
  "agent_version": "0.1.0",
  "latest_telemetry": {
    "cpu": 12.3,
    "memory": 45.1,
    "disk": 67.8,
    "load_1m": 0.45,
    "timestamp": 1779500000.0
  },
  "recent_tasks": [],
  "is_stale": false
}
```

A specific agent (use the auto-generated ID, not `bee1`):

```bash
curl -s http://<HUB_IP>:8100/api/v1/agents/bee1-a3f2c1 | python3 -m json.tool
```

Returns `404` if not found. Returns `503` if the hub's MQTT connection is not ready.

### 2. Agent logs

```bash
# systemd journal
sudo journalctl -u mqtt-agent -f

# Or stdout if running manually
```

Expected output (bare mode):

```
2024-01-01 12:00:00 [INFO] agent: Edge SRE Agent starting
2024-01-01 12:00:00 [INFO] agent: Agent ID: bee1-a3f2c1
2024-01-01 12:00:00 [INFO] agent: LLM mode: bare (telemetry only)
2024-01-01 12:00:01 [INFO] agent: Connected to MQTT broker at 192.168.1.100:1883
2024-01-01 12:00:01 [INFO] agent: Published discovery (retained)
```

Expected output (LLM mode):

```
2024-01-01 12:00:00 [INFO] agent: LLM mode: enabled
2024-01-01 12:00:05 [INFO] agent: LLM loaded: LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf
2024-01-01 12:05:01 [INFO] agent: Tool call: get_system_metrics({})
2024-01-01 12:05:02 [INFO] agent: LLM: All metrics within normal range. CPU 12%, memory 45%, disk 68%. No anomalies detected.
```

---

## Customizing the Agent

### Adding custom SRE tools

Edit `sre_prompt.py` — add entries to the `AVAILABLE_TOOLS` list and implement the corresponding handler in `edge_agent.py`'s `execute_tool_call()` function (line ~150). The tool definition format is OpenAI function-calling schema compatible.

### Changing the SRE persona

Edit `SRE_SYSTEM_PROMPT` in `sre_prompt.py`. The model uses ChatML format. Keep the READ-ONLY constraint explicit — the agent enforces it at the code level by only exposing read tools.

### Adjusting analysis frequency

```bash
# Run LLM analysis every 10 minutes
python3 agent.py --sre-interval 600
```

### Adding custom telemetry fields

Edit the `gather_metrics()` function in `edge_agent.py` to add custom metrics. Extra fields are passed through to the dashboard via the MQTT payload.

### Multiple agents on one device

Run multiple instances with different service files. Each auto-generates a unique agent ID from the same hostname + machine-id, so they share the same ID. To differentiate, override with `--agent-id`:

```bash
python3 agent.py --agent-id bee1-gpu     --name "Bee GPU Monitor"
python3 agent.py --agent-id bee1-docker  --name "Bee Docker Monitor"
```

---

## Troubleshooting

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| `Connection refused` | Wrong host/port, firewall, Mosquitto not running | `nc -zv <HUB_IP> 1883`, `docker compose ps mosquitto` on hub |
| `Not authorized` | Token mismatch or password file not mounted | Verify `MQTT_HUB_TOKEN` matches hub's `ingest-svc.env`. Verify `passwd` file is mounted in Mosquitto container |
| Agent not appearing in dashboard | Discovery message not arriving | Check `mosquitto_sub -t "agent/discovery"` from hub. Check `systemctl status mqtt-agent` on edge device |
| Agent shows offline after startup | LWT message fired prematurely or token invalid in payload | Verify `MQTT_HUB_TOKEN` is exported at launch. Check agent logs for token validation warnings on hub |
| LLM fails to load | Missing build tools or wrong Python version | `pip install llama-cpp-python` — requires cmake, gcc, python3-dev. Python 3.11+ |
| LLM runs but no findings | Model is loaded but analysis interval hasn't fired yet | Wait for `--sre-interval` seconds (default 300). Check logs for tool call output |
| Task not delivered | Agent ID in topic doesn't match subscribed topic | Tasks are routed to the auto-generated ID (`bee1-a3f2c1`), not just `bee1`. Check the dashboard for the actual agent ID |
| `503 Service Unavailable` from REST API | Hub MQTT client not initialized | Wait ~5s after hub startup. Check hub logs: `docker compose logs mqtt_hub` |
| `ImportError: sre_prompt` | Missing companion file | Copy both `edge_agent.py` AND `sre_prompt.py` to `/opt/mqtt-agent/` |

---

## MQTT Topic Reference

| Topic | QoS | Direction | Purpose |
|-------|-----|-----------|---------|
| `agent/discovery` | 1 (retained) | Agent → Hub | Registration, heartbeat, offline notification |
| `agent/telemetry` | 0 | Agent → Hub | Periodic CPU/memory/disk metrics |
| `agent/task/{agent_id}` | 1 | Hub → Agent | Direct task dispatch to a specific agent |
| `agent/task/{agent_id}/result` | 1 | Agent → Hub | Task execution result |
| `agent/global` | 1 | Hub → All | Broadcast message to all agents |

---

## JSON Contract Reference

### Discovery (`agent/discovery`)

```json
{
  "agent_id": "bee1-a3f2c1",
  "name": "Edge SRE Agent",
  "capabilities": ["shell", "monitor"],
  "status": "online",
  "uptime": 3600.0,
  "token": "MySecretToken123",
  "hostname": "bee1",
  "ip_addresses": {"eth0": "192.168.1.5", "wlan0": "192.168.1.6"},
  "mac_address": "b8:27:eb:12:34:56",
  "os_info": "Linux-6.8.0-45-generic-x86_64-with-glibc2.35",
  "cpu_info": {
    "model": "Intel(R) N100",
    "cores_physical": 4,
    "cores_logical": 4
  },
  "memory_total_gb": 15.6,
  "disk_total_gb": 238.5,
  "python_version": "3.11.2",
  "agent_version": "0.1.0"
}
```

### Telemetry (`agent/telemetry`)

```json
{
  "agent_id": "bee1",
  "cpu": 12.3,
  "memory": 45.1,
  "disk": 67.8,
  "uptime": 3600.0,
  "token": "MySecretToken123",
  "timestamp": 1779500000.0
}
```

### Task Request (`agent/task/{agent_id}`)

```json
{
  "task_id": "a1b2c3d4",
  "command": "ps aux --sort=-%cpu | head -6",
  "params": {},
  "origin": "hub"
}
```

### Task Result (`agent/task/{agent_id}/result`)

```json
{
  "task_id": "a1b2c3d4",
  "agent_id": "bee1",
  "status": "success",
  "output": "USER  PID  %CPU %MEM  VSZ  RSS TTY STAT START TIME COMMAND\n...",
  "error": ""
}
```
