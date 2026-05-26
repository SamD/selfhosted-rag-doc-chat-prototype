# MQTT Agent Hub

**A private, air-gapped multi-agent communication hub for LAN environments. Agents on edge devices announce themselves, report telemetry, take tasks, and optionally run a local LLM for autonomous system reliability analysis. No cloud, no certificates, no accounts.**

---

## Architecture

```
┌──────────────────┐       ┌───────────────┐       ┌──────────────────┐
│  Edge Agent      │ MQTT  │  Mosquitto    │ MQTT  │  Hub (FastAPI)   │
│  (bare metal)    │──────▶│  (broker)     │──────▶│  :8100           │
│                  │       │  :1883 (MQTT) │       │                  │
│  telemetry +     │       │  :9001 (WS)   │       │  registry +      │
│  tasks + SRE     │       └───────────────┘       │  REST API        │
│                  │                                │  dashboard serve │
└──────────────────┘                                └────────┬─────────┘
                                                              │
                                                     ┌────────▼─────────┐
                                                     │  Dashboard       │
                                                     │  (Astro SPA)     │
                                                     │  :4322           │
                                                     └──────────────────┘
```

### MQTT Topics

| Topic | QoS | Retained | Direction | Purpose |
|-------|-----|----------|-----------|---------|
| `agent/discovery` | 1 | yes | Agent → Hub | Registration with system identity (hostname, IP, MAC, OS, CPU, memory, disk) |
| `agent/telemetry` | 0 | no | Agent → Hub | Periodic CPU/memory/disk/load + optional SRE findings |
| `agent/task/{agent_id}` | 1 | no | Hub → Agent | Direct task dispatch to a specific agent |
| `agent/task/{agent_id}/result` | 1 | no | Agent → Hub | Task execution result |
| `agent/global` | 1 | no | Hub → All | Broadcast message to all connected agents |

All topic strings, env var names, and message schemas are defined in `shared/` — a single source of truth shared across the hub, agents, and tests.

---

## Quick Start

### Prerequisites

- Docker v2.20+
- The `shared/` directory must be alongside `mqtt_agent_hub/` at repo root (monorepo structure)

### Deploy the hub standalone

```bash
cd mqtt_agent_hub
cp .env.example .env              # edit MQTT_HUB_TOKEN
docker compose up                 # starts mosquitto + hub + dashboard
```

### Or deploy everything (RAG pipeline + agent hub)

```bash
# From repo root
docker compose up
```

The root `docker-compose.yaml` includes both `doc-ingest-chat/docker-compose.yaml` and `mqtt_agent_hub/docker-compose.yaml`.

### Access points

| Service | URL |
|---------|-----|
| Hub REST API | `http://localhost:8100/api/v1/health` |
| Agent dashboard | `http://localhost:4322` |
| Mosquitto MQTT | `mqtt://localhost:1883` |
| Mosquitto WebSocket | `ws://localhost:9001` |

The dashboard defaults to dark theme. A theme picker in the header provides 11 daisyUI themes: dark, light, corporate, synthwave, cyberpunk, forest, dracula, night, nord, dim, sunset. Selection persists in localStorage.

---

## Testing

Both the chat frontend and dashboard have `npm test` scripts that run `astro check && astro build`, plus shell-based smoke tests that verify daisyUI classes survive the build:

```bash
# Dashboard
cd mqtt_agent_hub/astro-dashboard
npm test && bash test.sh

# Chat frontend
cd astro-frontend
npm test && bash test.sh
```

---

## REST API

Base: `http://localhost:8100/api/v1`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/agents` | List all agents (with system identity + latest telemetry) |
| `GET` | `/agents/{id}` | Get a specific agent |
| `POST` | `/task/{agent_id}` | Dispatch a task to an agent |
| `POST` | `/global` | Broadcast a message to all agents |

### Example

```bash
# List all agents
curl -s http://localhost:8100/api/v1/agents | python3 -m json.tool

# Send a task to agent bee1-a3f2c1
curl -s -X POST http://localhost:8100/api/v1/task/bee1-a3f2c1 \
  -H "Content-Type: application/json" \
  -d '{"command": "ps aux --sort=-%cpu | head -6"}'

# Broadcast to all agents
curl -s -X POST http://localhost:8100/api/v1/global \
  -H "Content-Type: application/json" \
  -d '{"content": "Maintenance window starting in 5 minutes"}'
```

---

## Edge Agent Deployment

Agents run on bare-metal Debian minipcs. Full instructions in [docs/edge-agent.md](../docs/edge-agent.md).

### Quick deploy (bare mode — telemetry only)

```bash
# On the edge device
sudo apt install -y python3 python3-pip
pip install paho-mqtt psutil

# Copy agent + shared contracts from hub
scp mqtt_agent_hub/edge_agent.py user@<EDGE_IP>:/opt/mqtt-agent/agent.py
scp mqtt_agent_hub/sre_prompt.py user@<EDGE_IP>:/opt/mqtt-agent/sre_prompt.py
scp -r shared/ user@<EDGE_IP>:/opt/mqtt-agent/shared/

# Configure
sudo mkdir -p /opt/mqtt-agent
sudo tee /etc/default/mqtt-agent << 'EOF'
MQTT_BROKER_HOST=<HUB_IP>
MQTT_HUB_TOKEN=MySecretToken123
EOF

# Launch
MQTT_BROKER_HOST=<HUB_IP> MQTT_HUB_TOKEN=MySecretToken123 \
  python3 /opt/mqtt-agent/agent.py
```

### LLM mode (SRE analysis with LFM2.5)

```bash
# Install llama-cpp-python and download model
sudo apt install -y cmake gcc g++ python3-dev
pip install llama-cpp-python
wget -P /opt/mqtt-agent/models \
https://huggingface.co/NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF/resolve/main/LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf?download=true

# Add to env
echo 'LLM_PATH=/opt/mqtt-agent/models/LFM2.5-1.2B-Nova-Function-Calling.Q5_K_M.gguf' \
  | sudo tee -a /etc/default/mqtt-agent

# Restart agent — LLM analysis runs every 5 minutes
sudo systemctl restart mqtt-agent
```

---

## Agent Identity & Registration

On startup, each agent auto-generates a unique ID and publishes a discovery message:

```json
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
  "agent_version": "NOT_SET"
}
```

- **Agent ID format**: `{hostname}-{machine_id_hash_6chars}` — stable across restarts, unique per machine
- **Offline detection**: Mosquitto Last Will & Testament publishes `status: "offline"` on disconnect
- **Stale pruning**: Hub background task marks agents offline after 60s of no telemetry

---

## SRE Persona & Function Calling

When running in LLM mode, the agent loads an SRE system prompt and 8 function-calling tools:

| Tool | Description |
|------|-------------|
| `get_system_metrics` | CPU, memory, disk, load, uptime |
| `list_top_processes` | Top N processes by CPU |
| `list_top_processes_by_memory` | Top N processes by memory |
| `check_disk_health` | Partitions exceeding 80% usage |
| `find_stale_temp_files` | Temp files older than N days |
| `check_suspicious_processes` | Processes from /tmp, high CPU, unusual names |
| `check_network_connections` | Active listeners and connections |
| `report_finding` | Publish finding to telemetry channel |

The system prompt enforces a strict READ-ONLY constraint: the agent may inspect and report but must not modify, kill, or delete without explicit instruction.

---

## Environment Variables

### Hub (`mqtt_agent_hub/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER_HOST` | `localhost` | Mosquitto broker hostname/IP (native MQTT for hub + edge agents) |
| `MQTT_BROKER_PORT` | `1883` | Native MQTT port |
| `MQTT_WS_PORT` | `9001` | WebSocket port (Mosquitto listener + dashboard) |
| `MQTT_HUB_TOKEN` | `changeme` | Pre-shared authentication token |
| `HUB_PORT` | `8100` | Hub REST API port |

### Dashboard (`mqtt_agent_hub/.env` — `PUBLIC_` prefix = exposed to browser)

| Variable | Default | Description |
|----------|---------|-------------|
| `PUBLIC_MQTT_BROKER_HOST` | _(dashboard hostname)_ | Override when Mosquitto is on a different host than the dashboard |
| `PUBLIC_HUB_PORT` | `8100` | Hub REST API port (must match `HUB_PORT`) |
| `PUBLIC_MQTT_WS_PORT` | `9001` | MQTT WebSocket port (must match `MQTT_WS_PORT`) |
| `PUBLIC_MQTT_USERNAME` | `hub` | MQTT username for dashboard WebSocket auth |
| `PUBLIC_MQTT_PASSWORD` | `changeme` | MQTT password for dashboard WebSocket auth |

### Edge Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER_HOST` | `localhost` | Hub hostname/IP |
| `MQTT_BROKER_PORT` | `1883` | MQTT port |
| `MQTT_HUB_TOKEN` | `changeme` | Pre-shared secret |
| `LLM_PATH` | (none) | Path to GGUF model (enables SRE mode) |
| `AGENT_VERSION` | `NOT_SET` | Version string in discovery |

---

## Project Structure

```
mqtt_agent_hub/
├── pyproject.toml              # Python deps (fastapi, paho-mqtt, pydantic, psutil)
├── docker-compose.yaml         # Standalone: mosquitto + hub + dashboard
├── Dockerfile.hub              # python:3.12-slim
├── .env                        # MQTT configuration
├── mosquitto.conf              # Broker config (MQTT :1883 + WS :9001)
├── passwd / acl                # Mosquitto auth
├── hub_server.py               # FastAPI entry point
├── config.py                   # Runtime config (imports shared.*)
├── mqtt/
│   ├── client.py               # Async MQTT client (paho-mqtt VERSION2)
│   └── registry.py             # In-memory agent registry
├── api/
│   └── endpoints.py            # REST API router
├── edge_agent.py               # Edge agent (bare + LLM mode)
├── test_mqtt_publisher.py      # Reference/test agent
├── sre_prompt.py               # SRE persona + function-calling tools
├── astro-dashboard/            # Astro v6 + Tailwind v4 + daisyUI SPA
│   ├── src/components/ThemePicker.astro  # 11-theme selector
│   └── test.sh                  # Build + content smoke test
└── tests/                      # 19 pytest tests
```

---

## Development

```bash
# Install deps into venv
uv pip install -e ".[test]" --python .venv/bin/python

# Run tests
PYTHONPATH=/path/to/repo:. .venv/bin/python -m pytest mqtt_agent_hub/tests/ -v

# Run hub locally (for development)
PYTHONPATH=/path/to/repo:. python -m mqtt_agent_hub.hub_server

# Lint
ruff check mqtt_agent_hub/ shared/
```

### Running without Docker

Start Mosquitto manually on the host, then:

```bash
export MQTT_BROKER_HOST=localhost
export MQTT_HUB_TOKEN=devtoken
python -m mqtt_agent_hub.hub_server
```

The dashboard can be started separately:

```bash
cd mqtt_agent_hub/astro-dashboard
npm install

# With broker on localhost (default)
npm run dev

# With remote broker
PUBLIC_MQTT_BROKER_HOST=192.168.1.100 npm run dev
```

### Testing an agent locally

```bash
PYTHONPATH=/path/to/repo:. python mqtt_agent_hub/test_mqtt_publisher.py --agent-id test1 --name "Test Agent"
```

---

## Remote Broker Deployment

To run Mosquitto on a separate host from the dashboard and hub:

### Broker host

```bash
# Start only Mosquitto
docker compose up mosquitto
```

### Dashboard + hub host

```bash
# Configure .env to point at the remote broker
PUBLIC_MQTT_BROKER_HOST=192.168.1.100   # broker IP
MQTT_BROKER_HOST=192.168.1.100          # hub native MQTT connection
MQTT_BROKER_PORT=1883
PUBLIC_MQTT_WS_PORT=9001
PUBLIC_HUB_PORT=8100

# Start hub + dashboard (no mosquitto)
docker compose up mqtt_hub hub_dashboard
```

Edge agents connect to the same `MQTT_BROKER_HOST` as the hub. The dashboard connects via WebSocket to `PUBLIC_MQTT_BROKER_HOST` which is resolved in the browser — ensure the broker's WebSocket port is reachable from the browser's network.

---

## Token Auth

Authentication uses a single pre-shared token:

1. **Mosquitto level**: Password file (`passwd`) with user `hub` and the token hash. Agents authenticate as `username=hub, password=<token>`
2. **Application level**: Every discovery and telemetry payload includes a `token` field. The hub validates it against `MQTT_HUB_TOKEN` before accepting the message

---

## See Also

- [docs/edge-agent.md](../docs/edge-agent.md) — Complete edge agent deployment guide (bare metal, systemd, LLM mode)
- [docs/overview.md](../docs/overview.md) — RAG pipeline architecture
- [shared/](../shared/) — Single source of truth for topics, env names, defaults, and message models
