# NiFi Middleware

Apache NiFi is deployed as transparent middleware between Redis queues, providing flow orchestration, provenance tracking, and backpressure management. Workers continue using direct Redis calls — NiFi sits between `_input` and `_output` queues.

## Architecture

```
Workers (Producers)          NiFi Middleware              Workers (Consumers)
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│ ocr_utils.py    │─LPUSH──▶│ ocr_input        │         │                 │
│ whisper_utils.py│─LPUSH──▶│ whisper_input    │─NiFi───▶│ ocr_worker.py   │
│ producer_graph  │─RPUSH──▶│ chunk:0_input    │ Flow    │ whisper_worker  │
│                 │─RPUSH──▶│ chunk:1_input    │ Files   │ consumer_worker │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                           │                           │
        │                           ▼                           │
        │                  ┌──────────────────┐                 │
        └──────────────────│ ocr_output       │◀──BRPOP─────────┘
                           │ whisper_output   │◀──BRPOP─────────┐
                           │ chunk:0_output   │◀──BLPOP─────────┤
                           │ chunk:1_output   │◀──BLPOP─────────┘
                           └──────────────────┘
```

**Key Points:**
- Workers continue using direct Redis calls (no code changes to worker logic)
- NiFi sits between `_input` and `_output` queues
- Correlation IDs (`reply_key`) are preserved in message payloads
- Queue names are tracked as FlowFile attributes for observability
- The `nifi_bootstrap` service automatically deploys the flow on startup

## Components

### Python Processors

Located in `python/extensions/`:

- **RedisQueueConsumer** (`FlowFileSource`): Pops messages from Redis `_input` queues using BRPOP/BLPOP/LPOP and creates FlowFiles
- **RedisQueueProducer** (`FlowFileTransform`): Pushes FlowFile content to Redis `_output` queues using LPUSH/RPUSH

Both processors use the native Python `redis` library with connection pooling.

### NiFi Client

`nifi_client.py` - NiPyAPI wrapper for programmatic flow management:
- Connects to remote NiFi via REST API
- Supports SSL verification toggle (for self-signed certs)
- Basic authentication
- Creates process groups, processors, and connections
- Starts/stops flows and verifies health

### Bootstrap Service

`nifi_bootstrap.py` - One-shot service that deploys and starts the flow:
- Waits for NiFi to become available (with exponential backoff)
- Checks if "RAG Pipeline" process group exists
- Creates the process group with consumer/producer pairs for each queue
- Starts all processors
- Verifies flow health
- Exits (doesn't restart)

This service runs automatically as part of the Docker Compose stack.

## Prerequisites

- NiFi 2.x deployed and accessible
- Python extensions enabled in NiFi
- Redis accessible from both workers and NiFi
- `nipyapi>=1.0.0` installed (`uv pip install 'nipyapi>=1.0.0'`)

## Deployment

### 1. Deploy NiFi

**Option A: Docker (for testing/development)**

```bash
docker run -d \
  --name nifi \
  -p 8443:8443 \
  -e NIFI_WEB_HTTPS_PORT=8443 \
  -e NIFI_WEB_HTTPS_HOST=0.0.0.0 \
  -e NIFI_WEB_PROXY_HOST=<nifi-host> \
  -e SINGLE_USER_CREDENTIALS_USERNAME=admin \
  -e SINGLE_USER_CREDENTIALS_PASSWORD=admin1234567 \
  -e NIFI_PYTHON_EXTENSIONS_ENABLED=true \
  -e NIFI_PYTHON_MAX_TASKS=2 \
  apache/nifi:2.0.0
```

**Option B: Existing NiFi instance**

Ensure Python extensions are enabled in `nifi.properties`:
```properties
nifi.python.extensions.source.directory.default=./python/extensions
nifi.python.max.tasks=2
```

### 2. Deploy Python Processors

Copy processors to NiFi's Python extensions directory:

```bash
# From the project root
scp nifi/python/extensions/RedisQueueConsumer.py <nifi-host>:/opt/nifi/nifi-current/python/extensions/
scp nifi/python/extensions/RedisQueueProducer.py <nifi-host>:/opt/nifi/nifi-current/python/extensions/

# Restart NiFi to load new processors
ssh <nifi-host> "docker restart nifi"
```

Wait 30-60 seconds for NiFi to restart and load the processors.

### 3. Configure Environment

```bash
export NIFI_ENDPOINT="https://<nifi-host>:8443/nifi-api"
export NIFI_USERNAME="admin"
export NIFI_PASSWORD="<your-password>"
export NIFI_SSL_VERIFY="false"  # For self-signed certificates

# Redis configuration (used by bootstrap to set processor properties)
export REDIS_HOST="<redis-host>"
export REDIS_PORT="<redis-port>"
export REDIS_DB="0"
```

### 4. Start the Stack

```bash
./doc-ingest-chat/run-compose.sh --build
```

The `nifi_bootstrap` service will automatically:
1. Deploy the flow (if not exists) or use existing flow
2. Update all processor properties from environment variables (REDIS_HOST, REDIS_PORT, REDIS_DB)
3. Start all processors
4. Verify flow health

### 5. Verify Bootstrap Success

Check the bootstrap service logs:

```bash
docker logs nifi_bootstrap
```

Expected output:
```
✅ NiFi endpoint configured successfully
✅ NiFi is available. Root process group ID: <id>
✅ Created process group 'RAG Pipeline' (ID: <id>)
✅ Created consumer processor for ocr_processing_job_input
✅ Created producer processor for ocr_processing_job_output
✅ Created connection: ocr_processing_job_input → ocr_processing_job_output
... (for each queue)
✅ Flow started successfully: RAG Pipeline
✅ All processors are running
✅ Flow deployed, started, and healthy
```

### 6. Access NiFi UI

Open `https://<nifi-host>:8443/nifi` in a browser.

You should see:
- "RAG Pipeline" process group on the root canvas
- Inside it: consumer/producer pairs for each queue
- All processors showing "Running" status
- Queue depths visible on connections

## Operations

### Monitor Flow

**NiFi UI:**
- Right-click processor → "View data provenance" to see message flow
- Right-click connection → "List queue" to see queued FlowFiles
- Check processor status icons (green = running, yellow = invalid, red = stopped)

**Command line:**
```bash
# Check Redis queue depths
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN ocr_processing_job_input
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN ocr_processing_job_output

# Check bootstrap service logs
docker logs nifi_bootstrap
```

### Stop the Flow

```bash
cd nifi/
python -c "
from nifi_client import NifiClient
from nipyapi import canvas
import os

client = NifiClient(
    base_url=os.getenv('NIFI_ENDPOINT'),
    username=os.getenv('NIFI_USERNAME'),
    password=os.getenv('NIFI_PASSWORD'),
    ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
)

pg_id = client.check_flow_exists('RAG Pipeline')
if pg_id:
    canvas.schedule_process_group(pg_id, scheduled=False)
    print('✅ Stopped RAG Pipeline')
"
```

### Start the Flow

```bash
cd nifi/
python -c "
from nifi_client import NifiClient
import os

client = NifiClient(
    base_url=os.getenv('NIFI_ENDPOINT'),
    username=os.getenv('NIFI_USERNAME'),
    password=os.getenv('NIFI_PASSWORD'),
    ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
)

pg_id = client.check_flow_exists('RAG Pipeline')
if pg_id:
    client.start_process_group(pg_id)
    print('✅ Started RAG Pipeline')
"
```

### Delete the Flow

```bash
cd nifi/
python -c "
from nifi_client import NifiClient
import os

client = NifiClient(
    base_url=os.getenv('NIFI_ENDPOINT'),
    username=os.getenv('NIFI_USERNAME'),
    password=os.getenv('NIFI_PASSWORD'),
    ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
)

pg_id = client.check_flow_exists('RAG Pipeline')
if pg_id:
    client.delete_process_group(pg_id)
    print('✅ Deleted RAG Pipeline')
else:
    print('RAG Pipeline not found')
"
```

### Redeploy the Flow

```bash
# Delete existing flow
cd nifi/
python -c "
from nifi_client import NifiClient
import os

client = NifiClient(
    base_url=os.getenv('NIFI_ENDPOINT'),
    username=os.getenv('NIFI_USERNAME'),
    password=os.getenv('NIFI_PASSWORD'),
    ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
)

pg_id = client.check_flow_exists('RAG Pipeline')
if pg_id:
    client.delete_process_group(pg_id)
    print('Deleted old flow')
"

# Re-run bootstrap
python nifi_bootstrap.py
```

### Update Processor Properties (Day-2)

To update Redis connection properties on existing processors without redeploying:

```bash
cd nifi/
python -c "
import os
import requests
import urllib3
urllib3.disable_warnings()

nifi_endpoint = os.getenv('NIFI_ENDPOINT', 'https://192.168.30.67:7443/nifi-api')
username = os.getenv('NIFI_USERNAME', 'admin')
password = os.getenv('NIFI_PASSWORD', 'admin1234567')
ssl_verify = os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true'

# Get access token
token_resp = requests.post(
    f'{nifi_endpoint}/access/token',
    data={'username': username, 'password': password},
    headers={'Content-Type': 'application/x-www-form-urlencoded'},
    verify=ssl_verify
)
token = token_resp.text
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

# Get all processors in RAG Pipeline
pg_id = 'b338c2d2-019e-1000-cb06-6997035faf96'  # Update if different
procs_resp = requests.get(f'{nifi_endpoint}/flow/process-groups/{pg_id}/processors', headers=headers, verify=ssl_verify)
processors = procs_resp.json()['processors']

# Update Redis Host and Port on all processors
new_host = '192.168.30.67'
new_port = '6380'

for proc in processors:
    proc_id = proc['id']
    proc_name = proc['component']['name']
    version = proc['revision']['version']
    
    # Get current processor config
    get_resp = requests.get(f'{nifi_endpoint}/processors/{proc_id}', headers=headers, verify=ssl_verify)
    proc_data = get_resp.json()
    
    # Update properties
    props = proc_data['component']['config']['properties']
    if 'Redis Host' in props:
        props['Redis Host'] = new_host
    if 'Redis Port' in props:
        props['Redis Port'] = new_port
    
    # Put updated config
    proc_data['component']['config']['properties'] = props
    proc_data['revision']['version'] = version
    
    put_resp = requests.put(f'{nifi_endpoint}/processors/{proc_id}', json=proc_data, headers=headers, verify=ssl_verify)
    if put_resp.status_code == 200:
        print(f'✅ Updated {proc_name} ({proc_id})')
    else:
        print(f'❌ Failed to update {proc_name}: {put_resp.status_code} - {put_resp.text}')

print('\\n⚠️  Restart processors to apply changes:')
print('   - Stop all processors in NiFi UI or via API')
print('   - Start all processors in NiFi UI or via API')
"
```

**Alternative: Update via NiFi UI**
1. Open NiFi UI at `https://<nifi-host>:8443/nifi`
2. Double-click "RAG Pipeline" process group
3. Right-click each processor → Configure
4. Update "Redis Host" and "Redis Port" properties
5. Click Apply, then restart the processor

## Rollback to Direct Redis

If NiFi is causing issues, you can revert to direct Redis:

1. **Stop and delete the NiFi flow** (see "Delete the Flow" above)

2. **Revert worker code** to use base queue names (remove `_input`/`_output` suffixes):
   - `ocr_utils.py`: `lpush(REDIS_OCR_JOB_QUEUE, ...)` instead of `lpush(f"{REDIS_OCR_JOB_QUEUE}_input", ...)`
   - `whisper_utils.py`: `lpush(REDIS_WHISPER_JOB_QUEUE, ...)` instead of `lpush(f"{REDIS_WHISPER_JOB_QUEUE}_input", ...)`
   - `producer_graph.py`: `queue_name` instead of `f"{queue_name}_input"`
   - `ocr_worker.py`: `brpop(REDIS_OCR_JOB_QUEUE, ...)` instead of `brpop(f"{REDIS_OCR_JOB_QUEUE}_output", ...)`
   - `whisperx_worker.py`: `brpop(REDIS_WHISPER_JOB_QUEUE, ...)` instead of `brpop(f"{REDIS_WHISPER_JOB_QUEUE}_output", ...)`
   - `consumer_worker.py`: `blpop(queue_name, ...)` instead of `blpop(f"{queue_name}_output", ...)`

3. **Restart workers**:
   ```bash
   ./doc-ingest-chat/run-compose.sh --build
   ```

## Testing

### Unit Tests

```bash
# Test NiFi processors (mocks nifiapi)
.venv/bin/python -m pytest nifi/tests/test_redis_queue_consumer.py -v
.venv/bin/python -m pytest nifi/tests/test_redis_queue_producer.py -v

# Test NiFi client (mocks nipyapi)
.venv/bin/python -m pytest nifi/tests/test_nifi_client.py -v
```

### Integration Tests

Requires running Redis and NiFi:

```bash
# Test queue naming convention and round-trip
.venv/bin/python -m pytest nifi/tests/test_nifi_integration.py -v
```

### Manual Testing

1. Push a test message to an `_input` queue:
   ```bash
   docker exec -it $(docker ps -q -f name=redis) redis-cli LPUSH ocr_processing_job_input '{"job_id":"test","reply_key":"test:123"}'
   ```

2. Check NiFi UI for FlowFile movement

3. Pop from `_output` queue:
   ```bash
   docker exec -it $(docker ps -q -f name=redis) redis-cli BRPOP ocr_processing_job_output 5
   ```

## Troubleshooting

See [infra/operations/day-2.md](../infra/operations/day-2.md#9-nifi--middleware-troubleshooting) for detailed troubleshooting guide.

**Common issues:**
- Bootstrap service failed → Check `docker logs nifi_bootstrap`
- Processors not found → Deploy processor files and restart NiFi
- SSL errors → Set `NIFI_SSL_VERIFY=false` for self-signed certs
- Auth failures → Verify `NIFI_USERNAME`/`NIFI_PASSWORD` match NiFi config
- Messages not flowing → Check processor status in NiFi UI
- Invalid processors → Check processor properties in NiFi UI
- **Ghost processors** → See below

### Ghost Processors

**What are ghost processors?**
When a Python processor fails to load (syntax error, import error, missing dependency, or invalid base class), NiFi creates a "ghost" processor that appears in the flow but cannot run. These show as "Invalid" in the NiFi UI with errors like "Missing Processor" or "Processor is of type X, but this is not a valid Processor type".

**Common causes:**
- Python syntax errors in processor code
- Using wrong base class (e.g., `FlowFileProcessor` instead of `FlowFileSource`)
- Missing or incorrect imports
- Invalid `__init__` signature (NiFi passes `jvm` keyword argument)
- Using Java-style `PropertyDescriptor.Builder()` instead of direct constructor
- Passing Java Relationship object to `FlowFileSourceResult` (use string name instead)
- Default scheduling period is 1 minute (set to 100ms for near-real-time)

**How to fix:**

1. **Stop the process group:**
   ```bash
   export NIFI_ENDPOINT="https://192.168.30.67:7443/nifi-api"
   export NIFI_USERNAME="admin"
   export NIFI_PASSWORD="admin1234567"
   
   python3 -c "
   import requests
   import urllib3
   urllib3.disable_warnings()
   
   token = requests.post(f'$NIFI_ENDPOINT/access/token', 
                         data={'username': '$NIFI_USERNAME', 'password': '$NIFI_PASSWORD'}, 
                         verify=False).text
   headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
   
   pg_id = 'YOUR_PROCESS_GROUP_ID'
   requests.put(f'$NIFI_ENDPOINT/flow/process-groups/{pg_id}', 
                json={'id': pg_id, 'state': 'STOPPED'}, 
                headers=headers, verify=False)
   "
   ```

2. **Delete the ghost processors:**
   ```bash
   python3 -c "
   import requests
   import urllib3
   import time
   urllib3.disable_warnings()
   
   token = requests.post(f'$NIFI_ENDPOINT/access/token', 
                         data={'username': '$NIFI_USERNAME', 'password': '$NIFI_PASSWORD'}, 
                         verify=False).text
   headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
   
   ghost_ids = ['PROC_ID_1', 'PROC_ID_2', 'PROC_ID_3']
   for proc_id in ghost_ids:
       resp = requests.get(f'$NIFI_ENDPOINT/processors/{proc_id}', headers=headers, verify=False)
       if resp.status_code == 200:
           version = resp.json()['revision']['version']
           requests.delete(f'$NIFI_ENDPOINT/processors/{proc_id}?version={version}', 
                          headers=headers, verify=False)
   "
   ```

3. **Fix the processor code and redeploy:**
   ```bash
   scp nifi/python/extensions/RedisQueueConsumer.py vm-ace:/opt/nifi/python_extensions/
   ssh vm-ace "docker restart nifi-2.0-ai"
   ```

4. **Re-run bootstrap:**
   ```bash
   cd nifi/
   export NIFI_ENDPOINT="https://192.168.30.67:7443/nifi-api"
   export NIFI_USERNAME="admin"
   export NIFI_PASSWORD="admin1234567"
   export NIFI_SSL_VERIFY="false"
   export REDIS_HOST="192.168.30.67"
   export REDIS_PORT="6380"
   export REDIS_DB="0"
   python3 nifi_bootstrap.py
   ```

**Prevention:**
- Always test processor code locally with `python3 -m py_compile processor.py` before deploying
- Use correct base classes: `FlowFileSource` for sources, `FlowFileTransform` for transforms
- Accept `jvm=None, **kwargs` in `__init__` to handle NiFi's initialization
- Use direct `PropertyDescriptor()` constructor, not Java-style builders
- Use string relationship names in `FlowFileSourceResult` (e.g., `"success"`), not Java Relationship objects
- Set scheduling period to 100ms for consumer processors (default is 1 minute)

## Configuration Reference

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `NIFI_ENDPOINT` | NiFi REST API endpoint (must include `/nifi-api`) | `https://nifi-host:8443/nifi-api` |
| `NIFI_USERNAME` | NiFi username for basic auth | `admin` |
| `NIFI_PASSWORD` | NiFi password for basic auth | `admin1234567` |
| `NIFI_SSL_VERIFY` | SSL certificate verification | `false` |
| `REGISTRY_ENDPOINT` | NiFi Registry endpoint (optional) | `http://registry-host:18080/nifi-registry-api` |
| `REDIS_HOST` | Redis server hostname | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_DB` | Redis database number | `0` |

### Queue Names

Configured via environment variables:
- `REDIS_OCR_JOB_QUEUE` (default: `ocr_processing_job`)
- `REDIS_WHISPER_JOB_QUEUE` (default: `whisper_processing_job`)
- `QUEUE_NAMES` (default: `chunk_ingest_queue:0,chunk_ingest_queue:1`)

Each queue gets `_input` and `_output` suffixes for the NiFi sandwich pattern.

## Future Enhancements

- **Phase 2**: Workers push directly to NiFi input ports via REST API (full provenance)
- **Phase 3**: NiFi handles reply routing with correlation IDs
- **Phase 4**: NiFi Registry for flow versioning and deployment
- **Phase 5**: Native NiFi processors replace Python workers for specific tasks

## References

- [NiFi Documentation](https://nifi.apache.org/docs.html)
- [NiPyAPI Documentation](https://nipyapi.readthedocs.io/)
- [NiFi Python Processor Development](https://nifi.apache.org/docs/nifi-docs/html/developer-guide.html#python-extensions)
- [Quick Start Guide](../docs/quickstart.md#nifi-middleware-required)
- [Day 1 Operations](../infra/operations/day-1.md#7-nifi-middleware-required)
- [Day 2 Operations](../infra/operations/day-2.md#9-nifi--middleware-troubleshooting)
