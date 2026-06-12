#!/usr/bin/env python3
"""
NiFi Bootstrap Service

One-shot service that deploys and starts the RAG pipeline flow in NiFi.
This service runs once at startup, deploys the flow, and exits.

Steps:
1. Flush Redis queues
2. SCP processor code to NiFi host
3. Restart NiFi container
4. Wait for NiFi to become available
5. Delete all existing process groups
6. Create new RAG Pipeline flow
7. Update processor properties
8. Start the flow
9. Verify health

Uses NiPyAPI CI operations for reliable deployment.
"""
import logging
import os
import subprocess
import sys
import time

import nipyapi
import redis
import requests
from nifi_client import NifiClient
from nipyapi import canvas, config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


STATIC_QUEUES = [
    "ocr_processing_job_input",
    "ocr_processing_job_output",
    "ocr_reply_input",
    "ocr_reply_output",
    "whisper_processing_job_input",
    "whisper_processing_job_output",
    "whisper_reply_input",
    "whisper_reply_output",
    "retype_llm_job_input",
    "retype_llm_job_output",
    "retype_llm_reply_input",
    "retype_llm_reply_output",
]


def flush_redis():
    """Flush all pipeline Redis queues."""
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))

    logger.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    for attempt in range(1, 31):
        try:
            r.ping()
            break
        except redis.ConnectionError:
            if attempt == 30:
                logger.error("Failed to connect to Redis after 30 attempts")
                sys.exit(1)
            logger.info(f"Waiting for Redis... (attempt {attempt}/30)")
            time.sleep(2)

    logger.info("Connected to Redis")

    queue_names_str = os.environ.get("QUEUE_NAMES", "chunk_ingest_queue:0")
    queue_names = [q.strip() for q in queue_names_str.split(",") if q.strip()]
    chunk_queues = []
    for q in queue_names:
        chunk_queues.append(f"{q}_input")
        chunk_queues.append(f"{q}_output")

    all_queues = STATIC_QUEUES + chunk_queues
    total = 0
    for queue in all_queues:
        length = r.llen(queue)
        if length > 0:
            r.delete(queue)
            logger.info(f"Flushed {queue} ({length} messages)")
        else:
            logger.info(f"{queue} (empty)")
        total += length

    logger.info(f"Flushed {total} messages from {len(all_queues)} queues")


def deploy_processors():
    """SCP processor code to NiFi host and restart NiFi container."""
    ssh_host = os.environ.get("NIFI_SSH_HOST", "")
    if not ssh_host:
        logger.error("NIFI_SSH_HOST is required to deploy processors")
        sys.exit(1)

    extensions_dir = os.environ.get("NIFI_EXTENSIONS_DIR", "")
    if not extensions_dir:
        logger.error("NIFI_EXTENSIONS_DIR is required to deploy processors")
        sys.exit(1)
    container_name = os.environ.get("NIFI_CONTAINER_NAME", "nifi-2.0-ai")
    
    # Default to local path relative to nifi/ directory, or Docker path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_processors_dir = os.path.join(script_dir, "python", "extensions")
    processors_dir = os.environ.get("PROCESSORS_DIR", default_processors_dir)

    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    logger.info(f"Copying processors to {ssh_host}...")
    subprocess.run(
        f"ssh {ssh_opts} {ssh_host} 'mkdir -p {extensions_dir}'",
        shell=True, check=True
    )
    subprocess.run(
        f"scp {ssh_opts} {processors_dir}/*.py {ssh_host}:{extensions_dir}/",
        shell=True, check=True
    )

    logger.info(f"Restarting NiFi container {container_name}...")
    subprocess.run(
        f"ssh {ssh_opts} {ssh_host} 'docker restart {container_name}'",
        shell=True, check=False
    )


def configure_nifi_connection():
    """Configure NiPyAPI connection from environment variables."""
    nifi_endpoint = os.getenv("NIFI_ENDPOINT")
    if not nifi_endpoint:
        logger.error("❌ NIFI_ENDPOINT environment variable is required")
        sys.exit(1)

    nifi_username = os.getenv("NIFI_USERNAME")
    nifi_password = os.getenv("NIFI_PASSWORD")
    nifi_ssl_verify = os.getenv("NIFI_SSL_VERIFY", "false").lower() == "true"

    if not nifi_username or not nifi_password:
        logger.error("❌ NIFI_USERNAME and NIFI_PASSWORD environment variables are required")
        sys.exit(1)

    logger.info(f"Configuring NiFi connection to {nifi_endpoint}")

    # Configure NiPyAPI
    config.nifi_config.host = nifi_endpoint
    config.nifi_config.verify_ssl = nifi_ssl_verify
    config.nifi_config.username = nifi_username
    config.nifi_config.password = nifi_password

    if not nifi_ssl_verify:
        logger.warning("SSL verification disabled - using self-signed certificates")

    try:
        nipyapi.utils.set_endpoint(
            nifi_endpoint,
            ssl=nifi_endpoint.startswith("https"),
            login=True,
        )
        logger.info("✅ NiFi endpoint configured successfully")
    except Exception as e:
        logger.error(f"❌ Failed to configure NiFi endpoint: {e}")
        sys.exit(1)


def wait_for_nifi(max_retries=10, initial_delay=5):
    """Wait for NiFi to become available with exponential backoff."""
    logger.info(f"Waiting for NiFi to become available (max {max_retries} retries)")

    nifi_endpoint = os.getenv("NIFI_ENDPOINT")
    nifi_username = os.getenv("NIFI_USERNAME")
    nifi_password = os.getenv("NIFI_PASSWORD")
    ssl_verify = os.getenv("NIFI_SSL_VERIFY", "false").lower() == "true"

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # Use requests to check if NiFi is ready
            token_response = requests.post(
                f"{nifi_endpoint}/access/token",
                data={"username": nifi_username, "password": nifi_password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                verify=ssl_verify,
                timeout=5
            )
            if token_response.status_code == 201:
                logger.info(f"✅ NiFi is available at {nifi_endpoint}")
                return True
        except Exception as e:
            logger.warning(f"NiFi not available (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, 60)  # Exponential backoff, max 60s

    logger.error("❌ NiFi is not available after all retries")
    return False


def check_flow_exists(flow_name="RAG Pipeline"):
    """Check if the RAG Pipeline flow already exists."""
    logger.info(f"Checking if flow '{flow_name}' already exists")

    try:
        pgs = canvas.list_all_process_groups()
        for pg in pgs:
            if pg.component.name == flow_name:
                logger.info(f"✅ Flow '{flow_name}' already exists (ID: {pg.id})")
                return pg.id
        logger.info(f"Flow '{flow_name}' does not exist")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to check for existing flow: {e}")
        return None


def delete_all_process_groups():
    """Delete all process groups from the root canvas."""
    logger.info("Deleting all existing process groups...")

    try:
        import requests
        nifi_endpoint = os.getenv("NIFI_ENDPOINT")
        nifi_username = os.getenv("NIFI_USERNAME")
        nifi_password = os.getenv("NIFI_PASSWORD")
        ssl_verify = os.getenv("NIFI_SSL_VERIFY", "false").lower() == "true"

        # Get access token
        token_response = requests.post(
            f"{nifi_endpoint}/access/token",
            data={"username": nifi_username, "password": nifi_password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            verify=ssl_verify
        )
        token = token_response.text
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Get all process groups
        pgs = canvas.list_all_process_groups()
        deleted_count = 0

        for pg in pgs:
            pg_id = pg.id
            pg_name = pg.component.name
            logger.info(f"Deleting process group '{pg_name}' (ID: {pg_id})")

            try:
                # Stop the process group first
                stop_url = f"{nifi_endpoint}/flow/process-groups/{pg_id}"
                stop_data = {"id": pg_id, "state": "STOPPED"}
                requests.put(stop_url, json=stop_data, headers=headers, verify=ssl_verify)
                time.sleep(1)

                # Get current revision
                get_url = f"{nifi_endpoint}/process-groups/{pg_id}"
                resp = requests.get(get_url, headers=headers, verify=ssl_verify)
                if resp.status_code == 200:
                    version = resp.json()["revision"]["version"]

                    # Delete the process group
                    del_url = f"{nifi_endpoint}/process-groups/{pg_id}?version={version}"
                    resp = requests.delete(del_url, headers=headers, verify=ssl_verify)
                    if resp.status_code == 200:
                        logger.info(f"✅ Deleted '{pg_name}'")
                        deleted_count += 1
                    else:
                        logger.error(f"❌ Failed to delete '{pg_name}': {resp.status_code} - {resp.text}")
                else:
                    logger.error(f"❌ Failed to get revision for '{pg_name}': {resp.status_code}")

            except Exception as e:
                logger.error(f"❌ Error deleting '{pg_name}': {e}")

        logger.info(f"✅ Deleted {deleted_count} process group(s)")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to delete process groups: {e}")
        return False


def _create_queue_pair(client, pg_id, workflow, redis_host, redis_port, redis_db):
    """Create a forwarder → worker pair with error logger for a single queue."""
    # Forwarder (RedisQueueConsumer - reads from input queue)
    forwarder_id = client.create_consumer_processor(
        process_group_id=pg_id,
        queue_name=workflow["input_queue"],
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        socket_timeout=5,
        connection_timeout=5
    )
    client.update_processor_name(forwarder_id, workflow["forwarder_name"])
    client.update_processor_scheduling(forwarder_id, scheduling_period="100 ms", concurrent_tasks=1)
    logger.info(f"  ✅ {workflow['forwarder_name']} (ID: {forwarder_id})")
    time.sleep(1)

    # Worker (RedisQueueProducer - writes to output queue)
    worker_id = client.create_producer_processor(
        process_group_id=pg_id,
        queue_name=workflow["output_queue"],
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        push_operation="LPUSH"
    )
    client.update_processor_name(worker_id, workflow["worker_name"])
    logger.info(f"  ✅ {workflow['worker_name']} (ID: {worker_id})")
    time.sleep(1)

    # Error logger
    logger_name = f"{workflow['name']} Error Logger"
    logger_id = client.create_log_processor(
        process_group_id=pg_id,
        name=logger_name,
        log_level="error",
        log_prefix=f"[{workflow['name'].upper()} FAILURE]",
        log_message="${queue.error.message}"
    )
    logger.info(f"  ✅ {logger_name} (ID: {logger_id})")
    time.sleep(1)

    # Connections: forwarder → worker (success), worker → logger (failure)
    client.create_connection(pg_id, forwarder_id, worker_id, relationships=["success"])
    client.create_connection(pg_id, worker_id, logger_id, relationships=["failure"])


def create_rag_pipeline_flow():
    """Create the RAG Pipeline flow with separate process groups per component."""
    logger.info("Creating RAG Pipeline flow")

    try:
        client = NifiClient(
            base_url=os.getenv("NIFI_ENDPOINT"),
            username=os.getenv("NIFI_USERNAME"),
            password=os.getenv("NIFI_PASSWORD"),
            ssl_verify=os.getenv("NIFI_SSL_VERIFY", "false").lower() == "true"
        )

        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))

        # Top-level process group
        root_pg_id = client.create_process_group("RAG Pipeline")
        logger.info(f"✅ Created root process group 'RAG Pipeline' (ID: {root_pg_id})")

        # ── 1. OCR Processing ──────────────────────────────────────────
        ocr_pg_id = client.create_process_group("OCR Processing", parent_pg_id=root_pg_id)
        logger.info(f"📁 Created 'OCR Processing' group (ID: {ocr_pg_id})")
        for wf in [
            {"name": "OCR", "input_queue": "ocr_processing_job_input", "forwarder_name": "OCR Forwarder", "worker_name": "OCR Worker", "output_queue": "ocr_processing_job_output"},
            {"name": "OCR Reply", "input_queue": "ocr_reply_input", "forwarder_name": "OCR Reply Forwarder", "worker_name": "OCR Reply Worker", "output_queue": "ocr_reply_output"},
        ]:
            _create_queue_pair(client, ocr_pg_id, wf, redis_host, redis_port, redis_db)

        # ── 2. Whisper Processing ──────────────────────────────────────
        whisper_pg_id = client.create_process_group("Whisper Processing", parent_pg_id=root_pg_id)
        logger.info(f"📁 Created 'Whisper Processing' group (ID: {whisper_pg_id})")
        for wf in [
            {"name": "Whisper", "input_queue": "whisper_processing_job_input", "forwarder_name": "Whisper Forwarder", "worker_name": "Whisper Worker", "output_queue": "whisper_processing_job_output"},
            {"name": "Whisper Reply", "input_queue": "whisper_reply_input", "forwarder_name": "Whisper Reply Forwarder", "worker_name": "Whisper Reply Worker", "output_queue": "whisper_reply_output"},
        ]:
            _create_queue_pair(client, whisper_pg_id, wf, redis_host, redis_port, redis_db)

        # ── 3. Retype to Markdown LLM (supervisor LLM normalization) ────
        retype_pg_id = client.create_process_group("Retype to Markdown LLM", parent_pg_id=root_pg_id)
        logger.info(f"📁 Created 'Retype to Markdown LLM' group (ID: {retype_pg_id})")
        for wf in [
            {"name": "Retype LLM", "input_queue": "retype_llm_job_input", "forwarder_name": "Retype LLM Forwarder", "worker_name": "Retype LLM Worker", "output_queue": "retype_llm_job_output"},
            {"name": "Retype LLM Reply", "input_queue": "retype_llm_reply_input", "forwarder_name": "Retype LLM Reply Forwarder", "worker_name": "Retype LLM Reply Worker", "output_queue": "retype_llm_reply_output"},
        ]:
            _create_queue_pair(client, retype_pg_id, wf, redis_host, redis_port, redis_db)

        # ── 4. Chunk and Tokenize Consumer Forwarder (one-way to consumer) ──
        fwd_pg_id = client.create_process_group("Chunk and Tokenize Consumer", parent_pg_id=root_pg_id)
        logger.info(f"📁 Created 'Chunk and Tokenize Consumer' group (ID: {fwd_pg_id})")
        queue_names_str = os.getenv("QUEUE_NAMES", "chunk_ingest_queue:0")
        for queue_name in queue_names_str.split(","):
            queue_name = queue_name.strip()
            if not queue_name:
                continue
            wf = {
                "name": f"Chunk Forwarder ({queue_name})",
                "input_queue": f"{queue_name}_input",
                "forwarder_name": f"Chunk Forwarder ({queue_name})",
                "worker_name": f"Chunk Worker ({queue_name})",
                "output_queue": f"{queue_name}_output",
            }
            _create_queue_pair(client, fwd_pg_id, wf, redis_host, redis_port, redis_db)

        logger.info(f"✅ RAG Pipeline flow created successfully (ID: {root_pg_id})")
        return root_pg_id

    except Exception as e:
        logger.error(f"❌ Failed to create RAG Pipeline flow: {e}")
        raise


def update_processor_properties(pg_id):
    """Update processor properties from environment variables using NiPyAPI."""
    logger.info("Updating processor properties from environment variables")
    
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_db = os.getenv("REDIS_DB", "0")
        
        logger.info(f"Redis config: host={redis_host}, port={redis_port}, db={redis_db}")
        
        # Stop all processors first
        logger.info("Stopping processors to update properties...")
        canvas.schedule_process_group(pg_id, scheduled=False)
        time.sleep(2)
        
        # Get all processors (includes nested sub-groups since we deleted everything else)
        processors = canvas.list_all_processors()
        
        for proc in processors:
            proc_name = proc.component.name
            proc_type = proc.component.type
            
            # Only update Redis processors
            if 'RedisQueue' in proc_type:
                logger.info(f"Updating {proc_name} ({proc_type})")
                
                # Build updated config based on processor type
                if 'Consumer' in proc_type:
                    updated_config = nipyapi.nifi.ProcessorConfigDTO(
                        properties={
                            "Redis Host": redis_host,
                            "Redis Port": redis_port,
                            "Redis DB Index": redis_db
                        }
                    )
                else:
                    updated_config = nipyapi.nifi.ProcessorConfigDTO(
                        properties={
                            "Redis Host": redis_host,
                            "Redis Port": redis_port,
                            "Redis DB": redis_db
                        }
                    )
                
                # Update processor
                canvas.update_processor(proc, updated_config)
                logger.info(f"✅ Updated {proc_name}")
        
        logger.info("✅ All processor properties updated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update processor properties: {e}")
        return False


def start_flow(pg_id):
    """Start the process group and all child processors using NiPyAPI canvas operations."""
    logger.info(f"Starting process group (ID: {pg_id})")

    try:
        # First, stop all processors to ensure clean state
        logger.info("Stopping all processors first...")
        canvas.schedule_process_group(pg_id, scheduled=False)
        logger.info("✅ All processors stopped")
        
        # Wait a moment for processors to fully stop
        time.sleep(2)
        
        # Now start the entire process group using REST API
        # This sets the process group state to RUNNING with recursive=True
        import requests
        nifi_endpoint = os.getenv("NIFI_ENDPOINT")
        nifi_username = os.getenv("NIFI_USERNAME")
        nifi_password = os.getenv("NIFI_PASSWORD")
        ssl_verify = os.getenv("NIFI_SSL_VERIFY", "false").lower() == "true"
        
        # Get access token
        token_response = requests.post(
            f"{nifi_endpoint}/access/token",
            data={"username": nifi_username, "password": nifi_password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            verify=ssl_verify
        )
        token = token_response.text
        
        # Start the process group
        url = f"{nifi_endpoint}/flow/process-groups/{pg_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "id": pg_id,
            "state": "RUNNING"
        }
        
        response = requests.put(url, json=payload, headers=headers, verify=ssl_verify)
        
        if response.status_code == 200:
            logger.info("✅ Process group started successfully")
            return True
        else:
            logger.error(f"❌ Failed to start process group: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to start process group: {e}")
        return False


def verify_flow_health(pg_id):
    """Verify the process group health using NiPyAPI canvas operations."""
    logger.info(f"Verifying process group health (ID: {pg_id})")

    try:
        # Get the process group by ID
        pg = canvas.get_process_group(pg_id, identifier_type='id')
        if not pg:
            logger.error("❌ Process group not found")
            return False

        # Get all processors (includes nested sub-groups)
        processors = canvas.list_all_processors()
        
        total_count = len(processors)
        running_count = sum(1 for p in processors if p.status.run_status == 'Running')
        stopped_count = sum(1 for p in processors if p.status.run_status == 'Stopped')
        invalid_count = sum(1 for p in processors if p.status.run_status == 'Invalid')

        logger.info("Process group status:")
        logger.info(f"  - Total processors: {total_count}")
        logger.info(f"  - Running processors: {running_count}")
        logger.info(f"  - Stopped processors: {stopped_count}")
        logger.info(f"  - Invalid processors: {invalid_count}")

        if invalid_count > 0:
            logger.error("❌ Process group has invalid processors")
            return False

        if stopped_count > 0:
            logger.warning("⚠️  Process group has stopped processors")

        if running_count == total_count:
            logger.info("✅ All processors are running")
            return True
        else:
            logger.warning("⚠️  Not all processors are running")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to verify process group health: {e}")
        return False


def main():
    """Main bootstrap function."""
    logger.info("=" * 60)
    logger.info("NiFi Bootstrap Service Starting")
    logger.info("=" * 60)

    # Step 1: Flush Redis queues
    logger.info("--- Step 1: Flushing Redis Queues ---")
    flush_redis()

    # Step 2: Deploy processors and restart NiFi
    logger.info("--- Step 2: Deploying NiFi Processors ---")
    deploy_processors()

    # Wait for NiFi to be available (uses requests directly, no NiPyAPI needed)
    if not wait_for_nifi():
        sys.exit(1)

    # Configure NiFi connection (NiPyAPI setup, requires NiFi to be up)
    configure_nifi_connection()

    # Always delete and recreate to ensure flow matches current code
    existing_pg_id = check_flow_exists()
    if existing_pg_id:
        logger.info("Existing flow found, redeploying with current configuration...")

    # Delete all existing process groups for clean deployment
    if not delete_all_process_groups():
        logger.error(" Failed to delete existing process groups")
        sys.exit(1)

    # Create new flow
    try:
        pg_id = create_rag_pipeline_flow()
    except Exception as e:
        logger.error(f"❌ Failed to create flow: {e}")
        sys.exit(1)

    # Update processor properties from environment variables
    if not update_processor_properties(pg_id):
        logger.error("❌ Failed to update processor properties")
        sys.exit(1)

    # Start the flow
    if not start_flow(pg_id):
        logger.error("❌ Failed to start flow")
        sys.exit(1)

    # Wait for processors to start and become valid (Python code loading can be slow)
    for attempt in range(1, 21):
        logger.info(f"Checking health (attempt {attempt}/20)...")
        time.sleep(3)
        if verify_flow_health(pg_id):
            logger.info("✅ Flow deployed, started, and healthy")
            logger.info("=" * 60)
            logger.info("NiFi Bootstrap Service Complete")
            logger.info("=" * 60)
            sys.exit(0)

    logger.error("❌ Flow deployed but failed health check after 60s")
    sys.exit(1)


if __name__ == "__main__":
    main()
