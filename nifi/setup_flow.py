#!/usr/bin/env python3
"""
NiFi Flow Setup Script

Creates the RAG pipeline flow in NiFi using NiPyAPI.
Sets up RedisQueueConsumer and RedisQueueProducer processors for each queue.
"""
import logging
import os
import sys

from nifi_client import NifiClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_queue_names():
    """Get list of queue names from environment variables."""
    queue_names = []

    ocr_queue = os.getenv("REDIS_OCR_JOB_QUEUE", "ocr_processing_job")
    queue_names.append(ocr_queue)

    whisper_queue = os.getenv("REDIS_WHISPER_JOB_QUEUE", "whisper_processing_job")
    queue_names.append(whisper_queue)

    chunk_queues = os.getenv("QUEUE_NAMES", "chunk_ingest_queue:0,chunk_ingest_queue:1")
    for queue in chunk_queues.split(","):
        queue = queue.strip()
        if queue:
            queue_names.append(queue)

    return queue_names


def get_redis_config():
    """Get Redis connection configuration from environment."""
    return {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
    }


def setup_flow():
    """Set up the NiFi flow with consumer/producer pairs for each queue."""
    nifi_endpoint = os.getenv("NIFI_ENDPOINT")
    if not nifi_endpoint:
        logger.error("NIFI_ENDPOINT environment variable is required")
        sys.exit(1)

    nifi_username = os.getenv("NIFI_USERNAME")
    if not nifi_username:
        logger.error("NIFI_USERNAME environment variable is required")
        sys.exit(1)

    nifi_password = os.getenv("NIFI_PASSWORD")
    if not nifi_password:
        logger.error("NIFI_PASSWORD environment variable is required")
        sys.exit(1)

    nifi_ssl_verify = os.getenv("NIFI_SSL_VERIFY", "false").lower() == "true"
    registry_endpoint = os.getenv("REGISTRY_ENDPOINT")

    logger.info("Initializing NiFi client")
    client = NifiClient(
        base_url=nifi_endpoint,
        username=nifi_username,
        password=nifi_password,
        ssl_verify=nifi_ssl_verify,
        registry_url=registry_endpoint,
    )

    logger.info("Waiting for NiFi to become available")
    if not client.wait_for_nifi(max_retries=5):
        logger.error("NiFi is not available after retries")
        sys.exit(1)

    process_group_name = "RAG Pipeline"
    logger.info(f"Checking if process group '{process_group_name}' exists")
    pg_id = client.check_flow_exists(process_group_name)

    if pg_id:
        logger.info(f"Process group '{process_group_name}' already exists (ID: {pg_id})")
        logger.info("Skipping creation - flow already configured")
        sys.exit(0)

    logger.info(f"Creating process group '{process_group_name}'")
    pg_id = client.create_process_group(process_group_name)
    logger.info(f"Created process group with ID: {pg_id}")

    queue_names = get_queue_names()
    redis_config = get_redis_config()

    logger.info(f"Setting up {len(queue_names)} queue(s)")

    for queue_name in queue_names:
        logger.info(f"Processing queue: {queue_name}")

        input_queue = f"{queue_name}_input"
        output_queue = f"{queue_name}_output"

        logger.info(f"Creating consumer for {input_queue}")
        consumer_id = client.create_consumer_processor(
            process_group_id=pg_id,
            queue_name=input_queue,
            redis_host=redis_config["host"],
            redis_port=redis_config["port"],
            redis_db=redis_config["db"],
            pop_operation="BRPOP",
            timeout_seconds=5,
        )
        logger.info(f"Created consumer processor (ID: {consumer_id})")

        logger.info(f"Creating producer for {output_queue}")
        producer_id = client.create_producer_processor(
            process_group_id=pg_id,
            queue_name=output_queue,
            redis_host=redis_config["host"],
            redis_port=redis_config["port"],
            redis_db=redis_config["db"],
            push_operation="LPUSH",
        )
        logger.info(f"Created producer processor (ID: {producer_id})")

        logger.info("Creating connection from consumer to producer")
        connection_id = client.create_connection(
            process_group_id=pg_id,
            source_id=consumer_id,
            destination_id=producer_id,
            relationships=["success"],
        )
        logger.info(f"Created connection (ID: {connection_id})")

    logger.info("Starting process group")
    if not client.start_process_group(pg_id):
        logger.error("Failed to start process group")
        sys.exit(1)

    logger.info("Verifying flow health")
    if not client.verify_flow_health(pg_id):
        logger.error("Flow health check failed - some processors are not running")
        sys.exit(1)

    logger.info("Flow setup completed successfully")
    logger.info(f"Process group ID: {pg_id}")
    logger.info(f"Queues configured: {', '.join(queue_names)}")


if __name__ == "__main__":
    try:
        setup_flow()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Flow setup failed: {e}", exc_info=True)
        sys.exit(1)
