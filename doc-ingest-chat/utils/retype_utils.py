"""
Retype LLM utility — sends raw text to supervisor LLM via NiFi queues
and waits for normalized markdown reply.
"""

import json
import time
import uuid

from services.redis_service import get_redis_client
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest.retype_utils")

RETYPE_JOB_QUEUE = "retype_llm_job"
RETYPE_REPLY_QUEUE = "retype_llm_reply"


def send_raw_text_to_retype_llm(raw_content: str, idx: int = 0, trace_id: str = None) -> str:
    """Send raw text to the Retype to Markdown LLM worker via NiFi queues and wait for reply."""
    if trace_id:
        set_trace_id(trace_id)

    start_time = time.perf_counter()
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "idx": idx,
        "raw_content": raw_content,
        "trace_id": trace_id,
    }

    try:
        redis_client = get_redis_client()
        redis_client.lpush(f"{RETYPE_JOB_QUEUE}_input", json.dumps(job))
    except Exception as e:
        log.error(f"❌ Failed to submit retype LLM job to Redis: {e}")
        raise RuntimeError(f"Redis submission failed: {e}")

    # Wait for reply on retype_llm_reply_output, match by job_id
    reply_queue = f"{RETYPE_REPLY_QUEUE}_output"
    result = None
    wait_timeout = 600
    start_wait = time.time()

    while (time.time() - start_wait) < wait_timeout:
        result = redis_client.blpop(reply_queue, timeout=30)
        if result:
            _, data = result
            reply = json.loads(data)
            if reply.get("job_id") == job_id:
                result = reply
                break
            else:
                # Not our reply, put it back for another caller
                redis_client.lpush(reply_queue, data)
                result = None
                continue

        elapsed = int(time.time() - start_wait)
        log.info(f"⏳ Waiting for Retype LLM... batch {idx} ({elapsed}s elapsed)")

    if not result:
        raise TimeoutError(f"Retype LLM timeout after {wait_timeout}s for batch {idx}")

    roundtrip_ms = (time.perf_counter() - start_time) * 1000.0
    normalized = result.get("normalized_content", raw_content)
    log.info(f"🧠 Retype LLM reply for batch {idx}: {len(normalized)} chars ({roundtrip_ms:.0f}ms)")

    return normalized
