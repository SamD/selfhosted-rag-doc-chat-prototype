#!/usr/bin/env python3
"""
Retype to Markdown LLM Worker.
Reads raw text from retype_llm_job_output, calls supervisor LLM for normalization,
writes clean markdown to retype_llm_reply_input.
"""

import json
import multiprocessing
import os
import signal
import sys
import traceback
from multiprocessing import Manager, Pool

from config import settings
from services.redis_service import get_redis_client
from utils.logging_config import setup_pdf_logging
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest.retype_llm_worker")

RETYPE_JOB_QUEUE = "retype_llm_job"
RETYPE_REPLY_QUEUE = "retype_llm_reply"

# Same prompt as gatekeeper_logic.py process_chunk()
MARKDOWN_FORMATTER_PROMPT = (
    "You are a Markdown formatter. "
    "Convert ONLY the text contained between START_OF_RAW_TEXT and END_OF_RAW_TEXT into valid Markdown. "
    "Preserve all content and structure as much as possible. "
    "Use headings, bullet points, numbered lists, code blocks, and tables only when they fit the input. "
    "DO NOT summarize, infer, or add information. "
    "Return only Markdown, with no preface or explanation. "
    "Remove only characters that are clearly non-linguistic artifacts such as repeated symbols, "
    "encoding errors, or isolated non-alphanumeric strings. "
    "Stop immediately when you reach END_OF_RAW_TEXT.\n\n"
)


def call_supervisor_llm(raw_content: str, idx: int = 0) -> str:
    """Call the supervisor LLM to normalize raw text into clean markdown."""
    from utils.llm_setup import get_supervisor_llm
    from utils.text_utils import get_tokenizer

    llm = get_supervisor_llm()
    tokenizer = get_tokenizer()

    user_msg = f"{MARKDOWN_FORMATTER_PROMPT}START_OF_RAW_TEXT\n{raw_content}\nEND_OF_RAW_TEXT"

    # Enforce context limit
    context_limit = int(settings.SUPERVISOR_N_CTX * 0.8)
    encoded_prompt = tokenizer.encode(user_msg, add_special_tokens=True)
    if len(encoded_prompt) > context_limit:
        log.warning(f"⚠️ Batch {idx} too large ({len(encoded_prompt)} tokens), truncating to {context_limit}")
        truncated_tokens = tokenizer.encode(raw_content, add_special_tokens=False)[:context_limit - 200]
        raw_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        user_msg = f"{MARKDOWN_FORMATTER_PROMPT}START_OF_RAW_TEXT\n{raw_content}\nEND_OF_RAW_TEXT"

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": user_msg}],
        stop=["END_OF_RAW_TEXT"],
    )

    content = response["choices"][0]["message"]["content"] or ""

    # Some reasoning models (Qwen3, etc.) put output in reasoning_content
    if not content.strip():
        reasoning_content = response["choices"][0]["message"].get("reasoning_content") or ""
        if reasoning_content.strip():
            log.info(f"🧠 Batch {idx}: Using reasoning_content ({len(reasoning_content)} chars)")
            content = reasoning_content

    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    usage = response.get("usage", {})
    token_info = f"in={usage.get('prompt_tokens', '?')} out={usage.get('completion_tokens', '?')} reason={finish_reason}"

    stripped = content.strip()
    if not stripped:
        log.error(f"🈳 Batch {idx}: LLM returned empty ({token_info}), falling back to raw text")
        content = raw_content
    elif stripped.isspace():
        log.warning(f"⚠️ Batch {idx}: LLM returned whitespace ({token_info}), falling back to raw text")
        content = raw_content
    else:
        log.info(f"🤖 Batch {idx}: LLM returned {len(content)} chars ({token_info})")

    return content


def worker_task(job):
    """Process a single retype LLM job."""
    redis_client = get_redis_client()

    raw_content = job.get("raw_content", "")
    idx = job.get("idx", 0)
    trace_id = job.get("trace_id")
    job_id = job.get("job_id", "")

    if trace_id:
        set_trace_id(trace_id)

    if not raw_content:
        log.error(f"💥 Empty raw_content in job: {list(job.keys())}")
        return

    try:
        normalized = call_supervisor_llm(raw_content, idx)

        reply = {
            "job_id": job_id,
            "idx": idx,
            "normalized_content": normalized,
            "trace_id": trace_id,
        }
        redis_client.lpush(f"{RETYPE_REPLY_QUEUE}_input", json.dumps(reply))
        log.info(f"📤 Sent reply for batch {idx} ({len(normalized)} chars)")

    except Exception as e:
        log.error(f"💥 Retype LLM failed for batch {idx}: {e}")
        log.error(traceback.format_exc())
        # Send error reply so gatekeeper doesn't hang
        reply = {
            "job_id": job_id,
            "idx": idx,
            "normalized_content": raw_content,  # fallback to raw
            "trace_id": trace_id,
            "error": str(e),
        }
        redis_client.lpush(f"{RETYPE_REPLY_QUEUE}_input", json.dumps(reply))


def init_worker(lock):
    global queue_lock
    queue_lock = lock


SHUTDOWN = multiprocessing.Event()


def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating retype LLM shutdown...")
    SHUTDOWN.set()


def dispatcher(p, shared_state):
    """Main loop that pops jobs from retype_llm_job_output and dispatches to pool."""
    redis_client = get_redis_client()
    log.info(f"🛰️ Retype LLM Dispatcher listening on {RETYPE_JOB_QUEUE}_output...")

    while not SHUTDOWN.is_set():
        try:
            res = redis_client.brpop(f"{RETYPE_JOB_QUEUE}_output", timeout=5)
            if res:
                _, job_raw = res
                job = json.loads(job_raw)
                p.apply_async(worker_task, args=(job,))
        except json.JSONDecodeError:
            log.error(f"💥 Malformed Job: {job_raw}")
        except Exception as e:
            log.error(f"💥 Dispatcher error: {e}")


def main():
    """Main entry point."""
    print(f"🧠 Retype LLM Worker Debug: SUPERVISOR_LLM_ENDPOINTS={os.getenv('SUPERVISOR_LLM_ENDPOINTS')}")
    print(f"🧠 Retype LLM Worker Debug: Python={sys.executable}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    setup_pdf_logging()

    # Pre-load supervisor LLM
    from utils.llm_setup import get_supervisor_llm

    log.info("🏗️ Pre-loading supervisor LLM in parent process...")
    get_supervisor_llm()

    with Manager() as manager:
        shared_state = manager.dict()
        lock = manager.Lock()

        try:
            num_workers = 2
            log.info(f"🚀 Spawning {num_workers} retype LLM workers (maxtasksperchild=1)")

            with Pool(processes=num_workers, initializer=init_worker, initargs=(lock,), maxtasksperchild=1) as pool:
                dispatcher(pool, shared_state)
        except Exception as e:
            log.error(f"💥 Retype LLM Worker encountered error: {e}")
        finally:
            log.info("✅ Retype LLM worker pool terminated cleanly")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()
