import json
import os
import sys
import logging
from pathlib import Path

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("direct_integration")

# Add project paths
sys.path.insert(0, os.path.join(os.getcwd(), "doc-ingest-chat"))

from services.redis_service import get_redis_client
from workers.ocr_graph import run_ocr_graph
from workers.gatekeeper_logic import gatekeeper_extract_and_normalize, get_slug

def process_one_ocr_job():
    redis_client = get_redis_client()
    job_raw = redis_client.brpop("ocr_processing_job", timeout=2)
    if not job_raw:
        log.info("No OCR jobs in queue.")
        return False
    
    _, job_data = job_raw
    job = json.loads(job_data)
    log.info(f"⚙️ Manually processing OCR job for {job.get('rel_path')} page {job.get('page_num')}...")
    
    success = run_ocr_graph(job)
    log.info(f"✅ OCR Graph Success: {success}")
    return True

def run_integration():
    # 1. Process the pending OCR jobs first
    # (Since gatekeeper_extract_and_normalize enqueues and then waits,
    # we need to run this in a way that doesn't deadlock. 
    # Actually, gatekeeper_extract_and_normalize is what enqueues.
    # So we'll run the gatekeeper in a thread or just let it time out and retry.)
    
    # Better: Use the small scanned file
    file_path = "Docs/staging/scanned_test.pdf"
    if not os.path.exists(file_path):
        log.error(f"File not found: {file_path}")
        return

    p = Path(file_path)
    slug = get_slug(p.stem)
    metadata = {
        "id": "test-direct-uuid",
        "slug": slug,
        "source_file": file_path,
        "type": "pdf"
    }

    log.info(f"🚀 Starting Direct Integration for {file_path}")
    
    # We'll use a hack: start the OCR dispatcher in a background thread
    import threading
    def dispatcher_thread():
        log.info("🛰️ OCR Dispatcher Thread started.")
        redis_client = get_redis_client()
        processed = 0
        while processed < 5: # Limit to 5 jobs for test
            res = redis_client.brpop("ocr_processing_job", timeout=5)
            if res:
                _, data = res
                job = json.loads(data)
                log.info(f"🧠 Thread processing OCR job: {job['rel_path']} P{job['page_num']}")
                run_ocr_graph(job)
                processed += 1
        log.info("🛰️ OCR Dispatcher Thread exiting.")

    t = threading.Thread(target=dispatcher_thread, daemon=True)
    t.start()

    # Now run the gatekeeper logic which will enqueue and then WAIT for the thread to finish the job
    success = gatekeeper_extract_and_normalize(file_path, metadata)
    
    log.info(f"🏁 Integration Result: {success}")
    
    # Show the output file content
    output_files = list(Path("Docs").glob("scanned_test-*.md"))
    if output_files:
        out_p = output_files[0]
        log.info(f"📄 Output File: {out_p} ({out_p.stat().st_size} bytes)")
        with open(out_p, "r") as f:
            print("\n--- BEGIN OUTPUT ---")
            print(f.read())
            print("--- END OUTPUT ---\n")
    else:
        log.error("❌ No output file generated!")

if __name__ == "__main__":
    run_integration()
