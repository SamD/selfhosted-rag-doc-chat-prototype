import logging
import os
import sys

# Set these BEFORE any other imports to ensure settings picks them up correctly
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["SKIP_LOAD_DOTENV"] = "true"

import time

# Setup Environment
MODEL_PATH = "/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf"
PROJECT_ROOT = os.getcwd()
INGEST_DIR = os.path.join(PROJECT_ROOT, "Docs")
EMBEDDING_MODEL_PATH = "/home/samueldoyle/AI_LOCAL/e5-large-v2"

os.makedirs(INGEST_DIR, exist_ok=True)

os.environ["INGEST_FOLDER"] = INGEST_DIR
os.environ["EMBEDDING_MODEL_PATH"] = EMBEDDING_MODEL_PATH
os.environ["LLM_PATH"] = MODEL_PATH
os.environ["SUPERVISOR_LLM_PATH"] = MODEL_PATH
os.environ["LLAMA_N_CTX"] = "16384"

# Proper Logging Setup
log_path = os.path.join(PROJECT_ROOT, "full_document_normalization.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("full_normalization_runner")

# Add doc-ingest-chat to sys.path
sys.path.append(os.path.join(PROJECT_ROOT, "doc-ingest-chat"))

from workers.gatekeeper_logic import gatekeeper_extract_and_normalize  # noqa: E402


def verify_markdown(md_path):
    log.info(f"🔍 [Verification] Checking {md_path}...")
    if not os.path.exists(md_path):
        log.error("❌ Error: Output file not found.")
        return False

    size = os.path.getsize(md_path)
    log.info(f"📊 Final Size: {size / 1024 / 1024:.2f} MB")

    with open(md_path, "r", encoding="utf-8") as f:
        content_head = f.read(10000)

    if content_head.startswith("---") and "ID:" in content_head and "Slug:" in content_head:
        log.info("✅ Metadata Block Present.")
    else:
        log.error("❌ Metadata Block Missing or Corrupt.")
        return False

    if "# " in content_head:
        log.info("✅ Document Title found.")

    return True


def run_production():
    target_file = "Docs/outline_of_history_pt1.pdf"
    if not os.path.exists(target_file):
        log.error(f"❌ Error: {target_file} not found.")
        return

    log.info(f"🚀 Starting FULL Document Normalization: {target_file}")
    log.info(f"🤖 Using Model: {MODEL_PATH}")
    log.info(f"⏱️ Start Time: {time.ctime()}\n")

    start_t = time.time()

    try:
        from workers.gatekeeper_logic import get_slug; from pathlib import Path; slug = get_slug(Path(target_file).stem); md_path = os.path.join(INGEST_DIR, f"{slug}.md"); success, _ = gatekeeper_extract_and_normalize("full-norm-job", target_file, md_path)

        duration = time.time() - start_t
        if success:
            log.info(f"✅ FULL Normalization Complete in {duration / 3600:.2f} hours!")
            import glob

            md_files = glob.glob(os.path.join(INGEST_DIR, "outline-of-history-pt1-*.md"))
            if md_files:
                final_md = sorted(md_files, key=os.path.getmtime)[-1]
                if verify_markdown(str(final_md)):
                    log.info(f"🏁 Final Result: SUCCESS -> {final_md}")
                else:
                    log.error("🏁 Final Result: PARTIAL (Verification Failed)")
            else:
                log.error("❌ Error: Could not find output Markdown file.")
        else:
            log.error(f"❌ Normalization Failed after {duration / 60:.2f} minutes.")

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR: {e}", exc_info=True)
    finally:
        log.info(f"🏁 End Time: {time.ctime()}")


if __name__ == "__main__":
    run_production()
