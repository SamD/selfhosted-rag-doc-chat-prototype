import os
import subprocess
import sys
import time

# 1. Configuration (EXACT PATHS FROM DISK)
PROJECT_ROOT = os.getcwd()
PYTHON_EXE = sys.executable  # Path to the venv python
MODEL_PATH_PHI = "/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf"
INGEST_DIR = os.path.join(PROJECT_ROOT, "Docs")
EMBEDDING_MODEL_PATH = "/home/samueldoyle/AI_LOCAL/e5-large-v2"

# Environment variables common to all processes
ENV = os.environ.copy()
ENV["INGEST_FOLDER"] = INGEST_DIR
ENV["EMBEDDING_MODEL_PATH"] = EMBEDDING_MODEL_PATH
ENV["LLM_PATH"] = MODEL_PATH_PHI
ENV["SUPERVISOR_LLM_PATH"] = MODEL_PATH_PHI
ENV["LLAMA_N_CTX"] = "16384"
ENV["REDIS_HOST"] = "localhost"
ENV["REDIS_PORT"] = "6379"
ENV["SKIP_LOAD_DOTENV"] = "true"
ENV["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "doc-ingest-chat")


def main():
    print("🚀 Starting unified ingestion environment...")
    print(f"📁 INGEST_FOLDER: {INGEST_DIR}")
    print(f"🧬 EMBEDDING_MODEL_PATH: {EMBEDDING_MODEL_PATH}")
    print(f"🐍 Python: {PYTHON_EXE}")

    # 2. Start OCR Worker in background
    print("🛰️ Launching OCR Worker...")
    stdout_log = open("ocr_worker_stdout.log", "w", buffering=1)
    stderr_log = open("ocr_worker_stderr.log", "w", buffering=1)

    ocr_process = subprocess.Popen([PYTHON_EXE, "doc-ingest-chat/run_ocr_worker.py"], env=ENV, stdout=stdout_log, stderr=stderr_log)

    # Give it a moment to start
    time.sleep(10)
    if ocr_process.poll() is not None:
        print("❌ OCR Worker failed to start. Check ocr_worker_stderr.log")
        return
    else:
        print(f"✅ OCR Worker running (PID: {ocr_process.pid})")

    # 3. Start Normalization Script
    print("📖 Starting Full Normalization...")
    try:
        subprocess.run([PYTHON_EXE, "run_full_normalization.py"], env=ENV, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Normalization failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        print("🧹 Cleaning up processes...")
        ocr_process.terminate()
        try:
            ocr_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ocr_process.kill()
        stdout_log.close()
        stderr_log.close()


if __name__ == "__main__":
    main()
