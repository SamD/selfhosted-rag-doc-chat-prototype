import os
import sys
import time
import json
from pathlib import Path

# Setup Environment
MODEL_PATH = "/home/samueldoyle/AI_LOCAL/Models/Qwen-3/Qwen3-8B-Q6_K.gguf"
PROJECT_ROOT = os.getcwd()
INGEST_DIR = os.path.join(PROJECT_ROOT, "Docs") # Output in same dir as source
os.makedirs(INGEST_DIR, exist_ok=True)

os.environ["INGEST_FOLDER"] = INGEST_DIR
os.environ["EMBEDDING_MODEL_PATH"] = INGEST_DIR # Dummy
os.environ["LLM_PATH"] = MODEL_PATH
os.environ["SUPERVISOR_LLM_PATH"] = MODEL_PATH
os.environ["LLAMA_N_CTX"] = "16384" # Boosted for large chunks

# Redirect stdout to progress log
log_file = open("normalization_progress.log", "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

# Add doc-ingest-chat to sys.path
sys.path.append(os.path.join(PROJECT_ROOT, "doc-ingest-chat"))

from workers.gatekeeper_logic import gatekeeper_extract_and_normalize

def verify_markdown(md_path):
    print(f"\n🔍 [Verification] Checking {md_path}...")
    if not os.path.exists(md_path):
        print("❌ Error: Output file not found.")
        return False
    
    size = os.path.getsize(md_path)
    print(f"📊 Final Size: {size / 1024 / 1024:.2f} MB")
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Check Metadata
    if content.startswith("---") and "ID:" in content and "Slug:" in content:
        print("✅ Metadata Block Present.")
    else:
        print("❌ Metadata Block Missing or Corrupt.")
        return False
    
    # 2. Check for Chunk Markers
    chunk_markers = content.count("--- CHUNK")
    print(f"✅ Found {chunk_markers} chunk transitions.")
    
    # 3. Check for structural consistency
    if "# " in content:
        print("✅ Document Title found.")
    
    return True

def run_full_test():
    target_file = "Docs/outline_of_history_pt1.pdf"
    if not os.path.exists(target_file):
        print(f"❌ Error: {target_file} not found.")
        return

    print(f"🚀 Starting FULL Document Normalization: {target_file}")
    print(f"🤖 Using Model: {MODEL_PATH}")
    print(f"⏱️ Start Time: {time.ctime()}\n")

    start_t = time.time()
    
    try:
        success = gatekeeper_extract_and_normalize(target_file)
        
        duration = time.time() - start_t
        if success:
            print(f"\n✅ FULL Normalization Complete in {duration/3600:.2f} hours!")
            # Filename is slugified. outline_of_history_pt1 -> outline-of-history-pt1-XXXX.md
            # We look for files matching the pattern
            md_files = list(Path(INGEST_DIR).glob("outline-of-history-pt1-*.md"))
            if md_files:
                # Get most recent
                final_md = sorted(md_files, key=os.path.getmtime)[-1]
                if verify_markdown(str(final_md)):
                    print(f"🏁 Final Result: SUCCESS -> {final_md}")
                else:
                    print("🏁 Final Result: PARTIAL (Verification Failed)")
            else:
                print("❌ Error: Could not find output Markdown file.")
        else:
            print(f"❌ Normalization Failed after {duration/60:.2f} minutes.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n🏁 End Time: {time.ctime()}")
        log_file.close()

if __name__ == "__main__":
    run_full_test()
