import os
import sys
from pathlib import Path

# 1. Setup Environment
MODEL_PATH = "/home/samueldoyle/AI_LOCAL/Models/Qwen-3/Qwen3-8B-Q6_K.gguf"
PROJECT_ROOT = os.getcwd()
INGEST_DIR = os.path.join(PROJECT_ROOT, "test_normalization_output_parallel")
os.makedirs(INGEST_DIR, exist_ok=True)

os.environ["INGEST_FOLDER"] = INGEST_DIR
os.environ["EMBEDDING_MODEL_PATH"] = INGEST_DIR # Dummy
os.environ["LLM_PATH"] = MODEL_PATH
os.environ["SUPERVISOR_LLM_PATH"] = MODEL_PATH
os.environ["LLAMA_N_CTX"] = "8192"

# Add doc-ingest-chat to sys.path
sys.path.append(os.path.join(PROJECT_ROOT, "doc-ingest-chat"))

from workers.gatekeeper_logic import gatekeeper_extract_and_normalize

def run_test():
    target_file = "Docs/outline_of_history_pt1.pdf"
    if not os.path.exists(target_file):
        print(f"❌ Error: {target_file} not found.")
        return

    print(f"🚀 Starting Real-World PARALLEL Normalization Test on {target_file}")
    print(f"🤖 Using Model: {MODEL_PATH}")
    print(f"📂 Output Dir: {INGEST_DIR}\n")

    try:
        # We only want to process 1 tiny chunk for the test to be super fast.
        test_pdf_small = os.path.join(INGEST_DIR, "small_test.txt")
        with open(test_pdf_small, "w") as f:
            f.write("Chapter 1: The Beginning\n" + "Short test content for fast verification.") 

        print(f"📝 Created small test file: {test_pdf_small}")
        
        success = gatekeeper_extract_and_normalize(test_pdf_small)
        
        if success:
            print("\n✅ Parallel Normalization Success!")
            output_file = os.path.join(INGEST_DIR, "small-test.md")
            if os.path.exists(output_file):
                print(f"📄 Output file created: {output_file}")
                with open(output_file, "r") as f:
                    content = f.read()
                    print(f"📊 Output length: {len(content)} characters")
                    print("🔍 Preview (First 500 chars):")
                    print(content[:500])
            else:
                print("❌ Error: Output file not found.")
        else:
            print("❌ Normalization failed.")

    except Exception as e:
        print(f"\n❌ Error during normalization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
