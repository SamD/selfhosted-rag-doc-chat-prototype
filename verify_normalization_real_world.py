import os
import sys

# 1. Setup Environment
MODEL_PATH = "/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf"
PROJECT_ROOT = os.getcwd()
INGEST_DIR = os.path.join(PROJECT_ROOT, "test_normalization_output_parallel")
os.makedirs(INGEST_DIR, exist_ok=True)

os.environ["INGEST_FOLDER"] = INGEST_DIR
os.environ["EMBEDDING_MODEL_PATH"] = INGEST_DIR  # Dummy
os.environ["LLM_PATH"] = MODEL_PATH
os.environ["SUPERVISOR_LLM_PATH"] = MODEL_PATH
os.environ["LLAMA_N_CTX"] = "8192"

# Add doc-ingest-chat to sys.path
sys.path.append(os.path.join(PROJECT_ROOT, "doc-ingest-chat"))

from workers.gatekeeper_logic import gatekeeper_extract_and_normalize  # noqa: E402


def run_test():
    target_file = os.path.join(INGEST_DIR, "real_slice.txt")
    if not os.path.exists(target_file):
        print(f"❌ Error: {target_file} not found.")
        return

    print(f"🚀 Starting Real-World PARALLEL Normalization Test on {target_file}")
    print(f"🤖 Using Model: {MODEL_PATH}")
    print(f"📂 Output Dir: {INGEST_DIR}\n")

    try:
        success = gatekeeper_extract_and_normalize(target_file)

        if success:
            print("\n✅ Parallel Normalization Success!")
            import glob

            md_files = glob.glob(os.path.join(INGEST_DIR, "real-slice-*.md"))
            if md_files:
                output_file = md_files[0]
                print(f"📄 Output file created: {output_file}")
                with open(output_file, "r") as f:
                    content = f.read()
                    print(f"📊 Output length: {len(content)} characters")

                    if content.startswith("---") and "ID:" in content:
                        print("✅ Metadata block found.")

                    if "# " in content:
                        print("✅ H1 Title found.")

                    if "--- CHUNK 1 ---" in content:
                        print("✅ Found Chunk 1 transition!")
                        idx = content.find("--- CHUNK 1 ---")
                        print("🔍 Preview (Transition):")
                        print(content[idx : idx + 500])
                    else:
                        print("⚠️ Chunk 1 transition NOT found.")
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
