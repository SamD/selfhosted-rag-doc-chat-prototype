import os
import sys

from config import settings
from llama_cpp import Llama


def test_load():
    print(f"🚀 Attempting to load model: {settings.SUPERVISOR_LLM_PATH}")
    try:
        llm = Llama(
            model_path=settings.SUPERVISOR_LLM_PATH,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            flash_attn=True,
            verbose=True
        )
        print("✅ Model loaded successfully!")
        res = llm("Q: What is 2+2? A: ", max_tokens=10)
        print(f"Result: {res}")
    except Exception as e:
        print(f"💥 Failed to load model: {e}")

if __name__ == "__main__":
    os.environ["GGML_CUDA_GRAPH_OPT"] = "1"
    sys.path.insert(0, os.path.join(os.getcwd(), "doc-ingest-chat"))
    test_load()
