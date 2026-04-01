import os

from config.settings import LLAMA_CHAT_FORMAT, LLAMA_F16_KV, LLAMA_MAX_TOKENS, LLAMA_N_BATCH, LLAMA_N_CTX, LLAMA_N_GPU_LAYERS, LLAMA_N_THREADS, LLAMA_REPEAT_PENALTY, LLAMA_SEED, LLAMA_TEMPERATURE, LLAMA_TOP_K, LLAMA_TOP_P, LLAMA_VERBOSE, LLM_PATH


class LlamaParamStrategy:
    def __init__(self):
        pass

    def get_params(self):
        # Use os.getenv with the imported constant as the fallback/default.
        # This ensures we get the LIVE environment variable first.
        params = {
            "model_path": os.getenv("LLM_PATH", LLM_PATH),
            "n_ctx": int(os.getenv("LLAMA_N_CTX", LLAMA_N_CTX)),
            "n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", LLAMA_N_GPU_LAYERS)),
            "n_threads": int(os.getenv("LLAMA_N_THREADS", LLAMA_N_THREADS)),
            "n_batch": int(os.getenv("LLAMA_N_BATCH", LLAMA_N_BATCH)),
            "f16_kv": os.getenv("LLAMA_F16_KV", str(LLAMA_F16_KV)).lower() == "true",
            "temperature": float(os.getenv("LLAMA_TEMPERATURE", LLAMA_TEMPERATURE)),
            "top_k": int(os.getenv("LLAMA_TOP_K", LLAMA_TOP_K)),
            "top_p": float(os.getenv("LLAMA_TOP_P", LLAMA_TOP_P)),
            "repeat_penalty": float(os.getenv("LLAMA_REPEAT_PENALTY", LLAMA_REPEAT_PENALTY)),
            "max_tokens": int(os.getenv("LLAMA_MAX_TOKENS", LLAMA_MAX_TOKENS)),
            "chat_format": os.getenv("LLAMA_CHAT_FORMAT", LLAMA_CHAT_FORMAT),
            "verbose": os.getenv("LLAMA_VERBOSE", str(LLAMA_VERBOSE)).lower() == "true",
            "seed": int(os.getenv("LLAMA_SEED", LLAMA_SEED)),
            "use_mmap": False,  # Usually safer inside containers to avoid bus errors
        }

        # Force GPU layers to 0 if explicit disable is set
        if os.getenv("LLAMA_USE_GPU", "false").lower() == "false":
            params["n_gpu_layers"] = 0

        # Log exactly what is being sent to llama-cpp to catch the CLI's ghost values
        print(f"DEBUG [LlamaParamStrategy]: Path={params['model_path']}, GPU={params['n_gpu_layers']}, CTX={params['n_ctx']}")

        return params
