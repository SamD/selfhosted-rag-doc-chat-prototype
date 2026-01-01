import os

from config.settings import (
    LLAMA_CHAT_FORMAT,
    LLAMA_F16_KV,
    LLAMA_MAX_TOKENS,
    LLAMA_N_BATCH,
    LLAMA_N_CTX,
    LLAMA_N_GPU_LAYERS,
    LLAMA_N_THREADS,
    LLAMA_REPEAT_PENALTY,
    LLAMA_SEED,
    LLAMA_TEMPERATURE,
    LLAMA_TOP_K,
    LLAMA_TOP_P,
    LLAMA_VERBOSE,
    LLM_PATH,
)


class LlamaParamStrategy:
    def __init__(self):
        pass  # No retriever/LLM injected yet

    def get_params(self):
        params = {
            "model_path": LLM_PATH,
            "n_ctx": LLAMA_N_CTX,
            "n_gpu_layers": LLAMA_N_GPU_LAYERS,
            "n_threads": LLAMA_N_THREADS,
            "n_batch": LLAMA_N_BATCH,
            "f16_kv": LLAMA_F16_KV,
            "temperature": LLAMA_TEMPERATURE,
            "top_k": LLAMA_TOP_K,
            "top_p": LLAMA_TOP_P,
            "repeat_penalty": LLAMA_REPEAT_PENALTY,
            "max_tokens": LLAMA_MAX_TOKENS,
            "chat_format": LLAMA_CHAT_FORMAT,
            "verbose": LLAMA_VERBOSE,
            "seed": LLAMA_SEED,
        }
        if os.getenv("LLAMA_USE_GPU", "false").lower() == "false":
            del params["n_gpu_layers"]

        return params
