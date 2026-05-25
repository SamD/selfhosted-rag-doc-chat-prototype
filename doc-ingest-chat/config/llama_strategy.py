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

from shared.defaults import DEFAULT_LLAMA_USE_GPU
from shared.env_names import (
    ENV_LLAMA_CHAT_FORMAT,
    ENV_LLAMA_F16_KV,
    ENV_LLAMA_MAX_TOKENS,
    ENV_LLAMA_N_BATCH,
    ENV_LLAMA_N_CTX,
    ENV_LLAMA_N_GPU_LAYERS,
    ENV_LLAMA_N_THREADS,
    ENV_LLAMA_REPEAT_PENALTY,
    ENV_LLAMA_SEED,
    ENV_LLAMA_TEMPERATURE,
    ENV_LLAMA_TOP_K,
    ENV_LLAMA_TOP_P,
    ENV_LLAMA_USE_GPU,
    ENV_LLAMA_VERBOSE,
    ENV_LLM_PATH,
)


class LlamaParamStrategy:
    def __init__(self):
        pass

    def get_params(self):
        params = {
            "model_path": os.getenv(ENV_LLM_PATH, LLM_PATH),
            "n_ctx": int(os.getenv(ENV_LLAMA_N_CTX, str(LLAMA_N_CTX))),
            "n_gpu_layers": int(os.getenv(ENV_LLAMA_N_GPU_LAYERS, str(LLAMA_N_GPU_LAYERS))),
            "n_threads": int(os.getenv(ENV_LLAMA_N_THREADS, str(LLAMA_N_THREADS))),
            "n_batch": int(os.getenv(ENV_LLAMA_N_BATCH, str(LLAMA_N_BATCH))),
            "f16_kv": os.getenv(ENV_LLAMA_F16_KV, str(LLAMA_F16_KV)).lower() == "true",
            "temperature": float(os.getenv(ENV_LLAMA_TEMPERATURE, str(LLAMA_TEMPERATURE))),
            "top_k": int(os.getenv(ENV_LLAMA_TOP_K, str(LLAMA_TOP_K))),
            "top_p": float(os.getenv(ENV_LLAMA_TOP_P, str(LLAMA_TOP_P))),
            "repeat_penalty": float(os.getenv(ENV_LLAMA_REPEAT_PENALTY, str(LLAMA_REPEAT_PENALTY))),
            "max_tokens": int(os.getenv(ENV_LLAMA_MAX_TOKENS, str(LLAMA_MAX_TOKENS))),
            "chat_format": os.getenv(ENV_LLAMA_CHAT_FORMAT, LLAMA_CHAT_FORMAT),
            "verbose": os.getenv(ENV_LLAMA_VERBOSE, str(LLAMA_VERBOSE)).lower() == "true",
            "seed": int(os.getenv(ENV_LLAMA_SEED, str(LLAMA_SEED))),
            "use_mmap": False,
        }

        if os.getenv(ENV_LLAMA_USE_GPU, DEFAULT_LLAMA_USE_GPU).lower() == "false":
            params["n_gpu_layers"] = 0

        print(f"DEBUG [LlamaParamStrategy]: Path={params['model_path']}, GPU={params['n_gpu_layers']}, CTX={params['n_ctx']}")

        return params
