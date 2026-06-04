```shell
#!/bin/bash
set -eo pipefail

# mardown_writer_llama.sh

# Load global settings from environment (safe!)
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"
export MAIN_GPU="${MAIN_GPU:-1}"
export GGML_SYCL_DEVICE="${GGML_SYCL_DEVICE:-1}"
# Validate GPU settings
if [[ -n "$ONEAPI_DEVICE_SELECTOR" ]]; then
  echo "Using ONEAPI_DEVICE_SELECTOR: $ONEAPI_DEVICE_SELECTOR"
elif [[ -n "$MAIN_GPU" ]]; then
  echo "Using MAIN_GPU: $MAIN_GPU"
else
  echo "WARNING: No GPU selection specified. Using default." >&2
fi

export LLAMA_PORT="${LLAMA_PORT:-11435}"
if [[ -n "$LLAMA_PORT" ]]; then
  echo "Using LLAMA_PORT: $LLAMA_PORT"
else
  echo "WARNING: No GPU selection specified. Using default." >&2
fi


LLAMA_HOME="${LLAMA_HOME:-/mnt/shared/Projects/GitHub/AI/intel_llama.cpp/build-intel/bin}"
MODELS_HOME="${MODELS_HOME:-/mnt/shared/Models}"

source /opt/intel/oneapi/setvars.sh

# Enable accurate memory reporting
export ZES_ENABLE_SYSMAN=1

# Disable DNN fallback if you encounter issues (test both)
export g_ggml_sycl_disable_dnn=0   # Keep DNN enabled initially

    # --model $MODELS_HOME/Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf \
mkdir -p /tmp/slots
$LLAMA_HOME/llama-server \
    --model $MODELS_HOME/Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf \
    --n-gpu-layers 99 \
    --ctx-size 262144 \
    --batch-size 4096\
    --ubatch-size 2048 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --no-warmup \
    --fit on \
    --parallel 2 \
    --threads 8 \
    --threads-batch 6\
    --temp 0.1 \
    --top-k 0 \
    --top-p 1.0 \
    --min-p 0.45 \
    --n-predict -1\
    --reasoning off \
    --reasoning-budget -1 \
    --repeat-penalty 1.2 \
    --repeat-last-n 128 \
    --frequency-penalty 0.0 \
    --presence-penalty 0.0 \
    --split-mode none \
    --flash-attn on \
    --no-mmap \
    --mlock \
    --swa-full \
    --numa distribute \
    --cache-reuse 0 \
    --cache-ram  1024 \
    --cont-batching \
    --direct-io \
    --kv-unified \
    --no-context-shift \
    --spec-draft-n-max 2 \
    --spec-type draft-mtp \
    --spec-draft-p-min 0.60 \
    --spec-draft-ngl 99 \
    --slot-save-path /tmp/slots \
    --jinja \
    --metrics \
    --host 0.0.0.0 \
    --port $LLAMA_PORT

```