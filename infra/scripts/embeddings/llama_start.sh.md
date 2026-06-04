```shell
##!/bin/bash
set -eo pipefail

# run_mxbai_intel.sh

# Intel oneAPI environment (once)
source /opt/intel/oneapi/setvars.sh

# Your original GPU backend variables (all valid)
export SYCL_DEVICE_FILTER=level_zero
export GGML_SYCL_FPCW=1
export OverrideDefaultTimeout=0
export ZE_ENABLE_PCI_ID_ORDERING=1
export NEOReadDebugKeys=1
export DisableFenceTimeout=1
export EnableWalkerPartition=1
export ZES_ENABLE_SYSMAN=1
export ZE_INTEL_DRIVER_DATA=1

# Additional GPU performance hints for N100
export SYCL_CACHE_PERSISTENT=1           # cache JIT compilation
export ZE_AFFINITY_MASK=0                # use first subdevice (if any)

# Prevent CPU oversubscription (GPU still uses CPU for scheduling)
export OMP_NUM_THREADS=4                 # match CPU cores
export MKL_NUM_THREADS=1

LLAMA_HOME=/mnt/shared/Projects/GitHub/AI/intel_llama.cpp/build-intel/bin   # your SYCL build
MODELS_HOME=/mnt/shared/Models

$LLAMA_HOME/llama-server \
  --model $MODELS_HOME/MXBAI-embedded/mxbai-embed-large-v1.Q4_K_M.gguf \
  --ctx-size 8192\
  --pooling cls \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --flash-attn on \
  --threads 4 \
  --threads-batch 4 \
  --no-warmup \
  --kv_unified \
  --parallel -1 \
  --jinja \
  --cont-batching \
  --embedding \
  --host 0.0.0.0 \
  --port 11434

```