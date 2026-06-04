```shell
#!/bin/bash
set -eo pipefail

# run_docling_serve_nvidia.sh

# NVIDIA CUDA environment setup for CUDA 13.0
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 2. PyTorch/NVIDIA specific (adjusted for CUDA 13.0)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=1073741824

# 3. Hardware Strategy
export DOCLING_DEVICE="cuda"

# 4. Path Definitions
export DOCLING_HOME="/mnt/shared/Models/DOCLING"
export DOCLING_SERVE_HOME="/mnt/shared/Projects/GitHub/AI/docling-serve"
export DOCLING_SERVE_ARTIFACTS_PATH="$DOCLING_HOME/Docling-Models/docling_models_bundle"

# 5. NVIDIA CUDA 13.0 specific
export CUDA_MODULE_LOADING=LAZY
# CUDA 13.0 uses compute capability 9.0+ features by default
export NVIDIA_TF32_OVERRIDE=1

# 6. Virtual environment
export VIRTUAL_ENV=$HOME/aibin/venv
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/onnxruntime/capturable/lib:$LD_LIBRARY_PATH

# might be needed
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# 7. ONNX Runtime - CUDA 13.0 compatible
export ONNXRUNTIME_EXECUTION_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
export ORT_CUDA_CACHE_PATH="$DOCLING_HOME/ort_cuda_cache"
mkdir -p "$ORT_CUDA_CACHE_PATH"

# 8. RapidOCR with CUDA support
export DOCLING_SERVE_ALLOW_CUSTOM_OCR_CONFIG=true
export DOCLING_SERVE_DEFAULT_OCR_ENGINE=rapidocr
export DOCLING_SERVE_OCR_CUSTOM_CONFIG='{
    "kind": "rapidocr",
    "engine_name": "onnxruntime",
    "onnx_providers": ["CUDAExecutionProvider"],
    "onnx_provider_options": [
        {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
    ]
}'

# 9. Offline & Performance
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="/mnt/shared/Models/TORCH_CACHE"
export TOKENIZERS_PARALLELISM=false

# 10. Execution Config
export DOCLING_SERVE_WARM_UP=true
export DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1
export DOCLING_SERVE_MAX_SYNC_WAIT=600

# 11. Memory management for CUDA 13.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force GPU provider ordering
export RAPIDOCR_ORT_PROVIDERS="CUDAExecutionProvider,TensorrtExecutionProvider,CPUExecutionProvider"
export ORT_DISABLE_AZURE=1
export DOCLING_DEVICE="cuda:0"
export DOCLING_CUDA_USE_FLASH_ATTENTION2=true

# Batch sizes for better GPU utilization
export DOCLING_SERVE_LAYOUT_BATCH_SIZE=160
export DOCLING_SERVE_TABLE_BATCH_SIZE=10
export DOCLING_SERVE_OCR_BATCH_SIZE=20

# 12. Start Process
#pushd "$DOCLING_SERVE_HOME" || exit 1
pushd "$HOME/aibin" || exit 1
source ./venv/bin/activate


docling-serve run --no-enable-ui --timeout-keep-alive 600

deactivate
popd


```