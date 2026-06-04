```shell
#!/bin/bash
set -eo pipefail

# run_whisper_vulkan.sh

source /opt/intel/oneapi/setvars.sh
# Force the SYCL backend to be the primary 
export GGML_SYCL_DEVICE=1
# Disable SYCL graph execution to prevent hangs
export GGML_SYCL_DISABLE_GRAPH=1
# export GGML_SYCL_DEBUG=1
export ADIGPU_DEVICE_INDEX=1

# RUN
# sycl-ls

# Force SYCL to only use the Level-Zero GPU path
export ONEAPI_DEVICE_SELECTOR='level_zero:1'
# Disable the OpenCL fallback to ensure it either hits the GPU or crashes
export SYCL_DEVICE_FILTER='level_zero:1'

export ZES_ENABLE_SYSMAN=1
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/redist/lib/intel64

export ZES_ENABLE_SYSMAN=1


export WHISPER_HOME=/mnt/shared/Projects/GitHub/AI/whisper.cpp
export GGML_MODEL=/mnt/shared/Models/GGML-large-v3-turbo/ggml-large-v3-turbo.bin
export GGML_VAD=/mnt/shared/Models/GGML-large-v3-turbo/ggml-silero-vad.bin

# No ROCm overrides needed for Vulkan
$WHISPER_HOME/build-amd/bin/whisper-server \
  -m $GGML_MODEL \
   --convert \
   --device 0 \
   --threads 4 \
   --processors 4 \
   --vad \
   --vad-model "$GGML_VAD" \
   --vad-threshold 0.5 \
   --beam-size 2 \
   --best-of 2 \
   --tmp-dir /dev/shm \
   --print-progress \
   --host 0.0.0.0 \
   --port 1145

```