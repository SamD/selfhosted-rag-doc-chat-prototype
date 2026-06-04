```shell
##!/bin/bash
set -eo pipefail

# build_llamacpp_intel_cpu.sh

source /opt/intel/oneapi/setvars.sh

export SYCL_DEVICE_FILTER=level_zero
export GGML_SYCL_FPCW=1
export OverrideDefaultTimeout=0
export ZE_ENABLE_PCI_ID_ORDERING=1
export NEOReadDebugKeys=1
export DisableFenceTimeout=1
export EnableWalkerPartition=1

pushd /mnt/shared/Projects/GitHub/AI/intel_llama.cpp

# 2. Delete the poisoned cache
rm -rf build-intel/
# If using uv/mise and the build is hidden:
find . -name "CMakeCache.txt" -delete

git pull

cmake -B build-intel \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Intel10_64lp \
  -DCMAKE_CXX_FLAGS="-march=native -O3" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

  # -DLLAMA_OPENSSL=ON

# Compile using all available cores
cmake --build build-intel --config Release -j $(nproc)
popd


```