```shell
#!/bin/bash
set -eo pipefail

# Source the oneAPI environment
source /opt/intel/oneapi/setvars.sh --force

export CXXFLAGS="-O3 -march=native -mtune=native"
export CFLAGS="-O3 -march=native -mtune=native"

pushd /mnt/shared/Projects/GitHub/AI/intel_llama.cpp

# Clean out prior artifacts completely
rm -rf build-intel/
find . -name "CMakeCache.txt" -delete

git pull

# Configure with ALL SYCL optimizations
cmake -B build-intel \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=OFF \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
  -DGGML_SYCL_STMT=ON \
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/opt/intel/oneapi/mkl/2026.0 \
  -DLLAMA_LTO=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_C_FLAGS="$CFLAGS"

# Compile using all available threads
cmake --build build-intel --config Release -j $(nproc)
popd


```