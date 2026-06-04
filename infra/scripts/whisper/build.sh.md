```shell
#!/bin/bash
set -eo pipefail


# build_whisper_amd.sh

export ROCM_PATH=/opt/rocm
export PATH=$ROCm_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCm_PATH/lib:$LD_LIBRARY_PATH

pushd /mnt/shared/Projects/GitHub/AI/whisper.cpp

rm -rf build-amd/
find . -name "CMakeCache.txt" -delete


export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

pushd /mnt/shared/Projects/GitHub/AI/whisper.cpp

# Clean slate
rm -rf build-amd/
find . -name "CMakeCache.txt" -delete

git pull

# Explicitly turn HIP OFF and Vulkan ON
# This bypasses the problematic ROCm kernels entirely
cmake -B build-amd \
    -DGGML_HIP=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_VULKAN_CHECK_RESULTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF


cmake --build build-amd --config Release -j $(nproc)
popd

```