#!/bin/bash

# Compile advanced CUDA kernel for constraint-aware physics
CUDA_ARCH=${CUDA_ARCH:-75}

echo "Compiling advanced compute forces kernel for SM_${CUDA_ARCH}..."

nvcc \
    -arch=sm_${CUDA_ARCH} \
    -O3 \
    --use_fast_math \
    -ptx \
    -rdc=true \
    --compiler-options -fPIC \
    src/utils/advanced_compute_forces.cu \
    -o src/utils/advanced_compute_forces.ptx

chmod 644 src/utils/advanced_compute_forces.ptx

echo "Advanced kernel compilation complete."

# Also compile the dual-graph kernel if it exists
if [ -f "src/utils/compute_dual_graphs.cu" ]; then
    echo "Compiling dual graphs kernel..."
    nvcc \
        -arch=sm_${CUDA_ARCH} \
        -O3 \
        --use_fast_math \
        -ptx \
        -rdc=true \
        --compiler-options -fPIC \
        src/utils/compute_dual_graphs.cu \
        -o src/utils/compute_dual_graphs.ptx
    
    chmod 644 src/utils/compute_dual_graphs.ptx
    echo "Dual graphs kernel compilation complete."
fi