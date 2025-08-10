# PTX Compilation Guide

## Overview
This project uses NVIDIA CUDA kernels that are compiled to PTX (Parallel Thread Execution) files. To avoid compilation issues during Docker builds and runtime, we pre-compile all PTX files.

## Pre-compilation Process

### 1. Local Compilation
Run the pre-compilation script before building the Docker image:

```bash
# From the project root (/workspace/ext)
./scripts/precompile-ptx.sh
```

This script will:
- Find all `.cu` files in `src/utils/`
- Compile them to PTX files in `src/utils/ptx/`
- Use optimizations (`-O2 --use_fast_math`) for faster compilation
- Create stub files if compilation fails

### 2. Docker Build
The Dockerfile should copy the pre-compiled PTX files:

```dockerfile
# Copy pre-compiled PTX files
COPY src/utils/ptx/*.ptx /app/src/utils/ptx/

# Ensure correct permissions
RUN mkdir -p /app/src/utils/ptx && chmod -R 755 /app/src/utils/ptx
```

### 3. Build Configuration
The `build.rs` script automatically:
- Checks if PTX files exist in `src/utils/ptx/`
- Verifies they're newer than source `.cu` files
- Only recompiles if necessary

## PTX Files

Current kernels:
- `advanced_compute_forces.ptx` - Advanced force calculations with constraints
- `advanced_gpu_algorithms.ptx` - Complex visual analytics algorithms
- `compute_dual_graphs.ptx` - Dual graph computation
- `compute_forces.ptx` - Basic force-directed layout
- `dual_graph_unified.ptx` - Unified dual graph physics
- `unified_physics.ptx` - Unified physics engine
- `visual_analytics_core.ptx` - Core visual analytics

## Troubleshooting

### Compilation Timeout
Some kernels (e.g., `advanced_gpu_algorithms.cu`) are complex and may take long to compile. Solutions:
1. Use optimization flags: `-O2 --use_fast_math`
2. Split large kernels into smaller ones
3. Pre-compile on a powerful machine

### Missing PTX Files
If PTX files are missing:
1. Run `./scripts/precompile-ptx.sh`
2. Check CUDA installation: `nvcc --version`
3. Verify GPU architecture: `nvidia-smi`

### Docker Build Issues
Ensure the Dockerfile:
1. Copies PTX files before Rust build
2. Sets correct permissions
3. Doesn't trigger recompilation

## GPU Architecture
Current target: SM_86 (NVIDIA RTX A6000, RTX 30 series)

To compile for different architectures, modify:
- `scripts/precompile-ptx.sh`: Change `-arch=sm_86`
- `build.rs`: Update `SM_VERSION` constant

## Environment Variables
- `SKIP_CUDA_BUILD=1` - Skip CUDA compilation (not currently implemented)
- `CUDA_ARCH=sm_XX` - Override target architecture (future enhancement)