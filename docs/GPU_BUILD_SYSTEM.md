# GPU Build System Documentation

This document describes the comprehensive build system for GPU kernel compilation in the VisionFlow project.

## Overview

The build system integrates CUDA PTX compilation into the Cargo build process, supporting multiple GPU architectures with production-grade error handling and logging.

## Architecture

### Components

1. **`scripts/compile_ptx.sh`** - Main GPU kernel compilation script
2. **`build.rs`** - Cargo build integration
3. **`scripts/build-helper.sh`** - Comprehensive build management utilities

### Supported GPU Architectures

| Architecture | Compute Capability | Target GPUs |
|--------------|-------------------|-------------|
| SM_75 | 7.5 | RTX 20 series, Tesla T4 |
| SM_86 | 8.6 | RTX 30 series, A6000, A100 |
| SM_89 | 8.9 | RTX 40 series, H100 |
| SM_90 | 9.0 | H100, H200 |

Default target: **SM_86** (NVIDIA A6000)

## GPU Kernels

The build system compiles the following CUDA kernels:

| Kernel | Purpose | File |
|--------|---------|------|
| `compute_forces` | Original physics simulation | `src/utils/compute_forces.cu` |
| `visual_analytics_core` | Advanced visual analytics | `src/utils/visual_analytics_core.cu` |
| `advanced_gpu_algorithms` | High-performance algorithms | `src/utils/advanced_gpu_algorithms.cu` |
| `dual_graph_unified` | Unified dual graph processing | `src/utils/dual_graph_unified.cu` |

## Quick Start

### Prerequisites

1. **CUDA Toolkit** (11.0 or later)
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   
   # Or download from: https://developer.nvidia.com/cuda-toolkit
   ```

2. **NVIDIA Drivers** compatible with your CUDA version

3. **Rust toolchain** (1.70.0 or later)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

### Basic Usage

```bash
# Validate build environment
./scripts/build-helper.sh validate

# Compile all GPU kernels
./scripts/build-helper.sh compile

# Build the entire project
cargo build --release

# Check build status
./scripts/build-helper.sh status
```

## Detailed Usage

### Manual PTX Compilation

```bash
# Compile for A6000 (default)
./scripts/compile_ptx.sh

# Compile for RTX 4090
CUDA_ARCH=89 ./scripts/compile_ptx.sh

# Debug compilation
./scripts/compile_ptx.sh --debug

# Clean and recompile
./scripts/compile_ptx.sh --clean
```

### Cargo Integration

The build system automatically compiles PTX kernels during `cargo build`:

```bash
# Release build (optimized)
cargo build --release

# Debug build (with debug symbols)
cargo build

# Force recompilation
cargo clean && cargo build

# CPU-only build (skip GPU compilation)
cargo build --no-default-features --features cpu
```

### Build Helper Utilities

```bash
# Environment validation
./scripts/build-helper.sh validate

# Clean all artifacts
./scripts/build-helper.sh clean --force

# Run build system tests
./scripts/build-helper.sh test

# Performance benchmark
./scripts/build-helper.sh benchmark

# Diagnose issues
./scripts/build-helper.sh doctor

# Show detailed status
./scripts/build-helper.sh status
```

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CUDA_ARCH` | `86` | Target compute capability |
| `DEBUG` | `0` | Enable debug compilation |
| `FAIL_FAST` | `1` | Stop on first compilation error |
| `VERBOSE` | `0` | Enable verbose logging |

### Cargo Features

| Feature | Description |
|---------|-------------|
| `gpu` (default) | Enable GPU acceleration with CUDA |
| `cpu` | CPU-only mode, skip GPU compilation |

Example:
```bash
# CPU-only build
cargo build --no-default-features --features cpu

# Explicit GPU build
cargo build --features gpu
```

## Error Handling

### Common Issues and Solutions

#### 1. NVCC Not Found
```
ERROR: nvcc not found. Please install CUDA Toolkit.
```
**Solution**: Install CUDA Toolkit from NVIDIA's website or package manager.

#### 2. Compute Capability Mismatch
```
WARN: Unknown compute capability: SM_XX
```
**Solution**: Verify your GPU's compute capability and set `CUDA_ARCH` accordingly.

#### 3. Compilation Failures
```
ERROR: Failed to compile kernel_name
```
**Solution**: Check the detailed log in `logs/ptx_compilation.log` for specific errors.

#### 4. Permission Errors
```
ERROR: No write permissions to src/utils
```
**Solution**: Ensure proper file permissions or run with appropriate privileges.

### Debugging

Enable debug mode for detailed compilation information:

```bash
# Script-level debugging
DEBUG=1 ./scripts/compile_ptx.sh --debug

# Cargo-level debugging
RUST_LOG=debug cargo build

# Verbose build helper
./scripts/build-helper.sh compile --verbose --debug
```

### Logs

Build logs are stored in:
- `logs/ptx_compilation.log` - PTX compilation details
- `logs/rust.log` - Rust build information

## Advanced Configuration

### Custom Compilation Flags

Modify `scripts/compile_ptx.sh` to add custom NVCC flags:

```bash
# Add custom flags to nvcc_flags array
local nvcc_flags=(
    -arch=sm_${CUDA_ARCH}
    -O3
    --use_fast_math
    -ptx
    -rdc=true
    --compiler-options -fPIC
    # Add your custom flags here
    --maxrregcount=64
    --ftz=true
)
```

### Build Script Customization

Modify `build.rs` to customize the build process:

```rust
// Custom kernel detection
fn get_cuda_kernels(utils_dir: &Path) -> Vec<String> {
    // Custom logic for kernel discovery
}

// Custom compilation validation
fn verify_ptx_outputs(utils_dir: &Path, kernel_files: &[String]) {
    // Custom PTX validation logic
}
```

## Performance Optimization

### Compilation Performance

- Use `--clean` flag judiciously (only when needed)
- Set `FAIL_FAST=0` to compile all kernels even if some fail
- Use parallel compilation for multiple architectures

### Runtime Performance

- Target specific compute capability for optimal performance
- Use `--use_fast_math` for mathematical operations (enabled by default)
- Enable `-O3` optimization level (enabled by default)

## CI/CD Integration

### GitHub Actions Example

```yaml
name: GPU Build
on: [push, pull_request]

jobs:
  gpu-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install CUDA
        uses: Jimver/cuda-toolkit@v0.2.11
        with:
          cuda: '12.0'
      
      - name: Validate Build Environment
        run: ./scripts/build-helper.sh validate
      
      - name: Compile GPU Kernels
        run: ./scripts/build-helper.sh compile
      
      - name: Build Project
        run: cargo build --release
      
      - name: Run Tests
        run: ./scripts/build-helper.sh test
```

### Docker Integration

```dockerfile
# Multi-stage build for GPU support
FROM nvidia/cuda:12.0-devel-ubuntu22.04 as builder

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source
COPY . /workspace
WORKDIR /workspace

# Validate and build
RUN ./scripts/build-helper.sh validate
RUN ./scripts/build-helper.sh compile
RUN cargo build --release

# Runtime stage
FROM nvidia/cuda:12.0-runtime-ubuntu22.04
COPY --from=builder /workspace/target/release/webxr /usr/local/bin/
```

## Troubleshooting

### Build System Doctor

Run the diagnostic tool for automated issue detection:

```bash
./scripts/build-helper.sh doctor
```

This will check:
- CUDA installation and version
- GPU driver compatibility
- Disk space availability
- File permissions
- Project structure integrity

### Manual Diagnostics

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify kernel files
ls -la src/utils/*.cu

# Check PTX outputs
ls -la src/utils/*.ptx

# Test compilation manually
cd src/utils
nvcc -arch=sm_86 -ptx compute_forces.cu -o test.ptx
```

## Contributing

When adding new GPU kernels:

1. Place `.cu` files in `src/utils/`
2. Add kernel name to `compile_ptx.sh` kernels array
3. Update this documentation
4. Test with `./scripts/build-helper.sh test`

## Performance Benchmarking

```bash
# Run compilation benchmark
./scripts/build-helper.sh benchmark

# Profile individual kernels
nvprof ./target/release/webxr

# Memory usage analysis
nvidia-smi dmon
```

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [Cargo Build Scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html)
- [GPU Architecture Guide](https://developer.nvidia.com/cuda-gpus)