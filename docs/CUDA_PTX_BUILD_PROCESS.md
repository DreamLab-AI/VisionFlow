# Unified CUDA/PTX Build Process

## Overview
This document describes the single source of truth for building CUDA code to PTX in the VisionFlow system. All CUDA compilation is managed through `build.rs` during the Rust compilation process.

## Build Process

### 1. Single Source: build.rs
The `build.rs` script is the **only** place where CUDA code is compiled to PTX. This ensures consistency and eliminates confusion from multiple build paths.

**Location**: `/workspace/ext/build.rs`

**What it does**:
1. Compiles `src/utils/visionflow_unified.cu` to PTX format
2. Places the output at `src/utils/ptx/visionflow_unified.ptx`
3. Also compiles object files for linking Thrust wrappers
4. Handles both debug and release configurations
5. Respects the `CUDA_ARCH` environment variable (default: sm_86)

### 2. Docker Build Integration

#### Development (Dockerfile.dev)
- Sets `CUDA_ARCH` environment variable
- Runs `cargo build` which triggers `build.rs`
- PTX is generated in-situ during compilation
- No manual PTX compilation or copying needed

#### Production (Dockerfile.production)
- Sets `CUDA_ARCH` via build argument
- Runs `cargo build --release` which triggers `build.rs`
- Copies the generated PTX from builder to runtime stage:
  ```dockerfile
  COPY --from=builder /build/src/utils/ptx/visionflow_unified.ptx /app/src/utils/ptx/visionflow_unified.ptx
  ```

### 3. Runtime PTX Loading

The `graph_actor.rs` loads the PTX file from one of two locations:
1. **Production**: `/app/src/utils/ptx/visionflow_unified.ptx`
2. **Development**: `src/utils/ptx/visionflow_unified.ptx`

This simple two-path approach eliminates the need for multiple search paths.

## Key Files

| File | Purpose |
|------|---------|
| `build.rs` | Compiles CUDA to PTX (single source of truth) |
| `src/utils/visionflow_unified.cu` | CUDA kernel source code |
| `src/utils/ptx/visionflow_unified.ptx` | Generated PTX output (gitignored) |
| `src/actors/graph_actor.rs` | Loads PTX at runtime |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_ARCH` | 86 | CUDA compute capability (e.g., 86 for sm_86) |
| `CARGO_FEATURE_GPU` | - | Set by cargo when building with `--features gpu` |

## Build Commands

### Local Development
```bash
# PTX is built automatically during cargo build
CUDA_ARCH=86 cargo build --features gpu
```

### Docker Development
```bash
docker build -f Dockerfile.dev --build-arg CUDA_ARCH=86 -t visionflow-dev .
```

### Docker Production
```bash
docker build -f Dockerfile.production --build-arg CUDA_ARCH=86 -t visionflow .
```

## Important Notes

1. **Never manually compile PTX** - Always let `build.rs` handle it
2. **PTX files are gitignored** - They are build artifacts, not source code
3. **No redundant PTX compilation** - Removed all duplicate nvcc calls from Dockerfiles
4. **Single PTX path per environment** - Production vs Development, no complex search paths
5. **Build-time architecture selection** - Use `CUDA_ARCH` build arg/env var

## Troubleshooting

### PTX Not Found
- Check that `cargo build` completed successfully
- Verify `src/utils/ptx/` directory exists
- Ensure CUDA toolkit is installed (nvcc must be available)

### Wrong Architecture
- Set `CUDA_ARCH` correctly for your GPU
- Common values: 75 (RTX 2080), 86 (RTX 3090), 89 (RTX 4090)

### Build Failures
- Check `cargo:warning` messages from build.rs
- Ensure CUDA source file exists at `src/utils/visionflow_unified.cu`
- Verify nvcc is in PATH