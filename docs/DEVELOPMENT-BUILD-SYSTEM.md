# Development Build System - Architecture-Agnostic Rust Compilation

## Overview
The development environment now **ALWAYS builds the Rust backend inside the container on startup**. This ensures the binary is compiled for the container's specific architecture and CUDA configuration, avoiding architecture mismatches.

## Why This Change Was Necessary

### The Problem
- The previous multi-stage Docker build compiled the Rust binary during image creation
- This binary might not match the runtime container's architecture (e.g., different CUDA versions, CPU architectures)
- Code changes required manual rebuilding inside the container
- Physics settings weren't propagating because the old binary didn't have the fixes

### The Solution
- Removed multi-stage build from `Dockerfile.dev`
- Container now includes full Rust toolchain and build dependencies
- Every container startup triggers a fresh build with the exact architecture
- Your code changes (including physics fixes) are automatically compiled

## How It Works

### 1. Container Startup Flow
```
./launch.sh up
    ↓
Docker container starts
    ↓
dev-entrypoint.sh or supervisord runs
    ↓
rust-backend-wrapper.sh executes
    ↓
cargo build --release --features gpu
    ↓
Rust backend starts with fresh binary
```

### 2. Key Components

#### `Dockerfile.dev`
- Single-stage build (no more builder stage)
- Includes Rust toolchain and build tools
- Copies source code for compilation
- Pre-fetches dependencies with `cargo fetch`

#### `scripts/rust-backend-wrapper.sh`
- Wrapper script that builds before running
- Used by supervisord for process management
- Can skip rebuild with `SKIP_RUST_REBUILD=true`

#### `scripts/dev-entrypoint.sh`
- Alternative entry point for non-supervisord mode
- Also handles automatic rebuilding
- Provides detailed logging

## Usage

### Standard Development Workflow
```bash
# 1. Make your code changes
vim src/handlers/settings_handler.rs

# 2. Rebuild and restart
./launch.sh down
./launch.sh -f up  # -f forces Docker image rebuild if needed

# 3. Container automatically:
#    - Detects container architecture
#    - Builds Rust backend for that architecture
#    - Applies all your code changes
#    - Starts with the new binary
```

### First-Time Setup
```bash
# Clean everything and start fresh
./launch.sh clean  # Warning: removes all containers
./launch.sh -f up  # Force rebuild and start
```

### Quick Restart (Skip Rebuild)
```bash
# If you ONLY changed client code and want to skip Rust rebuild
SKIP_RUST_REBUILD=true ./launch.sh up
```

## Build Performance

### First Build
- Takes 2-3 minutes for full compilation
- Compiles all dependencies and CUDA kernels
- Optimizes for the specific GPU architecture

### Subsequent Builds
- Uses Cargo's incremental compilation
- Only rebuilds changed code
- Typically 30-60 seconds

### Optimizations
- `cargo fetch` pre-downloads dependencies in Docker image
- Build cache persists between container restarts
- Only clean builds when Docker image is rebuilt

## Architecture Detection

The build system automatically detects:
- GPU architecture (CUDA compute capability)
- CPU architecture (x86_64, ARM, etc.)
- Available CUDA version
- Compiler optimizations

This ensures optimal performance for your specific hardware.

## Monitoring the Build

### Watch Build Progress
```bash
# In another terminal while container is starting
docker logs -f visionflow_container
```

You'll see:
```
[RUST-WRAPPER] Rebuilding Rust backend with GPU support...
   Compiling webxr v0.1.0 (/app)
[RUST-WRAPPER] ✓ Rust backend rebuilt successfully
```

### Check Build Logs
```bash
docker exec visionflow_container cat /app/logs/rust.log
```

## Troubleshooting

### Build Fails
1. Check compilation errors in logs
2. Ensure CUDA toolkit is properly installed
3. Verify source code syntax

### Build Takes Too Long
1. First build is always slow (normal)
2. Use `SKIP_RUST_REBUILD=true` if no Rust changes
3. Check available disk space

### Wrong Architecture
This should not happen anymore! The build always matches the container.

## Benefits

1. **Architecture Compatibility**: Binary always matches container
2. **Automatic Updates**: Code changes applied on every restart
3. **No Manual Steps**: Just use `launch.sh` normally
4. **GPU Optimization**: Compiles for your specific GPU
5. **Development Speed**: No need to remember rebuild commands

## Impact on Physics Settings Issue

With this system:
1. Your physics propagation fixes are compiled on startup
2. The `propagate_physics_to_gpu` calls are included
3. Settings changes immediately affect the simulation
4. No manual rebuild or architecture mismatch issues

## Summary

The development environment now handles everything automatically:
- ✅ Architecture-specific compilation
- ✅ Automatic code compilation on startup
- ✅ Physics fixes applied without manual steps
- ✅ Single command to rebuild and run: `./launch.sh up`

Just write code and use `launch.sh` - the build system handles the rest!