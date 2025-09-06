# GPU Physics Fix for Knowledge Graph Nodes

## Problem
The knowledge graph nodes were locked in their initial distribution and not responding to force-directed physics simulation. Investigation revealed:
- GPU compute context was not being initialized
- PTX file was not being generated/found at runtime
- "No GPU compute context available for physics simulation" warnings in logs

## Root Causes
1. **PTX File Path Issue**: The build.rs script sets `VISIONFLOW_PTX_PATH` at compile time to a path that doesn't exist in the Docker container at runtime
2. **Missing GPU Initialization**: The advanced_gpu_context was never successfully initialized due to PTX loading failure
3. **No Fallback Mechanism**: When PTX file wasn't found, system failed silently without attempting to compile it

## Solution Implemented

### 1. Added Initialization Delay
Added a 2-second delay before GPU initialization to ensure system is ready:
```rust
// Add initialization delay as suggested
info!("Waiting 2 seconds before GPU initialization to ensure system is ready...");
tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
```

### 2. Runtime PTX Compilation Fallback
Created `compile_ptx_fallback()` function that:
- Searches for CUDA source file in multiple locations
- Compiles PTX on the fly using nvcc
- Uses CUDA_ARCH environment variable (defaults to 89 for RTX A6000)
- Returns compiled PTX content as string

### 3. Environment Configuration
Updated docker-compose.dev.yml to pass CUDA_ARCH at runtime:
```yaml
environment:
  - CUDA_ARCH=${CUDA_ARCH:-89} # Pass CUDA architecture for runtime PTX compilation
```

### 4. Fallback Integration
Modified GPU initialization to use fallback when primary PTX loading fails:
```rust
let ptx_content = match std::fs::read_to_string(ptx_path) {
    Ok(content) => content,
    Err(e) => {
        // Try to compile PTX on the fly as a fallback
        warn!("Attempting to compile PTX file on the fly as fallback...");
        compile_ptx_fallback().await?
    }
};
```

## Files Modified
- `/workspace/ext/src/actors/graph_actor.rs` - Added delay and fallback compilation
- `/workspace/ext/docker-compose.dev.yml` - Added CUDA_ARCH to runtime environment
- `/workspace/ext/.env` - Already had CUDA_ARCH=89 configured

## How It Works Now
1. System waits 2 seconds after startup for GPU to be ready
2. Attempts to load pre-compiled PTX file from build directory
3. If that fails, compiles PTX on the fly at runtime
4. Initializes advanced_gpu_context with compiled PTX
5. Physics simulation runs on GPU for force-directed graph layout

## Verification
After restart, check logs for:
- "Successfully compiled PTX file on the fly" (if fallback used)
- "Successfully initialized advanced GPU context"
- No more "No GPU compute context available" warnings
- Nodes should respond to physics forces and move dynamically

## CUDA Architecture Settings
- RTX A6000 uses Ada architecture (sm_89)
- RTX 30-series uses Ampere (sm_86)
- Set CUDA_ARCH in .env file to match your GPU

## Future Improvements
Consider:
- Pre-compiling PTX for common architectures
- Caching compiled PTX to avoid recompilation
- Better error reporting for GPU initialization failures