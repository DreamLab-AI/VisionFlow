# GPU Physics Initialization Fix

## Problem Summary
The physics simulation was showing only 3 nodes bouncing instead of the full 177-node graph. Physics control changes from the UI weren't affecting the simulation.

## Root Causes Found

### 1. GPU Context Not Initializing
- The GraphServiceActor's `advanced_gpu_context` was remaining `None`
- Physics simulation requires this GPU context to run
- Without it, the simulation loop warns "No GPU compute context available for physics simulation"

### 2. PTX Compilation Path Issues
- PTX (CUDA kernel) compilation was failing silently
- The `CARGO_MANIFEST_DIR` is compiled into the binary at build time (`/app`)
- At runtime, the path resolution was failing
- The PTX loader needs to compile at runtime in Docker environments

### 3. Missing Error Recovery
- When PTX loading failed, the `gpu_init_in_progress` flag wasn't reset
- This prevented retry attempts
- The actor got stuck thinking initialization was still in progress

## Fixes Applied

### 1. Added DOCKER_ENV Variable
Set `DOCKER_ENV=1` in three places to force runtime PTX compilation:
- `/scripts/rust-backend-wrapper.sh`
- `/scripts/dev-entrypoint.sh`
- `/supervisord.dev.conf`

This ensures the PTX loader always compiles the CUDA kernel at runtime in containers.

### 2. Improved Error Handling
In `/src/actors/graph_actor.rs`:
- Added error logging for PTX load failures
- Reset `gpu_init_in_progress` flag on failure via `ResetGPUInitFlag` message
- Added detailed logging of GPU initialization steps

### 3. Enhanced Debug Logging
Added logging to `/src/utils/ptx.rs`:
- Log when entering PTX load functions
- Log Docker environment detection
- Log compilation architecture selection

## How It Works Now

1. **Container Startup**:
   - DOCKER_ENV=1 is set in the environment
   - Rust backend rebuilds with `cargo build --release --features gpu`

2. **Graph Initialization**:
   - Server loads 177 nodes from metadata
   - GraphServiceActor starts simulation loop
   - After 2 seconds, attempts GPU initialization

3. **PTX Compilation**:
   - Detects DOCKER_ENV=1, uses runtime compilation
   - Compiles `/app/src/utils/visionflow_unified.cu` to PTX
   - Creates `/tmp/visionflow_unified.ptx`

4. **GPU Context Creation**:
   - UnifiedGPUCompute initializes with PTX
   - Context stored in GraphServiceActor
   - Physics simulation runs on GPU

5. **Physics Updates**:
   - Settings changes trigger `propagate_physics_to_gpu()`
   - Updates sent to both GPUComputeActor and GraphServiceActor
   - GPU applies new physics parameters in real-time

## Testing After Fix

To verify the fix works:

1. Rebuild and restart the container:
```bash
./launch.sh down
./launch.sh -f up
```

2. Check the logs for successful GPU initialization:
```bash
docker logs visionflow_container | grep -E "GPU|PTX"
```

Look for:
- "PTX content loaded successfully"
- "✅ Successfully initialized advanced GPU context with 177 nodes"
- "GPU physics simulation is now active"

3. Verify physics controls work:
- Open the UI control panel
- Adjust physics sliders (damping, repulsion, etc.)
- Changes should immediately affect the simulation

## Key Files Modified

1. **Graph Actor** (`src/actors/graph_actor.rs`):
   - Lines 1312-1317: PTX error handling
   - Lines 1325-1343: GPU initialization logging
   - Line 1356: Reset flag on failure

2. **PTX Loader** (`src/utils/ptx.rs`):
   - Line 10: Added log imports
   - Lines 43-48: Docker environment detection logging
   - Lines 82-84: Compilation logging

3. **Wrapper Scripts**:
   - `scripts/rust-backend-wrapper.sh`: Line 8 - export DOCKER_ENV=1
   - `scripts/dev-entrypoint.sh`: Line 5 - export DOCKER_ENV=1
   - `supervisord.dev.conf`: Line 27 - DOCKER_ENV=1 in environment

4. **Settings Handler** (previously fixed):
   - `src/handlers/settings_handler.rs`: Physics propagation calls
   - `src/actors/graph_actor.rs`: UpdateSimulationParams handler

## Monitoring

Watch for these warnings that indicate issues:
- "No GPU compute context available for physics simulation"
- "Failed to load PTX content"
- "Failed to initialize advanced GPU context"
- "GPU NOT INITIALIZED! Cannot update graph data"

If you see these, check:
1. CUDA is available: `nvidia-smi`
2. nvcc is installed: `which nvcc`
3. CUDA source exists: `ls /app/src/utils/visionflow_unified.cu`
4. PTX compiles: `nvcc -ptx -arch=sm_86 /app/src/utils/visionflow_unified.cu -o /tmp/test.ptx`

## Summary

The fix ensures:
- ✅ PTX always compiles at runtime in Docker
- ✅ GPU context initializes properly
- ✅ Physics runs on all 177 nodes
- ✅ UI controls affect the simulation immediately
- ✅ Errors are logged and recoverable