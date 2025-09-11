# Automatic Rust Backend Rebuild System

## Overview
The development environment now **automatically rebuilds the Rust backend on container startup** to ensure all code changes are applied. This fixes the major oversight where code changes weren't taking effect without manual rebuilding.

## How It Works

### 1. Container Startup
When you run `./launch.sh up` or restart the container:
- The `dev-entrypoint.sh` or `rust-backend-wrapper.sh` script runs
- It automatically executes `cargo build --release --features gpu`
- Your latest code changes are compiled into the binary
- The newly built binary is then started

### 2. What Was Changed

#### `/workspace/ext/Dockerfile.dev`
- Added Rust toolchain installation to the final image
- Added build dependencies (build-essential, pkg-config, libssl-dev)
- Copies source code and Cargo files for rebuilding
- Includes the wrapper script and supervisord config

#### `/workspace/ext/scripts/dev-entrypoint.sh`
- Added automatic rebuild step before starting Rust server
- Builds from `/app/target/release/webxr` after compilation
- Can be skipped with `SKIP_RUST_REBUILD=true` if needed

#### `/workspace/ext/scripts/rust-backend-wrapper.sh` (NEW)
- Wrapper script used by supervisord
- Ensures rebuild happens even when supervisord manages the process
- Provides clear logging of rebuild process

#### `/workspace/ext/supervisord.dev.conf`
- Updated to use `rust-backend-wrapper.sh` instead of direct binary
- Ensures rebuild happens on every supervisord restart

## Usage

### Normal Development Flow
```bash
# 1. Make code changes to src/*.rs files

# 2. Restart the container to apply changes
./launch.sh down
./launch.sh up

# The container will automatically:
# - Rebuild the Rust backend with your changes
# - Start the updated server
# - Apply physics propagation fixes and any other code changes
```

### Quick Restart (After Initial Build)
```bash
# If you want to skip the rebuild (e.g., only changed client code)
SKIP_RUST_REBUILD=true ./launch.sh up
```

### Monitoring the Rebuild
The rebuild process is logged to `/app/logs/rust.log`. You'll see:
```
[RUST-WRAPPER] Rebuilding Rust backend with GPU support to apply code changes...
[RUST-WRAPPER] ✓ Rust backend rebuilt successfully
```

## Benefits

1. **No Manual Steps**: Code changes are automatically compiled
2. **Consistent Environment**: Every restart ensures latest code is running
3. **Clear Feedback**: Logs show when rebuild happens and if it succeeds
4. **Development Speed**: No need to remember rebuild commands
5. **Physics Fixes Applied**: Your physics propagation fixes will now be active

## Performance Considerations

- First startup after code changes takes 1-2 minutes for rebuild
- Subsequent startups with no changes are faster (cargo uses incremental compilation)
- The rebuild only happens in development mode, not production

## Troubleshooting

### Rebuild Fails
If the rebuild fails, check `/app/logs/rust.log` for compilation errors:
```bash
docker exec visionflow_container cat /app/logs/rust.log | tail -100
```

### Skip Rebuild for Testing
If you need to test without rebuilding:
```bash
export SKIP_RUST_REBUILD=true
./launch.sh up
```

### Force Clean Rebuild
To force a complete rebuild from scratch:
```bash
# Remove the container and rebuild
./launch.sh down
./launch.sh clean  # Warning: removes all containers and volumes
./launch.sh -f up  # Force rebuild
```

## Impact on Physics Settings Issue

With this automatic rebuild system:
1. Your physics propagation fixes in `settings_handler.rs` and `graph_actor.rs` will be compiled
2. The server will start with the fixed code
3. Physics settings changes in the UI will immediately affect the simulation
4. No manual rebuild steps required!

## Next Steps

After rebuilding with this new system:
1. Start the environment: `./launch.sh up`
2. Wait for rebuild to complete (watch the logs)
3. Open the UI and test physics controls
4. Changes should now apply immediately without server restart