# VisionFlow Rust Development Guide

## Problem: Rust Code Changes Not Reflected in Container

We discovered that Rust source code changes were not being reflected in the running Docker container because the source directory was not mounted as a volume. The container was using a pre-built binary from the Docker image build time.

## Solution: Development Volume Mounts

We've implemented the following fixes:

### 1. Source Code Volume Mounts

Added to `docker-compose.dev.yml`:
```yaml
volumes:
  - ./src:/app/src              # Mount Rust source code
  - ./Cargo.toml:/app/Cargo.toml # Mount Cargo manifest
  - ./Cargo.lock:/app/Cargo.lock # Mount dependency lock file
```

### 2. Automatic Rebuild Check

The container now checks if source files have been modified since the last build:
- `scripts/check-rust-rebuild.sh` - Automatically runs on container start
- Compares modification times of source files vs binary
- Rebuilds if any source file is newer

### 3. Manual Rebuild Script

For immediate rebuilds during development:
```bash
./scripts/dev-rebuild-rust.sh
```

This script:
- Rebuilds the Rust binary inside the container
- Restarts the Rust server
- Shows live logs

## Development Workflow

### Option 1: Container Restart (Automatic)
1. Make changes to Rust source code
2. Restart the container: `docker compose -f docker-compose.dev.yml restart visionflow-xr`
3. The container will automatically detect changes and rebuild

### Option 2: Manual Rebuild (Faster)
1. Make changes to Rust source code
2. Run: `./scripts/dev-rebuild-rust.sh`
3. The script rebuilds and restarts only the Rust server

### Option 3: Full Rebuild (Clean)
1. For major changes or dependency updates
2. Run: `./scripts/dev.sh --no-cache`
3. This rebuilds the entire Docker image

## Troubleshooting

### Check if Rust is running:
```bash
docker compose -f docker-compose.dev.yml exec visionflow-xr ps aux | grep webxr
```

### View Rust logs:
```bash
docker compose -f docker-compose.dev.yml exec visionflow-xr tail -f /app/logs/rust_server.log
```

### Force rebuild inside container:
```bash
docker compose -f docker-compose.dev.yml exec visionflow-xr bash
cd /app
cargo build --features gpu
cp target/debug/webxr /app/webxr
```

## Architecture Notes

- The Rust binary (`webxr`) runs on port 3666
- GPU features are enabled by default
- Build artifacts are cached in `cargo-target-cache` volume
- The binary is built in debug mode for faster compilation

## Performance Considerations

- Initial rebuild after source mount takes longer (full compilation)
- Subsequent rebuilds are incremental and faster
- The `cargo-target-cache` volume preserves build artifacts between container restarts