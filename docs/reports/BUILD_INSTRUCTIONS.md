# VisionFlow Build Instructions

## Overview
VisionFlow has been simplified to remove legacy build artifacts. The Cargo.lock file is now automatically generated during the Docker build process.

## Quick Start

### Development Build
```bash
# Clean build (recommended after dependency updates)
./scripts/docker_clean_build.sh dev

# Or using docker-compose directly
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up
```

### Production Build
```bash
# Clean build for production
./scripts/docker_clean_build.sh production

# Or using docker-compose
docker-compose build
docker-compose up
```

## Build Process Improvements

### What Changed
1. **Removed Cargo.lock from version control** - It's now generated automatically
2. **Simplified Dockerfiles** - No longer copy non-existent Cargo.lock
3. **Updated dependencies** - All Rust crates updated to latest stable versions
4. **Removed parking_lot** - Using standard library synchronization primitives
5. **Updated CUDA** - Now using CUDA 12.8.1 for better GPU support

### Why These Changes
- **Cargo.lock**: Was causing build failures as a legacy artifact
- **Dependencies**: Security updates and performance improvements
- **Parking_lot removal**: Eliminates potential deadlock issues
- **Simplified build**: Faster builds with better caching

## Dependency Management

### Rust Dependencies
The project now uses flexible semantic versioning for dependencies:
- No exact version pinning (was `=x.y.z`, now `x.y`)
- Automatic security updates within compatible ranges
- Cargo.lock generated fresh on each clean build

### Key Updates
- Actix Web: 4.9 (was 4.5.1)
- Tungstenite: 0.24 (was 0.22)
- Cudarc: 0.12 (was 0.11)
- Reqwest: 0.12 (was 0.11)
- Many other security and performance updates

## Docker Environment

### Base Images
- Development: `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- Production: `nvidia/cuda:12.8.1-runtime-ubuntu22.04`

### Build Arguments
- `CUDA_ARCH`: GPU architecture (default: 86 for RTX 30 series/A6000)
- `FEATURES`: Rust features (default: gpu)

## Troubleshooting

### Build Fails with "Cargo.lock not found"
This is normal on first build. The lock file is generated automatically.

### Dependency Resolution Issues
```bash
# Clear Docker build cache and rebuild
./scripts/docker_clean_build.sh dev
```

### GPU Not Detected
Ensure NVIDIA drivers and Docker GPU support are installed:
```bash
# Check GPU availability
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Development Workflow

### After Updating Dependencies
1. Remove old Cargo.lock: `rm -f Cargo.lock`
2. Clean build: `./scripts/docker_clean_build.sh dev`
3. The new Cargo.lock will be generated automatically

### Mounting for Development
The docker-compose.dev.yml mounts source directories for hot-reload:
- `/app/src` - Rust source code
- `/app/client` - Frontend code
- `/app/Cargo.toml` - Dependency configuration

Note: Cargo.lock is NOT mounted - it's container-specific

## Performance Tips

### Build Caching
Docker layers are cached efficiently:
1. Base image and system dependencies
2. Rust and Node.js installation
3. Dependency fetch (`cargo fetch`)
4. Source code compilation

### Parallel Builds
Use BuildKit for faster builds:
```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev .
```

## Security Notes

### Dependency Scanning
Run security audit periodically:
```bash
./scripts/security_audit.sh
```

### Updates
Check for outdated dependencies:
```bash
cargo outdated
```

## Summary

The build process is now simpler and more maintainable:
- No manual Cargo.lock management
- Automatic dependency resolution
- Better security with flexible versioning
- Cleaner Docker builds
- Improved caching

For questions or issues, see the main README or open an issue on GitHub.