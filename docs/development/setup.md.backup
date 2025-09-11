# VisionFlow Development Setup Guide

*[Development](../index.md)*

This guide will help you set up your development environment for contributing to VisionFlow, with a focus on the Rust backend development workflow.

## Prerequisites

- Docker and Docker Compose
- Git
- Node.js (for frontend development)
- Rust toolchain (optional, for local development)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd visionflow
   ```

2. **Start development environment**
   ```bash
   ./scripts/dev.sh
   ```

This will start all services in development mode with hot reloading enabled.

## Development Environment Setup

### Container-Based Development (Recommended)

The project uses Docker containers for consistent development environments. All source code is mounted as volumes, allowing for real-time development without container rebuilds.

#### Key Components
- **Frontend**: React application with hot reloading
- **Backend**: Rust WebXR server on port 3666
- **GPU Compute**: CUDA-enabled processing
- **Databases**: Redis and other data stores

#### Volume Mounts
The development setup automatically mounts:
```yaml
volumes:
  - ./src:/app/src              # Rust source code
  - ./Cargo.toml:/app/Cargo.toml # Cargo manifest
  - ./Cargo.lock:/app/Cargo.lock # Dependency lock file
  - ./frontend:/app/frontend     # Frontend source
```

## Rust Development Workflow

### Automatic Rebuild Detection

The container automatically detects when Rust source files have been modified and rebuilds the binary. This happens:
- On container startup
- When the container is restarted
- Via the automatic rebuild check script

### Development Options

#### Option 1: Container Restart (Automatic Detection)
Best for: Small to medium changes

1. Make changes to Rust source code
2. Restart the container:
   ```bash
   docker compose -f docker-compose.dev.yml restart visionflow-xr
   ```
3. The container automatically detects changes and rebuilds

#### Option 2: Manual Rebuild (Fastest)
Best for: Iterative development with frequent changes

1. Make changes to Rust source code
2. Run the manual rebuild script:
   ```bash
   ./scripts/dev-rebuild-rust.sh
   ```

This script:
- Rebuilds only the Rust binary inside the container
- Restarts the Rust server process
- Shows live logs for immediate feedback
- Preserves build cache for faster subsequent builds

#### Option 3: Full Rebuild (Clean Slate)
Best for: Major changes, dependency updates, or troubleshooting

1. For major changes or when in doubt:
   ```bash
   ./scripts/dev.sh --no-cache
   ```
2. This rebuilds the entire Docker image from scratch

### Build Performance Notes

- **Initial Build**: Takes longer as all dependencies are compiled
- **Incremental Builds**: Much faster thanks to Rust's incremental compilation
- **Build Cache**: The `cargo-target-cache` volume preserves compiled artifacts
- **Debug Mode**: Development builds use debug mode for faster compilation

## Development Scripts

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `./scripts/dev.sh` | Start development environment | `./scripts/dev.sh [--no-cache]` |
| `./scripts/dev-rebuild-rust.sh` | Quick Rust rebuild | `./scripts/dev-rebuild-rust.sh` |
| `./scripts/check-rust-rebuild.sh` | Auto-rebuild check | Called automatically |

### Rebuild Script Details

The `dev-rebuild-rust.sh` script performs these steps:
1. Compiles Rust code with GPU features enabled
2. Copies the new binary to the correct location
3. Restarts the Rust server process
4. Displays real-time logs for verification

## Container Development Setup

### Services Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Rust WebXR    │    │   GPU Compute   │
│   (React)       │◄──►│   Server        │◄──►│   (CUDA)        │
│   Port: 3000    │    │   Port: 3666    │    │   Background    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Environment Variables

Key development environment variables:
```bash
RUST_LOG=debug                    # Enable debug logging
CUDA_VISIBLE_DEVICES=0            # GPU selection
DOCKER_BUILDKIT=1                 # Enable BuildKit for faster builds
```

### Port Mapping

| Service | Container Port | Host Port | Description |
|---------|---------------|-----------|-------------|
| Frontend | 3000 | 3000 | React development server |
| Rust WebXR | 3666 | 3666 | WebXR API server |
| WebSocket | 8080 | 8080 | Real-time communication |

## Troubleshooting

### Common Issues

#### Rust Changes Not Reflected

**Problem**: Code changes don't appear in the running application.

**Solutions**:
1. Check if the rebuild script ran successfully:
   ```bash
   docker compose -f docker-compose.dev.yml logs visionflow-xr
   ```

2. Manual rebuild:
   ```bash
   ./scripts/dev-rebuild-rust.sh
   ```

3. Full restart:
   ```bash
   docker compose -f docker-compose.dev.yml restart visionflow-xr
   ```

#### Build Failures

**Problem**: Compilation errors or build failures.

**Diagnosis**:
1. Check Rust server status:
   ```bash
   docker compose -f docker-compose.dev.yml exec visionflow-xr ps aux | grep webxr
   ```

2. View detailed logs:
   ```bash
   docker compose -f docker-compose.dev.yml exec visionflow-xr tail -f /app/logs/rust_server.log
   ```

3. Manual compilation check:
   ```bash
   docker compose -f docker-compose.dev.yml exec visionflow-xr bash
   cd /app
   cargo build --features gpu
   ```

#### Container Issues

**Problem**: Container won't start or crashes.

**Solutions**:
1. Clean rebuild:
   ```bash
   ./scripts/dev.sh --no-cache
   ```

2. Check Docker resources:
   ```bash
   docker system df
   docker system prune  # if needed
   ```

3. Verify GPU access (if using CUDA):
   ```bash
   nvidia-docker run --rm nvidia/cuda:11.0-base nvidia-smi
   ```

### Debug Commands

#### Container Inspection
```bash
# Enter the container
docker compose -f docker-compose.dev.yml exec visionflow-xr bash

# Check running processes
docker compose -f docker-compose.dev.yml exec visionflow-xr ps aux

# Monitor resource usage
docker compose -f docker-compose.dev.yml exec visionflow-xr top
```

#### Log Analysis
```bash
# All service logs
docker compose -f docker-compose.dev.yml logs -f

# Specific service logs
docker compose -f docker-compose.dev.yml logs -f visionflow-xr

# Rust application logs
docker compose -f docker-compose.dev.yml exec visionflow-xr tail -f /app/logs/rust_server.log
```

#### Build Cache Management
```bash
# Clear Rust build cache
docker volume rm visionflow_cargo-target-cache

# Rebuild with clean cache
./scripts/dev.sh --no-cache
```

## Performance Optimisation

### Development Performance Tips

1. **Use Manual Rebuild**: For frequent changes, use `./scripts/dev-rebuild-rust.sh` instead of container restarts
2. **Preserve Build Cache**: Don't remove the `cargo-target-cache` volume unless necessary
3. **Incremental Changes**: Make small, focused changes for faster compilation
4. **Debug Builds**: Development uses debug builds for speed; release builds for production

### Resource Monitoring

Monitor development resource usage:
```bash
# Container resource usage
docker stats

# Disk usage
docker system df

# Volume sizes
docker volume ls
docker volume inspect visionflow_cargo-target-cache
```

## Contributing Workflow

### Before Making Changes

1. **Update your branch**:
   ```bash
   git pull origin main
   ```

2. **Start development environment**:
   ```bash
   ./scripts/dev.sh
   ```

3. **Verify everything works**:
   - Visit http://localhost:3000 for frontend
   - Check http://localhost:3666/health for backend

### Making Changes

1. **Make your changes** to the relevant source files
2. **Test immediately** using the manual rebuild script
3. **Verify functionality** in the browser/application
4. **Check logs** for any errors or warnings

### Testing Changes

1. **Run automated tests** (when available):
   ```bash
   # Rust tests
   docker compose -f docker-compose.dev.yml exec visionflow-xr cargo test

   # Frontend tests
   docker compose -f docker-compose.dev.yml exec frontend npm test
   ```

2. **Manual testing**:
   - Test the specific feature you modified
   - Ensure no regression in existing functionality
   - Test on different screen sizes/devices if relevant

### Submitting Changes

1. **Clean build test**:
   ```bash
   ./scripts/dev.sh --no-cache
   ```

2. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

3. **Create pull request** following project guidelines

## Advanced Development Setup

### Local Rust Development

For development without containers:

1. **Install Rust**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential libssl-dev pkg-config

   # macOS
   brew install openssl pkg-config
   ```

3. **Run locally**:
   ```bash
   cd /path/to/rust/code
   cargo run --features gpu
   ```

### IDE Setup

#### Visual Studio Code
Recommended extensions:
- Rust Analyzer
- Docker
- GitLens
- Thunder Client (for API testing)

#### Configuration
```json
// .vscode/settings.json
{
    "rust-analyzer.cargo.features": ["gpu"],
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

## Getting Help

### Resources

- **Project Documentation**: `/docs/`
- **API Reference**: `/docs/API_REFERENCE.md`
- **Architecture Guide**: `/docs/architecture/`

### Common Commands Reference

```bash
# Quick development start
./scripts/dev.sh

# Rust hot reload
./scripts/dev-rebuild-rust.sh

# Clean rebuild
./scripts/dev.sh --no-cache

# View logs
docker compose -f docker-compose.dev.yml logs -f

# Enter container
docker compose -f docker-compose.dev.yml exec visionflow-xr bash

# Stop all services
docker compose -f docker-compose.dev.yml down
```

### Support

If you encounter issues not covered in this guide:

1. Check existing documentation in `/docs/`
2. Search for similar issues in project logs
3. Ask for help in the project communication channels
4. Create an issue with detailed reproduction steps

---

Happy coding! This setup is designed to give you a smooth development experience with fast iteration cycles.

## Related Topics

- [Debug System Architecture](../development/debugging.md)
- [Developer Configuration System](../DEV_CONFIG.md)
- [Development Documentation](../development/index.md)
- [Getting Started with VisionFlow](../getting-started.md)
- [Testing Documentation](../development/testing.md)
- [VisionFlow Documentation](../index.md)
- [Vite Development Routing Configuration Explained](../VITE_DEV_ROUTING_EXPLAINED.md)
- [dev-backend-api](../reference/agents/development/backend/dev-backend-api.md)
- [ops-cicd-github](../reference/agents/devops/ci-cd/ops-cicd-github.md)
