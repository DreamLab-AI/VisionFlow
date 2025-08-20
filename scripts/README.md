# VisionFlow Scripts

This directory contains launch and utility scripts for the VisionFlow application.

## Main Launch Script

### `launch.sh`
The unified launcher for all environments. Replaces all legacy launch scripts.

```bash
# Development
./launch.sh                    # Start dev environment (default)
./launch.sh -d                 # Start dev in background
./launch.sh logs              # View logs
./launch.sh shell             # Open shell in container

# Production
./launch.sh -p production      # Start production environment
./launch.sh -p prod down       # Stop production

# Maintenance
./launch.sh status            # Show container status
./launch.sh restart           # Restart environment
./launch.sh clean             # Clean all containers and volumes
```

## Supporting Scripts

### Container Scripts (Used Internally)

- **`dev-entrypoint.sh`** - Development container entrypoint, manages services via supervisord
- **`start.sh`** - Production container entrypoint
- **`check-rust-rebuild.sh`** - Checks if Rust needs rebuilding in dev
- **`dev-rebuild-rust.sh`** - Rebuilds Rust binary in dev container
- **`dev-exec-proxy.sh/`** - Directory for exec proxy functionality

### GPU/CUDA Scripts

- CUDA kernels are now compiled automatically by `build.rs` during cargo build (no separate script needed)

## Removed Legacy Scripts

The following scripts have been removed in favour of the unified `launch.sh`:

- `dev.sh` - Replaced by `launch.sh` (default profile)
- `launch-production.sh` - Replaced by `launch.sh -p production`
- `deploy-production.sh` - Replaced by `launch.sh -p production`
- `build-helper.sh` - Functionality integrated into launch.sh
- `container-helper.sh` - No longer needed
- `start-backend-with-claude-flow.sh` - Claude Flow now integrated
- `precompile-ptx.sh` - Replaced by compile_unified_ptx.sh
- `compile_ptx.sh` - Replaced by compile_unified_ptx.sh

## Environment Variables

Key variables used by the scripts:

```bash
# GPU
CUDA_ARCH=86              # GPU architecture (86 for RTX 30xx)
NVIDIA_VISIBLE_DEVICES=0  # GPU device selection

# Network
MCP_TCP_PORT=9500        # Claude Flow MCP port
SYSTEM_NETWORK_PORT=4000 # Rust backend port

# Development
VITE_DEV_SERVER_PORT=5173 # Vite dev server
VITE_HMR_PORT=5173       # Vite HMR (same as server)

# Logging
RUST_LOG=warn            # Rust log level
```

## Docker Profiles

The application uses Docker Compose profiles:

- **`dev`** - Development with hot reloading
- **`production`/`prod`** - Production optimised build

## Routing Architecture

### Development (Port 3001)
```
User Browser → Nginx:3001 → ├─ /api/* → Rust:4000
                            ├─ /ws* → Rust:4000
                            ├─ /__vite_hmr → Vite:5173
                            └─ /* → Vite:5173
```

### Production (Port 4000)
```
Cloudflare → Nginx:4000 → ├─ /api/* → Rust:3001
                         ├─ /ws* → Rust:3001
                         └─ /* → Static Files
```

## Troubleshooting

### Development Issues

1. **Port conflicts**: Check nothing else is using ports 3001, 4000, 5173
2. **Vite HMR not working**: Ensure /__vite_hmr is properly proxied
3. **GPU not detected**: Check CUDA_ARCH matches your GPU

### Production Issues

1. **Cloudflare tunnel**: Ensure CLOUDFLARE_TUNNEL_TOKEN is set
2. **WebSocket issues**: Check Cloudflare WebSocket timeout settings
3. **HTTPS redirects**: Cloudflare handles SSL termination

### Common Commands

```bash
# Check logs
docker logs visionflow_container

# Enter container
docker exec -it visionflow_container bash

# Check service status
./launch.sh status

# Full rebuild
./launch.sh -f build
```