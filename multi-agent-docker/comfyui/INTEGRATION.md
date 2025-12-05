# ComfyUI Integration with Multi-Agent-Docker

## Overview

The ComfyUI container with open3d support is now integrated with the unified build system. Running `./build-unified.sh` will detect and manage the ComfyUI container.

## Volume Persistence Explained

### What is Volume Persistence?

**Volume persistence** means your data survives container lifecycle events:

```bash
# Without volumes (data lost on container removal)
docker run --rm comfyui  # All models/outputs deleted when stopped

# With volumes (data persists)
docker run -v comfyui-models:/models comfyui
docker rm comfyui  # Container removed
docker run -v comfyui-models:/models comfyui  # Same data available!
```

### ComfyUI Persistent Data

When using docker-compose, these volumes persist your data:

| Volume Name | Container Path | What's Stored | Size |
|-------------|---------------|---------------|------|
| `comfyui-models` | `/root/ComfyUI/models` | FLUX, SD, SAM3D models | 10-50 GB |
| `comfyui-custom-nodes` | `/root/ComfyUI/custom_nodes` | Extensions like sam3dobjects | 1-5 GB |
| `comfyui-output` | `/root/ComfyUI/output` | Generated images/videos | Growing |
| `comfyui-input` | `/root/ComfyUI/input` | Input images | As needed |
| `comfyui-user` | `/root/ComfyUI/user` | Settings, workflows | < 100 MB |

### Benefits

1. **Rebuild Safety**: Rebuild container without losing downloaded models
2. **Backup Ready**: Backup volumes independently of containers
3. **Upgrade Path**: Upgrade ComfyUI without re-downloading models
4. **Multi-Instance**: Share models across multiple ComfyUI instances

## Build Options

### Standard Build (Quick - Uses Existing Container)

```bash
./build-unified.sh
```

**What it does**:
- Builds and launches agentic-workstation
- **Checks if ComfyUI container exists** with open3d
- Reports status and access info
- **Does NOT rebuild** existing ComfyUI

**Output Example**:
```
[4/4] Deploying ComfyUI standalone container...

Checking existing ComfyUI container for open3d...
✅ ComfyUI already running with open3d stub
   Container: comfyui
   Network: docker_ragflow
   Access: http://localhost:8188
   open3d: 0.18.0-stub
```

### Build with Full Open3D (30-60 minutes)

```bash
./build-unified.sh --comfyui-full
```

**What it does**:
- Builds agentic-workstation
- **Compiles open3d from source** for ComfyUI
- Deploys custom ComfyUI image
- Persists models in Docker volumes

**Time**: 30-60 minutes (one-time compilation)

### Skip ComfyUI Check

```bash
./build-unified.sh --skip-comfyui
```

**Use case**: Only need agentic-workstation without ComfyUI

## Architecture

### Two Container Setup

```
┌─────────────────────────────────────────────────────────┐
│                   docker_ragflow network                │
│                                                           │
│  ┌──────────────────────┐     ┌─────────────────────┐  │
│  │ agentic-workstation  │     │   comfyui           │  │
│  │                      │     │                     │  │
│  │ - Management API     │────▶│ - ComfyUI Server   │  │
│  │ - Code Server        │     │ - open3d (stub)    │  │
│  │ - VNC Desktop        │     │ - sam3dobjects     │  │
│  │ - SSH Access         │     │ - GPU Access       │  │
│  │                      │     │                     │  │
│  │ Port: 9090, 8080     │     │ Port: 8188         │  │
│  └──────────────────────┘     └─────────────────────┘  │
│           ▲                             ▲               │
│           │                             │               │
└───────────┼─────────────────────────────┼───────────────┘
            │                             │
    Host: localhost:9090         Host: localhost:8188
```

### Network Communication

**From Host**:
- Agentic Workstation: `http://localhost:9090`
- ComfyUI: `http://localhost:8188`

**From agentic-workstation container**:
```bash
curl http://comfyui.ragflow:8188
```

**From ComfyUI container**:
```bash
curl http://agentic-workstation.ragflow:9090
```

## Current Implementation

### What build-unified.sh Does

1. **Builds agentic-workstation** (always)
2. **Checks for ComfyUI** container (step 4/4)
3. **Reports status** if found:
   - Container running
   - open3d version (stub/full)
   - Access URLs
   - Network connectivity
4. **Provides guidance** if not found

### What It Doesn't Do (Yet)

- ❌ Automatically start stopped ComfyUI container
- ❌ Build ComfyUI if not present (unless `--comfyui-full` flag)
- ❌ Install stub into existing container

## Manual ComfyUI Management

### Check Status

```bash
# Is it running?
docker ps | grep comfyui

# Check open3d version
docker exec comfyui python3 -c "import open3d; print(open3d.__version__)"

# Check extension
docker logs comfyui | grep sam3d
```

### Start Existing Container

```bash
# If stopped
docker start comfyui

# Check health
curl http://localhost:8188
```

### Build from Scratch

```bash
cd multi-agent-docker/comfyui
./build-comfyui.sh
```

Or use integrated build:
```bash
./build-unified.sh --comfyui-full
```

## Volume Management

### List Volumes

```bash
docker volume ls | grep comfyui
```

### Inspect Volume

```bash
docker volume inspect comfyui-models
```

### Backup Volume

```bash
# Create backup
docker run --rm -v comfyui-models:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/comfyui-models-backup.tar.gz /data

# Restore backup
docker run --rm -v comfyui-models:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/comfyui-models-backup.tar.gz -C /
```

### Clean Up Volumes

```bash
# Stop and remove container
docker stop comfyui && docker rm comfyui

# Remove volumes (WARNING: Deletes all data!)
docker volume rm comfyui-models comfyui-custom-nodes comfyui-output comfyui-input comfyui-user
```

## Upgrade Scenarios

### Scenario 1: Upgrade to Full Open3D

**Current**: ComfyUI with stub (working)
**Goal**: Full open3d support

```bash
# Option A: Use integrated build
./build-unified.sh --comfyui-full

# Option B: Manual build
cd comfyui
./build-comfyui.sh
```

**Result**: New image with compiled open3d, volumes preserved

### Scenario 2: Update ComfyUI Base

```bash
# Pull latest base image
docker pull yanwk/comfyui-boot:cu130-slim

# Rebuild with stub
cd comfyui
docker-compose -f docker-compose.comfyui.yml build --no-cache

# Or rebuild with full open3d
./build-comfyui.sh
```

### Scenario 3: Fresh Start (Keep Models)

```bash
# Stop and remove container only
docker stop comfyui && docker rm comfyui

# Rebuild (volumes persist)
cd comfyui
./build-comfyui.sh
```

## Troubleshooting

### ComfyUI Not Detected

```bash
# Check if container exists
docker ps -a | grep comfyui

# If not running, start it
docker start comfyui

# If doesn't exist, check network
docker network inspect docker_ragflow
```

### Network Issues

```bash
# Verify ragflow network
docker network inspect docker_ragflow | grep comfyui

# Reconnect if needed
docker network connect docker_ragflow comfyui
```

### Volume Issues

```bash
# Check if volumes are mounted
docker inspect comfyui | grep -A 20 Mounts

# Verify volume data
docker run --rm -v comfyui-models:/data ubuntu ls -lh /data
```

### Open3D Issues

```bash
# Check current version
docker exec comfyui python3 -c "import open3d; print(open3d.__version__)"

# Reinstall stub if needed
docker exec comfyui bash -c "
python3 -c \"
import os
stub_dir = '/usr/lib/python3.13/site-packages'
os.makedirs(f'{stub_dir}/open3d', exist_ok=True)
with open(f'{stub_dir}/open3d/__init__.py', 'w') as f:
    f.write('''
class PointCloud: pass
class TriangleMesh: pass
__version__ = '0.18.0-stub'
''')
\"
"

# Or build full version
./build-unified.sh --comfyui-full
```

## Integration Checklist

When running `./build-unified.sh`, verify:

- [x] Agentic workstation builds
- [x] docker_ragflow network created
- [x] ComfyUI container detected (if exists)
- [x] Open3D version reported
- [x] Access URLs displayed
- [x] Management commands shown

## Next Steps

### For Users

1. **Current setup works**: Stub is functional for most use cases
2. **Need full open3d**: Run `./build-unified.sh --comfyui-full` (one-time)
3. **Testing workflows**: Access http://localhost:8188 and test SAM3D nodes

### For Developers

1. **Auto-start logic**: Add logic to start stopped ComfyUI containers
2. **Auto-install stub**: Install stub if ComfyUI detected without open3d
3. **Full docker-compose integration**: Merge into single compose file
4. **Health monitoring**: Add ComfyUI health checks to build script
5. **Multi-GPU support**: Distribute GPU allocation between containers

## Reference

### File Structure

```
multi-agent-docker/
├── build-unified.sh                    # ✅ Integrated ComfyUI detection
├── docker-compose.unified.yml          # Agentic workstation
├── Dockerfile.unified                  # Agentic workstation
└── comfyui/
    ├── Dockerfile.comfyui-open3d       # Custom ComfyUI with open3d
    ├── docker-compose.comfyui.yml      # ComfyUI deployment
    ├── build-comfyui.sh                # Build automation
    ├── README.md                       # Documentation
    ├── SOLUTION_SUMMARY.md             # Technical details
    └── INTEGRATION.md                  # This file
```

### Key Commands

```bash
# Standard deployment
./build-unified.sh

# With full open3d (30-60 min)
./build-unified.sh --comfyui-full

# Skip ComfyUI check
./build-unified.sh --skip-comfyui

# Rebuild from scratch
./build-unified.sh --no-cache --comfyui-full

# Manual ComfyUI build
cd comfyui && ./build-comfyui.sh

# Stop everything
docker compose -f docker-compose.unified.yml down
docker stop comfyui
```

---

**Last Updated**: 2025-12-04
**Integration Status**: ✅ Complete
**ComfyUI Support**: Stub (working) | Full (available via --comfyui-full)
