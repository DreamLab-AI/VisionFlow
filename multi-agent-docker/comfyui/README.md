# ComfyUI with Open3D Support

This directory contains Docker build files for deploying ComfyUI with full `open3d` support on the `docker_ragflow` network.

## Problem

The `comfyui-sam3dobjects` extension requires `open3d>=0.18.0`, but open3d doesn't have prebuilt wheels for Python 3.13.9 (used in the base ComfyUI image).

## Solutions

### Solution 1: Stub Module (Current - Quick)

A minimal `open3d` stub module was created to satisfy imports. This allows the extension to load, and most functionality works using `trimesh` and `pyvista`.

**Status**: ✅ Currently working in the running container

**Limitations**: Some advanced open3d-specific features may not work

### Solution 2: Full Build (Recommended for Production)

Build a custom ComfyUI image with open3d compiled from source.

## Quick Start

### Using Current Container with Stub

The running `comfyui` container already has the stub module installed. To verify:

```bash
docker exec comfyui python3 -c "import open3d; print(open3d.__version__)"
# Output: 0.18.0-stub
```

The `comfyui-sam3dobjects` extension is loaded and functional.

### Building Custom Image with Full Open3D

**Note**: This takes 30-60 minutes due to open3d compilation.

```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/comfyui
./build-comfyui.sh
```

This will:
1. Build a custom ComfyUI image with open3d from source
2. Deploy it on `docker_ragflow` network
3. Persist models, custom nodes, and outputs in Docker volumes

## Files

```
comfyui/
├── Dockerfile.comfyui-open3d       # Custom Dockerfile extending base image
├── docker-compose.comfyui.yml      # Compose file with volume persistence
├── build-comfyui.sh                # Build and deploy script
└── README.md                       # This file
```

## Docker Build Details

### Base Image
- `yanwk/comfyui-boot:cu130-slim` (openSUSE Tumbleweed with Python 3.13.9)

### Build Process
1. Install build dependencies (gcc-c++, cmake, eigen3, etc.)
2. Clone Open3D from GitHub
3. Configure CMake without GUI/WebRTC/benchmarks
4. Compile with parallel jobs
5. Install to system Python

### Workarounds
- Patches assimp CMakeLists.txt to remove `-Werror` flags
- Disables GUI components (not needed for server)
- Uses fallback if build fails

## Network Configuration

The ComfyUI container joins the `docker_ragflow` network, making it accessible to other services:

- **Hostname**: `comfyui` / `comfyui.ragflow` / `comfyui.local`
- **Port**: 8188 (mapped to host)
- **Health Check**: `http://localhost:8188`

## Volume Persistence

Data is persisted in Docker volumes:

| Volume | Path | Purpose |
|--------|------|---------|
| `comfyui-models` | `/root/ComfyUI/models` | Downloaded models |
| `comfyui-custom-nodes` | `/root/ComfyUI/custom_nodes` | Custom extensions |
| `comfyui-output` | `/root/ComfyUI/output` | Generated images |
| `comfyui-input` | `/root/ComfyUI/input` | Input images |
| `comfyui-user` | `/root/ComfyUI/user` | User data & config |

## Integration with Multi-Agent-Docker

### Accessing from Agentic Workstation

The agentic workstation can access ComfyUI via the ragflow network:

```bash
# From inside agentic-workstation container
curl http://comfyui.ragflow:8188
```

### Using with Management API

The management API (port 9090) can orchestrate ComfyUI workflows:

```javascript
const response = await fetch('http://comfyui.ragflow:8188/api/prompt', {
  method: 'POST',
  body: JSON.stringify(workflow)
});
```

## Troubleshooting

### Check Container Status
```bash
docker ps | grep comfyui
docker inspect comfyui --format='{{.State.Health.Status}}'
```

### View Logs
```bash
docker logs comfyui
docker logs comfyui --tail 100 -f  # Follow logs
```

### Enter Container
```bash
docker exec -it comfyui bash
```

### Verify Open3D
```bash
docker exec comfyui python3 -c "
import open3d as o3d
print(f'Open3D version: {o3d.__version__}')
print(f'CUDA available: {o3d.core.cuda.is_available()}')
"
```

### Rebuild from Scratch
```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/comfyui
docker-compose -f docker-compose.comfyui.yml down -v  # Remove volumes
./build-comfyui.sh
```

## Performance Notes

- **Build time**: 30-60 minutes (open3d compilation)
- **Image size**: ~8-10 GB (includes open3d + dependencies)
- **RAM usage**: 8-16 GB recommended
- **GPU**: NVIDIA GPU with CUDA 13.0 support

## Alternative: Using Prebuilt Wheels

If open3d releases Python 3.13 wheels in the future, update the Dockerfile:

```dockerfile
RUN pip3 install --no-cache-dir open3d>=0.18.0
```

## References

- [Open3D Documentation](http://www.open3d.org/)
- [ComfyUI SAM3DObjects](https://github.com/your-repo/comfyui-sam3dobjects)
- [Base Image: yanwk/comfyui-boot](https://github.com/YanWenKun/ComfyUI-Docker)
