# ComfyUI SAM3D Docker Setup

## Overview

ComfyUI container with SAM3D (Segment Anything 3D) nodes pre-configured for 3D object generation from images.

**Location**: `multi-agent-docker/comfyui/`

## Quick Start

```bash
cd multi-agent-docker/comfyui
docker compose -f docker-compose.comfyui.yml up -d
```

**Web UI**: http://localhost:8188

## Architecture

- **Base image**: `yanwk/comfyui-boot:cu130-slim` (Python 3.13, PyTorch 2.9.1+cu130)
- **SAM3D venv**: Isolated Python 3.10 environment with PyTorch 2.4.1+cu124
- **GPU**: NVIDIA RTX A6000 (48GB VRAM)

### Why Isolated Environment?

SAM3D requires PyTorch 2.4.x with specific CUDA libraries. The main ComfyUI runs on Python 3.13/PyTorch 2.9.1. The isolated venv prevents conflicts.

## Build Issues & Fixes (Dec 2025)

### 1. PyTorch3D wheel 403 error

**Problem**: Official Facebook PyTorch3D wheels at `dl.fbaipublicfiles.com` return 403 Forbidden.

**Fix**: Use third-party prebuilt wheels from miropsota:
```dockerfile
RUN uv pip install --python ./_env/bin/python \
    pytorch3d==0.7.9+pt2.4.1cu124 \
    --extra-index-url https://miropsota.github.io/torch_packages_builder
```

### 2. segment-anything-2 package name

**Problem**: Package is `sam2`, not `segment-anything-2`.

**Fix**: `pip install sam2`

### 3. sam2 overwrites torch version

**Problem**: sam2 pulls torch 2.9.1, breaking pytorch3d compatibility.

**Fix**: Reinstall pinned torch after sam2:
```dockerfile
RUN uv pip install --python ./_env/bin/python \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124 --reinstall
```

### 4. CUDA library path for subprocess

**Problem**: SAM3D worker subprocess can't find CUDA libs (`cublasLtCreate` error).

**Fix**: Patch `subprocess_bridge.py` to set `LD_LIBRARY_PATH` with nvidia lib paths from the venv. Applied automatically at build time.

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./storage-models` | `/root/ComfyUI/models` | Model weights |
| `./storage-output` | `/root/ComfyUI/output` | Generated files |
| `./storage-input` | `/root/ComfyUI/input` | Input images |
| `comfyui-custom-nodes` (volume) | `/root/ComfyUI/custom_nodes` | Extensions |

## SAM3D Models

Located at `./storage-models/sam3d/hf/checkpoints/`:
- `ss_generator.ckpt` (6.3GB) - Sparse structure generator
- `slat_generator.ckpt` (4.6GB) - SLAT latent generator
- `slat_decoder_mesh.ckpt` (347MB) - Mesh decoder
- `slat_decoder_gs.ckpt` (164MB) - Gaussian splatting decoder
- `pipeline.yaml` - Pipeline config

## SAM3D Nodes

**Pipeline** (in order):
1. `LoadSAM3DModel` - Load model (tag: `hf`)
2. `SAM3D_DepthEstimate` - Depth estimation (~2s)
3. `SAM3DSparseGen` - Sparse structure (~3s)
4. `SAM3DSLATGen` - SLAT generation (~60s)
5. `SAM3DGaussianDecode` or `SAM3DMeshDecode` (~15s)
6. `SAM3DTextureBake` - Optional texture baking (~30-60s)

**Export**: `SAM3DExportPLY`, `SAM3DExportMesh`

## Troubleshooting

### Container keeps restarting
Check for conflicting custom nodes in the volume:
```bash
docker volume rm comfyui_comfyui-custom-nodes
docker compose -f docker-compose.comfyui.yml up -d
```

### VRAM issues
Set `use_gpu_cache: False` in LoadSAM3DModel node.

### Check logs
```bash
docker logs comfyui --tail 100
```

## Network

- Joins `docker_ragflow` network
- Aliases: `comfyui.ragflow`, `comfyui.local`
