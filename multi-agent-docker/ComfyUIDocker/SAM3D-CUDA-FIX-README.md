# SAM3D CUDA Fix for ComfyUI

## Problem

The SAM3D Objects custom node for ComfyUI uses an isolated Python 3.10 virtual environment, while the main ComfyUI container uses Python 3.13. The worker subprocess that runs SAM3D inference was unable to find CUDA libraries, resulting in:

```
Invalid handle. Cannot load symbol cublasLtCreate
Worker process closed unexpectedly
```

## Solution

This fix ensures the SAM3D worker subprocess has the correct `LD_LIBRARY_PATH` pointing to CUDA libraries in the isolated venv.

### Components

1. **subprocess_bridge.py patch** - Adds `_get_worker_env()` method that:
   - Detects Python version in SAM3D venv (python3.10)
   - Builds LD_LIBRARY_PATH with all NVIDIA library locations
   - Passes environment to subprocess.Popen()

2. **Startup script** (`fix-sam3d-on-startup.sh`) - Automatically applies the patch when container starts

3. **Manual fix script** (`fix-sam3d-cuda.sh`) - For testing and manual application

### Files Modified

- `/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/nodes/subprocess_bridge.py`
  - Added `import os`
  - Added `_get_worker_env()` method
  - Modified `subprocess.Popen()` to include `env` parameter

### CUDA Libraries Included

The patch adds these NVIDIA library paths from the venv:
- nvidia-cublas
- nvidia-cuda-cupti
- nvidia-cuda-nvrtc
- nvidia-cuda-runtime
- nvidia-cudnn
- nvidia-cufft
- nvidia-curand
- nvidia-cusolver
- nvidia-cusparse
- nvidia-nccl
- nvidia-nvjitlink
- nvidia-nvtx

### Verification

After applying the fix, SAM3D workflows should complete successfully:

```bash
# Test workflow
curl -X POST http://localhost:8188/prompt \
  -H "Content-Type: application/json" \
  -d @sam3d_workflow.json

# Check logs for success
docker logs comfyui 2>&1 | grep "SAM3DObjects.*Added.*CUDA"

# Expected output:
# [SAM3DObjects] Added 12 CUDA library paths to worker environment
# [SAM3DObjects] MeshDecode completed: /root/ComfyUI/output/.../mesh.glb
```

### Build Integration

To integrate into Docker build:

```dockerfile
# Copy fix scripts
COPY scripts/fix-sam3d-cuda.sh /runner-scripts/
COPY scripts/patch-sam3d-bridge.py /runner-scripts/
COPY runner-scripts/fix-sam3d-on-startup.sh /runner-scripts/

# Make executable
RUN chmod +x /runner-scripts/*.sh

# Use enhanced entrypoint (optional)
CMD ["bash","/runner-scripts/entrypoint-with-sam3d-fix.sh"]
```

Or call the fix script from your existing entrypoint:

```bash
# In entrypoint.sh, after ComfyUI starts:
if [ -f /runner-scripts/fix-sam3d-on-startup.sh ]; then
    bash /runner-scripts/fix-sam3d-on-startup.sh
fi
```

### Performance

- **Model**: SAM3D Objects (facebook/sam-3d-objects)
- **Hardware**: NVIDIA RTX A6000 (48GB VRAM)
- **Workflow time**: ~90 seconds total
  - Depth estimation: ~5s
  - Sparse generation: ~3s
  - SLAT generation: ~60s
  - Mesh decode: ~15s
- **Output**: 13MB GLB mesh file + point cloud + metadata

### Tested On

- Base image: `yanwk/comfyui-boot:cu130-slim`
- ComfyUI version: 0.3.75
- Python: 3.13 (main) + 3.10 (SAM3D venv)
- CUDA: 13.0
- PyTorch: 2.9.1+cu130

## Credits

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- SAM3D Objects: https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects
- Fix developed for: Turbo Flow Claude multi-agent environment
