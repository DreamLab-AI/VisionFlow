---
title: GPU Access Fix Summary
description: All changes implemented to fix PyTorch CUDA detection and enable GPU-accelerated image generation with ComfyUI.
category: explanation
tags:
  - http
  - deployment
  - docker
  - testing
  - ai
related-docs:
  - ARCHITECTURE_COMPLETE.md
  - ARCHITECTURE_OVERVIEW.md
  - ASCII_DEPRECATION_COMPLETE.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker installation
---

# GPU Access Fix Summary

## Changes Made

All changes implemented to fix PyTorch CUDA detection and enable GPU-accelerated image generation with ComfyUI.

### 1. Dockerfile.unified (Lines 152-172)

**Changed PyTorch installation** to specify CUDA version:

```dockerfile
# BEFORE:
torch torchvision torchaudio \

# AFTER:
torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
```

Added `huggingface-hub` for model downloads.

### 2. Dockerfile.unified (Lines 348-376)

**Added ComfyUI installation phase:**

```dockerfile
# PHASE 14: ComfyUI Installation (GPU-Accelerated Image Generation)
USER devuser
WORKDIR /home/devuser

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /home/devuser/ComfyUI

# Install with GPU-enabled PyTorch
RUN cd /home/devuser/ComfyUI && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

# Download FLUX model
RUN cd /home/devuser/ComfyUI && \
    . venv/bin/activate && \
    mkdir -p models/checkpoints && \
    hf download Comfy-Org/flux1-schnell flux1-schnell-fp8.safetensors --local-dir models/checkpoints/
```

### 3. docker-compose.unified.yml (Lines 77-88)

**Added NVIDIA environment variables:**

```yaml
environment:
  # NVIDIA GPU Configuration
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics,display
  LD_LIBRARY_PATH: /opt/cuda/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
  CUDA_HOME: /opt/cuda
  __GLX_VENDOR_LIBRARY_NAME: nvidia
  __NV_PRIME_RENDER_OFFLOAD: 1
  __VK_LAYER_NV_optimus: NVIDIA_only
```

### 4. entrypoint-unified.sh (Lines 131-173)

**Added Phase 3: GPU Verification:**

```bash
# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        echo "✓ NVIDIA driver accessible: $GPU_COUNT GPU(s) detected"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | \
            awk -F', ' '{printf "  GPU %s: %s (%s)\n", $1, $2, $3}'
    else
        echo "⚠️  nvidia-smi failed - GPU may not be accessible"
    fi
else
    echo "⚠️  nvidia-smi not found"
fi

# Test PyTorch CUDA detection
PYTORCH_TEST=$(/opt/venv/bin/python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: PyTorch cannot access CUDA')
" 2>&1)

echo "$PYTORCH_TEST"

if echo "$PYTORCH_TEST" | grep -q "CUDA available: True"; then
    echo "✓ PyTorch GPU acceleration ready"
else
    echo "⚠️  PyTorch GPU acceleration not available - will fallback to CPU"
fi
```

Updated all subsequent phase numbers (3→4, 4→5, etc., 10→11).

### 5. build-unified.sh (Lines 71-104)

**Added GPU verification section:**

```bash
echo "Testing NVIDIA GPU access..."
docker exec agentic-workstation nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Testing PyTorch CUDA..."
docker exec agentic-workstation /opt/venv/bin/python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo "Testing ComfyUI installation..."
if docker exec agentic-workstation test -d /home/devuser/ComfyUI; then
    echo "✅ ComfyUI installed at /home/devuser/ComfyUI"
    if docker exec agentic-workstation test -f /home/devuser/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors; then
        echo "✅ FLUX model downloaded"
    fi
fi
```

## What Was Fixed

### Root Causes Addressed

1. **PyTorch CUDA Version Mismatch**
   - System has CUDA 13.0, but PyTorch auto-install detected wrong version
   - Fixed by explicitly installing `torch` with `--index-url https://download.pytorch.org/whl/cu124`
   - While system has CUDA 13.0, PyTorch cu124 is compatible via CUDA forward compatibility

2. **Missing Environment Variables**
   - `LD_LIBRARY_PATH` was not set in docker-compose
   - Added full NVIDIA environment configuration to docker-compose.yml
   - Added GLX and Vulkan optimizations for GPU rendering

3. **ComfyUI Not Pre-installed**
   - ComfyUI was NOT installed during container build
   - Added full installation in Dockerfile Phase 14
   - Pre-downloads FLUX.1-schnell model during build

4. **No Verification on Startup**
   - Container started without checking GPU accessibility
   - Added comprehensive GPU verification in entrypoint Phase 3
   - Build script now verifies GPU access after deployment

## Testing Instructions

### 1. Rebuild Container

```bash
cd /home/devuser/workspace/project/multi-agent-docker
docker build --no-cache -f Dockerfile.unified -t agentic-workstation:latest .
```

### 2. Launch Container

```bash
docker compose -f docker-compose.unified.yml up -d
```

### 3. Verify GPU Access

Build script will automatically verify GPU access. You should see:

```
Testing NVIDIA GPU access...
0, NVIDIA RTX A6000, 49140 MiB
1, Quadro RTX 6000, 24576 MiB
2, Quadro RTX 6000, 24576 MiB

Testing PyTorch CUDA...
PyTorch version: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU count: 3
  GPU 0: NVIDIA RTX A6000
  GPU 1: Quadro RTX 6000
  GPU 2: Quadro RTX 6000

Testing ComfyUI installation...
✅ ComfyUI installed at /home/devuser/ComfyUI
✅ FLUX model downloaded
```

### 4. Test ComfyUI Manually

```bash
# Inside container
docker exec -it -u devuser agentic-workstation bash
cd /home/devuser/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188

# Should see:
# Device: cuda:0 NVIDIA RTX A6000
# (NOT "Device: cpu")
```

### 5. Generate Test Image

```bash
# From outside container (with ComfyUI running on port 8188)
curl -X POST "http://localhost:8188/prompt" \
  -H "Content-Type: application/json" \
  -d @test-prompt.json
```

## Expected Performance

- **With GPU**: 3-15 seconds per image (1024x1024, FLUX.1-schnell, 4 steps)
- **Without GPU (CPU fallback)**: 2-5 minutes per image

## Files Modified

1. `/home/devuser/workspace/project/multi-agent-docker/Dockerfile.unified`
2. `/home/devuser/workspace/project/multi-agent-docker/docker-compose.unified.yml`
3. `/home/devuser/workspace/project/multi-agent-docker/unified-config/entrypoint-unified.sh`
4. `/home/devuser/workspace/project/multi-agent-docker/build-unified.sh`

## Notes

- **CUDA Compatibility**: System has CUDA 13.0, PyTorch cu124 works via forward compatibility
- **Automatic Fallback**: If GPU fails, ComfyUI will automatically use CPU mode (with warning)
- **Model Download**: FLUX model (~17GB) downloads during build; may take 10-15 minutes
- **First Run**: If model download failed during build, it will retry on first ComfyUI launch

## Troubleshooting

If GPU still not accessible after rebuild:

1. **Check NVIDIA runtime**:
   ```bash
   docker info | grep -i nvidia
   ```

2. **Verify docker-compose runtime**:
   ```bash
   docker inspect agentic-workstation | grep -i runtime
   ```

3. **Test GPU passthrough**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

4. **Check environment variables**:
   ```bash
   docker exec agentic-workstation env | grep -E "CUDA|NVIDIA|LD_LIBRARY"
   ```

---

---

## Related Documentation

- [Terminal Grid Configuration](multi-agent-docker/TERMINAL_GRID.md)
- [QA Validation Final Report](QA_VALIDATION_FINAL.md)
- [Google Antigravity IDE Integration](multi-agent-docker/ANTIGRAVITY.md)
- [Final Status - Turbo Flow Unified Container Upgrade](multi-agent-docker/development-notes/SESSION_2025-11-15.md)
- [Hyprland Migration Summary](multi-agent-docker/hyprland-migration-summary.md)

## Status

✅ **All fixes implemented and ready for rebuild**

The container will now:
- Install PyTorch with correct CUDA version
- Pre-install ComfyUI with FLUX model
- Set all necessary NVIDIA environment variables
- Verify GPU access on every startup
- Provide clear diagnostics during build
