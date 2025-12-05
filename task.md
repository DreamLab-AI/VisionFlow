# GPU Access Issue in Docker Container

## Problem Summary
ComfyUI and PyTorch cannot access NVIDIA GPUs despite GPUs being visible to nvidia-smi and device files existing in /dev/.

## Environment Details

### Host GPU Hardware
```
nvidia-smi output shows:
- GPU 0: NVIDIA RTX A6000 (49140 MiB VRAM)
- GPU 1: Quadro RTX 6000 (24576 MiB VRAM)
- GPU 2: Quadro RTX 6000 (24576 MiB VRAM)
- Driver Version: 580.105.08
- CUDA Version: 13.0
```

### Container Environment
- Hostname: `agentic-workstation`
- OS: Linux (CachyOS/Arch Linux base)
- Python: 3.13.7
- CUDA Toolkit: /opt/cuda (CUDA 13.0)
- CUDA Device Files: Present in /dev/ (nvidia0, nvidia1, nvidia2, nvidiactl, nvidia-uvm, etc.)

### PyTorch Detection Failure
```python
import torch
torch.cuda.is_available()  # Returns: False
torch.cuda.device_count()  # Returns: 0
```

**Installed PyTorch:**
- Version: 2.6.0+cu124
- CUDA version: 12.4
- Installation method: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

**Original PyTorch (also failed):**
- Version: 2.7.1+cu118
- CUDA version: 11.8

### CUDA Environment Variables
```bash
echo $LD_LIBRARY_PATH    # Empty/not set
echo $CUDA_VISIBLE_DEVICES  # Empty/not set
echo $CUDA_HOME          # /opt/cuda
```

### CUDA Libraries Present
- Location: `/opt/cuda/lib64/`
- Contains: libcublas, libcudart, libcudnn, etc.
- Version: CUDA 13.0 (based on library versions like libcudart.so.13.0.96)

## Root Cause Analysis

### Issue 1: PyTorch CUDA Version Mismatch
- System has CUDA 13.0
- PyTorch built for CUDA 11.8 (original) or CUDA 12.4 (reinstalled)
- PyTorch cu124 doesn't detect CUDA 13.0 runtime

### Issue 2: Missing LD_LIBRARY_PATH
- Environment variable `LD_LIBRARY_PATH` not set
- PyTorch cannot find CUDA libraries in `/opt/cuda/lib64/`
- Attempted fix: `export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH` - still failed

### Issue 3: Possible NVIDIA Container Toolkit Issues
- Docker may not have `--gpus all` or `--runtime=nvidia` configuration
- NVIDIA Container Runtime may not be properly configured
- Container may not have proper device permissions despite /dev/nvidia* existing

### Issue 4: PyTorch Binary Compatibility
- PyTorch pre-built binaries may not be compatible with CUDA 13.0
- May need to build PyTorch from source for CUDA 13.0 support

## Impact

### Current Workaround
- ComfyUI runs in CPU mode: `python main.py --listen 0.0.0.0 --port 8188 --cpu`
- Image generation works but extremely slow (2-5 minutes per image vs 3-15 seconds on GPU)

### What Doesn't Work
- GPU-accelerated image generation with FLUX models
- Any CUDA-dependent deep learning operations
- Fast inference for ComfyUI workflows

## Required Fixes for Docker Build

### Fix 1: Install PyTorch with Correct CUDA Version
**Location:** `/home/devuser/workspace/project/multi-agent-docker/Dockerfile.unified`

Need to either:
- **Option A:** Install PyTorch nightly build with CUDA 13.x support
- **Option B:** Build PyTorch from source for CUDA 13.0
- **Option C:** Downgrade CUDA in container to 12.4 and match PyTorch cu124

### Fix 2: Set Environment Variables in Dockerfile
```dockerfile
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/opt/cuda
ENV PATH=/opt/cuda/bin:$PATH
```

### Fix 3: Ensure NVIDIA Runtime Configuration
**Location:** `/home/devuser/workspace/project/multi-agent-docker/docker-compose.unified.yml`

Verify docker-compose includes:
```yaml
services:
  agentic-workstation:
    runtime: nvidia
    # OR
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Fix 4: Pre-install ComfyUI with GPU Support
**Location:** `/home/devuser/workspace/project/multi-agent-docker/Dockerfile.unified`

Add to Dockerfile:
```dockerfile
# Install ComfyUI with GPU support
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /home/devuser/ComfyUI && \
    cd /home/devuser/ComfyUI && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt && \
    chown -R devuser:devuser /home/devuser/ComfyUI

# Download FLUX model
RUN cd /home/devuser/ComfyUI && \
    . venv/bin/activate && \
    mkdir -p models/checkpoints && \
    hf download Comfy-Org/flux1-schnell flux1-schnell-fp8.safetensors --local-dir models/checkpoints/
```

### Fix 5: Verify GPU Access on Container Startup
**Location:** `/home/devuser/workspace/project/multi-agent-docker/unified-config/entrypoint-unified.sh`

Add GPU verification:
```bash
# Verify GPU access
echo "=== GPU Verification ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"

# Test PyTorch CUDA
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || echo "WARNING: PyTorch CUDA test failed"
```

## Testing Steps

After implementing fixes:

1. **Rebuild container:**
   ```bash
   cd /home/devuser/workspace/project/multi-agent-docker
   docker build -f Dockerfile.unified -t agentic-workstation:latest .
   ```

2. **Verify GPU access:**
   ```bash
   docker run --gpus all -it agentic-workstation:latest nvidia-smi
   ```

3. **Test PyTorch CUDA:**
   ```bash
   docker run --gpus all -it agentic-workstation:latest \
     python3 -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Test ComfyUI:**
   ```bash
   # Should NOT require --cpu flag
   cd /home/devuser/ComfyUI
   source venv/bin/activate
   python main.py --listen 0.0.0.0 --port 8188
   # Check logs for "Device: cuda" instead of "Device: cpu"
   ```

## Files to Modify

1. `/home/devuser/workspace/project/multi-agent-docker/Dockerfile.unified`
2. `/home/devuser/workspace/project/multi-agent-docker/docker-compose.unified.yml`
3. `/home/devuser/workspace/project/multi-agent-docker/unified-config/entrypoint-unified.sh`
4. `/home/devuser/workspace/project/multi-agent-docker/build-unified.sh` (if needed)

## Priority
**HIGH** - GPU acceleration is essential for practical ComfyUI usage and other AI workloads.

## Status
- [x] Problem identified
- [x] Root causes analyzed
- [ ] Docker build files updated
- [ ] Container rebuilt
- [ ] GPU access verified
- [ ] ComfyUI tested with GPU

## Notes
- The first cat image generated in CPU mode took ~3 minutes
- With GPU this should take 3-15 seconds
- This affects all AI workloads in the container, not just ComfyUI
