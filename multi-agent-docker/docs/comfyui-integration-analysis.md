# ComfyUI Integration Analysis for Multi-Agent-Docker

**Analysis Date**: 2025-11-30
**Target System**: Arch Linux (CachyOS) Multi-Agent Workstation
**Source System**: openSUSE Tumbleweed ComfyUI Docker
**ComfyUI Version**: Latest stable (tagged release)
**CUDA Version**: 13.0

---

## Executive Summary

This document provides a comprehensive analysis for integrating ComfyUI (Stable Diffusion WebUI) into the existing multi-agent-docker Arch Linux workstation. The source ComfyUI Docker implementation uses openSUSE Tumbleweed with zypper package management, requiring systematic translation to Arch Linux pacman equivalents.

**Key Findings**:
- ComfyUI requires **Python 3.13** with PyTorch and CUDA 13.0
- **150+ Python packages** needed across 3 tiers (pak3, pak5, pak7)
- Port **8188** required for ComfyUI web interface
- **No major conflicts** with existing services
- CUDA environment already configured in base system
- Primary challenge: openSUSE → Arch Linux package translation

---

## 1. System Requirements Analysis

### 1.1 Python Environment

**ComfyUI Requirements**:
- Python 3.13 (with GIL)
- Python development headers
- Pip, wheel, setuptools
- Cython for compilation

**Current System** (Dockerfile.unified):
```dockerfile
# PHASE 1: Base packages
python python-pip rustup
```

**Gap Analysis**:
- Current: Python 3.x (default Arch version, likely 3.12.x)
- Required: Python 3.13 specifically
- **Action Required**: Install Python 3.13 from AUR or build from source

### 1.2 CUDA and GPU Support

**ComfyUI CUDA Stack**:
- CUDA Toolkit 13.0
- cuDNN (CUDA Deep Neural Network library)
- NVIDIA GPU drivers
- Multiple nvidia-* PyTorch packages

**Current System** (Dockerfile.unified lines 68-89):
```dockerfile
# PHASE 2: CUDA Development Environment
RUN pacman -S --noconfirm \
        cuda cuda-tools \
        cudnn \
        nvidia-utils \
        libglvnd \
    && rm -rf /var/cache/pacman/pkg/*

ENV CUDA_HOME=/opt/cuda
ENV PATH=/opt/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH}
```

**Compatibility Assessment**:
- ✅ CUDA toolkit installed
- ✅ cuDNN present
- ✅ NVIDIA utilities configured
- ✅ LD_LIBRARY_PATH configured
- ⚠️ Need to verify CUDA version matches 13.0

---

## 2. Package Translation: openSUSE → Arch Linux

### 2.1 Core Development Tools

| openSUSE Package | Arch Linux Equivalent | Status | Notes |
|------------------|----------------------|--------|-------|
| `python313-devel` | `python` (3.x headers included) | ✅ Installed | Arch bundles headers |
| `python313-pip` | `python-pip` | ✅ Installed | - |
| `python313-wheel` | `python-wheel` | Need install | - |
| `python313-setuptools` | `python-setuptools` | Need install | - |
| `python313-Cython` | `cython` | Need install | - |
| `gcc15` | `gcc` (latest) | ✅ Installed | base-devel |
| `gcc15-c++` | `g++` | ✅ Installed | base-devel |
| `make` | `make` | ✅ Installed | base-devel |
| `ninja` | `ninja` | Need install | Build system |
| `aria2` | `aria2` | Need install | Download accelerator |

### 2.2 Computer Vision Libraries

| openSUSE Package | Arch Linux Equivalent | Status | Notes |
|------------------|----------------------|--------|-------|
| `python313-opencv` | `python-opencv` | Available | Extra repo |
| `opencv` | `opencv` | Available | Extra repo |
| `opencv-devel` | `opencv` (headers included) | Available | Bundled |
| `Mesa-libGL1` | `mesa` | ✅ Installed | - |
| `Mesa-libEGL-devel` | `mesa` | ✅ Installed | - |
| `libgthread-2_0-0` | `glib2` | ✅ Installed | Core lib |
| `libQt5OpenGL5` | `qt5-base` | Available | - |

### 2.3 Media Processing

| openSUSE Package | Arch Linux Equivalent | Status | Notes |
|------------------|----------------------|--------|-------|
| `python313-ffmpeg-python` | `python-ffmpeg-python` | Available (AUR) | - |
| `python313-imageio` | `python-imageio` | Available (pip) | - |
| `ffmpeg` | `ffmpeg` | ✅ Installed | - |
| `x264` | `x264` | Available | Video codec |
| `x265` | `x265` | Available | HEVC codec |

### 2.4 Git and Version Control

| openSUSE Package | Arch Linux Equivalent | Status | Notes |
|------------------|----------------------|--------|-------|
| `python313-GitPython` | `python-gitpython` | Available (pip) | - |
| `python313-pygit2` | `python-pygit2` | Available | - |
| `git` | `git` | ✅ Installed | - |

### 2.5 Scientific Computing (openSUSE system packages)

| openSUSE Package | Arch Linux Approach | Notes |
|------------------|---------------------|-------|
| `python313-matplotlib` | Install via pip | Arch prefers pip for Python packages |
| `python313-numba-devel` | Install via pip | Same |
| `python313-pandas` | Install via pip | Same |
| `python313-scikit-image` | Install via pip | Same |
| `python313-scikit-learn` | Install via pip | Same |
| `python313-scipy` | Install via pip | Same |

**Rationale**:
- openSUSE uses system packages for Python libraries (verified by openSUSE repos)
- Arch Linux philosophy: system packages for core tools, pip for Python libraries
- Better compatibility with PyTorch from pip

### 2.6 Utilities and Tools

| openSUSE Package | Arch Linux Equivalent | Status | Notes |
|------------------|----------------------|--------|-------|
| `bison` | `bison` | Available | Parser generator |
| `gawk` | `gawk` | ✅ Installed | GNU awk |
| `ninja` | `ninja` | Available | Build system |
| `aria2` | `aria2` | Available | Download manager |
| `findutils` | `findutils` | ✅ Installed | Core utils |
| `fish` | `fish` | Available | Shell |
| `fd` | `fd` | ✅ Installed | Find alternative |
| `fuse` | `fuse3` | Available | Filesystem |
| `vim` | `vim` | ✅ Installed | - |
| `which` | `which` | ✅ Installed | - |

---

## 3. Python Dependencies Analysis

### 3.1 PyTorch Installation Strategy

**ComfyUI Approach** (Dockerfile lines 119-180):
1. Dry-run pip install to extract package list
2. Install nvidia-* packages sequentially (5 at a time)
3. Install torch, torchvision, torchaudio separately
4. Install triton separately
5. Configure LD_LIBRARY_PATH for .so files

**Package Index**: `https://download.pytorch.org/whl/cu130`

**Key nvidia-* Packages** (from dry-run output):
- nvidia-cublas-cu13
- nvidia-cudnn-cu13
- nvidia-cuda-runtime-cu13
- nvidia-cuda-nvrtc-cu13
- nvidia-cufft-cu13
- nvidia-curand-cu13
- nvidia-cusolver-cu13
- nvidia-cusparse-cu13
- nvidia-nccl-cu13
- nvidia-nvtx-cu13

**LD_LIBRARY_PATH Configuration**:
```bash
/usr/local/lib64/python3.13/site-packages/torch/lib
/usr/local/lib/python3.13/site-packages/cusparselt/lib
/usr/local/lib/python3.13/site-packages/nvidia/*/lib
```

### 3.2 ComfyUI Dependencies (pak3.txt - Essentials)

**39 Core Packages**:
```
accelerate, diffusers, ftfy, huggingface-hub
imageio[ffmpeg], joblib, kornia, matplotlib
nvidia-ml-py, omegaconf, onnx, onnxruntime-gpu
opencv-contrib-python-headless, pandas
pilgram, pillow, pygit2, python-ffmpeg
regex, scikit-build-core, scikit-image
scikit-learn, scipy, timm, torchmetrics
transformers

# Hand-picks
compel, lark                    # For smZNodes
cupy-cuda13x                    # Frame Interpolation
fairscale                       # APISR
torchdiffeq                     # DepthFM
```

**Installation Method**: `pip install -r pak3.txt`

### 3.3 Extended Dependencies (pak5.txt - 67 packages)

**Categories**:
- **Data Processing**: addict, aenum, dill, numba, numexpr
- **Web/Network**: aiohttp, requests, httpx
- **Image Processing**: albumentations, blend-modes, color-matcher, pillow
- **ML/AI**: einops, peft, segment-anything, spandrel, ultralytics
- **Git Integration**: GitPython, PyGithub
- **File Handling**: filelock, protobuf, pyyaml, safetensors, toml
- **Database**: SQLAlchemy, alembic
- **CLI/Logging**: rich, rich-argparse, loguru, typer, tqdm
- **QR Codes**: qrcode[pil]
- **Background Removal**: rembg, transparent-background
- **3D Processing**: trimesh[easy]
- **Package Management**: uv (fast pip alternative)
- **Code Quality**: black, yapf

**Installation Method**: `pip install -r pak5.txt`

### 3.4 Special Git Packages (pak7.txt - 11 packages)

**Direct from Git Repositories**:
```
dlib
facexlib
insightface
git+https://github.com/openai/CLIP.git
git+https://github.com/cozy-comfyui/cozy_comfyui@main#egg=cozy_comfyui
git+https://github.com/cozy-comfyui/cozy_comfy@main#egg=cozy_comfy
git+https://github.com/facebookresearch/sam2.git
git+https://github.com/ltdrdata/cstr.git
git+https://github.com/ltdrdata/ffmpy.git
git+https://github.com/ltdrdata/img2texture.git
```

**Special Requirements**:
- `dlib`: Requires CMake and build tools
- `facexlib`: Face recognition models
- `insightface`: Advanced face analysis
- `CLIP`: OpenAI's image-text model
- `SAM2`: Segment Anything Model 2

**Installation Method**: `pip install -r pak7.txt`

---

## 4. ComfyUI Application Structure

### 4.1 Directory Layout

**Pre-loaded Bundle** (Dockerfile line 227):
```
/default-comfyui-bundle/
├── ComfyUI/
│   ├── main.py                    # Application entry point
│   ├── requirements.txt           # ComfyUI core deps
│   ├── custom_nodes/
│   │   └── ComfyUI-Manager/
│   │       └── requirements.txt   # Manager deps
│   └── models/
│       └── vae/                   # VAE models (preloaded)
```

**Runtime Location** (entrypoint.sh copies to):
```
/root/ComfyUI/
├── main.py
├── input/                         # User inputs
├── output/                        # Generated outputs
├── user/
│   └── default/
│       ├── workflows/             # User workflows
│       └── ComfyUI-Manager/
│           └── config.ini         # Force PIP (not UV)
```

### 4.2 Volume Mounts

**docker-compose.yml Configuration**:
```yaml
volumes:
  - "./storage:/root"                                 # Persistent ComfyUI data
  - "./storage-models/models:/root/ComfyUI/models"    # Model storage
  - "./storage-models/hf-hub:/root/.cache/huggingface/hub"
  - "./storage-models/torch-hub:/root/.cache/torch/hub"
  - "./storage-user/input:/root/ComfyUI/input"
  - "./storage-user/output:/root/ComfyUI/output"
  - "./storage-user/workflows:/root/ComfyUI/user/default/workflows"
```

**Recommended Mapping for Multi-Agent-Docker**:
```yaml
volumes:
  - "/home/devuser/models/comfyui:/opt/comfyui"                    # Application
  - "/home/devuser/models/comfyui-models:/opt/comfyui/models"      # Models (large)
  - "/home/devuser/.cache/huggingface:/root/.cache/huggingface"    # HF cache
  - "/home/devuser/.cache/torch:/root/.cache/torch"                # Torch cache
  - "/home/devuser/workspace/comfyui-input:/opt/comfyui/input"
  - "/home/devuser/workspace/comfyui-output:/opt/comfyui/output"
```

### 4.3 Preloaded Models

**VAE Decoders** (preload-cache.sh lines 36-41):
- `taesdxl_decoder.pth` - SDXL decoder
- `taesd_decoder.pth` - SD 1.5 decoder
- `taesd3_decoder.pth` - SD 3 decoder
- `taef1_decoder.pth` - FLUX.1 decoder

**Source**: GitHub madebyollin/taesd repository

---

## 5. Script Migration Plan

### 5.1 Entrypoint Script Adaptation

**Current Structure** (entrypoint.sh):
```bash
#!/bin/bash
set -e

# 1. Run user's set-proxy script
# 2. Copy ComfyUI from /default-comfyui-bundle to /root/ComfyUI
# 3. Run user's pre-start script
# 4. Set Python environment variables
# 5. Start ComfyUI: python3 ./ComfyUI/main.py --listen --port 8188 ${CLI_ARGS}
```

**Required Modifications for Multi-Agent-Docker**:
```bash
#!/bin/bash
set -e

# 1. Skip proxy setup (not needed in multi-agent-docker)
# 2. Copy ComfyUI from /opt/comfyui-bundle to /opt/comfyui (if not exists)
# 3. Create user directories if needed
# 4. Set Python 3.13 environment:
#    - PYTHONPYCACHEPREFIX=/home/devuser/.cache/pycache
#    - PIP_USER=false (use venv instead)
#    - PATH includes Python 3.13
# 5. Activate venv if using virtual environment
# 6. Start ComfyUI as devuser (not root):
#    python3.13 /opt/comfyui/main.py --listen --port 8188 ${CLI_ARGS}
```

**Security Enhancement**:
- Run as `devuser` instead of `root`
- Use systemd service or supervisord entry
- Proper file permissions

### 5.2 Pre-start Script Template

**Location**: `/home/devuser/.config/comfyui/pre-start.sh`

**Purpose**: User customization before ComfyUI starts

**Example Use Cases**:
- Install additional custom nodes
- Update models
- Configure environment
- Run cleanup tasks

### 5.3 Model Cache Preloading

**Current Script** (preload-cache.sh):
```bash
# Clone ComfyUI stable version
git clone 'https://github.com/comfyanonymous/ComfyUI.git'
git reset --hard "$(git tag | grep -e '^v' | sort -V | tail -1)"

# Clone ComfyUI-Manager
cd custom_nodes
git clone --depth=1 https://github.com/Comfy-Org/ComfyUI-Manager.git

# Force PIP usage (not UV)
cat <<EOF > /default-comfyui-bundle/ComfyUI/user/default/ComfyUI-Manager/config.ini
[default]
use_uv = False
EOF

# Download VAE models
cd /default-comfyui-bundle/ComfyUI/models/vae
aria2c 'https://github.com/madebyollin/taesd/raw/refs/heads/main/taesdxl_decoder.pth'
aria2c 'https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_decoder.pth'
aria2c 'https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd3_decoder.pth'
aria2c 'https://github.com/madebyollin/taesd/raw/refs/heads/main/taef1_decoder.pth'
```

**Adaptation for Multi-Agent-Docker**:
- Run during Dockerfile build phase
- Store in `/opt/comfyui-bundle`
- Use curl/wget instead of aria2 (or install aria2)

---

## 6. Port and Network Configuration

### 6.1 Port Mapping

**ComfyUI Required Port**:
- **8188**: Web interface (HTTP)

**Current Multi-Agent-Docker Ports**:
| Port | Service | Status |
|------|---------|--------|
| 22 → 2222 | SSH | In use |
| 5901 | VNC | In use |
| 8080 | code-server | In use |
| 9090 | Management API | In use |
| 9600 | Z.AI (internal) | In use |
| **8188** | **Available** | ✅ FREE |

**Conflict Assessment**: ✅ **No conflicts** - Port 8188 available

### 6.2 Service Integration

**Supervisord Entry** (add to supervisord.unified.conf):
```ini
[program:comfyui]
command=/home/devuser/.local/bin/start-comfyui.sh
directory=/opt/comfyui
user=devuser
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=/var/log/comfyui.log
stdout_logfile_maxbytes=10MB
environment=
    CUDA_HOME="/opt/cuda",
    PATH="/opt/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    LD_LIBRARY_PATH="/opt/cuda/lib64",
    PYTHONPYCACHEPREFIX="/home/devuser/.cache/pycache"
priority=700
```

**Priority Order**:
- 10: dbus
- 50: sshd
- 100: xvnc
- 200: xfce4
- 300: management-api
- 400: code-server
- 500: claude-zai
- 600: gemini-flow
- **700: comfyui** (new)
- 900: tmux-autostart

### 6.3 Docker Network Configuration

**Current Network** (docker-compose.unified.yml):
```yaml
networks:
  ragflow:
    external: true
```

**ComfyUI Addition**:
```yaml
services:
  turbo-unified:
    # ... existing config
    ports:
      - "2222:22"
      - "5901:5901"
      - "8080:8080"
      - "9090:9090"
      - "8188:8188"  # Add ComfyUI port
```

---

## 7. CUDA Environment Analysis

### 7.1 Current CUDA Configuration

**Dockerfile.unified** (lines 66-89):
```dockerfile
RUN pacman -S --noconfirm \
        cuda cuda-tools \
        cudnn \
        nvidia-utils \
        libglvnd

ENV CUDA_HOME=/opt/cuda
ENV PATH=/opt/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH}
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __VK_LAYER_NV_optimus=NVIDIA_only
ENV LIBGL_ALWAYS_INDIRECT=0
```

### 7.2 ComfyUI CUDA Requirements

**LD_LIBRARY_PATH** (Dockerfile line 190-207):
```bash
LD_LIBRARY_PATH="/usr/local/lib64/python3.13/site-packages/torch/lib:\
/usr/local/lib/python3.13/site-packages/cusparselt/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_cupti/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cufile/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/curand/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusparselt/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvshmem/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvtx/lib"
```

### 7.3 Compatibility Matrix

| Component | ComfyUI Requirement | Multi-Agent Status | Action |
|-----------|--------------------|--------------------|--------|
| CUDA Toolkit | 13.0 | Installed (verify version) | Check version |
| cuDNN | Compatible with CUDA 13 | Installed | ✅ OK |
| GCC | 15 (CUDA 13.0 compatible) | Latest Arch gcc | ✅ OK |
| Python | 3.13 | 3.x (upgrade needed) | Install 3.13 |
| PyTorch | cu130 build | Not installed | Install from pip |
| nvidia-utils | CUDA 13.0 compatible | Installed | ✅ OK |

### 7.4 Environment Variable Merging

**Combined Configuration**:
```bash
# Base CUDA (existing)
CUDA_HOME=/opt/cuda
PATH=/opt/cuda/bin:${PATH}

# PyTorch library paths (add for ComfyUI)
LD_LIBRARY_PATH=/opt/cuda/lib64:\
/usr/local/lib/python3.13/site-packages/torch/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/*/lib:${LD_LIBRARY_PATH}

# OpenGL (existing)
__GLX_VENDOR_LIBRARY_NAME=nvidia
__NV_PRIME_RENDER_OFFLOAD=1
__VK_LAYER_NV_optimus=NVIDIA_only
LIBGL_ALWAYS_INDIRECT=0

# Python cache (ComfyUI)
PYTHONPYCACHEPREFIX=/home/devuser/.cache/pycache
```

---

## 8. Service Conflict Analysis

### 8.1 Resource Requirements

**ComfyUI Resource Profile**:
- **CPU**: 2-4 cores (PyTorch inference)
- **RAM**: 8-16 GB (model loading)
- **GPU VRAM**: 4-12 GB (depending on model)
- **Disk**: 10-50 GB (models + cache)
- **Network**: HTTP server on port 8188

**Current Multi-Agent Services**:
| Service | CPU | RAM | GPU | Notes |
|---------|-----|-----|-----|-------|
| management-api | Low | ~100MB | No | Node.js API |
| code-server | Medium | ~500MB | No | Web IDE |
| claude-zai | Low | ~200MB | No | Worker pool |
| gemini-flow | Low | ~150MB | No | Orchestration |
| xfce4 | Low | ~300MB | Yes | Desktop |

**Conflict Assessment**:
- ✅ **No CPU conflicts** - ComfyUI will be dominant GPU workload
- ⚠️ **GPU contention** - Ensure CUDA_VISIBLE_DEVICES if multiple GPU users
- ⚠️ **RAM usage** - Recommend 16GB+ total system RAM
- ✅ **Network** - Port 8188 available

### 8.2 Process Priority

**Recommended Priority**:
1. **SSH/dbus** (critical infrastructure)
2. **ComfyUI** (GPU workload, high resource usage)
3. **Desktop/VNC** (user interface)
4. **Management APIs** (lower priority)

**Supervisord Priority Adjustment**:
```ini
# Critical
[program:dbus]        priority=10
[program:sshd]        priority=50

# High priority (GPU workload)
[program:comfyui]     priority=100

# Medium priority
[program:xvnc]        priority=200
[program:xfce4]       priority=300

# Lower priority
[program:management-api]  priority=400
[program:code-server]     priority=500
[program:claude-zai]      priority=600
[program:gemini-flow]     priority=700
[program:tmux-autostart]  priority=900
```

### 8.3 Shared Library Conflicts

**Potential Conflicts**:
| Library | ComfyUI | Multi-Agent | Resolution |
|---------|---------|-------------|------------|
| OpenCV | python-opencv-headless | opencv (GUI) | ⚠️ Use headless in venv |
| FFmpeg | System package | ffmpeg (installed) | ✅ Compatible |
| CUDA libs | nvidia-* (pip) | cuda (pacman) | ✅ Coexist via LD_LIBRARY_PATH |
| Qt5 | libQt5OpenGL5 | qt5-base | ✅ Compatible |
| Mesa/GL | OpenGL support | mesa (installed) | ✅ Compatible |

**Resolution Strategy**:
- Use Python virtual environment for ComfyUI packages
- Keep system packages for shared dependencies
- Configure LD_LIBRARY_PATH to prioritize pip nvidia-* libs

---

## 9. Integration Recommendations

### 9.1 Installation Strategy

**Phase 1: System Packages** (Dockerfile additions):
```dockerfile
# Add to PHASE 1: Base System Packages
RUN pacman -S --noconfirm \
    # (existing packages...)
    # ComfyUI system dependencies
    python-wheel python-setuptools \
    cython ninja aria2 \
    qt5-base \
    x264 x265 \
    bison \
    && rm -rf /var/cache/pacman/pkg/*
```

**Phase 2: Python 3.13 Installation**:
```dockerfile
# Add as separate phase after PHASE 6
# PHASE 6.5: Python 3.13 for ComfyUI
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tar.xz && \
    tar -xf Python-3.13.0.tar.xz && \
    cd Python-3.13.0 && \
    ./configure --prefix=/usr/local/python3.13 --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/python3.13/bin/python3.13 /usr/bin/python3.13 && \
    ln -s /usr/local/python3.13/bin/pip3.13 /usr/bin/pip3.13 && \
    cd /tmp && rm -rf Python-3.13.0*
```

**Phase 3: PyTorch Installation**:
```dockerfile
# Install PyTorch with CUDA 13.0 support
RUN pip3.13 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
```

**Phase 4: ComfyUI Dependencies**:
```dockerfile
# Copy requirement files
COPY ComfyUIDocker/builder-scripts/pak3.txt /tmp/comfyui-pak3.txt
COPY ComfyUIDocker/builder-scripts/pak5.txt /tmp/comfyui-pak5.txt
COPY ComfyUIDocker/builder-scripts/pak7.txt /tmp/comfyui-pak7.txt

# Install in order
RUN pip3.13 install --no-cache-dir -r /tmp/comfyui-pak3.txt && \
    pip3.13 install --no-cache-dir -r /tmp/comfyui-pak5.txt && \
    pip3.13 install --no-cache-dir -r /tmp/comfyui-pak7.txt && \
    rm /tmp/comfyui-pak*.txt
```

**Phase 5: ComfyUI Application**:
```dockerfile
# Clone and preload ComfyUI
RUN mkdir -p /opt/comfyui-bundle && \
    cd /opt/comfyui-bundle && \
    git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git reset --hard "$(git tag | grep -e '^v' | sort -V | tail -1)" && \
    cd custom_nodes && \
    git clone --depth=1 https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    mkdir -p /opt/comfyui-bundle/ComfyUI/user/default/ComfyUI-Manager && \
    echo "[default]\nuse_uv = False" > /opt/comfyui-bundle/ComfyUI/user/default/ComfyUI-Manager/config.ini

# Preload VAE models
RUN mkdir -p /opt/comfyui-bundle/ComfyUI/models/vae && \
    cd /opt/comfyui-bundle/ComfyUI/models/vae && \
    wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesdxl_decoder.pth && \
    wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_decoder.pth && \
    wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd3_decoder.pth && \
    wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taef1_decoder.pth

# Install ComfyUI requirements
RUN pip3.13 install --no-cache-dir \
    -r /opt/comfyui-bundle/ComfyUI/requirements.txt \
    -r /opt/comfyui-bundle/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Set ownership
RUN chown -R devuser:devuser /opt/comfyui-bundle
```

### 9.2 Startup Script

**Location**: `/home/devuser/.local/bin/start-comfyui.sh`

```bash
#!/bin/bash
set -eu

echo "========================================="
echo "[ComfyUI] Starting ComfyUI for Multi-Agent-Docker"
echo "========================================="

# Environment setup
export CUDA_HOME=/opt/cuda
export PATH=/opt/cuda/bin:/usr/local/python3.13/bin:${PATH}
export PYTHONPYCACHEPREFIX=/home/devuser/.cache/pycache
export PIP_ROOT_USER_ACTION=ignore

# Configure LD_LIBRARY_PATH for PyTorch
export LD_LIBRARY_PATH="/opt/cuda/lib64:\
/usr/local/lib/python3.13/site-packages/torch/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Copy bundle to workspace if needed
WORKSPACE=/opt/comfyui
if [ ! -f "$WORKSPACE/main.py" ]; then
    echo "[INFO] Initializing ComfyUI workspace..."
    mkdir -p "$WORKSPACE"
    cp -a /opt/comfyui-bundle/ComfyUI/. "$WORKSPACE/"
    echo "[INFO] ComfyUI workspace initialized."
else
    echo "[INFO] Using existing ComfyUI workspace."
fi

# Run pre-start hook if exists
if [ -f /home/devuser/.config/comfyui/pre-start.sh ]; then
    echo "[INFO] Running pre-start hook..."
    bash /home/devuser/.config/comfyui/pre-start.sh
fi

# Start ComfyUI
cd "$WORKSPACE"
echo "[INFO] Starting ComfyUI on port 8188..."
exec python3.13 main.py --listen --port 8188 ${CLI_ARGS:-}
```

### 9.3 File Locations Summary

**Build-time Files** (in Dockerfile):
- `/opt/comfyui-bundle/` - Pristine ComfyUI clone (never modified)
- `/tmp/comfyui-pak*.txt` - Temporary requirement files

**Runtime Files**:
- `/opt/comfyui/` - Active ComfyUI instance
- `/home/devuser/models/comfyui-models/` - Model storage (large files)
- `/home/devuser/.cache/huggingface/` - Hugging Face cache
- `/home/devuser/.cache/torch/` - Torch hub cache
- `/home/devuser/.cache/pycache/` - Python bytecode cache
- `/home/devuser/.config/comfyui/` - User configuration
- `/var/log/comfyui.log` - Service logs

**User Scripts**:
- `/home/devuser/.local/bin/start-comfyui.sh` - Startup script
- `/home/devuser/.config/comfyui/pre-start.sh` - Pre-start hook (optional)

---

## 10. Potential Issues and Mitigations

### 10.1 Python Version Conflicts

**Issue**: System Python (3.12.x) vs ComfyUI Python (3.13)

**Mitigation**:
- Use `python3.13` explicitly in all scripts
- Configure supervisord to use full path: `/usr/local/python3.13/bin/python3.13`
- Keep system Python for other services

### 10.2 CUDA Library Conflicts

**Issue**: System CUDA vs pip-installed nvidia-* packages

**Mitigation**:
- LD_LIBRARY_PATH prioritizes pip packages first
- System CUDA remains available as fallback
- Test with: `python3.13 -c "import torch; print(torch.cuda.is_available())"`

### 10.3 OpenCV Conflicts

**Issue**: opencv-headless (ComfyUI) vs opencv (system with GUI support)

**Mitigation**:
- Install opencv-contrib-python-headless in Python 3.13 environment
- Keep system opencv for GUI tools (Blender, QGIS)
- Separate Python package namespaces

### 10.4 Disk Space

**Issue**: Models and cache can consume 50+ GB

**Mitigation**:
- Use separate volume for `/home/devuser/models/`
- Configure model pruning in ComfyUI-Manager
- Monitor with: `df -h /home/devuser/models`

### 10.5 Memory Pressure

**Issue**: Large models can consume 16+ GB RAM

**Mitigation**:
- Configure ComfyUI to use GPU memory for models
- Set `--lowvram` or `--novram` flags in CLI_ARGS if needed
- Monitor with: `nvidia-smi` and `htop`

### 10.6 Network Security

**Issue**: Port 8188 exposed without authentication

**Mitigation**:
- Use nginx reverse proxy with authentication
- Configure ComfyUI to bind to localhost only (remove --listen)
- Access via SSH tunnel: `ssh -L 8188:localhost:8188 devuser@host`

---

## 11. Testing and Validation

### 11.1 CUDA Verification

```bash
# Check CUDA toolkit
nvcc --version
# Expected: CUDA 13.0

# Check PyTorch CUDA support
python3.13 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
# Expected: CUDA available: True, CUDA version: 13.0, Device count: 1+

# Check GPU visibility
nvidia-smi
# Expected: GPU list with processes
```

### 11.2 Package Installation Verification

```bash
# Verify Python 3.13
python3.13 --version
# Expected: Python 3.13.0

# Check essential packages
python3.13 -c "import torch, torchvision, torchaudio; print('PyTorch OK')"
python3.13 -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python3.13 -c "import PIL; print(f'Pillow {PIL.__version__}')"
python3.13 -c "import diffusers; print('Diffusers OK')"
python3.13 -c "import transformers; print('Transformers OK')"

# Check ComfyUI imports
cd /opt/comfyui
python3.13 -c "import comfy; print('ComfyUI core OK')"
```

### 11.3 Service Startup Test

```bash
# Test startup script
/home/devuser/.local/bin/start-comfyui.sh &
# Wait 10 seconds
curl http://localhost:8188
# Expected: HTML response or 200 OK

# Check supervisord
sudo supervisorctl status comfyui
# Expected: RUNNING

# Check logs
sudo supervisorctl tail -f comfyui
# Expected: No errors, "Starting server" message
```

### 11.4 End-to-End Test

```bash
# Access web interface
# Navigate to: http://localhost:8188

# Expected:
# - ComfyUI web UI loads
# - Can create workflow
# - Can load models (if installed)
# - GPU detected in settings
```

---

## 12. Performance Optimization

### 12.1 PyTorch JIT Compilation

```bash
# Enable PyTorch JIT for faster inference
export PYTORCH_JIT=1
```

### 12.2 CUDA Memory Management

```bash
# CLI args for memory management
CLI_ARGS="--lowvram"         # For 6-8GB VRAM
CLI_ARGS="--normalvram"      # For 8-12GB VRAM
CLI_ARGS="--highvram"        # For 12+ GB VRAM
CLI_ARGS="--cpu"             # CPU-only mode (slow)
```

### 12.3 Model Caching

```bash
# Preload models to cache
# Configure in ComfyUI-Manager settings:
# - Enable "Auto-update models"
# - Set cache location: /home/devuser/models/comfyui-models
```

---

## 13. Maintenance and Updates

### 13.1 ComfyUI Updates

```bash
# Update ComfyUI core
cd /opt/comfyui
git fetch origin
git pull

# Update custom nodes
cd custom_nodes/ComfyUI-Manager
git pull

# Update Python packages
pip3.13 install --upgrade -r requirements.txt
```

### 13.2 Model Management

```bash
# Model locations
/opt/comfyui/models/checkpoints/     # Stable Diffusion checkpoints
/opt/comfyui/models/loras/          # LoRA models
/opt/comfyui/models/vae/            # VAE models
/opt/comfyui/models/embeddings/     # Textual inversions

# Cleanup old models
# Use ComfyUI-Manager web interface
```

### 13.3 Cache Cleanup

```bash
# Clear Python cache
rm -rf /home/devuser/.cache/pycache/*

# Clear Hugging Face cache (careful - redownloads models)
rm -rf /home/devuser/.cache/huggingface/*

# Clear Torch hub cache
rm -rf /home/devuser/.cache/torch/*
```

---

## 14. Architecture Decision Records

### ADR-001: Python 3.13 Installation Strategy

**Decision**: Build Python 3.13 from source as altinstall

**Rationale**:
- ComfyUI requires Python 3.13 specifically
- Arch Linux stable repo has Python 3.12.x
- AUR packages may lag behind or break
- Altinstall prevents conflicts with system Python

**Alternatives Considered**:
- Use AUR package: Risk of breakage, dependency conflicts
- Use pyenv: Added complexity, not ideal for containers
- Wait for Arch to update: Unpredictable timeline

**Consequences**:
- Manual compilation adds build time (~5-10 minutes)
- Need to maintain Python 3.13 separately
- Clear separation from system Python

### ADR-002: Package Management Strategy

**Decision**: Use pip for all Python packages, pacman only for system dependencies

**Rationale**:
- openSUSE uses system packages, but Arch philosophy differs
- PyTorch from pip ensures CUDA 13.0 compatibility
- Easier to match exact versions from pak3/5/7.txt
- Virtual environment not needed (container isolation)

**Alternatives Considered**:
- Mix pacman + pip: Version conflicts, fragmentation
- Use pacman python-* packages: Missing many dependencies
- Use conda/mamba: Overkill for container deployment

**Consequences**:
- 150+ pip packages to install
- Longer build times
- Consistent with upstream ComfyUI deployment

### ADR-003: Service User and Permissions

**Decision**: Run ComfyUI as `devuser` instead of `root`

**Rationale**:
- Security best practice
- Matches multi-agent-docker pattern
- File permissions align with workspace ownership
- SSH access works seamlessly

**Alternatives Considered**:
- Run as root (original ComfyUI): Security risk
- Create dedicated comfyui user: Adds complexity

**Consequences**:
- Need to ensure devuser has GPU access (already configured)
- File paths use /home/devuser instead of /root
- Supervisord runs as devuser

### ADR-004: Model Storage Location

**Decision**: Store models in `/home/devuser/models/comfyui-models/`

**Rationale**:
- Centralized model storage
- Easy to mount as separate volume
- Shared with other AI tools (Blender, etc.)
- Survives ComfyUI updates

**Alternatives Considered**:
- Store in /opt/comfyui/models: Harder to isolate volume
- Store in /root (original): Inconsistent with user model

**Consequences**:
- Need symlinks or config to point ComfyUI to model location
- 10-50GB disk space required

---

## 15. Next Steps

### Phase 1: Preparation (No Build Required)
1. ✅ Review this analysis document
2. Create package installation checklist
3. Prepare Python 3.13 build configuration
4. Plan disk space allocation (50GB recommended)

### Phase 2: Dockerfile Modifications
1. Add system packages (ninja, aria2, qt5-base, x264, x265)
2. Add Python 3.13 build phase
3. Install PyTorch with CUDA 13.0
4. Install pak3/pak5/pak7 requirements
5. Clone and preload ComfyUI bundle
6. Add startup scripts

### Phase 3: Configuration Files
1. Create supervisord entry for ComfyUI
2. Create start-comfyui.sh startup script
3. Update docker-compose.yml with port 8188
4. Add volume mounts for models and cache
5. Create pre-start.sh template

### Phase 4: Build and Test
1. Build Docker image: `docker build -f Dockerfile.unified -t turbo-flow-unified:latest .`
2. Start container: `docker-compose -f docker-compose.unified.yml up -d`
3. Verify CUDA: `docker exec turbo-flow-unified nvidia-smi`
4. Verify Python 3.13: `docker exec -u devuser turbo-flow-unified python3.13 --version`
5. Check ComfyUI service: `docker exec turbo-flow-unified supervisorctl status comfyui`
6. Access web UI: `http://localhost:8188`

### Phase 5: Documentation
1. Update CLAUDE.md with ComfyUI instructions
2. Create ComfyUI usage guide for devpod users
3. Document model installation process
4. Add troubleshooting section

---

## 16. Cost-Benefit Analysis

### Benefits

**Capabilities Added**:
- Stable Diffusion image generation
- Text-to-image workflows
- Image-to-image transformations
- ControlNet support
- LoRA model integration
- Custom node ecosystem

**Integration Value**:
- Single unified development container
- Shared CUDA resources
- Consistent environment management
- Claude Code can script ComfyUI workflows

### Costs

**Build Time**:
- Python 3.13 compilation: ~5-10 minutes
- PyTorch + dependencies: ~15-30 minutes
- ComfyUI clone: ~2-5 minutes
- **Total**: ~20-45 minutes added to build

**Disk Space**:
- Python 3.13: ~500 MB
- PyTorch + CUDA packages: ~8 GB
- ComfyUI application: ~2 GB
- VAE models (preloaded): ~200 MB
- **Total**: ~11 GB base + models (10-50 GB variable)

**Runtime Resources**:
- RAM: +8-16 GB (model loading)
- GPU VRAM: 4-12 GB (depending on usage)
- CPU: +10-30% (during inference)

**Maintenance**:
- Regular ComfyUI updates
- Model management
- Custom node updates
- Python package updates

### Risk Assessment

**Low Risk**:
- ✅ Port conflict (8188 free)
- ✅ Service conflicts (isolated supervisord entry)
- ✅ CUDA compatibility (already configured)

**Medium Risk**:
- ⚠️ Build failures (Python 3.13 compilation)
- ⚠️ Package conflicts (150+ pip packages)
- ⚠️ Disk space (models can grow large)

**High Risk**:
- ⚠️ Memory pressure (requires 16+ GB RAM for large models)
- ⚠️ GPU VRAM limitations (may need --lowvram flag)

**Mitigation**:
- Test build in stages
- Monitor disk usage
- Configure memory limits
- Use progressive model loading

---

## Appendix A: Complete Package List

### A.1 System Packages (pacman)

```bash
# Already installed (base-devel)
base-devel git vim nano curl wget make gcc g++

# Need to install
python-wheel python-setuptools cython
ninja aria2
qt5-base
x264 x265
bison

# Optional (already installed)
ffmpeg opencv python-opencv
```

### A.2 Python Packages (pip3.13)

**Tier 1: PyTorch** (pak-pytorch.txt)
```
# Core PyTorch
torch==2.x.x+cu130
torchvision==0.x.x+cu130
torchaudio==2.x.x+cu130
triton
pytorch-triton

# NVIDIA CUDA packages (installed automatically with PyTorch)
nvidia-cublas-cu13
nvidia-cudnn-cu13
nvidia-cuda-runtime-cu13
nvidia-cuda-nvrtc-cu13
nvidia-cufft-cu13
nvidia-curand-cu13
nvidia-cusolver-cu13
nvidia-cusparse-cu13
nvidia-nccl-cu13
nvidia-nvtx-cu13
```

**Tier 2: Essentials** (pak3.txt)
```
accelerate
diffusers
ftfy
huggingface-hub
imageio[ffmpeg]
joblib
kornia
matplotlib
nvidia-ml-py
omegaconf
onnx
onnxruntime-gpu
opencv-contrib-python-headless
pandas
pilgram
pillow
pygit2
python-ffmpeg
regex
scikit-build-core
scikit-image
scikit-learn
scipy
timm
torchmetrics
transformers
compel
lark
cupy-cuda13x
fairscale
torchdiffeq
```

**Tier 3: Extended** (pak5.txt)
```
addict aenum aiohttp albumentations alembic av
black blend-modes cachetools chardet
clip-interrogator color-matcher colour-science
deepdiff dill einops filelock fvcore
gguf GitPython hydra-core imageio-ffmpeg
importlib-metadata iopath loguru mss
numba numexpr peft piexif pixeloe
protobuf psutil py-cpuinfo pydantic
pydantic-settings pydub PyGithub pymatting
python-dateutil pyyaml qrcode[pil]
rembg requirements-parser rich rich-argparse
safetensors segment-anything sentencepiece
simpleeval spandrel SQLAlchemy tokenizers
toml torchsde tqdm transparent-background
trimesh[easy] typer typing-extensions
ultralytics uv webcolors yacs yapf yarl
```

**Tier 4: Git Packages** (pak7.txt)
```
dlib
facexlib
insightface
git+https://github.com/openai/CLIP.git
git+https://github.com/cozy-comfyui/cozy_comfyui@main
git+https://github.com/cozy-comfyui/cozy_comfy@main
git+https://github.com/facebookresearch/sam2.git
git+https://github.com/ltdrdata/cstr.git
git+https://github.com/ltdrdata/ffmpy.git
git+https://github.com/ltdrdata/img2texture.git
```

### A.3 Models (preloaded)

```
# VAE Decoders
https://github.com/madebyollin/taesd/raw/refs/heads/main/taesdxl_decoder.pth
https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_decoder.pth
https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd3_decoder.pth
https://github.com/madebyollin/taesd/raw/refs/heads/main/taef1_decoder.pth
```

---

## Appendix B: Environment Variables Reference

```bash
# CUDA Configuration
CUDA_HOME=/opt/cuda
PATH=/opt/cuda/bin:/usr/local/python3.13/bin:${PATH}

# PyTorch Library Paths
LD_LIBRARY_PATH=/opt/cuda/lib64:\
/usr/local/lib/python3.13/site-packages/torch/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_cupti/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cufile/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/curand/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusparselt/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvshmem/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvtx/lib

# OpenGL/Vulkan (already configured)
__GLX_VENDOR_LIBRARY_NAME=nvidia
__NV_PRIME_RENDER_OFFLOAD=1
__VK_LAYER_NV_optimus=NVIDIA_only
LIBGL_ALWAYS_INDIRECT=0

# Python Configuration
PYTHONPYCACHEPREFIX=/home/devuser/.cache/pycache
PIP_ROOT_USER_ACTION=ignore

# ComfyUI CLI Arguments
CLI_ARGS=""                    # Default: auto-detect
CLI_ARGS="--lowvram"          # For 6-8GB VRAM
CLI_ARGS="--normalvram"       # For 8-12GB VRAM
CLI_ARGS="--highvram"         # For 12+ GB VRAM
CLI_ARGS="--cpu"              # CPU-only (slow)
```

---

## Appendix C: Directory Structure

```
/opt/comfyui-bundle/           # Build-time bundle (pristine)
├── ComfyUI/
│   ├── main.py
│   ├── requirements.txt
│   ├── custom_nodes/
│   │   └── ComfyUI-Manager/
│   └── models/
│       └── vae/
│           ├── taesdxl_decoder.pth
│           ├── taesd_decoder.pth
│           ├── taesd3_decoder.pth
│           └── taef1_decoder.pth

/opt/comfyui/                  # Runtime instance
├── main.py
├── custom_nodes/
├── input/
├── output/
├── models/
│   ├── checkpoints/
│   ├── loras/
│   ├── vae/
│   └── embeddings/
└── user/
    └── default/
        ├── workflows/
        └── ComfyUI-Manager/
            └── config.ini

/home/devuser/models/
├── comfyui-models/            # Large model storage
│   ├── checkpoints/
│   ├── loras/
│   └── vae/
└── shared-models/             # Other AI tools

/home/devuser/.cache/
├── pycache/                   # Python bytecode
├── huggingface/               # HF model cache
│   └── hub/
└── torch/                     # Torch hub
    └── hub/

/home/devuser/.config/
└── comfyui/
    ├── pre-start.sh           # User hook
    └── config.json            # User settings

/var/log/
└── comfyui.log               # Service logs
```

---

## Appendix D: Supervisord Configuration

```ini
[program:comfyui]
command=/home/devuser/.local/bin/start-comfyui.sh
directory=/opt/comfyui
user=devuser
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=/var/log/comfyui.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
environment=
    CUDA_HOME="/opt/cuda",
    PATH="/opt/cuda/bin:/usr/local/python3.13/bin:/usr/local/bin:/usr/bin:/bin",
    LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/local/lib/python3.13/site-packages/torch/lib:/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib",
    PYTHONPYCACHEPREFIX="/home/devuser/.cache/pycache",
    PIP_ROOT_USER_ACTION="ignore",
    __GLX_VENDOR_LIBRARY_NAME="nvidia",
    __NV_PRIME_RENDER_OFFLOAD="1",
    __VK_LAYER_NV_optimus="NVIDIA_only"
priority=700
stopwaitsecs=30
```

---

## Conclusion

This analysis provides a complete roadmap for integrating ComfyUI into the multi-agent-docker Arch Linux workstation. The integration is **feasible with moderate complexity**, requiring:

1. **Python 3.13 installation** (build from source)
2. **150+ Python packages** (pip installation)
3. **System package additions** (12 pacman packages)
4. **Startup script creation** (supervisord integration)
5. **Port and volume configuration** (docker-compose updates)

**No major conflicts** exist with current services, and the CUDA environment is already compatible. The primary challenge is the **build time** (~30-45 minutes) and **disk space** (~11 GB base + models).

**Recommended approach**: Implement in phases, test after each major addition, and monitor resource usage.

---

## Appendix E: SaladTechnologies comfyui-api Integration

### E.1 Overview

**Repository**: https://github.com/SaladTechnologies/comfyui-api
**Version**: 1.13.5 (latest stable)
**Purpose**: REST API wrapper for stateless ComfyUI operations
**Tech Stack**: TypeScript, Fastify, WebSocket
**Default Port**: 3000

### E.2 Key Features

1. **Stateless API**: No session management, pure request-response
2. **Synchronous Mode**: Base64-encoded images in response
3. **Async Mode**: Webhook callbacks with job completion
4. **Storage Backends**:
   - S3-compatible storage
   - Azure Blob Storage
   - HTTP endpoints
   - Hugging Face repositories
5. **Model Management**: LRU cache, model manifest, auto-download
6. **Image Processing**: Server-side image transformations (Sharp)
7. **Swagger Docs**: Self-documenting API at `/docs`

### E.3 Dependencies

**Node.js Packages** (package.json):
```json
{
  "@aws-sdk/client-s3": "^3.820.0",
  "@azure/identity": "^4.13.0",
  "@azure/storage-blob": "^12.28.0",
  "@fastify/swagger": "^9.5.0",
  "@fastify/swagger-ui": "^5.2.2",
  "fastify": "^5.3.3",
  "fastify-type-provider-zod": "^4.0.2",
  "sharp": "^0.34.5",
  "typescript": "^5.8.3",
  "undici": "^7.10.0",
  "ws": "^8.18.2",
  "yaml": "^2.8.1",
  "zod": "^3.25.36"
}
```

**System Dependencies**:
- Node.js 23.11.1 ✅ (already installed)
- libvips (for Sharp image processing)

### E.4 Installation

**Option 1: Pre-built Binary** (Recommended):
```dockerfile
# Add to PHASE 13: Application Files
ARG COMFYUI_API_VERSION=1.13.5
RUN mkdir -p /opt/comfyui-api && \
    curl -fsSL "https://github.com/SaladTechnologies/comfyui-api/releases/download/${COMFYUI_API_VERSION}/comfyui-api" \
         -o /opt/comfyui-api/comfyui-api && \
    chmod +x /opt/comfyui-api/comfyui-api && \
    chown -R devuser:devuser /opt/comfyui-api
```

**Option 2: Build from Source**:
```dockerfile
RUN git clone https://github.com/SaladTechnologies/comfyui-api.git /opt/comfyui-api && \
    cd /opt/comfyui-api && \
    npm install && \
    npm run build && \
    chown -R devuser:devuser /opt/comfyui-api
```

**System Package Required**:
```dockerfile
# Add to PHASE 1: Base packages
RUN pacman -S --noconfirm libvips
```

### E.5 Configuration

**Environment Variables**:
```bash
# Required
PORT=3000
COMFYUI_URL=http://localhost:8188

# Optional - AWS S3
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=...

# Optional - Azure Blob
AZURE_STORAGE_ACCOUNT=...
AZURE_STORAGE_KEY=...
AZURE_CONTAINER_NAME=...

# Optional - LRU Cache
MAX_CACHE_SIZE=50000000000  # 50GB
CACHE_DIR=/home/devuser/.cache/comfyui

# Optional - Webhooks
WEBHOOK_SECRET=...
```

### E.6 Supervisord Service

```ini
[program:comfyui-api]
command=/opt/comfyui-api/comfyui-api
user=devuser
directory=/opt/comfyui-api
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=/var/log/comfyui-api.log
stdout_logfile_maxbytes=10MB
environment=
    PORT="3000",
    COMFYUI_URL="http://localhost:8188",
    MAX_CACHE_SIZE="50000000000",
    CACHE_DIR="/home/devuser/.cache/comfyui",
    NODE_ENV="production"
priority=710
stopwaitsecs=10
```

### E.7 API Endpoints

**Health/Probes**:
- `GET /health` - Basic health check
- `GET /ready` - Readiness probe (checks ComfyUI connection)

**Documentation**:
- `GET /docs` - Swagger UI interactive docs

**Workflow Execution**:
- `POST /prompt` - Execute ComfyUI workflow
  - Sync mode: Returns base64 images in response
  - Async mode: Sends webhook on completion

**Model Management**:
- `GET /models` - List available models
- `POST /models/download` - Download model from manifest

### E.8 Example Usage

**Synchronous (Base64 response)**:
```bash
curl -X POST http://localhost:3000/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "3": {
        "inputs": {
          "seed": 42,
          "steps": 20,
          "cfg": 7.5,
          "sampler_name": "euler",
          "scheduler": "normal",
          "denoise": 1
        },
        "class_type": "KSampler"
      }
    }
  }'
```

**Asynchronous (Webhook callback)**:
```bash
curl -X POST http://localhost:3000/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {...},
    "webhook": {
      "url": "https://example.com/webhook",
      "secret": "my-webhook-secret"
    }
  }'
```

### E.9 Port Configuration

**Updated Port Mappings**:
```yaml
# docker-compose.unified.yml
ports:
  - "2222:22"        # SSH
  - "5901:5901"      # VNC
  - "8080:8080"      # code-server
  - "9090:9090"      # Management API
  - "8188:8188"      # ComfyUI Web UI (NEW)
  - "3000:3000"      # comfyui-api (NEW)
```

**Dockerfile EXPOSE**:
```dockerfile
# PHASE 16: Ports & Volumes
EXPOSE 22 5901 8080 9090 8188 3000
```

### E.10 Integration with Management API

**Create**: `/opt/management-api/routes/comfyui.js`

```javascript
module.exports = async function (fastify, opts) {
  const comfyuiApiUrl = process.env.COMFYUI_API_URL || 'http://localhost:3000';

  // Proxy to comfyui-api
  fastify.post('/api/comfyui/prompt', async (request, reply) => {
    const response = await fetch(`${comfyuiApiUrl}/prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request.body)
    });
    return response.json();
  });

  // Get models
  fastify.get('/api/comfyui/models', async (request, reply) => {
    const response = await fetch(`${comfyuiApiUrl}/models`);
    return response.json();
  });

  // Health check
  fastify.get('/api/comfyui/health', async (request, reply) => {
    const [apiHealth, uiHealth] = await Promise.all([
      fetch(`${comfyuiApiUrl}/health`).then(r => r.json()).catch(() => ({ status: 'down' })),
      fetch('http://localhost:8188/').then(r => ({ status: r.ok ? 'up' : 'down' })).catch(() => ({ status: 'down' }))
    ]);
    return {
      api: apiHealth,
      ui: uiHealth,
      overall: apiHealth.status === 'ok' && uiHealth.status === 'up' ? 'healthy' : 'unhealthy'
    };
  });
};
```

### E.11 tmux Workspace Addition

**Add Windows 8-9 to tmux-autostart.sh**:

```bash
# Window 8: ComfyUI
tmux new-window -t workspace:8 -n "ComfyUI"
tmux send-keys -t workspace:8 'cd /opt/comfyui' C-m
tmux send-keys -t workspace:8 'echo "=== ComfyUI Workspace ==="' C-m
tmux send-keys -t workspace:8 'echo "Web UI: http://localhost:8188"' C-m
tmux send-keys -t workspace:8 'echo "API: http://localhost:3000"' C-m
tmux send-keys -t workspace:8 'sudo supervisorctl tail -f comfyui' C-m

# Window 9: ComfyUI API
tmux new-window -t workspace:9 -n "ComfyUI-API"
tmux send-keys -t workspace:9 'cd /opt/comfyui-api' C-m
tmux send-keys -t workspace:9 'echo "=== ComfyUI API Workspace ==="' C-m
tmux send-keys -t workspace:9 'echo "Swagger Docs: http://localhost:3000/docs"' C-m
tmux send-keys -t workspace:9 'sudo supervisorctl tail -f comfyui-api' C-m
```

### E.12 Testing

```bash
# Test comfyui-api health
curl http://localhost:3000/health
# Expected: {"status":"ok"}

# Test ComfyUI connection
curl http://localhost:3000/ready
# Expected: {"status":"ready","comfyui":"connected"}

# Open Swagger docs
xdg-open http://localhost:3000/docs
```

### E.13 Use Cases

1. **Batch Processing**: Submit multiple workflows programmatically
2. **CI/CD Integration**: Automated image generation in pipelines
3. **Webhook Integration**: Async processing with callback notifications
4. **Cloud Storage**: Upload results directly to S3/Azure
5. **Model Management**: Auto-download models from manifest
6. **Image-to-Image**: Server-side image preprocessing with Sharp

### E.14 Performance

**Benchmarks** (from SaladTechnologies):
- Startup time: ~2 seconds
- Overhead per request: ~5-10ms
- Concurrent requests: Limited by ComfyUI queue
- WebSocket latency: ~1-2ms

### E.15 Security

**Recommendations**:
1. Use HTTPS in production (add reverse proxy)
2. Enable webhook signature validation
3. Restrict CORS origins
4. Use environment variables for secrets (never commit)
5. Rate-limit API endpoints

**Nginx Reverse Proxy Example**:
```nginx
server {
    listen 443 ssl;
    server_name comfyui-api.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

**Document Version**: 1.1
**Last Updated**: 2025-11-30
**Author**: System Architecture Designer
**Status**: Ready for Implementation (Updated with comfyui-api integration)
