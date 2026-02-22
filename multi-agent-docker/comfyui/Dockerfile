################################################################################
# ComfyUI with SAM3D CUDA Support
# Based on yanwk/comfyui-boot:cu130-slim with SAM3D environment pre-configured
################################################################################

# SECURITY: Pin to digest for production builds (e.g., yanwk/comfyui-boot@sha256:abc123...)
FROM yanwk/comfyui-boot:cu130-slim

LABEL maintainer="Multi-Agent-Docker"
LABEL description="ComfyUI with SAM3D CUDA support pre-configured"

# Install uv for faster Python package management
RUN pip install uv

# Clone SAM3D into the default bundle location so it gets copied on first run
WORKDIR /default-comfyui-bundle/ComfyUI/custom_nodes
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects.git comfyui-sam3dobjects

# Create Python 3.10 venv for SAM3D using uv
WORKDIR /default-comfyui-bundle/ComfyUI/custom_nodes/comfyui-sam3dobjects
RUN uv venv --python 3.10 --seed _env

# Install PyTorch 2.4.1 + CUDA 12.4 into SAM3D venv using uv
# This is the big download - do it at build time
RUN uv pip install --python ./_env/bin/python \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch3D from third-party wheel repository (official FB links are 403)
# Using miropsota's prebuilt wheels: https://github.com/facebookresearch/pytorch3d/discussions/1752
RUN uv pip install --python ./_env/bin/python \
    pytorch3d==0.7.9+pt2.4.1cu124 \
    --extra-index-url https://miropsota.github.io/torch_packages_builder

# Install spconv for sparse convolutions (required by SAM3D)
# Must be installed BEFORE other deps that might pull wrong CUDA version
RUN uv pip install --python ./_env/bin/python \
    "spconv-cu124>=2.3.8"

# Install all SAM3D dependencies from requirements_env.txt
# Skip packages that need special handling (pytorch3d already installed, utils3d from git)
# NOTE: sam2 removed - it pulls torch 2.9.x and breaks pytorch3d compatibility
RUN uv pip install --python ./_env/bin/python \
    fvcore iopath torchsde \
    "numpy>=1.24.0,<2.0" "scipy>=1.11.0" \
    "timm>=0.9.16" transformers diffusers "lightning>=2.0.0" \
    "scikit-image>=0.23.1" "imageio>=2.36.0" "Pillow>=10.0.0" \
    trimesh "point-cloud-utils>=0.29.5" plyfile \
    "matplotlib>=3.9.0" "seaborn>=0.13.0" "plotly>=5.24.0" \
    "hydra-core>=1.3.2" "easydict>=1.13" "loguru>=0.7.2" \
    huggingface-hub "python-igraph>=1.0.0" "tqdm>=4.67.0" \
    "optree>=0.11.0" "astor>=0.8.0" \
    "einops-exts>=0.0.4" "roma>=1.5.1" "xatlas>=0.0.9" "pymeshfix>=0.16.0" \
    "gdown>=5.2.0" "zstandard>=0.22.0" "ninja>=1.10.0" jaxtyping \
    open3d

# Install utils3d from git (required by vendored sam3d_objects)
RUN uv pip install --python ./_env/bin/python \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Install gsplat from prebuilt wheels (for Gaussian splatting rendering)
# Uses gsplat.studio wheel index for PyTorch 2.4 + CUDA 12.4
RUN uv pip install --python ./_env/bin/python \
    rich jaxtyping && \
    uv pip install --python ./_env/bin/python \
    "gsplat>=1.4.0" \
    --index-url https://docs.gsplat.studio/whl/pt24cu124

# Install nvdiffrast from HuggingFace (for mesh rendering)
RUN uv pip install --python ./_env/bin/python \
    "https://huggingface.co/spaces/microsoft/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl"

# Reinstall pinned torch version after dependencies may have upgraded it
RUN uv pip install --python ./_env/bin/python \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --reinstall

# Apply CUDA bridge patch to subprocess_bridge.py
RUN python3 << 'PYEOF'
import os
from pathlib import Path

bridge_file = Path("/default-comfyui-bundle/ComfyUI/custom_nodes/comfyui-sam3dobjects/nodes/subprocess_bridge.py")
content = bridge_file.read_text()

if "_get_worker_env" in content:
    print("Already patched")
    exit(0)

lines = content.split("\n")
new_lines = []
import_added = False
method_added = False

for i, line in enumerate(lines):
    # Add import os after import subprocess
    if not import_added and line.strip() == "import subprocess":
        new_lines.append(line)
        new_lines.append("import os")
        import_added = True
        continue

    # Add method before start_worker
    if not method_added and line.strip().startswith("def start_worker("):
        method = """    def _get_worker_env(self) -> dict:
        \"\"\"Get environment with CUDA libraries for worker process.\"\"\"
        env = os.environ.copy()
        node_root = self.worker_script.parent
        venv_lib = node_root / "_env" / "lib"
        if venv_lib.exists():
            python_dirs = [d for d in venv_lib.iterdir() if d.name.startswith("python")]
            if python_dirs:
                site_packages = python_dirs[0] / "site-packages"
                nvidia_libs = []
                for lib_name in ["cublas", "cuda_cupti", "cuda_nvrtc", "cuda_runtime",
                                "cudnn", "cufft", "curand", "cusolver", "cusparse",
                                "nccl", "nvjitlink", "nvtx"]:
                    lib_path = site_packages / "nvidia" / lib_name / "lib"
                    if lib_path.exists():
                        nvidia_libs.append(str(lib_path))
                if nvidia_libs:
                    cuda_path = ":".join(nvidia_libs)
                    existing = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = f"{cuda_path}:{existing}" if existing else cuda_path
                    print(f"[SAM3DObjects] Added {len(nvidia_libs)} CUDA library paths")
        return env
"""
        new_lines.append(method)
        method_added = True

    new_lines.append(line)

# Replace Popen call
content = "\n".join(new_lines)
old_popen = """self.process = subprocess.Popen(
                [str(self.python_exe), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )"""

new_popen = """# Build environment with CUDA libraries
            worker_env = self._get_worker_env()

            self.process = subprocess.Popen(
                [str(self.python_exe), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=worker_env,
            )"""

content = content.replace(old_popen, new_popen)
bridge_file.write_text(content)
print(f"Patch applied: import_added={import_added}, method_added={method_added}")
PYEOF

# Verify patch was applied
RUN grep -q "_get_worker_env" /default-comfyui-bundle/ComfyUI/custom_nodes/comfyui-sam3dobjects/nodes/subprocess_bridge.py \
    && echo "SAM3D CUDA patch verified"

# Copy additional fix scripts
WORKDIR /runner-scripts
COPY scripts/patch-sam3d-bridge.py /runner-scripts/
COPY scripts/fix-sam3d-cuda.sh /runner-scripts/
COPY runner-scripts/fix-sam3d-on-startup.sh /runner-scripts/
RUN chmod +x /runner-scripts/*.sh 2>/dev/null || true

# Reset to root workdir
USER root
VOLUME /root
WORKDIR /root
EXPOSE 8188
ENV CLI_ARGS=""
CMD ["bash", "/runner-scripts/entrypoint.sh"]
