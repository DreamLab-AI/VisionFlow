#!/bin/bash
# Startup script to fix SAM3D CUDA environment
# This runs after ComfyUI starts and SAM3D custom node is installed

SAM3D_DIR="/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects"
BRIDGE_FILE="$SAM3D_DIR/nodes/subprocess_bridge.py"

# Wait for SAM3D to be installed
MAX_WAIT=60
WAITED=0
while [ ! -f "$BRIDGE_FILE" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ ! -f "$BRIDGE_FILE" ]; then
    echo "[Startup] SAM3D not found, skipping fix"
    exit 0
fi

echo "[Startup] Checking SAM3D CUDA fix..."

# Check if already patched
if grep -q "_get_worker_env" "$BRIDGE_FILE" && grep -q "import os" "$BRIDGE_FILE"; then
    echo "[Startup] SAM3D already patched"
    exit 0
fi

echo "[Startup] Applying SAM3D CUDA fix..."

# Ensure os is imported
if ! grep -q "^import os" "$BRIDGE_FILE"; then
    sed -i '/^import subprocess/a import os' "$BRIDGE_FILE"
fi

# Apply patch using Python
python3 << 'PYEOF'
import re
from pathlib import Path

bridge_file = Path("/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/nodes/subprocess_bridge.py")
content = bridge_file.read_text()

if "_get_worker_env" in content:
    print("[Startup] Patch already applied")
    exit(0)

# Add env parameter to Popen
old_popen = """            self.process = subprocess.Popen(
                [str(self.python_exe), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )"""

new_popen = """            # Build environment with CUDA libraries
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

# Add the _get_worker_env method
method_code = '''    def _get_worker_env(self) -> dict:
        """Get environment with CUDA libraries for worker process."""
        env = os.environ.copy()

        # Add CUDA libraries from isolated venv
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
                    print(f"[SAM3DObjects] Added {len(nvidia_libs)} CUDA library paths to worker environment")

        return env

'''

# Insert method after __init__ (before @classmethod)
classmethod_pos = content.find('    @classmethod')
if classmethod_pos > 0:
    content = content[:classmethod_pos] + method_code + content[classmethod_pos:]

bridge_file.write_text(content)
print("[Startup] SAM3D CUDA fix applied successfully")
PYEOF

echo "[Startup] SAM3D fix complete"
