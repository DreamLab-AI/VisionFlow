#!/usr/bin/env python3
"""Patch SAM3D subprocess bridge to include CUDA environment"""

import re
from pathlib import Path

def patch_bridge(bridge_file: Path):
    """Patch the subprocess_bridge.py to include CUDA libraries in worker env"""

    print(f"Patching {bridge_file}")

    # Read original file
    with open(bridge_file, 'r') as f:
        content = f.read()

    # Backup original
    backup = bridge_file.with_suffix('.py.backup')
    if not backup.exists():
        with open(backup, 'w') as f:
            f.write(content)
        print(f"Created backup: {backup}")

    # Check if already patched
    if '_get_worker_env' in content:
        print("Already patched!")
        return

    # Add import os if not present
    if 'import os' not in content:
        content = content.replace('import subprocess', 'import subprocess\nimport os')

    # Find the Popen call and add env parameter
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

    # Write patched file
    with open(bridge_file, 'w') as f:
        f.write(content)

    print("Patch applied successfully!")

if __name__ == '__main__':
    bridge_file = Path('/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/nodes/subprocess_bridge.py')
    patch_bridge(bridge_file)
