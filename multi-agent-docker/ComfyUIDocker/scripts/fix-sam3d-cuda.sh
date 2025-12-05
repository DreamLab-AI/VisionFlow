#!/bin/bash
# Fix SAM3D CUDA environment
# This script ensures the SAM3D isolated venv has proper CUDA library paths

set -e

SAM3D_VENV="/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/_env"

if [ ! -d "$SAM3D_VENV" ]; then
    echo "SAM3D venv not found at $SAM3D_VENV"
    exit 0
fi

echo "=== Fixing SAM3D CUDA Environment ==="

# Find Python version in venv
PYTHON_VERSION=$(ls $SAM3D_VENV/lib/ | grep python | head -1)
echo "Python version: $PYTHON_VERSION"

# Build LD_LIBRARY_PATH for SAM3D venv
NVIDIA_LIBS=""
for lib in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft curand cusolver cusparse nccl nvjitlink nvtx; do
    LIB_PATH="$SAM3D_VENV/lib/$PYTHON_VERSION/site-packages/nvidia/$lib/lib"
    if [ -d "$LIB_PATH" ]; then
        NVIDIA_LIBS="$NVIDIA_LIBS:$LIB_PATH"
    fi
done

# Remove leading colon
NVIDIA_LIBS="${NVIDIA_LIBS#:}"

echo "NVIDIA library paths: $NVIDIA_LIBS"

# Create activation script that sets LD_LIBRARY_PATH
cat > "$SAM3D_VENV/bin/activate-cuda" << EOF
#!/bin/bash
# Source this after activating the venv to get CUDA libraries
export LD_LIBRARY_PATH="$NVIDIA_LIBS:\${LD_LIBRARY_PATH}"
echo "CUDA libraries added to LD_LIBRARY_PATH"
EOF

chmod +x "$SAM3D_VENV/bin/activate-cuda"

# Create a wrapper script for the inference worker
WORKER_SCRIPT="/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/inference_worker.py"
WRAPPER_SCRIPT="/root/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/inference_worker_wrapper.sh"

cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
# Wrapper that sets LD_LIBRARY_PATH before running the worker
export LD_LIBRARY_PATH="$NVIDIA_LIBS:\${LD_LIBRARY_PATH}"
exec $SAM3D_VENV/bin/python $WORKER_SCRIPT "\$@"
EOF

chmod +x "$WRAPPER_SCRIPT"

echo "=== SAM3D CUDA Fix Complete ==="
echo "Created: $SAM3D_VENV/bin/activate-cuda"
echo "Created: $WRAPPER_SCRIPT"
echo ""
echo "To use manually:"
echo "  source $SAM3D_VENV/bin/activate"
echo "  source $SAM3D_VENV/bin/activate-cuda"
