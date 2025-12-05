#!/bin/bash
# Enhanced entrypoint that includes SAM3D CUDA fix
# This wraps the original entrypoint and applies fixes after ComfyUI starts

set -e

echo "===================================================================================="
echo "Starting ComfyUI with SAM3D CUDA support..."
echo "===================================================================================="

# Start ComfyUI in background
/runner-scripts/entrypoint.sh &
COMFYUI_PID=$!

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to start..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Warning: ComfyUI did not start within ${MAX_WAIT}s"
fi

# Apply SAM3D fix if available
if [ -f /runner-scripts/fix-sam3d-on-startup.sh ]; then
    echo "Applying SAM3D CUDA fix..."
    bash /runner-scripts/fix-sam3d-on-startup.sh
fi

# Keep ComfyUI running
wait $COMFYUI_PID
