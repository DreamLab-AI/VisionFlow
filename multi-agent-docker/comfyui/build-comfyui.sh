#!/usr/bin/env bash
# Build and deploy ComfyUI with full open3d support

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "Building ComfyUI with open3d support"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Build a custom ComfyUI image with open3d from source"
echo "  2. Deploy it on the docker_ragflow network"
echo "  3. Persist models, custom nodes, and outputs"
echo ""
echo "Build time: 30-60 minutes (open3d compilation)"
echo ""

# Check if ragflow network exists
if ! docker network inspect docker_ragflow >/dev/null 2>&1; then
    echo "Error: docker_ragflow network not found"
    echo "Please create it first: docker network create docker_ragflow"
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a | grep -q comfyui; then
    echo "Stopping existing ComfyUI container..."
    docker stop comfyui 2>/dev/null || true
    docker rm comfyui 2>/dev/null || true
fi

# Build the image
echo ""
echo "Building ComfyUI image with open3d..."
echo "This will take 30-60 minutes due to open3d compilation."
echo ""

docker-compose -f docker-compose.comfyui.yml build

# Start the container
echo ""
echo "Starting ComfyUI container..."
echo ""

docker-compose -f docker-compose.comfyui.yml up -d

# Wait for health check
echo ""
echo "Waiting for ComfyUI to be healthy..."
for i in {1..60}; do
    if docker inspect comfyui --format='{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy"; then
        echo "ComfyUI is healthy!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Warning: ComfyUI didn't become healthy within 5 minutes"
        echo "Check logs: docker logs comfyui"
    fi
    sleep 5
done

echo ""
echo "======================================================================"
echo "ComfyUI deployed successfully!"
echo "======================================================================"
echo ""
echo "Access ComfyUI at: http://localhost:8188"
echo "Container name: comfyui"
echo "Network: docker_ragflow"
echo ""
echo "Useful commands:"
echo "  docker logs comfyui          # View logs"
echo "  docker exec -it comfyui bash # Enter container"
echo "  docker-compose -f docker-compose.comfyui.yml down  # Stop"
echo ""
