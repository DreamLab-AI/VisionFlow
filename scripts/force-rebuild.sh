#!/bin/bash
# Force complete rebuild with nginx

set -e

echo "🔨 Force Rebuilding VisionFlow Development Environment..."
echo ""
echo "This will completely rebuild from scratch to ensure nginx is installed"
echo ""


# Stop and remove container
echo "📦 Stopping and removing container..."
docker stop visionflow_container 2>/dev/null || true
docker rm visionflow_container 2>/dev/null || true

# Remove the image completely
echo "🗑️  Removing all related images..."
docker rmi ext-webxr-dev 2>/dev/null || true
docker rmi visionflow_container 2>/dev/null || true
docker images | grep "none" | awk '{print $3}' | xargs -r docker rmi 2>/dev/null || true

# Clear Docker build cache for this project
echo "🧹 Clearing Docker build cache..."
docker builder prune -f 2>/dev/null || true

# Rebuild from scratch
echo "🏗️  Building new container from scratch (this will take a few minutes)..."
docker compose --profile dev build --no-cache --progress=plain

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "To start the environment, run:"
echo "  ./scripts/launch.sh"
echo ""
echo "Or directly:"
echo "  docker compose --profile dev up"