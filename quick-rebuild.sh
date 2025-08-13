#!/bin/bash
# Quick rebuild script to fix nginx PID issue

set -e

echo "🔨 Quick Rebuild - Fixing Nginx PID File Issue"
echo "=============================================="
echo ""

cd /workspace/ext

# Stop and remove existing container
echo "📦 Stopping existing container..."
docker stop visionflow_container 2>/dev/null || true
docker rm visionflow_container 2>/dev/null || true

# Remove old image to force rebuild
echo "🗑️  Removing old image..."
docker rmi ext-webxr-dev 2>/dev/null || true
docker rmi visionflow-dev:latest 2>/dev/null || true

# Build with docker compose if available, otherwise use docker directly
echo "🏗️  Building new image..."
if command -v docker-compose &> /dev/null; then
    docker-compose --profile dev build --no-cache
elif docker compose version &> /dev/null; then
    docker compose --profile dev build --no-cache
else
    echo "Building directly with Docker..."
    docker build -f Dockerfile.dev -t visionflow-dev:latest \
        --build-arg CUDA_ARCH=86 \
        --no-cache .
fi

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "Starting the container..."

# Try to start with docker compose first, fall back to direct run
if command -v docker-compose &> /dev/null; then
    docker-compose --profile dev up -d
elif docker compose version &> /dev/null; then
    docker compose --profile dev up -d
else
    echo "Starting with Docker run..."
    docker run -d \
        --name visionflow_container \
        -p 3001:3001 \
        -v $(pwd)/nginx.dev.conf:/etc/nginx/nginx.conf:ro \
        -v $(pwd)/supervisord.dev.conf:/app/supervisord.dev.conf:ro \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/logs/nginx:/var/log/nginx \
        --env-file .env \
        visionflow-dev:latest
fi

echo ""
echo "🚀 Container started!"
echo ""
echo "Check logs with:"
echo "  docker logs -f visionflow_container"
echo ""
echo "Check nginx specifically:"
echo "  docker exec visionflow_container tail -f /var/log/nginx/error.log"