#!/bin/bash
# Simple rebuild and run script for development

set -e

echo "🔨 Rebuilding and running VisionFlow Development Environment..."
echo ""

cd /workspace/ext

# Stop existing container
echo "📦 Stopping existing container..."
docker stop visionflow_container 2>/dev/null || true
docker rm visionflow_container 2>/dev/null || true

# Build the development image
echo "🏗️  Building development image..."
docker build -f Dockerfile.dev -t visionflow-dev:latest \
    --build-arg CUDA_ARCH=86 .

# Run the container
echo "🚀 Starting container..."
docker run -d \
    --name visionflow_container \
    -p 3001:3001 \
    -p 4000:4000 \
    -p 5173:5173 \
    -p 24678:24678 \
    -v $(pwd)/client:/app/client \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/Cargo.toml:/app/Cargo.toml \
    -v $(pwd)/Cargo.lock:/app/Cargo.lock \
    -v $(pwd)/data/markdown:/app/data/markdown \
    -v $(pwd)/data/metadata:/app/data/metadata \
    -v $(pwd)/data/user_settings:/app/user_settings \
    -v $(pwd)/data/settings.yaml:/app/settings.yaml \
    -v $(pwd)/nginx.dev.conf:/etc/nginx/nginx.conf:ro \
    -v $(pwd)/logs/nginx:/var/log/nginx \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/supervisord.dev.conf:/app/supervisord.dev.conf:ro \
    --env-file .env \
    --env NODE_ENV=development \
    --env VITE_DEV_SERVER_PORT=5173 \
    --env VITE_API_PORT=4000 \
    --env VITE_HMR_PORT=24678 \
    --env RUST_LOG_REDIRECT=true \
    --env SYSTEM_NETWORK_PORT=4000 \
    --env CLAUDE_FLOW_HOST=multi-agent-container \
    --env MCP_TCP_PORT=9500 \
    --env MCP_TRANSPORT=tcp \
    visionflow-dev:latest

echo ""
echo "✅ Container started!"
echo ""
echo "View logs with:"
echo "  docker logs -f visionflow_container"
echo ""
echo "Access the application at:"
echo "  http://localhost:3001"