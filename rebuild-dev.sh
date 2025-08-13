#!/bin/bash
# Quick rebuild script for development environment

set -e

echo "🔨 Rebuilding VisionFlow Development Environment..."
echo ""
echo "This will:"
echo "  1. Stop existing containers"
echo "  2. Remove orphan containers (powerdev)"
echo "  3. Rebuild with nginx installed"
echo ""

# Stop existing containers
echo "📦 Stopping containers..."
docker compose --profile dev down --remove-orphans 2>/dev/null || true

# Remove the old image to force rebuild
echo "🗑️  Removing old image..."
docker rmi visionflow_container 2>/dev/null || true

# Rebuild with no cache to ensure nginx gets installed
echo "🏗️  Building new container (this may take a few minutes)..."
docker compose --profile dev build --no-cache

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "To start the environment, run:"
echo "  ./scripts/launch.sh"
echo ""
echo "Or directly:"
echo "  docker compose --profile dev up"