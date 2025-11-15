#!/bin/bash
# Quick rebuild and restart script

set -e

echo "🏗️  Rebuilding Turbo Flow Unified Container..."
echo ""
echo "This will:"
echo "  1. Rebuild Docker image with latest configs"
echo "  2. Restart container with new image"
echo "  3. Verify all services start correctly"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

cd "$(dirname "$0")"

echo "📦 Building image..."
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .

echo "🔄 Restarting container..."
docker compose -f docker-compose.unified.yml down
docker compose -f docker-compose.unified.yml up -d

echo "⏳ Waiting for services to start (30s)..."
sleep 30

echo "✅ Checking service status..."
docker exec agentic-workstation /opt/venv/bin/supervisorctl status

echo ""
echo "🎉 Rebuild complete!"
echo ""
echo "Connect via VNC: vnc://localhost:5901"
echo "Password: turboflow"
echo ""
echo "Or attach to tmux:"
echo "  docker exec -it -u devuser agentic-workstation tmux attach -t workspace"
echo ""
