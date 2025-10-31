#!/bin/bash
# Quick build and launch script for Turbo Flow Unified Container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Enable BuildKit for better caching and parallel builds
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Parse arguments
BUILD_ARGS=""
if [[ "$*" == *"--no-cache"* ]]; then
    BUILD_ARGS="--no-cache"
    echo "🔄 Building without cache..."
fi

echo "========================================"
echo "  AGENTIC WORKSTATION - BUILD & LAUNCH"
echo "========================================"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env from template"
        echo ""
        echo "⚠️  IMPORTANT: Edit .env and add your API keys before continuing!"
        echo ""
        read -p "Press Enter to continue (or Ctrl+C to exit and edit .env)..."
    else
        echo "❌ Error: .env.example not found"
        exit 1
    fi
fi

# Verify skills exist
echo "Verifying skills..."
if [[ ! -d "skills/docker-manager" ]]; then
    echo "⚠️  Warning: Docker Manager skill not found"
else
    echo "✅ Docker Manager skill found"
fi

if [[ ! -d "skills/chrome-devtools" ]]; then
    echo "⚠️  Warning: Chrome DevTools skill not found"
else
    echo "✅ Chrome DevTools skill found"
fi

echo ""

# Build the container
echo "[1/3] Building Docker image..."
docker build $BUILD_ARGS -f Dockerfile.unified -t agentic-workstation:latest .

echo ""
echo "[2/3] Launching container..."
docker compose -f docker-compose.unified.yml up -d

echo ""
echo "[3/3] Waiting for services to start..."
sleep 10

# Check services
echo ""
echo "Service Status:"
docker exec agentic-workstation supervisorctl status

echo ""
echo "========================================"
echo "  ✅ AGENTIC WORKSTATION RUNNING"
echo "========================================"
echo ""

# Verify skills installation
echo "Verifying skills installation..."
if docker exec agentic-workstation test -f /home/devuser/.claude/skills/docker-manager/SKILL.md 2>/dev/null; then
    echo "✅ Docker Manager skill installed"
else
    echo "⚠️  Docker Manager skill not found in container"
fi

if docker exec agentic-workstation test -f /home/devuser/.claude/skills/chrome-devtools/SKILL.md 2>/dev/null; then
    echo "✅ Chrome DevTools skill installed"
else
    echo "⚠️  Chrome DevTools skill not found in container"
fi

# Verify Docker socket
if docker exec agentic-workstation test -S /var/run/docker.sock 2>/dev/null; then
    echo "✅ Docker socket mounted"
else
    echo "⚠️  Docker socket not found - Docker Manager will not work"
fi

echo ""
echo "Access Methods:"
echo "  SSH:        ssh -p 2222 devuser@localhost  (password: turboflow)"
echo "  VNC:        vnc://localhost:5901           (password: turboflow)"
echo "  code-server: http://localhost:8080"
echo "  API:        http://localhost:9090/health"
echo "  Swagger:    http://localhost:9090/documentation"
echo ""
echo "Skills:"
echo "  Test Docker Manager: docker exec -it agentic-workstation /home/devuser/.claude/skills/docker-manager/test-skill.sh"
echo "  From Claude:        'Use Docker Manager to check VisionFlow status'"
echo ""
echo "Management Commands:"
echo "  View logs:  docker compose -f docker-compose.unified.yml logs -f"
echo "  Stop:       docker compose -f docker-compose.unified.yml down"
echo "  Restart:    docker compose -f docker-compose.unified.yml restart"
echo "  Shell:      docker exec -it agentic-workstation zsh"
echo ""
