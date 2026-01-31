#!/bin/bash
# ============================================================================
# AGENTIC WORKSTATION - Canonical Build System v3.0
# ============================================================================
#
# VERSION:     3.0.0
# UPDATED:     2026-01-31
#
# This is the CANONICAL build script for the unified agentic development
# workstation. Use this script to build and launch the container.
#
# USAGE:
#   ./build-unified.sh                 # Standard build
#   ./build-unified.sh --no-cache      # Force full rebuild
#   ./build-unified.sh --skip-comfyui  # Skip ComfyUI check
#   ./build-unified.sh --comfyui-full  # Build ComfyUI with full open3d
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build version
BUILD_VERSION="3.0.0"
BUILD_DATE="2026-01-31"

# Enable BuildKit for better caching and parallel builds
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Parse arguments
BUILD_ARGS=""
SKIP_COMFYUI=false
BUILD_COMFYUI_FULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            BUILD_ARGS="--no-cache"
            echo "🔄 Building without cache..."
            shift
            ;;
        --skip-comfyui)
            SKIP_COMFYUI=true
            echo "⏭️  Skipping ComfyUI deployment..."
            shift
            ;;
        --comfyui-full)
            BUILD_COMFYUI_FULL=true
            echo "🔨 Will build ComfyUI with full open3d (takes 30-60 min)..."
            shift
            ;;
        --version|-v)
            echo "Agentic Workstation Build System v${BUILD_VERSION} (${BUILD_DATE})"
            exit 0
            ;;
        --help|-h)
            echo "Agentic Workstation Build System v${BUILD_VERSION}"
            echo ""
            echo "Usage: ./build-unified.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cache      Force full rebuild without Docker cache"
            echo "  --skip-comfyui  Skip ComfyUI deployment check"
            echo "  --comfyui-full  Build ComfyUI with full open3d support"
            echo "  --version, -v   Show version information"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   AGENTIC WORKSTATION v${BUILD_VERSION} - Canonical Build System        ║"
echo "║   Claude Flow V3 | 62+ Skills | Multi-Agent Orchestration       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
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

# Check if ragflow network exists, create if needed
echo "Checking docker_ragflow network..."
if ! docker network inspect docker_ragflow >/dev/null 2>&1; then
    echo "Creating docker_ragflow network..."
    docker network create docker_ragflow
    echo "✅ Network created"
else
    echo "✅ Network exists"
fi

echo ""

# Build the agentic workstation container
echo "[1/4] Building Agentic Workstation Docker image..."
docker build $BUILD_ARGS -f Dockerfile.unified -t agentic-workstation:latest .

echo ""
echo "[2/4] Launching Agentic Workstation..."
docker compose -f docker-compose.unified.yml up -d

echo ""
echo "[3/4] Waiting for services to start..."
sleep 10

# Check services
echo ""
echo "Service Status:"
docker exec agentic-workstation /opt/venv/bin/supervisorctl status

echo ""
echo "========================================"
echo "  GPU VERIFICATION"
echo "========================================"
echo ""

# Test GPU access
echo "Testing NVIDIA GPU access..."
docker exec agentic-workstation nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
    echo "⚠️  GPU not accessible - check NVIDIA runtime configuration"

echo ""
echo "Testing PyTorch CUDA..."
docker exec agentic-workstation /opt/venv/bin/python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  WARNING: PyTorch cannot access CUDA')
    print('   Image generation will be CPU-only and very slow')
" 2>/dev/null || echo "⚠️  PyTorch test failed"

echo ""
echo "Testing ComfyUI installation..."
if docker exec agentic-workstation test -d /home/devuser/ComfyUI; then
    echo "✅ ComfyUI installed at /home/devuser/ComfyUI"
    if docker exec agentic-workstation test -f /home/devuser/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors; then
        echo "✅ FLUX model downloaded"
    else
        echo "⚠️  FLUX model not found - will download on first use"
    fi
else
    echo "⚠️  ComfyUI not installed"
fi

echo ""
echo "Testing ComfyUI service..."
if docker exec agentic-workstation /opt/venv/bin/supervisorctl status comfyui | grep -q RUNNING; then
    echo "✅ ComfyUI service running (port 8188)"

    # Wait a moment for it to fully start
    sleep 3

    # Test if ComfyUI is responding
    if docker exec agentic-workstation curl -s http://localhost:8188/system_stats >/dev/null 2>&1; then
        echo "✅ ComfyUI API responding"

        # Show device info
        DEVICE_INFO=$(docker exec agentic-workstation curl -s http://localhost:8188/system_stats 2>/dev/null | \
            docker exec -i agentic-workstation /opt/venv/bin/python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Device: {data['devices'][0]['name']} ({data['devices'][0]['type']})\")" 2>/dev/null || echo "Device info unavailable")
        echo "   $DEVICE_INFO"
    else
        echo "⚠️  ComfyUI not responding yet (may still be starting)"
    fi
else
    echo "⚠️  ComfyUI service not running"
    echo "   Check logs: docker exec agentic-workstation /opt/venv/bin/supervisorctl tail -f comfyui"
fi

echo ""
echo "========================================"
echo "  COMFYUI STANDALONE DEPLOYMENT"
echo "========================================"
echo ""

# Deploy standalone ComfyUI with open3d support
if [ "$SKIP_COMFYUI" = true ]; then
    echo "⏭️  Skipping standalone ComfyUI deployment (--skip-comfyui flag)"
elif [ "$BUILD_COMFYUI_FULL" = true ]; then
    echo "[4/4] Building ComfyUI with full open3d support..."
    echo "⚠️  This will take 30-60 minutes!"
    echo ""
    cd comfyui
    ./build-comfyui.sh
    cd ..
else
    echo "[4/4] Deploying ComfyUI standalone container..."
    echo ""

    # Check if existing comfyui container has open3d stub
    if docker ps -a | grep -q "^comfyui"; then
        echo "Checking existing ComfyUI container for open3d..."
        if docker exec comfyui python3 -c "import open3d; print(open3d.__version__)" 2>/dev/null | grep -q "stub"; then
            echo "✅ ComfyUI already running with open3d stub"
            echo "   Container: comfyui"
            echo "   Network: docker_ragflow"
            echo "   Access: http://localhost:8188"
            echo "   open3d: $(docker exec comfyui python3 -c 'import open3d; print(open3d.__version__)' 2>/dev/null)"
            echo ""
            echo "To rebuild with full open3d: ./build-unified.sh --comfyui-full"
        else
            echo "⚠️  ComfyUI running but open3d not detected"
            echo "   Install stub: docker exec comfyui python3 -m pip install trimesh pyvista"
        fi
    else
        echo "⚠️  Standalone ComfyUI container not found"
        echo ""
        echo "To deploy ComfyUI with open3d stub (quick):"
        echo "  1. Current container already has stub: docker exec comfyui python3 -c 'import open3d'"
        echo ""
        echo "To build with full open3d support (30-60 min):"
        echo "  ./build-unified.sh --comfyui-full"
        echo ""
        echo "Or manually:"
        echo "  cd comfyui && ./build-comfyui.sh"
    fi
fi

echo ""
echo "========================================"
echo "  ✅ DEPLOYMENT COMPLETE"
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
echo "========================================"
echo "  ACCESS INFORMATION"
echo "========================================"
echo ""
echo "Agentic Workstation:"
echo "  SSH:         ssh -p 2222 devuser@localhost  (password: turboflow)"
echo "  VNC:         vnc://localhost:5901           (password: turboflow)"
echo "  code-server: http://localhost:8080"
echo "  API:         http://localhost:9090/health"
echo "  Swagger:     http://localhost:9090/documentation"
echo ""

# Check ComfyUI standalone status
if docker ps | grep -q "comfyui"; then
    echo "ComfyUI Standalone (with open3d):"
    echo "  Web UI:      http://localhost:8188"
    echo "  From ragflow: http://comfyui.ragflow:8188"
    echo "  Container:   comfyui"
    OPEN3D_VER=$(docker exec comfyui python3 -c "import open3d; print(open3d.__version__)" 2>/dev/null || echo "not installed")
    echo "  open3d:      $OPEN3D_VER"
    if [[ "$OPEN3D_VER" == *"stub"* ]]; then
        echo "               (stub - basic functionality)"
        echo "               To build full: ./build-unified.sh --comfyui-full"
    fi
    echo ""
fi

echo "SSH Credentials:"
./unified-config/scripts/ssh-setup.sh status 2>/dev/null || echo "  Use: ./unified-config/scripts/ssh-setup.sh for SSH management"
echo ""
echo "Skills:"
echo "  Test Docker Manager: docker exec -it agentic-workstation /home/devuser/.claude/skills/docker-manager/test-skill.sh"
echo "  From Claude:         'Use Docker Manager to check ComfyUI status'"
echo ""
echo "Management Commands:"
echo "  View logs:   docker compose -f docker-compose.unified.yml logs -f"
echo "  Stop all:    docker compose -f docker-compose.unified.yml down && docker stop comfyui"
echo "  Restart:     docker compose -f docker-compose.unified.yml restart"
echo "  Shell:       docker exec -it agentic-workstation zsh"
echo "  ComfyUI:     docker exec -it comfyui bash"
echo ""
echo "ComfyUI Management:"
echo "  View logs:   docker logs comfyui -f"
echo "  Restart:     docker restart comfyui"
echo "  Stop:        docker stop comfyui"
echo "  Remove:      docker stop comfyui && docker rm comfyui"
echo ""
echo "Build Options:"
echo "  ./build-unified.sh                 # Standard build"
echo "  ./build-unified.sh --no-cache      # Force rebuild"
echo "  ./build-unified.sh --skip-comfyui  # Skip ComfyUI deployment check"
echo "  ./build-unified.sh --comfyui-full  # Build ComfyUI with full open3d (30-60 min)"
echo ""
