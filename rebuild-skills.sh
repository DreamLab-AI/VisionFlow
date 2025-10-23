#!/usr/bin/env bash
# Quick rebuild script for agentic-workstation with new skills
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Rebuild Agentic Workstation with Skills              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verify we're in project root
if [[ ! -f "docker-compose.unified.yml" ]]; then
    error "Must run from project root (AR-AI-Knowledge-Graph/)"
    exit 1
fi

# Check skills exist
if [[ ! -d "multi-agent-docker/skills/docker-manager" ]]; then
    error "Docker Manager skill not found"
    exit 1
fi

if [[ ! -d "multi-agent-docker/skills/chrome-devtools" ]]; then
    error "Chrome DevTools skill not found"
    exit 1
fi

log "Docker Manager skill found ✓"
log "Chrome DevTools skill found ✓"

# Check if container is running
if docker ps | grep -q agentic-workstation; then
    warning "Agentic workstation is currently running"
    read -p "Stop and rebuild? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Cancelled"
        exit 0
    fi

    log "Stopping container..."
    docker stop agentic-workstation || true
    docker rm agentic-workstation || true
    success "Container stopped"
fi

# Build
log "Building agentic-workstation..."
echo ""

# Disable BuildKit to avoid --allow flag bug in Docker Compose 2.40.x
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

if docker compose -f docker-compose.unified.yml build --no-cache agentic-workstation; then
    success "Build completed!"
else
    error "Build failed"
    exit 1
fi

# Start
echo ""
read -p "Start container now? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    log "Starting container..."

    if docker compose -f docker-compose.unified.yml up -d agentic-workstation; then
        success "Container started!"

        # Wait and verify
        log "Waiting for container..."
        sleep 5

        # Check skills
        if docker exec agentic-workstation test -f /home/devuser/.claude/skills/docker-manager/SKILL.md && \
           docker exec agentic-workstation test -f /home/devuser/.claude/skills/chrome-devtools/SKILL.md; then
            echo ""
            success "Skills installed successfully!"
            echo ""
            echo "✓ Docker Manager skill"
            echo "✓ Chrome DevTools skill"
            echo ""
            echo "Access container:"
            echo "  SSH: ssh devuser@localhost -p 2222 (password: turboflow)"
            echo "  Exec: docker exec -it agentic-workstation /bin/zsh"
            echo ""
            echo "Test Docker Manager:"
            echo "  docker exec -it agentic-workstation /home/devuser/.claude/skills/docker-manager/test-skill.sh"
            echo ""
        else
            warning "Could not verify skill installation"
        fi
    else
        error "Failed to start container"
        exit 1
    fi
else
    log "Start manually with:"
    echo "  docker compose -f docker-compose.unified.yml up -d agentic-workstation"
fi

echo ""
success "Done!"
