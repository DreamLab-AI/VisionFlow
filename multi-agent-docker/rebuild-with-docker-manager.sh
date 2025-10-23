#!/usr/bin/env bash
# Rebuild agentic-workstation container with Docker Manager skill
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root where docker-compose.unified.yml is located
cd "$PROJECT_ROOT"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Rebuild Agentic Workstation with Docker Manager      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if skill exists
if [[ ! -d "multi-agent-docker/skills/docker-manager" ]]; then
    error "Docker Manager skill not found at multi-agent-docker/skills/docker-manager"
    exit 1
fi

log "Docker Manager skill found ✓"

# Verify skill files
REQUIRED_FILES=(
    "multi-agent-docker/skills/docker-manager/SKILL.md"
    "multi-agent-docker/skills/docker-manager/README.md"
    "multi-agent-docker/skills/docker-manager/tools/docker_manager.py"
    "multi-agent-docker/skills/docker-manager/tools/visionflow_ctl.sh"
    "multi-agent-docker/skills/docker-manager/config/docker-auth.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        error "Required file missing: $file"
        exit 1
    fi
done

log "All required skill files present ✓"

# Check if container is running
if docker ps | grep -q agentic-workstation; then
    warning "Agentic workstation is currently running"
    read -p "Stop and rebuild? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Cancelled by user"
        exit 0
    fi

    log "Stopping agentic-workstation..."
    docker stop agentic-workstation || true
    docker rm agentic-workstation || true
    success "Container stopped"
fi

# Check for docker-compose file
COMPOSE_FILE="docker-compose.unified.yml"
if [[ ! -f "$COMPOSE_FILE" ]]; then
    COMPOSE_FILE="docker-compose.yml"
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    error "No docker-compose file found"
    exit 1
fi

log "Using compose file: $COMPOSE_FILE"

# Build the container
log "Building agentic-workstation with Docker Manager skill..."
echo ""

# Disable BuildKit to avoid --allow flag bug in Docker Compose 2.40.x
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

if docker compose -f "$COMPOSE_FILE" build --no-cache agentic-workstation; then
    success "Build completed successfully"
else
    error "Build failed"
    exit 1
fi

# Ask to start container
echo ""
read -p "Start the container now? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    log "Starting agentic-workstation..."

    if docker compose -f "$COMPOSE_FILE" up -d agentic-workstation; then
        success "Container started"

        # Wait for container to be ready
        log "Waiting for container to be ready..."
        sleep 5

        # Verify skill installation
        log "Verifying Docker Manager skill installation..."
        if docker exec agentic-workstation test -f /home/devuser/.claude/skills/docker-manager/SKILL.md; then
            success "Docker Manager skill installed successfully!"
            echo ""
            echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║              Installation Successful!                  ║${NC}"
            echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo "Access the container:"
            echo "  SSH: ssh devuser@localhost -p 2222 (password: turboflow)"
            echo "  Exec: docker exec -it agentic-workstation /bin/zsh"
            echo ""
            echo "Test the skill:"
            echo "  docker exec -it agentic-workstation /home/devuser/.claude/skills/docker-manager/test-skill.sh"
            echo ""
            echo "Use from Claude Code:"
            echo "  'Use Docker Manager to check VisionFlow status'"
            echo ""
        else
            warning "Skill files not found in container - may need manual verification"
        fi
    else
        error "Failed to start container"
        exit 1
    fi
else
    log "Container not started. Start manually with:"
    echo "  docker compose -f $COMPOSE_FILE up -d agentic-workstation"
fi

echo ""
success "Done!"
