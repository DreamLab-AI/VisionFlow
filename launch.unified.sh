#!/bin/bash
# VisionFlow Unified Launch Script - Simple, unified launcher for all environments
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.unified.yml"
CONTAINER_NAME="visionflow_container"

# Default values
COMMAND="${1:-up}"
ENVIRONMENT="${2:-dev}"

# Logging functions
log() {
    echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"
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

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Show help
show_help() {
    cat << EOF
${GREEN}╔════════════════════════════════════════════════════════════╗
║         VisionFlow Unified Launch Script                   ║
╚════════════════════════════════════════════════════════════╝${NC}

${YELLOW}Usage:${NC}
    ./launch.unified.sh [COMMAND] [ENVIRONMENT]

${YELLOW}Commands:${NC}
    ${GREEN}up${NC}         Start the environment (default)
    ${GREEN}down${NC}       Stop and remove containers
    ${GREEN}build${NC}      Build containers
    ${GREEN}rebuild${NC}    Rebuild containers (no cache)
    ${GREEN}logs${NC}       Show container logs (follow mode)
    ${GREEN}shell${NC}      Open interactive shell in container
    ${GREEN}restart${NC}    Restart the environment
    ${GREEN}status${NC}     Show container status and URLs
    ${GREEN}clean${NC}      Clean all containers, volumes, and images

${YELLOW}Environments:${NC}
    ${GREEN}dev${NC}        Development environment (default)
                - BUILD_TARGET=development
                - Verbose logging enabled
                - Hot reload enabled
                - No restart policy

    ${GREEN}prod${NC}       Production environment
                - BUILD_TARGET=production
                - Minimal logging
                - Restart policy: unless-stopped
                - Optimized builds

${YELLOW}Examples:${NC}
    ./launch.unified.sh                    ${CYAN}# Start dev environment${NC}
    ./launch.unified.sh up dev             ${CYAN}# Start dev environment${NC}
    ./launch.unified.sh build prod         ${CYAN}# Build production${NC}
    ./launch.unified.sh rebuild prod       ${CYAN}# Rebuild production (no cache)${NC}
    ./launch.unified.sh logs dev           ${CYAN}# View dev logs${NC}
    ./launch.unified.sh shell prod         ${CYAN}# Open prod shell${NC}
    ./launch.unified.sh restart dev        ${CYAN}# Restart dev${NC}
    ./launch.unified.sh clean              ${CYAN}# Clean everything${NC}

${YELLOW}Environment Files:${NC}
    .env.dev       Development configuration
    .env.prod      Production configuration

${YELLOW}GPU Support:${NC}
    Automatically detected via nvidia-smi
    Enabled in containers when GPU is available

EOF
}

# Validate command
validate_command() {
    case "$COMMAND" in
        up|down|build|rebuild|logs|shell|restart|status|clean|help|-h|--help)
            if [[ "$COMMAND" == "help" ]] || [[ "$COMMAND" == "-h" ]] || [[ "$COMMAND" == "--help" ]]; then
                show_help
                exit 0
            fi
            ;;
        *)
            error "Invalid command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        dev|prod)
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            echo "Valid environments: dev, prod"
            exit 1
            ;;
    esac
}

# Load environment-specific configuration
load_env_config() {
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"

    if [[ -f "$env_file" ]]; then
        success "Loading environment config: .env.$ENVIRONMENT"
        set -a
        source "$env_file"
        set +a
    else
        warning "Environment file not found: $env_file"
        if [[ -f "$PROJECT_ROOT/.env" ]]; then
            info "Using default .env file"
            set -a
            source "$PROJECT_ROOT/.env"
            set +a
        else
            error "No .env file found. Please create .env.$ENVIRONMENT or .env"
            exit 1
        fi
    fi
}

# Set environment-specific variables
set_environment_vars() {
    case "$ENVIRONMENT" in
        dev)
            export BUILD_TARGET="development"
            export COMPOSE_PROFILES="dev"
            export LOG_LEVEL="debug"
            export RESTART_POLICY="no"
            info "Environment: Development"
            info "  - Verbose logging enabled"
            info "  - Hot reload enabled"
            info "  - No restart policy"
            ;;
        prod)
            export BUILD_TARGET="production"
            export COMPOSE_PROFILES="prod"
            export LOG_LEVEL="info"
            export RESTART_POLICY="unless-stopped"
            info "Environment: Production"
            info "  - Minimal logging"
            info "  - Optimized builds"
            info "  - Restart policy: unless-stopped"
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    success "Docker: $(docker --version)"

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    success "Docker Compose: $(docker compose version)"

    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    success "Compose file: docker-compose.unified.yml"

    success "Prerequisites check complete"
}

# Detect and validate GPU
detect_gpu() {
    log "Detecting GPU..."

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
        if [[ -n "$GPU_INFO" ]]; then
            success "GPU detected: $GPU_INFO"
            export GPU_AVAILABLE="true"
            export NVIDIA_RUNTIME="nvidia"

            # Check NVIDIA Docker runtime
            if docker info 2>/dev/null | grep -q nvidia; then
                success "NVIDIA Docker runtime: Available"
            else
                warning "NVIDIA Docker runtime not found in Docker info"
                info "Install nvidia-container-toolkit for GPU passthrough"
            fi
        else
            warning "NVIDIA GPU not detected"
            export GPU_AVAILABLE="false"
        fi
    else
        warning "nvidia-smi not found - GPU support disabled"
        export GPU_AVAILABLE="false"
    fi
}

# Docker Compose wrapper
docker_compose() {
    cd "$PROJECT_ROOT"
    docker compose -f "$COMPOSE_FILE" --profile "$COMPOSE_PROFILES" "$@"
}

# Build containers
build_containers() {
    log "Building containers for $ENVIRONMENT environment..."

    local build_args=()

    if [[ "$COMMAND" == "rebuild" ]]; then
        info "Rebuild mode: Using --no-cache"
        build_args+=("--no-cache")
    fi

    # Add GPU support if available
    if [[ "${GPU_AVAILABLE:-false}" == "true" ]]; then
        info "Building with GPU support"
        build_args+=("--build-arg" "FEATURES=gpu")
    else
        info "Building without GPU support"
    fi

    # Disable BuildKit to avoid compatibility issues
    export DOCKER_BUILDKIT=0
    export COMPOSE_DOCKER_CLI_BUILD=0

    docker_compose build "${build_args[@]}"
    success "Build complete for $ENVIRONMENT environment"
}

# Start environment
start_environment() {
    log "Starting $ENVIRONMENT environment..."

    # Build if containers don't exist
    if ! docker images | grep -q "visionflow"; then
        info "Container images not found. Building first..."
        build_containers
    fi

    # Conditionally start cloudflared based on environment
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        info "Development mode: Skipping cloudflared tunnel (local access only)"
        docker_compose up -d --scale cloudflared=0
    else
        info "Production mode: Starting cloudflared tunnel"
        docker_compose up -d
    fi

    # Wait for containers to be ready
    sleep 3

    success "Environment started in background"
    echo ""
    show_service_urls
    echo ""
    info "View logs with: ${GREEN}./launch.unified.sh logs $ENVIRONMENT${NC}"
    info "Stop with: ${GREEN}./launch.unified.sh down $ENVIRONMENT${NC}"
}

# Stop environment
stop_environment() {
    log "Stopping $ENVIRONMENT environment..."
    docker_compose down
    success "Environment stopped"
}

# Restart environment
restart_environment() {
    log "Restarting $ENVIRONMENT environment..."
    stop_environment
    sleep 2
    start_environment
}

# Show logs
show_logs() {
    log "Showing logs for $ENVIRONMENT environment..."
    info "Press Ctrl+C to exit log view"
    echo ""
    docker_compose logs -f
}

# Open shell
open_shell() {
    log "Opening interactive shell in $ENVIRONMENT container..."

    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        error "Container is not running. Start it first with: ./launch.unified.sh up $ENVIRONMENT"
        exit 1
    fi

    info "Entering container shell..."
    docker exec -it "$CONTAINER_NAME" /bin/bash || docker exec -it "$CONTAINER_NAME" /bin/sh
}

# Show status
show_status() {
    log "Container status for $ENVIRONMENT environment:"
    echo ""
    docker_compose ps
    echo ""

    if docker ps -q -f name="$CONTAINER_NAME" &> /dev/null; then
        show_service_urls

        # Show resource usage
        echo ""
        log "Resource usage:"
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep "$CONTAINER_NAME" || true
    else
        warning "Container is not running"
        info "Start with: ${GREEN}./launch.unified.sh up $ENVIRONMENT${NC}"
    fi
}

# Show service URLs
show_service_urls() {
    log "Service URLs:"

    if [[ "$ENVIRONMENT" == "dev" ]]; then
        echo "  ${GREEN}Vite Dev:${NC}      http://localhost:3001"
        echo "  ${GREEN}Web UI:${NC}        http://localhost:4000 (→ 3001)"
    else
        echo "  ${GREEN}Web UI:${NC}        http://localhost:4000"
    fi

    echo "  ${GREEN}WebSocket:${NC}     ws://localhost:4000/ws"
    echo "  ${GREEN}Claude Flow:${NC}   tcp://localhost:9500"

    # Check for cloudflared tunnel
    if docker ps 2>/dev/null | grep -q cloudflared-tunnel; then
        echo ""
        success "Cloudflared tunnel: Active"
        echo "  ${GREEN}Public URL:${NC}    https://www.visionflow.info"
    fi
}

# Clean everything
clean_all() {
    warning "This will remove ALL VisionFlow containers, volumes, and images"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo

    if [[ "$REPLY" == "yes" ]]; then
        log "Cleaning all VisionFlow resources..."

        # Stop all containers for both environments
        for env in dev prod; do
            export COMPOSE_PROFILES="$env"
            log "Stopping $env environment..."
            docker_compose down -v 2>/dev/null || true
        done

        # Remove images
        log "Removing VisionFlow images..."
        docker images | grep visionflow | awk '{print $3}' | xargs -r docker rmi -f || true

        # Clean build cache
        log "Cleaning build cache..."
        docker builder prune -f

        success "Cleanup complete"
    else
        info "Cleanup cancelled"
    fi
}

# Show banner
show_banner() {
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         VisionFlow Unified Launcher                        ║${NC}"
    echo -e "${GREEN}║  Command:     ${CYAN}$(printf '%-42s' "$COMMAND")${GREEN}║${NC}"
    echo -e "${GREEN}║  Environment: ${YELLOW}$(printf '%-42s' "$ENVIRONMENT")${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Main execution
main() {
    # Validate inputs
    validate_command
    validate_environment

    # Show banner
    show_banner

    # Load configuration
    load_env_config
    set_environment_vars

    # Execute command
    case "$COMMAND" in
        up)
            check_prerequisites
            detect_gpu
            start_environment
            ;;
        down)
            stop_environment
            ;;
        build)
            check_prerequisites
            detect_gpu
            build_containers
            ;;
        rebuild)
            check_prerequisites
            detect_gpu
            build_containers
            ;;
        logs)
            show_logs
            ;;
        shell)
            open_shell
            ;;
        restart)
            check_prerequisites
            detect_gpu
            restart_environment
            ;;
        status)
            show_status
            ;;
        clean)
            clean_all
            ;;
    esac
}

# Run main function
main
