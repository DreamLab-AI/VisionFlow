#!/bin/bash
# VisionFlow Launch Script - Modern unified launcher for all environments
set -euo pipefail

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_NAME="visionflow_container"

# Default values
PROFILE="dev"
ACTION="up"
NO_CACHE=false
DETACHED=false
VERBOSE=false
FORCE_REBUILD=false

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

# Help text
show_help() {
    cat << EOF
${GREEN}VisionFlow Launch Script${NC}

${YELLOW}Usage:${NC}
    ./launch.sh [OPTIONS] [COMMAND]

${YELLOW}Commands:${NC}
    up          Start the environment (default)
    down        Stop and remove containers
    restart     Restart the environment
    logs        Show container logs
    shell       Open shell in container
    status      Show container status
    build       Build containers only
    clean       Clean all containers and volumes

${YELLOW}Options:${NC}
    -p, --profile PROFILE    Environment profile (dev|production|prod) [default: dev]
    -d, --detached          Run in background
    -v, --verbose           Verbose output
    -f, --force-rebuild     Force rebuild containers
    -n, --no-cache          Build without cache
    -h, --help              Show this help message

${YELLOW}Examples:${NC}
    ./launch.sh                    # Start development environment
    ./launch.sh -p production      # Start production environment
    ./launch.sh -d logs            # Show logs in detached mode
    ./launch.sh clean              # Clean everything
    ./launch.sh -p dev restart     # Restart development environment

${YELLOW}Environment Variables:${NC}
    CUDA_ARCH               GPU architecture (default: 86)
    MCP_TCP_PORT           Claude Flow port (default: 9500)
    NVIDIA_VISIBLE_DEVICES  GPU device selection
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--profile)
                PROFILE="$2"
                shift 2
                ;;
            -d|--detached)
                DETACHED=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            -n|--no-cache)
                NO_CACHE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            up|down|restart|logs|shell|status|build|clean)
                ACTION="$1"
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi

    # Check for .env file
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        warning ".env file not found"
        if [[ -f "$PROJECT_ROOT/.env_template" ]]; then
            info "Creating .env from template..."
            cp "$PROJECT_ROOT/.env_template" "$PROJECT_ROOT/.env"
            warning "Please edit .env and add your API keys"
        else
            error "No .env or .env_template found"
            exit 1
        fi
    fi

    # Check GPU availability for non-CPU profiles
    if [[ "$PROFILE" != "cpu" ]]; then
        if ! nvidia-smi &> /dev/null; then
            warning "NVIDIA GPU not detected. GPU features will be disabled"
        else
            info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
        fi
    fi

    success "Prerequisites check complete"
}

# Compile CUDA kernels
compile_cuda() {
    if [[ "$PROFILE" != "cpu" ]] && nvidia-smi &> /dev/null; then
        log "Compiling CUDA kernels..."
        if [[ -f "$SCRIPT_DIR/compile_unified_ptx.sh" ]]; then
            if "$SCRIPT_DIR/compile_unified_ptx.sh"; then
                success "CUDA kernels compiled"
            else
                warning "CUDA compilation failed, continuing without GPU acceleration"
            fi
        else
            info "CUDA compilation script not found, will compile in container"
        fi
    fi
}

# Docker Compose wrapper
docker_compose() {
    local compose_args=()

    if [[ "$VERBOSE" == true ]]; then
        compose_args+=("--verbose")
    fi

    cd "$PROJECT_ROOT"
    docker compose --profile "$PROFILE" "${compose_args[@]}" "$@"
}

# Start environment
start_environment() {
    log "Starting $PROFILE environment..."

    local up_args=()

    if [[ "$DETACHED" == true ]]; then
        up_args+=("-d")
    fi

    if [[ "$FORCE_REBUILD" == true ]] || [[ "$NO_CACHE" == true ]]; then
        log "Building containers..."
        local build_args=()

        if [[ "$NO_CACHE" == true ]]; then
            build_args+=("--no-cache")
        fi

        DOCKER_BUILDKIT=1 docker_compose build "${build_args[@]}"
    fi

    docker_compose up "${up_args[@]}"

    if [[ "$DETACHED" == true ]]; then
        success "Environment started in background"
        info "View logs with: ./launch.sh logs"
        info "Stop with: ./launch.sh down"
    fi
}

# Stop environment
stop_environment() {
    log "Stopping $PROFILE environment..."
    docker_compose down
    success "Environment stopped"
}

# Restart environment
restart_environment() {
    stop_environment
    start_environment
}

# Show logs
show_logs() {
    log "Showing logs for $PROFILE environment..."
    docker_compose logs -f
}

# Open shell
open_shell() {
    log "Opening shell in container..."
    docker exec -it "$CONTAINER_NAME" /bin/bash
}

# Show status
show_status() {
    log "Container status:"
    docker compose --profile "$PROFILE" ps

    if docker ps -q -f name="$CONTAINER_NAME" &> /dev/null; then
        echo ""
        log "Service URLs:"
        echo "  Web UI:        http://localhost:4000"
        echo "  WebSocket:     ws://localhost:4000/ws"
        echo "  Claude Flow:   tcp://localhost:9500"

        if [[ "$PROFILE" == "dev" ]]; then
            echo "  Vite Dev:      http://localhost:5173"
        fi
    fi
}

# Build only
build_only() {
    log "Building containers for $PROFILE..."

    local build_args=()
    if [[ "$NO_CACHE" == true ]]; then
        build_args+=("--no-cache")
    fi

    DOCKER_BUILDKIT=1 docker_compose build "${build_args[@]}"
    success "Build complete"
}

# Clean everything
clean_all() {
    warning "This will remove all containers, volumes, and images for this project"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Cleaning all project resources..."

        # Stop all containers
        docker_compose down -v

        # Remove images
        docker compose --profile dev --profile production images -q | xargs -r docker rmi -f

        # Clean build cache
        docker builder prune -f

        success "Cleanup complete"
    else
        info "Cleanup cancelled"
    fi
}

# Cleanup on exit
cleanup() {
    if [[ "$ACTION" == "up" ]] && [[ "$DETACHED" != true ]]; then
        log "Shutting down gracefully..."
        docker_compose stop
    fi
}

# Main execution
main() {
    parse_args "$@"

    # Validate profile
    case "$PROFILE" in
        dev|production|prod)
            ;;
        *)
            error "Invalid profile: $PROFILE"
            echo "Valid profiles: dev, production, prod"
            exit 1
            ;;
    esac

    # Show banner
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         VisionFlow Launcher            ║${NC}"
    echo -e "${GREEN}║  Profile: ${YELLOW}$(printf '%-28s' "$PROFILE")${GREEN}║${NC}"
    echo -e "${GREEN}║  Action:  ${CYAN}$(printf '%-28s' "$ACTION")${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo

    # Set trap for cleanup
    trap cleanup EXIT INT TERM

    # Execute action
    case "$ACTION" in
        up)
            check_prerequisites
            compile_cuda
            start_environment
            ;;
        down)
            stop_environment
            ;;
        restart)
            check_prerequisites
            compile_cuda
            restart_environment
            ;;
        logs)
            show_logs
            ;;
        shell)
            open_shell
            ;;
        status)
            show_status
            ;;
        build)
            check_prerequisites
            compile_cuda
            build_only
            ;;
        clean)
            clean_all
            ;;
        *)
            error "Invalid action: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"