#!/bin/bash
#
# Agentic Flow + Z.AI Combined Docker System Startup Script
# Connects to RAGFlow network for unified AI infrastructure
#
# Usage:
#   ./start-agentic-flow.sh [options]
#
# Options:
#   --build         Force rebuild containers
#   --no-ragflow    Skip RAGFlow network connection
#   --stop          Stop all services
#   --restart       Restart all services
#   --logs          Show container logs
#   --status        Show service status
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
ENV_FILE="$COMPOSE_DIR/.env"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if RAGFlow network exists
check_ragflow_network() {
    if docker network inspect docker_ragflow >/dev/null 2>&1; then
        log_success "RAGFlow network detected"
        return 0
    else
        log_warning "RAGFlow network not found"
        return 1
    fi
}

# Connect to RAGFlow network
connect_to_ragflow() {
    if [ "$SKIP_RAGFLOW" = true ]; then
        log_info "Skipping RAGFlow network connection"
        return
    fi

    if check_ragflow_network; then
        log_info "Connecting containers to RAGFlow network..."

        # Connect agentic-flow container
        if docker ps -q -f name=agentic-flow-cachyos >/dev/null 2>&1; then
            docker network connect docker_ragflow agentic-flow-cachyos 2>/dev/null || log_warning "agentic-flow-cachyos already connected"
        fi

        # Connect claude-zai container
        if docker ps -q -f name=claude-zai-service >/dev/null 2>&1; then
            docker network connect docker_ragflow claude-zai-service 2>/dev/null || log_warning "claude-zai-service already connected"
        fi

        log_success "Containers connected to RAGFlow network"
    else
        log_warning "RAGFlow not running. Services will use internal network only."
        log_info "To enable RAGFlow integration, start RAGFlow first:"
        log_info "  cd /path/to/ragflow && docker compose up -d"
    fi
}

# Validate environment
check_environment() {
    log_info "Checking environment configuration..."

    if [ ! -f "$ENV_FILE" ]; then
        log_warning "Environment file not found: $ENV_FILE"
        log_info "Creating from template..."
        cp "$COMPOSE_DIR/.env.example" "$ENV_FILE"
        log_success "Created $ENV_FILE - Please configure your API keys"
    fi

    # Check for required API keys
    if ! grep -q "ZAI_API_KEY=.*[^[:space:]]" "$ENV_FILE" || grep -q "ZAI_API_KEY=your_z_ai_api_key_here" "$ENV_FILE"; then
        log_warning "ZAI_API_KEY not configured in $ENV_FILE"
        log_info "Z.AI features will be unavailable. Get your key from https://z.ai/"
    fi

    if ! grep -q "GOOGLE_API_KEY=AIza" "$ENV_FILE"; then
        log_warning "GOOGLE_API_KEY not configured properly"
        log_info "Web summary tool may not work. Get your key from https://console.cloud.google.com/"
    fi

    log_success "Environment validated"
}

# Build containers
build_containers() {
    log_info "Building Docker containers..."
    cd "$COMPOSE_DIR"

    if [ "$FORCE_BUILD" = true ]; then
        docker-compose -f docker-compose.yml build --no-cache
    else
        docker-compose -f docker-compose.yml build
    fi

    log_success "Containers built successfully"
}

# Start services
start_services() {
    log_info "Starting Agentic Flow services..."
    cd "$COMPOSE_DIR"

    docker-compose -f docker-compose.yml up -d

    log_success "Services started"

    # Wait for services to be healthy
    log_info "Waiting for services to be ready..."
    sleep 5

    # Check claude-zai health
    if docker ps -q -f name=claude-zai-service -f status=running >/dev/null 2>&1; then
        log_info "Testing claude-zai health endpoint..."
        if curl -sf http://localhost:9600/health >/dev/null 2>&1; then
            log_success "claude-zai service is healthy"
        else
            log_warning "claude-zai service may not be ready yet"
        fi
    fi

    # Check main container
    if docker ps -q -f name=agentic-flow-cachyos -f status=running >/dev/null 2>&1; then
        log_success "agentic-flow-cachyos container is running"
    else
        log_error "agentic-flow-cachyos container failed to start"
    fi
}

# Stop services
stop_services() {
    log_info "Stopping Agentic Flow services..."
    cd "$COMPOSE_DIR"

    docker-compose -f docker-compose.yml down

    log_success "Services stopped"
}

# Show logs
show_logs() {
    log_info "Showing container logs..."
    cd "$COMPOSE_DIR"

    docker-compose -f docker-compose.yml logs -f --tail=100
}

# Show status
show_status() {
    log_info "Service Status:"
    echo ""

    cd "$COMPOSE_DIR"
    docker-compose -f docker-compose.yml ps

    echo ""
    log_info "Network Connections:"

    if docker ps -q -f name=agentic-flow-cachyos >/dev/null 2>&1; then
        echo "agentic-flow-cachyos networks:"
        docker inspect agentic-flow-cachyos --format '{{range $net, $v := .NetworkSettings.Networks}}  - {{$net}}{{"\n"}}{{end}}'
    fi

    if docker ps -q -f name=claude-zai-service >/dev/null 2>&1; then
        echo "claude-zai-service networks:"
        docker inspect claude-zai-service --format '{{range $net, $v := .NetworkSettings.Networks}}  - {{$net}}{{"\n"}}{{end}}'
    fi

    echo ""
    log_info "Health Checks:"

    # Check management API
    if curl -sf http://localhost:9090/health >/dev/null 2>&1; then
        log_success "Management API (9090): healthy"
    else
        log_warning "Management API (9090): not responding"
    fi

    # Check claude-zai
    if curl -sf http://localhost:9600/health >/dev/null 2>&1; then
        log_success "Claude-ZAI API (9600): healthy"
    else
        log_warning "Claude-ZAI API (9600): not responding"
    fi
}

# Open shell in container
open_shell() {
    if ! docker ps -q -f name=agentic-flow-cachyos -f status=running >/dev/null 2>&1; then
        log_error "Container not running. Start it first with: $0"
        exit 1
    fi

    log_info "Opening shell in agentic-flow-cachyos..."
    docker exec -it agentic-flow-cachyos zsh
}

# Clean up old resources
clean_resources() {
    log_info "Cleaning up Docker resources..."

    # Stop and remove containers
    cd "$COMPOSE_DIR"
    docker-compose -f docker-compose.yml down -v 2>/dev/null || true

    # Remove images
    log_info "Removing agentic-flow images..."
    docker rmi cachyos-agentic-flow-cachyos 2>/dev/null || log_info "Main image not found"
    docker rmi cachyos-claude-zai 2>/dev/null || log_info "Z.AI image not found"

    # Prune unused resources
    log_info "Pruning unused Docker resources..."
    docker system prune -f

    log_success "Cleanup complete"
}

# Run test suite
run_tests() {
    log_info "Running validation tests..."

    if [ ! -d "$COMPOSE_DIR/../test" ]; then
        log_warning "No test directory found"
        return
    fi

    cd "$COMPOSE_DIR"

    # Find and run test Dockerfiles
    local test_count=0
    for test_file in Dockerfile.test-*; do
        if [ -f "$test_file" ]; then
            test_count=$((test_count + 1))
            local test_name=$(basename "$test_file" | sed 's/Dockerfile.test-//')

            log_info "Running test: $test_name"

            if docker build -f "$test_file" -t "agentic-flow-test-$test_name" .; then
                log_success "Test passed: $test_name"
            else
                log_error "Test failed: $test_name"
            fi
        fi
    done

    if [ $test_count -eq 0 ]; then
        log_warning "No test files found (Dockerfile.test-*)"
    else
        log_success "Completed $test_count tests"
    fi
}

# Display usage
usage() {
    cat << EOF
Agentic Flow + Z.AI Combined Docker System

Usage: $0 [options]

Options:
    --build         Force rebuild containers
    --no-ragflow    Skip RAGFlow network connection
    --stop          Stop all services
    --restart       Restart all services
    --logs          Show container logs
    --status        Show service status
    --shell         Open interactive shell in container
    --clean         Clean up Docker resources (containers, images, volumes)
    --test          Run validation test suite
    -h, --help      Show this help message

Environment:
    Configure API keys in: $ENV_FILE

Services:
    - agentic-flow-cachyos:  Main orchestration container (port 9090)
    - claude-zai-service:    Z.AI semantic processing (port 9600)

Networks:
    - agentic-network:       Internal bridge network
    - docker_ragflow:        RAGFlow integration (optional)

Examples:
    # First time setup
    $0 --build

    # Regular startup
    $0

    # Check status
    $0 --status

    # Open shell
    $0 --shell

    # Clean everything
    $0 --clean

    # Run tests
    $0 --test

    # View logs
    $0 --logs

    # Restart services
    $0 --restart

EOF
}

# Parse arguments
FORCE_BUILD=false
SKIP_RAGFLOW=false
ACTION="start"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --no-ragflow)
            SKIP_RAGFLOW=true
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --restart)
            ACTION="restart"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --shell)
            ACTION="shell"
            shift
            ;;
        --clean)
            ACTION="clean"
            shift
            ;;
        --test)
            ACTION="test"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
echo ""
log_info "=== Agentic Flow + Z.AI Docker System ==="
echo ""

case $ACTION in
    start)
        check_environment
        if [ "$FORCE_BUILD" = true ]; then
            build_containers
        fi
        start_services
        connect_to_ragflow
        echo ""
        log_success "=== System Started ==="
        echo ""
        log_info "Access points:"
        log_info "  Management API:  http://localhost:9090"
        log_info "  Claude-ZAI API:  http://localhost:9600"
        log_info "  VNC Desktop:     http://localhost:6901 (if enabled)"
        echo ""
        log_info "Next steps:"
        log_info "  - Check status:  $0 --status"
        log_info "  - View logs:     $0 --logs"
        log_info "  - Shell access:  docker exec -it agentic-flow-cachyos zsh"
        echo ""
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        check_environment
        start_services
        connect_to_ragflow
        log_success "Services restarted"
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    shell)
        open_shell
        ;;
    clean)
        read -p "This will remove all containers, images, and volumes. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            clean_resources
        else
            log_info "Cleanup cancelled"
        fi
        ;;
    test)
        run_tests
        ;;
esac
