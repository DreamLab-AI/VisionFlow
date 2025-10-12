#!/bin/bash
#
# Comprehensive test runner for CachyOS and Z.AI Docker system
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
CONTAINER_NAME="${CONTAINER_NAME:-agentic-flow-cachyos}"
ZAI_CONTAINER="${ZAI_CONTAINER:-claude-zai-service}"
SKIP_DOCKER_CHECK="${SKIP_DOCKER_CHECK:-false}"

# Check if containers are running
check_containers() {
    log_info "Checking Docker containers..."

    if ! docker ps -q -f name=$CONTAINER_NAME -f status=running >/dev/null 2>&1; then
        log_error "Container $CONTAINER_NAME is not running"
        log_info "Start it with: ./start-agentic-flow.sh"
        exit 1
    fi

    if ! docker ps -q -f name=$ZAI_CONTAINER -f status=running >/dev/null 2>&1; then
        log_error "Container $ZAI_CONTAINER is not running"
        log_info "Start it with: ./start-agentic-flow.sh"
        exit 1
    fi

    log_success "All containers are running"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."

    local retries=30
    while [ $retries -gt 0 ]; do
        if curl -sf http://localhost:9090/health >/dev/null 2>&1 && \
           curl -sf http://localhost:9600/health >/dev/null 2>&1; then
            log_success "All services are ready"
            return 0
        fi

        retries=$((retries - 1))
        echo -n "."
        sleep 1
    done

    log_error "Services failed to become ready"
    exit 1
}

# Install test dependencies
setup_tests() {
    log_info "Setting up test environment..."

    cd "$SCRIPT_DIR"

    if [ ! -d "node_modules" ]; then
        log_info "Installing test dependencies..."
        npm install
    fi

    log_success "Test environment ready"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."

    cd "$SCRIPT_DIR"
    npm run test:unit

    log_success "Unit tests passed"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."

    cd "$SCRIPT_DIR"
    npm run test:integration

    log_success "Integration tests passed"
}

# Run E2E tests
run_e2e_tests() {
    log_info "Running end-to-end tests..."

    cd "$SCRIPT_DIR"
    npm run test:e2e

    log_success "E2E tests passed"
}

# Run performance benchmarks
run_performance_tests() {
    log_info "Running performance benchmarks..."

    cd "$SCRIPT_DIR"
    npm run test:performance

    log_success "Performance tests completed"
}

# Run load tests
run_load_tests() {
    log_info "Running load tests..."

    cd "$SCRIPT_DIR"

    if ! command -v artillery >/dev/null 2>&1; then
        log_warning "Artillery not installed, skipping load tests"
        log_info "Install with: npm install -g artillery"
        return
    fi

    npm run test:load

    log_success "Load tests completed"
}

# Run security tests
run_security_tests() {
    log_info "Running security tests..."

    cd "$SCRIPT_DIR"

    # npm audit
    log_info "Running npm audit..."
    npm audit || log_warning "npm audit found vulnerabilities"

    # Snyk (if available)
    if command -v snyk >/dev/null 2>&1; then
        log_info "Running Snyk security scan..."
        snyk test || log_warning "Snyk found vulnerabilities"
    else
        log_warning "Snyk not installed, skipping"
        log_info "Install with: npm install -g snyk"
    fi

    log_success "Security tests completed"
}

# Generate coverage report
generate_coverage() {
    log_info "Generating coverage report..."

    cd "$SCRIPT_DIR"
    npm run coverage

    log_success "Coverage report generated at tests/coverage/lcov-report/index.html"
}

# Display usage
usage() {
    cat << EOF
Comprehensive Test Runner for CachyOS and Z.AI Docker System

Usage: $0 [options]

Options:
    --all               Run all tests (default)
    --unit              Run unit tests only
    --integration       Run integration tests only
    --e2e               Run end-to-end tests only
    --performance       Run performance benchmarks only
    --load              Run load tests only
    --security          Run security tests only
    --coverage          Generate coverage report
    --skip-setup        Skip dependency installation
    --skip-docker       Skip Docker container checks
    -h, --help          Show this help message

Examples:
    # Run all tests
    $0

    # Run specific test suites
    $0 --unit --integration

    # Run with coverage
    $0 --coverage

    # Skip Docker checks (for CI)
    $0 --skip-docker

EOF
}

# Parse arguments
RUN_ALL=true
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_E2E=false
RUN_PERFORMANCE=false
RUN_LOAD=false
RUN_SECURITY=false
GENERATE_COV=false
SKIP_SETUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --unit)
            RUN_ALL=false
            RUN_UNIT=true
            shift
            ;;
        --integration)
            RUN_ALL=false
            RUN_INTEGRATION=true
            shift
            ;;
        --e2e)
            RUN_ALL=false
            RUN_E2E=true
            shift
            ;;
        --performance)
            RUN_ALL=false
            RUN_PERFORMANCE=true
            shift
            ;;
        --load)
            RUN_ALL=false
            RUN_LOAD=true
            shift
            ;;
        --security)
            RUN_ALL=false
            RUN_SECURITY=true
            shift
            ;;
        --coverage)
            GENERATE_COV=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER_CHECK=true
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
log_info "=== Agentic Flow Test Suite ==="
echo ""

# Pre-flight checks
if [ "$SKIP_DOCKER_CHECK" != "true" ]; then
    check_containers
    wait_for_services
fi

# Setup
if [ "$SKIP_SETUP" != "true" ]; then
    setup_tests
fi

# Run tests based on flags
if [ "$RUN_ALL" = "true" ]; then
    run_unit_tests
    run_integration_tests
    run_e2e_tests
    run_performance_tests
    run_load_tests
    run_security_tests
else
    [ "$RUN_UNIT" = "true" ] && run_unit_tests
    [ "$RUN_INTEGRATION" = "true" ] && run_integration_tests
    [ "$RUN_E2E" = "true" ] && run_e2e_tests
    [ "$RUN_PERFORMANCE" = "true" ] && run_performance_tests
    [ "$RUN_LOAD" = "true" ] && run_load_tests
    [ "$RUN_SECURITY" = "true" ] && run_security_tests
fi

# Coverage
if [ "$GENERATE_COV" = "true" ]; then
    generate_coverage
fi

echo ""
log_success "=== All Tests Complete ==="
echo ""
log_info "Next steps:"
log_info "  - View coverage: open tests/coverage/lcov-report/index.html"
log_info "  - Check metrics: curl http://localhost:9090/metrics"
log_info "  - View logs: ./start-agentic-flow.sh --logs"
echo ""
