#!/bin/bash
# Validation Script for Docker Manager Skill
# Tests that docker-manager skill dependencies and permissions are correct
# Usage: ./validate-docker-manager.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test function wrapper
run_test() {
    local test_name="$1"
    local test_command="$2"

    ((TESTS_RUN++))
    log_info "Test $TESTS_RUN: $test_name"

    if eval "$test_command"; then
        log_success "$test_name"
        return 0
    else
        log_error "$test_name"
        return 1
    fi
}

echo "========================================"
echo "  Docker Manager Validation Tests"
echo "========================================"
echo ""

# ============================================================================
# Test 1: Python docker library installed
# ============================================================================

run_test "Python docker library installed" \
    "python3 -c 'import docker; print(\"Version:\", docker.__version__)' 2>&1"

# ============================================================================
# Test 2: Docker library version >= 7.0.0
# ============================================================================

check_docker_version() {
    local version=$(python3 -c 'import docker; print(docker.__version__)' 2>/dev/null)
    local major=$(echo "$version" | cut -d. -f1)
    if [ "$major" -ge 7 ]; then
        echo "Docker library version: $version (>= 7.0.0)"
        return 0
    else
        echo "Docker library version: $version (< 7.0.0)"
        return 1
    fi
}

run_test "Docker library version >= 7.0.0" "check_docker_version"

# ============================================================================
# Test 3: Docker socket exists
# ============================================================================

run_test "Docker socket exists" \
    "[ -S /var/run/docker.sock ]"

# ============================================================================
# Test 4: Docker socket permissions
# ============================================================================

check_docker_socket_perms() {
    local perms=$(stat -c "%a" /var/run/docker.sock 2>/dev/null)
    if [ "$perms" = "666" ] || [ "$perms" = "660" ] || [ "$perms" = "777" ]; then
        echo "Docker socket permissions: $perms (OK)"
        return 0
    else
        echo "Docker socket permissions: $perms (restrictive, may not work)"
        return 1
    fi
}

run_test "Docker socket permissions" "check_docker_socket_perms"

# ============================================================================
# Test 5: devuser in docker group
# ============================================================================

run_test "devuser in docker group" \
    "groups devuser | grep -q docker"

# ============================================================================
# Test 6: Docker CLI access
# ============================================================================

run_test "Docker CLI access (docker ps)" \
    "docker ps > /dev/null 2>&1"

# ============================================================================
# Test 7: docker-manager skill exists
# ============================================================================

run_test "docker-manager skill directory exists" \
    "[ -d /home/devuser/.claude/skills/docker-manager ]"

# ============================================================================
# Test 8: docker-manager Python script exists
# ============================================================================

run_test "docker_manager.py exists" \
    "[ -f /home/devuser/.claude/skills/docker-manager/tools/docker_manager.py ]"

# ============================================================================
# Test 9: docker_manager.py is executable
# ============================================================================

run_test "docker_manager.py is executable" \
    "[ -x /home/devuser/.claude/skills/docker-manager/tools/docker_manager.py ]"

# ============================================================================
# Test 10: Python can connect to Docker socket
# ============================================================================

test_docker_connection() {
    python3 << 'EOF'
import docker
import sys
try:
    client = docker.from_env()
    info = client.info()
    print(f"Connected to Docker: {info['ServerVersion']}")
    sys.exit(0)
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)
EOF
}

run_test "Python Docker SDK can connect" "test_docker_connection"

# ============================================================================
# Test 11: docker_manager.py can list available operations
# ============================================================================

test_docker_manager_help() {
    cd /home/devuser/.claude/skills/docker-manager/tools
    local result=$(python3 docker_manager.py 2>&1)
    if echo "$result" | grep -q "available"; then
        echo "docker_manager.py help output found"
        return 0
    else
        echo "docker_manager.py did not show help"
        return 1
    fi
}

run_test "docker_manager.py shows available operations" "test_docker_manager_help"

# ============================================================================
# Test 12: docker_manager.py can discover containers
# ============================================================================

test_container_discover() {
    cd /home/devuser/.claude/skills/docker-manager/tools
    local result=$(python3 docker_manager.py container_discover 2>&1)
    if echo "$result" | grep -q '"success": true'; then
        echo "Container discovery successful"
        return 0
    else
        echo "Container discovery failed: $result"
        return 1
    fi
}

run_test "docker_manager.py container_discover works" "test_container_discover"

# ============================================================================
# Test 13: VisionFlow container exists (if running)
# ============================================================================

test_visionflow_exists() {
    if docker ps -a --format "{{.Names}}" | grep -q "visionflow_container"; then
        echo "VisionFlow container found"
        return 0
    else
        log_warning "VisionFlow container not found (this is OK if not running VisionFlow)"
        return 0  # Don't fail this test
    fi
}

run_test "VisionFlow container check" "test_visionflow_exists"

# ============================================================================
# Test 14: docker_manager.py can get VisionFlow status (if running)
# ============================================================================

test_visionflow_status() {
    if ! docker ps -a --format "{{.Names}}" | grep -q "visionflow_container"; then
        log_warning "VisionFlow not running, skipping status check"
        return 0
    fi

    cd /home/devuser/.claude/skills/docker-manager/tools
    local result=$(python3 docker_manager.py visionflow_status 2>&1)
    if echo "$result" | grep -q '"success": true'; then
        echo "VisionFlow status check successful"
        return 0
    else
        echo "VisionFlow status check failed: $result"
        return 1
    fi
}

run_test "docker_manager.py visionflow_status works" "test_visionflow_status"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "  Test Results Summary"
echo "========================================"
echo -e "Total tests run:    ${BLUE}$TESTS_RUN${NC}"
echo -e "Tests passed:       ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed:       ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Docker Manager skill is ready to use."
    echo ""
    echo "Try it with Claude Code:"
    echo "  claude"
    echo "  > Use docker-manager to check VisionFlow status"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed!${NC}"
    echo ""
    echo "Please review the failed tests above."
    echo ""
    echo "Common fixes:"
    echo "  1. Install docker library: pip3 install --break-system-packages docker"
    echo "  2. Fix socket permissions: sudo chmod 666 /var/run/docker.sock"
    echo "  3. Check docker group: groups devuser"
    echo ""
    exit 1
fi
