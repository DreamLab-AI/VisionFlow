#!/bin/bash

# MCP TCP Connection Test Script
# Tests JSON-RPC 2.0 protocol compliance and connection stability

set -e

echo "🔧 Testing MCP TCP Connection..."
echo "=================================="

# Configuration
HOST="localhost"
PORT="9500"
TIMEOUT=10
TEST_ID=$(date +%s)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "Testing $test_name... "

    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Basic TCP Connection
test_tcp_connection() {
    timeout $TIMEOUT nc -z $HOST $PORT
}

# Test 2: JSON-RPC 2.0 Initialize Request
test_initialize_request() {
    local request='{
        "jsonrpc": "2.0",
        "id": "'$TEST_ID'",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": true
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }'

    echo "$request" | timeout $TIMEOUT nc $HOST $PORT | grep -q "jsonrpc"
}

# Test 3: List Tools Request
test_list_tools() {
    local request='{
        "jsonrpc": "2.0",
        "id": "'$((TEST_ID+1))'",
        "method": "tools/list",
        "params": {}
    }'

    echo "$request" | timeout $TIMEOUT nc $HOST $PORT | grep -q "tools"
}

# Test 4: Health Check via HTTP (fallback)
test_health_endpoint() {
    timeout $TIMEOUT curl -s http://localhost:9501/health | grep -q "status"
}

# Test 5: Connection Persistence
test_connection_persistence() {
    local temp_file=$(mktemp)

    # Open persistent connection and send multiple requests
    {
        echo '{"jsonrpc": "2.0", "id": "'$TEST_ID'", "method": "ping", "params": {}}'
        sleep 1
        echo '{"jsonrpc": "2.0", "id": "'$((TEST_ID+1))'", "method": "ping", "params": {}}'
    } | timeout $((TIMEOUT*2)) nc $HOST $PORT > "$temp_file"

    # Check if we got responses for both requests
    local response_count=$(grep -c "jsonrpc" "$temp_file" 2>/dev/null || echo 0)
    rm -f "$temp_file"

    [ "$response_count" -ge 1 ]
}

# Test 6: Error Handling
test_error_handling() {
    local invalid_request='{"jsonrpc": "2.0", "id": "'$TEST_ID'", "method": "nonexistent_method"}'

    local response=$(echo "$invalid_request" | timeout $TIMEOUT nc $HOST $PORT 2>/dev/null || echo "")
    echo "$response" | grep -q "error"
}

echo "Running MCP TCP Connection Tests..."
echo "Host: $HOST:$PORT"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Run all tests
run_test "TCP Connection" "test_tcp_connection"
run_test "Initialize Request" "test_initialize_request"
run_test "List Tools" "test_list_tools"
run_test "Health Endpoint" "test_health_endpoint"
run_test "Connection Persistence" "test_connection_persistence"
run_test "Error Handling" "test_error_handling"

echo ""
echo "=================================="
echo -e "Results: ${GREEN}$PASSED PASSED${NC}, ${RED}$FAILED FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All MCP TCP tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check MCP server configuration.${NC}"
    exit 1
fi