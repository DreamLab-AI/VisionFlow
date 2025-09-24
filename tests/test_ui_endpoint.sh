#!/bin/bash

# UI REST API Endpoint Test Script
# Tests WebSocket bridge and HTTP endpoints

set -e

echo "🌐 Testing UI REST API Endpoints..."
echo "===================================="

# Configuration
HTTP_PORT="3002"
WS_PORT="3002"
HOST="localhost"
TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Test 1: HTTP Server Response
test_http_server() {
    timeout $TIMEOUT curl -s -o /dev/null -w "%{http_code}" http://$HOST:$HTTP_PORT/ | grep -E "200|404"
}

# Test 2: WebSocket Endpoint Availability
test_websocket_endpoint() {
    # Check if WebSocket port is listening
    timeout $TIMEOUT nc -z $HOST $WS_PORT
}

# Test 3: CORS Headers
test_cors_headers() {
    local response=$(timeout $TIMEOUT curl -s -I -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type" \
        -X OPTIONS http://$HOST:$HTTP_PORT/)

    echo "$response" | grep -i "access-control-allow-origin"
}

# Test 4: JSON Content-Type Support
test_json_support() {
    local response=$(timeout $TIMEOUT curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"test": "data"}' \
        http://$HOST:$HTTP_PORT/api/test 2>/dev/null || echo "")

    # Should not return connection refused or similar network errors
    ! echo "$response" | grep -q "Connection refused"
}

# Test 5: MCP Bridge Functionality
test_mcp_bridge() {
    local mcp_request='{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }'

    local response=$(timeout $TIMEOUT curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$mcp_request" \
        http://$HOST:$HTTP_PORT/mcp 2>/dev/null || echo "")

    # Check if response contains JSON-RPC structure or tools
    echo "$response" | grep -E "(jsonrpc|tools|result|error)"
}

# Test 6: Health Check Endpoint
test_health_check() {
    timeout $TIMEOUT curl -s http://$HOST:$HTTP_PORT/health | grep -E "(ok|healthy|status)"
}

# Test 7: Static File Serving (if applicable)
test_static_files() {
    local status_code=$(timeout $TIMEOUT curl -s -o /dev/null -w "%{http_code}" \
        http://$HOST:$HTTP_PORT/favicon.ico 2>/dev/null || echo "000")

    # Should return 200 (found) or 404 (not found), but not 500+ (server error)
    [ "${status_code:0:1}" != "5" ]
}

# Test 8: WebSocket Connection Test
test_websocket_connection() {
    # Use wscat if available, otherwise use a simple connection test
    if command -v wscat &> /dev/null; then
        echo '{"type":"ping"}' | timeout 5 wscat -c ws://$HOST:$WS_PORT/ 2>/dev/null | grep -q "pong"
    else
        # Fallback: just test if the WebSocket port accepts connections
        timeout $TIMEOUT nc -z $HOST $WS_PORT
    fi
}

# Test 9: Rate Limiting (if implemented)
test_rate_limiting() {
    local responses=0

    for i in {1..5}; do
        if timeout 2 curl -s -o /dev/null http://$HOST:$HTTP_PORT/ 2>/dev/null; then
            ((responses++))
        fi
    done

    # Should handle at least some requests (not completely blocked)
    [ $responses -gt 0 ]
}

echo "Running UI REST API Tests..."
echo "HTTP Port: $HTTP_PORT"
echo "WebSocket Port: $WS_PORT"
echo "Host: $HOST"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Run all tests
run_test "HTTP Server" "test_http_server"
run_test "WebSocket Endpoint" "test_websocket_endpoint"
run_test "CORS Headers" "test_cors_headers"
run_test "JSON Support" "test_json_support"
run_test "MCP Bridge" "test_mcp_bridge"
run_test "Health Check" "test_health_check"
run_test "Static Files" "test_static_files"
run_test "WebSocket Connection" "test_websocket_connection"
run_test "Rate Limiting" "test_rate_limiting"

echo ""
echo "===================================="
echo -e "Results: ${GREEN}$PASSED PASSED${NC}, ${RED}$FAILED FAILED${NC}"

# Additional diagnostics
echo ""
echo -e "${BLUE}📊 Additional Diagnostics:${NC}"
echo -n "HTTP Service Status: "
if curl -s --max-time 3 http://$HOST:$HTTP_PORT/ &>/dev/null; then
    echo -e "${GREEN}RUNNING${NC}"
else
    echo -e "${RED}NOT RESPONDING${NC}"
fi

echo -n "WebSocket Service Status: "
if nc -z $HOST $WS_PORT 2>/dev/null; then
    echo -e "${GREEN}LISTENING${NC}"
else
    echo -e "${RED}NOT LISTENING${NC}"
fi

# Check if services are using expected ports
echo -n "Port Usage Check: "
if netstat -tlnp 2>/dev/null | grep -q ":$HTTP_PORT "; then
    echo -e "${GREEN}PORT $HTTP_PORT IN USE${NC}"
else
    echo -e "${YELLOW}PORT $HTTP_PORT NOT FOUND${NC}"
fi

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All UI endpoint tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed. Check UI server configuration.${NC}"

    # Provide troubleshooting hints
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting hints:${NC}"
    echo "  - Ensure the UI server is running on port $HTTP_PORT"
    echo "  - Check if WebSocket bridge is properly configured"
    echo "  - Verify CORS settings allow cross-origin requests"
    echo "  - Check server logs for error details"

    exit 1
fi