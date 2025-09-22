#!/bin/bash

echo "=== Testing Playwright MCP Connection Stack ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test functions
test_service() {
    local service=$1
    local port=$2
    local name=$3
    
    echo -n "Testing $name on port $port... "
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓ Connected${NC}"
        return 0
    else
        echo -e "${RED}✗ Not available${NC}"
        return 1
    fi
}

test_health() {
    local port=$1
    local name=$2
    
    echo -n "Testing $name health endpoint... "
    if curl -sf http://127.0.0.1:$port >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Not responding${NC}"
        return 1
    fi
}

echo "1. Checking services status:"
echo "----------------------------"
supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status | grep playwright || echo "No playwright services found"

echo ""
echo "2. Testing network connectivity:"
echo "--------------------------------"
test_service 9879 "Playwright MCP Proxy (local)"
test_service 9880 "Playwright Proxy Health"

echo ""
echo "3. Testing cross-container connectivity:"
echo "---------------------------------------"
echo -n "Checking if GUI container is reachable... "
if ping -c 1 -W 1 gui-tools-service >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Reachable${NC}"
    
    echo -n "Testing Playwright MCP in GUI container... "
    if nc -z gui-tools-service 9879 2>/dev/null; then
        echo -e "${GREEN}✓ Connected${NC}"
    else
        echo -e "${YELLOW}⚠ Not responding (GUI container may be starting)${NC}"
    fi
else
    echo -e "${RED}✗ Not reachable${NC}"
fi

echo ""
echo "4. Testing MCP protocol:"
echo "------------------------"
echo -n "Sending test request to Playwright MCP... "
TEST_RESPONSE=$(echo '{"jsonrpc":"2.0","id":"test","method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test"}}}' | nc -w 2 localhost 9879 2>/dev/null | head -n 1)

if echo "$TEST_RESPONSE" | grep -q "jsonrpc"; then
    echo -e "${GREEN}✓ MCP protocol working${NC}"
else
    echo -e "${RED}✗ No valid response${NC}"
fi

echo ""
echo "5. Checking logs for errors:"
echo "----------------------------"
if [ -f /app/mcp-logs/playwright-proxy.log ]; then
    echo "Recent proxy logs:"
    tail -n 5 /app/mcp-logs/playwright-proxy.log | sed 's/^/  /'
else
    echo "No proxy logs found"
fi

echo ""
echo "6. Configuration check:"
echo "-----------------------"
if grep -q "playwright" /workspace/.mcp.json 2>/dev/null; then
    echo -e "${GREEN}✓ Playwright configured in .mcp.json${NC}"
else
    echo -e "${YELLOW}⚠ Playwright not found in .mcp.json${NC}"
fi

echo ""
echo "=== Summary ==="
echo "To use Playwright:"
echo "1. Ensure GUI container is running: docker-compose ps gui-tools-service"
echo "2. Check proxy status: playwright-proxy-status"
echo "3. In Claude: Use the 'playwright' MCP tools"
echo "4. View browser visually: VNC on port 5901"