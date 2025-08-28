#!/bin/bash

# Quick MCP Connection Test
# Minimal script for fast validation of MCP connectivity

MCP_HOST="${1:-multi-agent-container}"
MCP_PORT="${2:-9500}"

echo "Quick MCP Test - $MCP_HOST:$MCP_PORT"
echo "========================================"

# Test 1: Basic connectivity
echo -n "1. TCP Connection... "
if timeout 2 bash -c "echo > /dev/tcp/$MCP_HOST/$MCP_PORT" 2>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED - Cannot connect to $MCP_HOST:$MCP_PORT"
    exit 1
fi

# Test 2: Initialize
echo -n "2. MCP Initialize... "
INIT_RESPONSE=$(echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"QuickTest","version":"1.0"},"capabilities":{"tools":{"listChanged":true}}}}' | \
    timeout 3 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | head -1)

if echo "$INIT_RESPONSE" | grep -q '"result".*"protocolVersion"'; then
    VERSION=$(echo "$INIT_RESPONSE" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo "✓ OK (version: $VERSION)"
else
    echo "✗ FAILED"
    exit 1
fi

# Test 3: List tools
echo -n "3. List Tools... "
TOOLS_RESPONSE=$((
    echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"QuickTest","version":"1.0"},"capabilities":{"tools":{"listChanged":true}}}}'
    sleep 0.5
    echo '{"jsonrpc":"2.0","id":"tools","method":"tools/list","params":{}}'
) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"tools"')

if echo "$TOOLS_RESPONSE" | grep -q '"tools":\['; then
    TOOL_COUNT=$(echo "$TOOLS_RESPONSE" | grep -o '"name":"[^"]*"' | wc -l)
    echo "✓ OK ($TOOL_COUNT tools available)"
else
    echo "✗ FAILED"
    exit 1
fi

# Test 4: Swarm init
echo -n "4. Swarm Init... "
SWARM_RESPONSE=$((
    echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"QuickTest","version":"1.0"},"capabilities":{"tools":{"listChanged":true}}}}'
    sleep 0.5
    echo '{"jsonrpc":"2.0","id":"swarm","method":"swarm_init","params":{"topology":"mesh","maxAgents":2,"strategy":"balanced"}}'
) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"swarm"')

if echo "$SWARM_RESPONSE" | grep -q '"success":true.*"swarmId"'; then
    SWARM_ID=$(echo "$SWARM_RESPONSE" | grep -o '"swarmId":"[^"]*"' | cut -d'"' -f4)
    echo "✓ OK (swarm: $SWARM_ID)"
    
    # Test 5: Swarm status (if swarm created)
    echo -n "5. Swarm Status... "
    STATUS_RESPONSE=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"QuickTest","version":"1.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"status","method":"swarm_status","params":{"swarmId":"'$SWARM_ID'"}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"status"')
    
    if echo "$STATUS_RESPONSE" | grep -q '"success":true'; then
        echo "✓ OK"
    else
        echo "✗ FAILED"
    fi
else
    echo "✗ FAILED"
    echo "5. Swarm Status... SKIPPED"
fi

echo "========================================"
echo "✅ MCP connection is working!"