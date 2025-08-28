#!/bin/bash

echo "Testing direct MCP connection to port 9500..."

# Simple test to verify MCP server is listening and responding
test_mcp_direct() {
    local host="${1:-localhost}"
    local port="${2:-9500}"
    
    echo "Testing connection to $host:$port"
    
    # Test 1: Check if port is open
    timeout 2 bash -c "echo '' > /dev/tcp/$host/$port" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Port $port is open"
    else
        echo "❌ Cannot connect to port $port"
        return 1
    fi
    
    # Test 2: Send initialize request
    local request='{"jsonrpc":"2.0","id":"test-1","method":"initialize","params":{"protocolVersion":"1.0.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
    
    echo "Sending MCP initialize request..."
    response=$(echo "$request" | nc -w 5 "$host" "$port" 2>/dev/null | head -1)
    
    if [ -n "$response" ]; then
        echo "✅ Received response: $response"
        
        # Check if it's a valid JSON response
        echo "$response" | jq . >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Valid JSON response"
            return 0
        else
            echo "⚠️ Response is not valid JSON"
            return 1
        fi
    else
        echo "❌ No response received"
        return 1
    fi
}

# Test from inside container perspective
echo "=== Testing from container perspective ==="
test_mcp_direct "multi-agent-container" 9500

echo ""
echo "=== Testing localhost ==="
test_mcp_direct "localhost" 9500

echo ""
echo "=== Network diagnostics ==="
# Check what's listening on port 9500
netstat -tlnp 2>/dev/null | grep 9500 || ss -tlnp | grep 9500

echo ""
echo "=== Process check ==="
ps aux | grep -E "(mcp|claude-flow)" | grep -v grep

echo ""
echo "Done. If tests pass, the MCP server is reachable."