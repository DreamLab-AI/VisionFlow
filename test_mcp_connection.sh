#!/bin/bash

echo "Testing MCP connection to multi-agent-container:9500"
echo "====================================================="

# Test 1: Direct TCP connection
echo -e "\n1. Testing direct TCP connection..."
timeout 2 bash -c 'echo "{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"params\":{},\"id\":\"test\"}" | nc multi-agent-container 9500 | head -1'
if [ $? -eq 0 ]; then
    echo "✅ Direct TCP connection successful"
else
    echo "❌ Direct TCP connection failed"
fi

# Test 2: Check what IP the container resolves to
echo -e "\n2. Resolving multi-agent-container hostname..."
getent hosts multi-agent-container
if [ $? -eq 0 ]; then
    echo "✅ Hostname resolution successful"
else
    echo "❌ Hostname resolution failed"
fi

# Test 3: Show current environment
echo -e "\n3. Current environment variables:"
echo "CLAUDE_FLOW_HOST=${CLAUDE_FLOW_HOST:-not set}"
echo "MCP_HOST=${MCP_HOST:-not set}"
echo "MCP_TCP_PORT=${MCP_TCP_PORT:-not set}"

echo -e "\n====================================================="
echo "If the external container shows connection to multi-agent-container:9500,"
echo "it needs to be restarted to pick up the correct CLAUDE_FLOW_HOST"
echo "environment variable pointing to multi-agent-container."
echo ""
echo "The external container should have:"
echo "  CLAUDE_FLOW_HOST=multi-agent-container"
echo "  MCP_TCP_PORT=9500"