#!/bin/bash

echo "Testing MCP Connection to multi-agent-container (172.18.0.4:9500)"
echo "================================================================"

# Test 1: Initialize connection
echo -e "\n1. Testing initialization..."
INIT_RESULT=$(echo '{"jsonrpc":"2.0","id":"init-test","method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}' | nc 172.18.0.4 9500 -w 2)

if [[ $INIT_RESULT == *"result"* ]]; then
    echo "✅ Initialization successful"
    echo "Response: $INIT_RESULT"
else
    echo "❌ Initialization failed"
    exit 1
fi

# Test 2: List agents
echo -e "\n2. Testing agent_list..."
AGENT_LIST=$(echo '{"jsonrpc":"2.0","id":"list-test","method":"tools/call","params":{"name":"agent_list","arguments":{"filter":"all"}}}' | nc 172.18.0.4 9500 -w 2)

if [[ $AGENT_LIST == *"jsonrpc"* ]]; then
    echo "✅ Agent list call successful"
    echo "Response: $AGENT_LIST"
else
    echo "❌ Agent list failed"
fi

echo -e "\n✅ MCP Connection Test Complete"