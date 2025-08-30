#!/bin/bash

# Direct MCP Test - Simple verification of persistent server
# Run from VisionFlow container: bash test_mcp_direct.sh

HOST="${1:-multi-agent-container}"
PORT="${2:-9500}"

echo "Testing MCP Persistent Server at $HOST:$PORT"
echo "=========================================="
echo ""

echo "1. Testing swarm init:"
echo '{"jsonrpc":"2.0","id":"init1","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"mesh"}}}' | nc -w 2 $HOST $PORT
echo ""
echo ""

echo "2. Testing agent spawn:"
echo '{"jsonrpc":"2.0","id":"spawn1","method":"tools/call","params":{"name":"agent_spawn","arguments":{"type":"coordinator"}}}' | nc -w 2 $HOST $PORT
echo ""
echo ""

echo "3. Testing agent list (should show spawned agent):"
echo '{"jsonrpc":"2.0","id":"list1","method":"tools/call","params":{"name":"agent_list","arguments":{"filter":"all"}}}' | nc -w 2 $HOST $PORT
echo ""
echo ""

echo "4. Testing swarm status:"
echo '{"jsonrpc":"2.0","id":"status1","method":"tools/call","params":{"name":"swarm_status","arguments":{"verbose":true}}}' | nc -w 2 $HOST $PORT
echo ""
echo ""

echo "5. Testing swarm destroy:"
echo '{"jsonrpc":"2.0","id":"destroy1","method":"tools/call","params":{"name":"swarm_destroy","arguments":{}}}' | nc -w 2 $HOST $PORT
echo ""
echo ""

echo "Test complete! If you see JSON responses above, the persistent server is working."