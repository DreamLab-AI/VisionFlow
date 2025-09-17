#!/bin/bash
# Final test of the fixed MCP server

echo "=== Testing Fixed MCP Server ==="
echo ""

# Wait for server to be ready
echo "Waiting for MCP server to be ready..."
sleep 3

echo "1. Testing agent spawn..."
SPAWN_RESULT=$(echo '{"jsonrpc":"2.0","id":"spawn-1","method":"tools/call","params":{"name":"agent_spawn","arguments":{"type":"coordinator","name":"RealAgent1","swarmId":"test-swarm"}}}' | timeout 10 nc localhost 9500 2>/dev/null | tail -1)

if [ -z "$SPAWN_RESULT" ]; then
    echo "❌ No response from agent_spawn"
else
    echo "✅ Got response from agent_spawn"

    # Extract agent ID if possible
    AGENT_ID=$(echo "$SPAWN_RESULT" | grep -o '"agentId":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$AGENT_ID" ]; then
        echo "   Created agent: $AGENT_ID"
    fi
fi

echo ""
echo "2. Testing agent_list..."
sleep 2

LIST_RESULT=$(echo '{"jsonrpc":"2.0","id":"list-1","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | timeout 10 nc localhost 9500 2>/dev/null | tail -1)

if [ -z "$LIST_RESULT" ]; then
    echo "❌ No response from agent_list"
else
    echo "✅ Got response from agent_list"

    # Check for mock data
    if echo "$LIST_RESULT" | grep -q '"id":"agent-1"'; then
        echo "❌ STILL RETURNING MOCK DATA (agent-1, agent-2, agent-3)"
    else
        echo "✅ NOT returning mock data!"

        # Check what we got
        if echo "$LIST_RESULT" | grep -q '"agents":\[\]'; then
            echo "   Empty agent list (might need to spawn agents first)"
        elif echo "$LIST_RESULT" | grep -q "RealAgent"; then
            echo "   ✅✅✅ Found real spawned agents!"
        else
            echo "   Response received but unclear content"
        fi
    fi
fi

echo ""
echo "3. Raw test results:"
echo "Spawn result: $SPAWN_RESULT" | head -c 200
echo ""
echo "List result: $LIST_RESULT" | head -c 200
echo ""
echo ""
echo "=== Test Complete ==="