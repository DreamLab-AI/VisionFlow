#!/bin/bash
# Test MCP TCP connection to multi-agent-container

echo "Testing MCP TCP connection to multi-agent-container..."
echo ""

# Test basic connectivity
echo "1. Testing TCP connectivity on port 9500..."
nc -zv multi-agent-container 9500 2>&1 || {
    echo "❌ Cannot connect to multi-agent-container:9500"
    exit 1
}
echo "✅ TCP connection successful!"

echo ""
echo "2. Sending test JSON-RPC message..."
# Send a simple JSON-RPC request to check if it's an MCP server
echo '{"jsonrpc":"2.0","method":"ping","id":1}' | nc -w 2 multi-agent-container 9500 2>/dev/null || {
    echo "Note: No response (server might not be a simple echo server)"
}

echo ""
echo "3. Environment variables:"
echo "   CLAUDE_FLOW_HOST: ${CLAUDE_FLOW_HOST:-not set}"
echo "   MCP_TCP_PORT: ${MCP_TCP_PORT:-not set}"
echo "   MCP_TRANSPORT: ${MCP_TRANSPORT:-not set}"

echo ""
echo "Summary:"
echo "- The multi-agent-container IS running and accessible"
echo "- TCP port 9500 is open and accepting connections"
echo "- The ClaudeFlowActorTcp client exists in the code but may not be started"
echo ""
echo "To enable the connection:"
echo "1. The ClaudeFlowActor needs to be started in app_state.rs"
echo "2. Or use direct TCP connection in the initialize_multi_agent handler"