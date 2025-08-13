#!/bin/bash
# Test MCP TCP connection using IP address

echo "Testing MCP TCP connection to multi-agent-container via IP..."
echo ""

# Test basic connectivity using IP
echo "1. Testing TCP connectivity on 172.18.0.10:9500..."
nc -zv 172.18.0.10 9500 2>&1 || {
    echo "❌ Cannot connect to 172.18.0.10:9500"
    exit 1
}
echo "✅ TCP connection successful!"

echo ""
echo "2. Sending test JSON-RPC message..."
# Send a simple JSON-RPC request
echo '{"jsonrpc":"2.0","method":"ping","id":1}' | nc -w 2 172.18.0.10 9500 2>/dev/null || {
    echo "Note: No response (server might not be a simple echo server)"
}

echo ""
echo "Summary:"
echo "- Multi-agent-container IS accessible via IP: 172.18.0.10"
echo "- TCP port 9500 is open and accepting connections"
echo ""
echo "The backend should now be able to connect after restart."