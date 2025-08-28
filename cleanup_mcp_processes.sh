#!/bin/bash

echo "=== MCP Process Cleanup Script ==="
echo "Current MCP processes:"
ps aux | grep -E "(mcp|claude-flow)" | grep -v grep | wc -l
echo ""

# Kill all MCP-related processes except the main TCP server
echo "Killing duplicate MCP processes..."

# Kill all npm exec claude-flow processes
pkill -f "npm exec claude-flow" 2>/dev/null
pkill -f "claude-flow mcp start" 2>/dev/null
pkill -f "ruv-swarm mcp" 2>/dev/null
pkill -f "flow-nexus mcp" 2>/dev/null

# Wait a moment
sleep 2

# Kill any remaining node processes related to MCP (except the main TCP server on port 9500)
ps aux | grep -E "claude-flow.*mcp" | grep -v "mcp-tcp-server" | awk '{print $2}' | xargs -r kill -9 2>/dev/null

echo "Cleanup complete."
echo ""
echo "Remaining processes:"
ps aux | grep -E "(mcp|claude-flow)" | grep -v grep

echo ""
echo "File descriptor usage:"
lsof 2>/dev/null | wc -l

echo ""
echo "Checking port 9500..."
netstat -tlnp 2>/dev/null | grep 9500 || ss -tlnp | grep 9500

echo ""
echo "=== Restarting clean MCP TCP server ==="

# Kill the old TCP server
pkill -f "mcp-tcp-server.js" 2>/dev/null

# Wait for port to be released
sleep 2

# Start fresh TCP server
nohup node /app/core-assets/scripts/mcp-tcp-server.js > /tmp/mcp-tcp-server.log 2>&1 &
echo "Started new MCP TCP server (PID: $!)"

# Wait for it to start
sleep 2

# Test the connection
echo ""
echo "Testing cleaned up connection..."
timeout 2 bash -c "echo '' > /dev/tcp/localhost/9500" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ MCP TCP server is running on port 9500"
else
    echo "❌ MCP TCP server failed to start"
fi

echo ""
echo "Done. The system should now have clean file descriptors."