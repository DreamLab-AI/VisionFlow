#!/bin/bash
# Test spawning agents via docker exec (visionflow pattern)

set -e

CONTAINER="multi-agent-container"
MCP_HOST="172.18.0.4"
MCP_PORT="9500"

echo "🚀 Testing VisionFlow Agent Spawn Pattern"
echo "=========================================="
echo ""

# 1. Check container is running
echo "1️⃣  Checking container status..."
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Container $CONTAINER is not running"
    exit 1
fi
echo "✅ Container is running"
echo ""

# 2. Spawn agent via docker exec
echo "2️⃣  Spawning agent via docker exec..."
AGENT_NAME="test-agent-$(date +%s)"
TASK="create hello world in python"

docker exec "$CONTAINER" bash -c "
export CLAUDE_FLOW_DB_PATH=/workspace/.swarm/agent-${AGENT_NAME}.db
cd /workspace
echo '🤖 Spawning agent: $AGENT_NAME'
echo '📝 Task: $TASK'

# Spawn agent using claude-flow
npx claude-flow@latest spawn \
  --agent-name '$AGENT_NAME' \
  --task '$TASK' \
  --background &

SPAWN_PID=\$!
echo '✅ Agent spawned with PID: '\$SPAWN_PID

# Wait a moment for agent to initialize
sleep 2

# Show agent process
ps aux | grep -E 'claude-flow|$AGENT_NAME' | grep -v grep || echo 'No agent processes found'
"

echo ""
echo "3️⃣  Agent spawned successfully"
echo ""

# 3. Connect to TCP MCP to monitor
echo "4️⃣  Connecting to TCP MCP endpoint to monitor agent..."
echo "    MCP Server: $MCP_HOST:$MCP_PORT"
echo ""

# Simple netcat test first
if command -v nc >/dev/null 2>&1; then
    echo "Testing TCP connectivity..."
    if timeout 2 nc -zv "$MCP_HOST" "$MCP_PORT" 2>&1; then
        echo "✅ TCP MCP endpoint is reachable"
    else
        echo "❌ Cannot reach TCP MCP endpoint"
        exit 1
    fi
else
    echo "⚠️  netcat not available, skipping connectivity test"
fi

echo ""
echo "5️⃣  Checking database files for isolation..."
docker exec "$CONTAINER" bash -c 'ls -lh /workspace/.swarm/*.db 2>/dev/null || echo "No DB files found"'

echo ""
echo "6️⃣  Checking for database lock conflicts..."
docker exec "$CONTAINER" bash -c '
for db in /workspace/.swarm/*.db; do
    if [ -f "$db" ]; then
        echo "📊 $(basename $db):"
        fuser "$db" 2>/dev/null | while read pid; do
            ps -p $pid -o pid,comm,args --no-headers 2>/dev/null || echo "  Process $pid"
        done
    fi
done
' || echo "⚠️  Cannot check locks (fuser not available)"

echo ""
echo "✅ Test complete!"
echo ""
echo "Summary:"
echo "  - Agent spawned: $AGENT_NAME"
echo "  - TCP MCP: $MCP_HOST:$MCP_PORT"
echo "  - Database: /workspace/.swarm/agent-${AGENT_NAME}.db"
echo ""
echo "To monitor via Node.js MCP client:"
echo "  node scripts/test-tcp-mcp.js"
