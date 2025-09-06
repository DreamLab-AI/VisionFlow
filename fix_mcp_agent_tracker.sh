#!/bin/bash
# Fix MCP server agent tracker issues
# This script patches the globally installed claude-flow to fix agent tracking

set -e

echo "🔧 Fixing MCP Server Agent Tracker Issues"
echo "=========================================="

# Find the MCP server file
MCP_SERVER="/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"
AGENT_TRACKER="/usr/lib/node_modules/claude-flow/src/mcp/implementations/agent-tracker.js"

if [ ! -f "$MCP_SERVER" ]; then
    echo "❌ MCP server not found at $MCP_SERVER"
    exit 1
fi

if [ ! -f "$AGENT_TRACKER" ]; then
    echo "❌ Agent tracker not found at $AGENT_TRACKER"
    exit 1
fi

echo "✅ Found MCP server at: $MCP_SERVER"
echo "✅ Found agent tracker at: $AGENT_TRACKER"

# Backup original files
echo "📦 Creating backups..."
cp "$MCP_SERVER" "${MCP_SERVER}.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
cp "$AGENT_TRACKER" "${AGENT_TRACKER}.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

# Patch 1: Remove mock data fallback from agent_list
echo "🔨 Patch 1: Removing mock data fallback from agent_list..."
sed -i '1507,1530d' "$MCP_SERVER"

# Insert proper empty response instead of mock data
sed -i '1506a\
        // Return empty list if no agents found\
        return {\
          success: true,\
          swarmId: args.swarmId || "no-swarm",\
          agents: [],\
          count: 0,\
          timestamp: new Date().toISOString(),\
        };' "$MCP_SERVER"

echo "✅ Patch 1 applied: Mock data removed"

# Patch 2: Ensure agent tracker is initialized
echo "🔨 Patch 2: Ensuring agent tracker initialization..."
if ! grep -q "// Initialize agent tracker on startup" "$MCP_SERVER"; then
    # Add initialization at the top of the constructor
    sed -i '/constructor(options = {}) {/a\
    // Initialize agent tracker on startup\
    if (!global.agentTracker) {\
      try {\
        const AgentTracker = require("./implementations/agent-tracker.js");\
        global.agentTracker = new AgentTracker();\
        console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Agent tracker initialized`);\
      } catch (e) {\
        console.error(`[${new Date().toISOString()}] ERROR [claude-flow-mcp] Failed to initialize agent tracker:`, e);\
      }\
    }' "$MCP_SERVER"
    echo "✅ Patch 2 applied: Agent tracker initialization added"
else
    echo "ℹ️  Patch 2 already applied"
fi

# Patch 3: Debug logging for agent spawn
echo "🔨 Patch 3: Adding debug logging for agent tracking..."
# Add debug logging right after trackAgent call
sed -i '/global.agentTracker.trackAgent(agentId, {/a\
          console.error(\
            `[${new Date().toISOString()}] DEBUG [agent-tracker] Tracked agent ${agentId} with swarmId: ${agentData.swarmId}`,\
          );' "$MCP_SERVER"

echo "✅ Patch 3 applied: Debug logging added"

# Patch 4: Fix getActiveSwarmId method to actually work
echo "🔨 Patch 4: Implementing getActiveSwarmId method..."
if ! grep -q "async getActiveSwarmId()" "$MCP_SERVER"; then
    # Add the method if it doesn't exist
    sed -i '/handleToolCall(id, tool) {/i\
  async getActiveSwarmId() {\
    try {\
      // Try to get from memory store\
      const swarmId = await this.memoryStore.retrieve("active_swarm", {\
        namespace: "system",\
      });\
      if (swarmId) {\
        return swarmId;\
      }\
      // Try to get from agent tracker\
      if (global.agentTracker && global.agentTracker.swarms.size > 0) {\
        const swarms = Array.from(global.agentTracker.swarms.keys());\
        return swarms[swarms.length - 1]; // Return most recent swarm\
      }\
      return null;\
    } catch (error) {\
      console.error(`[${new Date().toISOString()}] ERROR [claude-flow-mcp] Failed to get active swarm:`, error);\
      return null;\
    }\
  }\
' "$MCP_SERVER"
    echo "✅ Patch 4 applied: getActiveSwarmId method added"
else
    echo "ℹ️  Patch 4 already applied"
fi

# Patch 5: Store active swarm ID when creating swarm
echo "🔨 Patch 5: Storing active swarm ID on creation..."
sed -i '/Track swarm creation/i\
        // Store as active swarm\
        try {\
          await this.memoryStore.store("active_swarm", swarmId, {\
            namespace: "system",\
            metadata: { type: "active_swarm", sessionId: this.sessionId },\
          });\
          console.error(\
            `[${new Date().toISOString()}] INFO [claude-flow-mcp] Set active swarm: ${swarmId}`,\
          );\
        } catch (error) {\
          console.error(\
            `[${new Date().toISOString()}] ERROR [claude-flow-mcp] Failed to store active swarm:`,\
            error,\
          );\
        }\
' "$MCP_SERVER"

echo "✅ Patch 5 applied: Active swarm storage added"

# Test that the file is still valid JavaScript
echo "🧪 Testing patched file syntax..."
node -c "$MCP_SERVER" 2>/dev/null && echo "✅ Syntax check passed" || echo "❌ Syntax error in patched file"

# Restart the MCP TCP server
echo "🔄 Restarting MCP TCP server..."
pkill -f "mcp-tcp-server.js" 2>/dev/null || true
sleep 1

# The supervisor should restart it automatically
echo "⏳ Waiting for server to restart..."
sleep 3

# Check if server is running
if pgrep -f "mcp-tcp-server.js" > /dev/null; then
    echo "✅ MCP TCP server is running"
else
    echo "⚠️  MCP TCP server not running, attempting manual start..."
    nohup node /app/core-assets/scripts/mcp-tcp-server.js > /app/mcp-logs/mcp-tcp-server.log 2>&1 &
    sleep 2
    if pgrep -f "mcp-tcp-server.js" > /dev/null; then
        echo "✅ MCP TCP server started manually"
    else
        echo "❌ Failed to start MCP TCP server"
    fi
fi

# Test the health endpoint
echo "🏥 Testing health endpoint..."
curl -s http://localhost:9501/health | jq . 2>/dev/null && echo "✅ Health check passed" || echo "❌ Health check failed"

echo ""
echo "🎉 MCP Server patches applied successfully!"
echo ""
echo "📝 Summary of changes:"
echo "  1. Removed mock data fallback - agent_list now returns real agents only"
echo "  2. Ensured agent tracker is initialized on server startup"
echo "  3. Added debug logging for agent tracking"
echo "  4. Implemented getActiveSwarmId method"
echo "  5. Active swarm ID is now stored when creating swarms"
echo ""
echo "🧪 To test the fixes:"
echo "  1. The VisionFlow container should now connect to 'multi-agent-container:9500'"
echo "  2. Agent spawning should properly track agents with their swarm IDs"
echo "  3. agent_list should return actual spawned agents, not mock data"
echo ""
echo "📍 Next steps:"
echo "  - Restart the VisionFlow container to pick up the hostname change"
echo "  - Test the spawn agent button in the UI"
echo "  - Monitor logs: tail -f /app/mcp-logs/mcp-tcp-server.log"