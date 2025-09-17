#!/bin/bash
# Fix MCP Server Patches - Apply the critical patches to make agent tracking work

echo "🔧 Applying MCP Server Patches to fix agent tracking..."

# Find the MCP server file
MCP_SERVER="/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"

if [ ! -f "$MCP_SERVER" ]; then
    echo "❌ MCP server not found at $MCP_SERVER"
    echo "Searching for alternative location..."
    MCP_SERVER=$(find /usr -name "mcp-server.js" -path "*/claude-flow/*" 2>/dev/null | head -1)
    if [ -z "$MCP_SERVER" ]; then
        MCP_SERVER=$(find /home -name "mcp-server.js" -path "*/claude-flow/*" 2>/dev/null | head -1)
    fi

    if [ -z "$MCP_SERVER" ] || [ ! -f "$MCP_SERVER" ]; then
        echo "❌ Could not find mcp-server.js"
        exit 1
    fi
fi

echo "✅ Found MCP server at: $MCP_SERVER"

# Create backup
BACKUP="${MCP_SERVER}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$MCP_SERVER" "$BACKUP"
echo "📦 Created backup at: $BACKUP"

# Patch 1: Fix hardcoded version
echo "🔧 Patch 1: Fixing hardcoded version..."
if grep -q "this.version = '2.0.0-alpha.59'" "$MCP_SERVER"; then
    sed -i "s/this.version = '2.0.0-alpha.59'/\/\/ PATCHED: Dynamic version\n    try { this.version = require('..\/..\/package.json').version; } catch (e) { this.version = '2.0.0-alpha.101'; }/" "$MCP_SERVER"
    echo "✅ Applied version patch"
else
    echo "⚠️  Version already patched or different"
fi

# Patch 2: Fix agent_list to return real agents instead of mock data
echo "🔧 Patch 2: Fixing agent_list to use real database..."

# Create a temporary patch file with the correct agent_list implementation
cat > /tmp/agent_list_patch.js << 'PATCH_EOF'
      async agent_list(args = {}) {
        // PATCHED: Query real agents from memory store instead of mock data
        try {
          console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] agent_list called with args:`, args);

          // Get all entries from memory store
          const allEntries = await this.memoryStore.list();
          console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Found ${allEntries.length} total memory entries`);

          // Filter for agent entries
          const agentEntries = allEntries.filter(entry => {
            return entry.key && (
              entry.key.startsWith('agent:') ||
              entry.key.startsWith('agent_') ||
              entry.key.includes('agent')
            );
          });

          console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Found ${agentEntries.length} agent entries`);

          // Parse agent data
          const agents = [];
          for (const entry of agentEntries) {
            try {
              let agentData;
              if (typeof entry.value === 'string') {
                agentData = JSON.parse(entry.value);
              } else {
                agentData = entry.value;
              }

              // Extract agent info
              agents.push({
                id: agentData.agentId || agentData.id || entry.key.split(':').pop(),
                name: agentData.name || 'Unknown',
                type: agentData.type || agentData.agent_type || 'unknown',
                status: agentData.status || 'active',
                capabilities: agentData.capabilities || [],
                swarmId: agentData.swarmId || args.swarmId || 'default'
              });
            } catch (e) {
              console.error(`Failed to parse agent entry ${entry.key}:`, e);
            }
          }

          // If no real agents found, check if there are any spawned agents in memory
          if (agents.length === 0) {
            console.error(`[${new Date().toISOString()}] WARN [claude-flow-mcp] No real agents found, returning empty list`);
            return {
              success: true,
              swarmId: args.swarmId || 'default',
              agents: [],
              count: 0,
              timestamp: new Date().toISOString(),
              message: 'No agents currently spawned. Use agent_spawn to create agents.'
            };
          }

          console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Returning ${agents.length} real agents`);

          return {
            success: true,
            swarmId: args.swarmId || 'default',
            agents: agents,
            count: agents.length,
            timestamp: new Date().toISOString()
          };

        } catch (error) {
          console.error(`[${new Date().toISOString()}] ERROR [claude-flow-mcp] agent_list failed:`, error);
          return {
            success: false,
            error: error.message,
            agents: [],
            count: 0,
            timestamp: new Date().toISOString()
          };
        }
      }
PATCH_EOF

# Apply the agent_list patch by replacing the existing function
echo "🔧 Applying agent_list patch..."

# First, check if we can find the agent_list function
if grep -q "async agent_list" "$MCP_SERVER"; then
    echo "Found agent_list function, replacing it..."

    # Use a Python script to do the complex replacement
    python3 << 'PYTHON_EOF'
import re
import sys

mcp_server_path = "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"

# Read the original file
with open(mcp_server_path, 'r') as f:
    content = f.read()

# Read the patch
with open('/tmp/agent_list_patch.js', 'r') as f:
    patch_content = f.read()

# Find and replace the agent_list function
# Match from "async agent_list" to the closing brace of the function
pattern = r'async agent_list\([^)]*\)\s*{[^{}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}'

# Check if pattern exists
if re.search(pattern, content):
    # Replace with our patched version
    new_content = re.sub(pattern, patch_content.strip(), content, count=1)

    # Write back
    with open(mcp_server_path, 'w') as f:
        f.write(new_content)

    print("✅ Successfully replaced agent_list function")
else:
    print("⚠️  Could not find agent_list pattern to replace")
    # Try a simpler approach - look for the mock response
    if '// Fallback mock response' in content:
        print("Found mock response comment, attempting targeted replacement...")
        # This is more complex, would need line-by-line processing
        print("⚠️  Manual intervention may be needed")
PYTHON_EOF

else
    echo "⚠️  agent_list function not found in expected format"
fi

# Patch 3: Fix agent_spawn to properly persist agents
echo "🔧 Patch 3: Ensuring agent_spawn persists to memory store..."

# Add logging to agent_spawn if not present
if grep -q "async agent_spawn" "$MCP_SERVER"; then
    # Add detailed logging to agent_spawn
    sed -i '/async agent_spawn/,/^  }$/ {
        /const agentId = /a\        console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Spawning agent ${agentId} of type ${args.type}`);
        /await this.memoryStore.set/a\        console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Persisted agent ${agentId} to memory store`);
    }' "$MCP_SERVER" 2>/dev/null || echo "⚠️  Could not add agent_spawn logging"
fi

# Patch 4: Fix the TCP server to use the same memory database
TCP_SERVER="/app/core-assets/scripts/mcp-tcp-server.js"
if [ -f "$TCP_SERVER" ]; then
    echo "🔧 Patch 4: Fixing TCP server database path..."

    # Ensure it uses the shared database
    if ! grep -q "CLAUDE_FLOW_DB_PATH" "$TCP_SERVER"; then
        # Add environment variable for shared database
        sed -i "/spawn.*claude-flow/,/});/ {
            s/env: {/env: {\n          CLAUDE_FLOW_DB_PATH: '\/workspace\/.swarm\/memory.db',/
        }" "$TCP_SERVER" 2>/dev/null || echo "⚠️  Could not patch TCP server env"
    fi
    echo "✅ TCP server configured to use shared database"
else
    echo "⚠️  TCP server not found at $TCP_SERVER"
fi

# Create the shared database directory if it doesn't exist
if [ ! -d "/workspace/.swarm" ]; then
    mkdir -p /workspace/.swarm
    chmod 777 /workspace/.swarm
    echo "✅ Created shared database directory: /workspace/.swarm"
fi

# Restart the MCP TCP server to apply patches
echo "🔄 Restarting MCP TCP server..."
pkill -f "mcp-tcp-server.js" 2>/dev/null
sleep 2

# Start it again (it should auto-restart via supervisor or we can start manually)
if command -v supervisorctl >/dev/null 2>&1; then
    supervisorctl restart mcp-tcp-server 2>/dev/null || echo "⚠️  Could not restart via supervisorctl"
else
    # Try to start it manually in background
    nohup node /app/core-assets/scripts/mcp-tcp-server.js > /workspace/ext/logs/mcp-tcp.log 2>&1 &
    echo "✅ Started MCP TCP server manually"
fi

echo ""
echo "✅ MCP patches applied!"
echo ""
echo "Test with these commands:"
echo "  1. Spawn agent: echo '{\"jsonrpc\":\"2.0\",\"id\":\"spawn-1\",\"method\":\"tools/call\",\"params\":{\"name\":\"agent_spawn\",\"arguments\":{\"type\":\"coordinator\",\"name\":\"TestCoord\"}}}' | nc localhost 9500 | tail -1"
echo "  2. List agents: echo '{\"jsonrpc\":\"2.0\",\"id\":\"list-1\",\"method\":\"tools/call\",\"params\":{\"name\":\"agent_list\",\"arguments\":{}}}' | nc localhost 9500 | tail -1"
echo ""