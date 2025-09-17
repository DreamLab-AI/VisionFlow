#!/bin/bash
# Manual fix for agent_list function

echo "🔧 Manually fixing agent_list function..."

MCP_SERVER="/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"

# Check if file exists
if [ ! -f "$MCP_SERVER" ]; then
    echo "❌ MCP server file not found!"
    exit 1
fi

# Create backup
BACKUP="${MCP_SERVER}.backup.manual.$(date +%s)"
cp "$MCP_SERVER" "$BACKUP"
echo "✅ Created backup: $BACKUP"

# Check if agent_list exists
if ! grep -q "async agent_list" "$MCP_SERVER"; then
    echo "❌ agent_list function not found!"

    # Look for where tools are implemented
    echo "Searching for tools implementation..."
    grep -n "case 'agent" "$MCP_SERVER" | head -5
    exit 1
fi

# Use Node.js to fix it since Python has permission issues
cat > /tmp/fix-agent-list.js << 'EOF'
const fs = require('fs');

const filePath = '/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js';
let content = fs.readFileSync(filePath, 'utf8');

// Check if already patched
if (content.includes('PATCHED: Query real agents from memory store')) {
    console.log('✅ Already patched!');
    process.exit(0);
}

// Find the agent_list method in the tools handler
// It's likely in a switch/case statement
const searchPattern = /case ['"]agent_list['"]:[\s\S]*?return\s*{[\s\S]*?};/;
const match = content.match(searchPattern);

if (match) {
    console.log('Found agent_list in switch/case at position', match.index);

    const replacement = `case 'agent_list':
      // PATCHED: Query real agents from memory store
      console.error(\`[\${new Date().toISOString()}] DEBUG agent_list called\`);
      try {
        const allEntries = await this.memoryStore.list();
        console.error(\`[\${new Date().toISOString()}] DEBUG Found \${allEntries.length} total entries\`);

        const agents = [];
        for (const entry of allEntries) {
          if (entry.key && entry.key.includes('agent')) {
            console.error(\`[\${new Date().toISOString()}] DEBUG Processing entry: \${entry.key}\`);
            try {
              const data = typeof entry.value === 'string' ? JSON.parse(entry.value) : entry.value;
              agents.push({
                id: data.agentId || data.id || entry.key.split(':').pop(),
                name: data.name || 'Unknown',
                type: data.type || 'unknown',
                status: data.status || 'active',
                capabilities: data.capabilities || []
              });
            } catch (e) {
              console.error(\`Failed to parse \${entry.key}:\`, e);
            }
          }
        }

        console.error(\`[\${new Date().toISOString()}] DEBUG Returning \${agents.length} real agents\`);

        return {
          success: true,
          swarmId: args.swarmId || 'default',
          agents: agents,
          count: agents.length,
          timestamp: new Date().toISOString()
        };
      } catch (error) {
        console.error('agent_list error:', error);
        return {
          success: false,
          agents: [],
          error: error.message,
          timestamp: new Date().toISOString()
        };
      }`;

    content = content.replace(searchPattern, replacement);
    console.log('Replaced agent_list case');
} else {
    // Try to find it as a method
    const methodPattern = /async\s+agent_list\s*\([^)]*\)\s*{[\s\S]*?^  }/m;
    const methodMatch = content.match(methodPattern);

    if (methodMatch) {
        console.log('Found agent_list as method');
        // This would need more complex replacement
        console.log('⚠️  Found as method but complex replacement needed');
    } else {
        console.log('❌ Could not find agent_list implementation!');

        // Search for mock data to locate it
        if (content.includes('agent-1') && content.includes('coordinator-1')) {
            console.log('Found mock data location, needs manual fix');

            // Find the line numbers
            const lines = content.split('\n');
            lines.forEach((line, i) => {
                if (line.includes('agent-1') || line.includes('coordinator-1')) {
                    console.log(`Line ${i+1}: ${line.trim().substring(0, 80)}`);
                }
            });
        }
        process.exit(1);
    }
}

// Write the fixed content
fs.writeFileSync(filePath, content);
console.log('✅ File updated successfully');
EOF

# Run the Node.js fix
node /tmp/fix-agent-list.js

echo ""
echo "Testing the fix..."

# Kill and restart MCP TCP server
pkill -f "mcp-tcp-server" 2>/dev/null
sleep 2

echo "Starting MCP TCP server..."
nohup node /app/core-assets/scripts/mcp-tcp-server.js > /tmp/mcp-tcp.log 2>&1 &
echo "Started with PID $!"

sleep 5

# Test it
echo "Testing agent_list..."
TEST=$(echo '{"jsonrpc":"2.0","id":"test","method":"tools/call","params":{"name":"agent_spawn","arguments":{"type":"coordinator","name":"TestAgent"}}}' | timeout 3 nc localhost 9500 2>/dev/null | tail -1)

if [ -n "$TEST" ]; then
    echo "✅ MCP server responding"

    # Now test agent_list
    LIST=$(echo '{"jsonrpc":"2.0","id":"test","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | timeout 3 nc localhost 9500 2>/dev/null | tail -1)

    if echo "$LIST" | grep -q "agent-1"; then
        echo "❌ Still returning mock data!"
    else
        echo "✅ Mock data appears to be fixed!"
    fi
else
    echo "❌ MCP server not responding"
fi

echo ""
echo "Check logs at: /tmp/mcp-tcp.log"