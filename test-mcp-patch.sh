#!/bin/bash
# Test MCP Patching - Debug Version

set -e  # Exit on error

echo "🔧 Testing MCP Server Patches..."

# 1. Check if claude-flow is installed
echo "Checking claude-flow installation..."
if command -v claude-flow >/dev/null 2>&1; then
    echo "✅ claude-flow found at: $(which claude-flow)"
    echo "   Version: $(claude-flow --version 2>/dev/null || echo 'unknown')"
else
    echo "❌ claude-flow not found"
    exit 1
fi

# 2. Find MCP server file
echo ""
echo "Finding MCP server file..."
MCP_SERVER=""

# Check multiple possible locations
for possible_path in \
    "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js" \
    "/usr/local/lib/node_modules/claude-flow/src/mcp/mcp-server.js" \
    "$(npm root -g 2>/dev/null)/claude-flow/src/mcp/mcp-server.js" \
; do
    echo "  Checking: $possible_path"
    if [ -f "$possible_path" ]; then
        MCP_SERVER="$possible_path"
        echo "  ✅ Found!"
        break
    fi
done

if [ -z "$MCP_SERVER" ]; then
    echo "  Searching filesystem..."
    MCP_SERVER=$(find /usr /home -name "mcp-server.js" -path "*/claude-flow/src/mcp/*" 2>/dev/null | head -1)
fi

if [ -z "$MCP_SERVER" ] || [ ! -f "$MCP_SERVER" ]; then
    echo "❌ MCP server not found!"
    exit 1
fi

echo "✅ MCP server found at: $MCP_SERVER"

# 3. Check current state
echo ""
echo "Current state of MCP server:"
echo "  Has PATCHED markers: $(grep -c "PATCHED" "$MCP_SERVER" 2>/dev/null || echo '0')"
echo "  Has hardcoded version: $(grep -c "2.0.0-alpha.59" "$MCP_SERVER" 2>/dev/null || echo '0')"
echo "  Has mock fallback: $(grep -c "agent-1.*agent-2.*agent-3" "$MCP_SERVER" 2>/dev/null || echo '0')"

# 4. Test current behavior
echo ""
echo "Testing current agent_list behavior..."
TEST_RESULT=$(echo '{"jsonrpc":"2.0","id":"test","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | timeout 3 nc localhost 9500 2>/dev/null | tail -1)

if [ -z "$TEST_RESULT" ]; then
    echo "⚠️  MCP server not responding on port 9500"
else
    if echo "$TEST_RESULT" | grep -q '"id":"agent-1"'; then
        echo "❌ Currently returning MOCK data (agent-1, agent-2, agent-3)"
        echo "   Need to apply patches!"
    else
        echo "✅ Not returning mock data"

        # Check if it returns real agents or empty
        if echo "$TEST_RESULT" | grep -q '"agents":\[\]'; then
            echo "   Returns empty agent list (no agents spawned)"
        else
            echo "   Returns real agents!"
            echo "$TEST_RESULT" | python3 -m json.tool 2>/dev/null | head -20
        fi
    fi
fi

# 5. Apply patches if needed
echo ""
read -p "Do you want to apply patches? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Applying patches..."

    # Create backup
    BACKUP="${MCP_SERVER}.backup.$(date +%s)"
    cp "$MCP_SERVER" "$BACKUP"
    echo "✅ Backup created: $BACKUP"

    # Apply Python patch for agent_list
    cat > /tmp/fix_agent_list.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import re
import sys

file_path = sys.argv[1]
print(f"Processing: {file_path}")

with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if "PATCHED: Query real agents from memory store" in content:
    print("Already patched!")
    sys.exit(0)

# Find agent_list function
new_agent_list = '''async agent_list(args = {}) {
    // PATCHED: Query real agents from memory store
    console.error(`[${new Date().toISOString()}] DEBUG agent_list called with args:`, args);
    try {
      const allEntries = await this.memoryStore.list();
      console.error(`[${new Date().toISOString()}] DEBUG Found ${allEntries.length} total entries`);

      const agents = allEntries
        .filter(e => {
          const matches = e.key && (e.key.includes('agent') || e.key.includes('Agent'));
          if (matches) console.error(`[${new Date().toISOString()}] DEBUG Agent entry: ${e.key}`);
          return matches;
        })
        .map(e => {
          try {
            const data = typeof e.value === 'string' ? JSON.parse(e.value) : e.value;
            return {
              id: data.agentId || data.id || e.key,
              name: data.name || 'Unknown',
              type: data.type || 'unknown',
              status: data.status || 'active',
              capabilities: data.capabilities || []
            };
          } catch (err) {
            console.error(`Failed to parse ${e.key}:`, err);
            return null;
          }
        })
        .filter(Boolean);

      console.error(`[${new Date().toISOString()}] DEBUG Returning ${agents.length} agents`);

      return {
        success: true,
        agents: agents,
        count: agents.length,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('agent_list error:', error);
      return { success: false, agents: [], error: error.message };
    }
  }'''

# Try to find and replace the function
# First try exact match
pattern1 = r'async agent_list\([^)]*\)\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}'
match = re.search(pattern1, content, re.DOTALL)

if match:
    print(f"Found agent_list at position {match.start()}")
    content = content[:match.start()] + new_agent_list + content[match.end():]
    print("Replaced agent_list function")
else:
    print("Could not find agent_list with regex, trying line-by-line search...")

    # Try to find by searching for the function declaration
    lines = content.split('\n')
    start_idx = -1
    for i, line in enumerate(lines):
        if 'async agent_list' in line:
            start_idx = i
            print(f"Found agent_list at line {i}")
            break

    if start_idx >= 0:
        # Find the end of the function (matching braces)
        brace_count = 0
        end_idx = start_idx
        in_function = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
                in_function = True
            if '}' in line:
                brace_count -= line.count('}')

            if in_function and brace_count == 0:
                end_idx = i
                break

        if end_idx > start_idx:
            print(f"Function ends at line {end_idx}")
            # Replace the function
            new_lines = lines[:start_idx] + new_agent_list.split('\n') + lines[end_idx+1:]
            content = '\n'.join(new_lines)
            print("Replaced agent_list function via line search")
        else:
            print("ERROR: Could not find function end")
            sys.exit(1)
    else:
        print("ERROR: Could not find agent_list function!")
        sys.exit(1)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✅ Successfully patched agent_list")
PYTHON_EOF

    python3 /tmp/fix_agent_list.py "$MCP_SERVER"

    # Restart MCP server
    echo "Restarting MCP TCP server..."
    pkill -f "mcp-tcp-server.js" 2>/dev/null || true
    sleep 2

    # The supervisor should restart it, or we can start manually
    if ! pgrep -f "mcp-tcp-server.js" >/dev/null; then
        echo "Starting MCP TCP server manually..."
        nohup node /app/core-assets/scripts/mcp-tcp-server.js > /workspace/ext/logs/mcp-tcp.log 2>&1 &
    fi

    sleep 3

    # Test again
    echo ""
    echo "Testing after patches..."
    TEST_RESULT=$(echo '{"jsonrpc":"2.0","id":"test","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | timeout 3 nc localhost 9500 2>/dev/null | tail -1)

    if echo "$TEST_RESULT" | grep -q '"id":"agent-1"'; then
        echo "❌ STILL returning mock data - patch may have failed"
    else
        echo "✅ Mock data fixed!"
    fi
fi

echo ""
echo "Done!"