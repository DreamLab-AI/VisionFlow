# Fixes Required for /app/setup-workspace.sh

## Problem
The MCP server patches in `/app/setup-workspace.sh` are not being applied correctly, causing the system to return mock agent data instead of real agents.

## Root Causes

1. **Path Detection Issue**: The script looks for the MCP server but doesn't handle all cases
2. **Patch Application Failure**: The sed commands may not match the actual file content
3. **agent_list Function**: Still returns mock data instead of querying the memory store

## Required Changes to /app/setup-workspace.sh

### 1. Fix the patch_mcp_server() function path detection (line ~380-390)

**Current:**
```bash
local mcp_server_path=""
# First, try to find the globally installed version
if [ -f "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js" ]; then
    mcp_server_path="/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"
else
    # Fall back to searching in npm cache
    mcp_server_path=$(find /home/ubuntu/.npm/_npx -name "mcp-server.js" -path "*/claude-flow/src/mcp/*" 2>/dev/null | head -1)
fi
```

**Change to:**
```bash
local mcp_server_path=""
# Check multiple possible locations
for possible_path in \
    "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js" \
    "/usr/local/lib/node_modules/claude-flow/src/mcp/mcp-server.js" \
    "$(npm root -g)/claude-flow/src/mcp/mcp-server.js" \
; do
    if [ -f "$possible_path" ]; then
        mcp_server_path="$possible_path"
        break
    fi
done

# If still not found, search
if [ -z "$mcp_server_path" ]; then
    mcp_server_path=$(find /usr /home -name "mcp-server.js" -path "*/claude-flow/src/mcp/*" 2>/dev/null | head -1)
fi
```

### 2. Fix the agent_list patch (line ~460-530)

The current patch tries to replace the mock fallback but the sed command is too complex.

**Replace the entire agent_list patching section with:**

```bash
# Create a Python script to properly replace the function
cat > /tmp/fix_agent_list.py << 'PYTHON_PATCH'
#!/usr/bin/env python3
import re
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"

with open(file_path, 'r') as f:
    content = f.read()

# Find the agent_list function and replace it
new_agent_list = '''async agent_list(args = {}) {
    // PATCHED: Query real agents from memory store
    try {
      const allEntries = await this.memoryStore.list();
      const agents = allEntries
        .filter(e => e.key && e.key.includes('agent'))
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
          } catch { return null; }
        })
        .filter(Boolean);

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

# Replace the function
pattern = r'async agent_list\([^)]*\)\s*{[^}]*(?:{[^}]*}[^}]*)*}'
if re.search(pattern, content):
    content = re.sub(pattern, new_agent_list, content, count=1)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Patched agent_list")
else:
    print("⚠️ Could not find agent_list to patch")
PYTHON_PATCH

python3 /tmp/fix_agent_list.py "$mcp_server_path"
```

### 3. Ensure TCP server uses shared database (line ~570-600)

**Add after the TCP server patch:**

```bash
# Ensure shared database directory exists with correct permissions
mkdir -p /workspace/.swarm
chmod 777 /workspace/.swarm

# Set environment variable globally
echo "export CLAUDE_FLOW_DB_PATH=/workspace/.swarm/memory.db" >> /etc/profile.d/claude-flow.sh
```

### 4. Add verification step (at the end of patch_mcp_server)

```bash
# Verify the patches were applied
echo "🔍 Verifying patches..."
if grep -q "PATCHED" "$mcp_server_path"; then
    echo "✅ Patches verified in $mcp_server_path"
else
    echo "❌ Patches NOT applied correctly!"
    echo "   Manual intervention required"
fi

# Test the agent_list function
echo "🧪 Testing agent_list..."
test_result=$(echo '{"jsonrpc":"2.0","id":"test","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | timeout 2 nc localhost 9500 2>/dev/null | tail -1)

if echo "$test_result" | grep -q '"id":"agent-1"'; then
    echo "❌ Still returning mock data!"
else
    echo "✅ Mock data fixed!"
fi
```

## Quick Fix Script

Run this to apply the fixes immediately:

```bash
sudo /workspace/ext/fix-mcp-patches.sh
```

This will:
1. Find the correct mcp-server.js location
2. Replace the agent_list function to query real memory store
3. Fix the TCP server to use shared database
4. Restart the services

## Testing After Fix

```bash
# 1. Spawn a real agent
echo '{"jsonrpc":"2.0","id":"spawn-1","method":"tools/call","params":{"name":"agent_spawn","arguments":{"type":"coordinator","name":"RealAgent1"}}}' | nc localhost 9500 | tail -1

# 2. List agents (should show RealAgent1, not mock agent-1)
echo '{"jsonrpc":"2.0","id":"list-1","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | nc localhost 9500 | tail -1
```

If agent_list returns `RealAgent1` instead of `agent-1`, the fix worked!