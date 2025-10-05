#!/bin/bash
# Configure Claude MCP with isolated database paths
set -e

CLAUDE_CONFIG="/home/dev/.claude/.claude.json"
CLAUDE_DIR="/home/dev/.claude"

echo "🔧 Configuring Claude MCP with database isolation..."

# Ensure .claude directory exists
mkdir -p "$CLAUDE_DIR"

# Create or update .claude.json with isolated DB paths
if [ -f "$CLAUDE_CONFIG" ]; then
    echo "📝 Updating existing Claude config..."

    # Backup existing config
    cp "$CLAUDE_CONFIG" "${CLAUDE_CONFIG}.bak.$(date +%s)"

    # Update claude-flow MCP config with isolated DB
    jq '.mcpServers["claude-flow"].env.CLAUDE_FLOW_DB_PATH = "/workspace/.swarm/claude-local.db"' \
        "$CLAUDE_CONFIG" > "${CLAUDE_CONFIG}.tmp" && mv "${CLAUDE_CONFIG}.tmp" "$CLAUDE_CONFIG"

    # Update command to use wrapper instead of npx @latest
    jq '.mcpServers["claude-flow"].command = "/workspace/claude-flow" |
        .mcpServers["claude-flow"].args = ["mcp", "start"]' \
        "$CLAUDE_CONFIG" > "${CLAUDE_CONFIG}.tmp" && mv "${CLAUDE_CONFIG}.tmp" "$CLAUDE_CONFIG"

    echo "✅ Updated existing config"
else
    echo "📝 Creating new Claude config..."

    # Create base config with MCP servers
    cat > "$CLAUDE_CONFIG" << 'EOF'
{
  "mcpServers": {
    "claude-flow": {
      "command": "/workspace/claude-flow",
      "args": ["mcp", "start"],
      "type": "stdio",
      "env": {
        "CLAUDE_FLOW_DB_PATH": "/workspace/.swarm/claude-local.db"
      }
    }
  }
}
EOF

    echo "✅ Created new config"
fi

# Also update workspace .mcp.json
if [ -f "/workspace/.mcp.json" ]; then
    echo "📝 Updating workspace MCP config..."
    jq '.mcpServers["claude-flow"].env.CLAUDE_FLOW_DB_PATH = "/workspace/.swarm/claude-local.db"' \
        /workspace/.mcp.json > /workspace/.mcp.json.tmp && \
        mv /workspace/.mcp.json.tmp /workspace/.mcp.json
fi

# Update settings.json to isolate hook DB paths
SETTINGS_FILE="/workspace/.claude/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
    echo "📝 Updating Claude settings hooks with DB isolation..."

    # Backup
    cp "$SETTINGS_FILE" "${SETTINGS_FILE}.bak.$(date +%s)"

    # Add CLAUDE_FLOW_DB_PATH prefix to all hook commands
    jq '
    (.hooks.PreToolUse[]?.hooks[]?.command // empty) |=
        if test("npx claude-flow") and (test("CLAUDE_FLOW_DB_PATH") | not) then
            "CLAUDE_FLOW_DB_PATH=/workspace/.swarm/claude-hooks.db " + .
        else
            .
        end |
    (.hooks.PostToolUse[]?.hooks[]?.command // empty) |=
        if test("npx claude-flow") and (test("CLAUDE_FLOW_DB_PATH") | not) then
            "CLAUDE_FLOW_DB_PATH=/workspace/.swarm/claude-hooks.db " + .
        else
            .
        end
    ' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"

    echo "✅ Updated hook DB isolation"
fi

# Set proper ownership
chown -R dev:dev "$CLAUDE_DIR" /workspace/.mcp.json /workspace/.claude 2>/dev/null || true

echo "✅ Claude MCP configuration complete"
echo ""
echo "Database isolation:"
echo "  - TCP Server:    /workspace/.swarm/tcp-server.db"
echo "  - Claude MCP:    /workspace/.swarm/claude-local.db"
echo "  - Hook calls:    /workspace/.swarm/claude-hooks.db"
