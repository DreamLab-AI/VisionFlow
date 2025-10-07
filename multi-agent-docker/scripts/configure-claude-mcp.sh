#!/bin/bash
# Configure Claude MCP with isolated database paths
set -e

CLAUDE_CONFIG="/home/dev/.claude/.claude.json"
CLAUDE_DIR="/home/dev/.claude"

echo "🔧 Configuring Claude MCP with database isolation..."

# Ensure .claude directory exists
mkdir -p "$CLAUDE_DIR"

# Also update /home/dev/.mcp.json if it exists
if [ -f "/home/dev/.mcp.json" ]; then
    echo "📝 Updating /home/dev/.mcp.json..."
    cp "/home/dev/.mcp.json" "/home/dev/.mcp.json.bak.$(date +%s)"

    jq '.mcpServers["claude-flow"].command = "node" |
        .mcpServers["claude-flow"].args = ["/app/scripts/stdio-to-tcp-bridge.js"] |
        .mcpServers["claude-flow"].env.MCP_HOST = "127.0.0.1" |
        .mcpServers["claude-flow"].env.MCP_PORT = "9500" |
        del(.mcpServers["claude-flow"].env.CLAUDE_FLOW_DB_PATH)' \
        /home/dev/.mcp.json > /home/dev/.mcp.json.tmp && \
        mv /home/dev/.mcp.json.tmp /home/dev/.mcp.json

    chown dev:dev /home/dev/.mcp.json
    echo "✅ Updated /home/dev/.mcp.json"
fi

# Create or update .claude.json with isolated DB paths
if [ -f "$CLAUDE_CONFIG" ]; then
    echo "📝 Updating existing Claude config..."

    # Backup existing config
    cp "$CLAUDE_CONFIG" "${CLAUDE_CONFIG}.bak.$(date +%s)"

    # Update claude-flow MCP config to use TCP bridge
    jq '.mcpServers["claude-flow"].command = "node" |
        .mcpServers["claude-flow"].args = ["/app/scripts/stdio-to-tcp-bridge.js"] |
        .mcpServers["claude-flow"].env.MCP_HOST = "127.0.0.1" |
        .mcpServers["claude-flow"].env.MCP_PORT = "9500" |
        del(.mcpServers["claude-flow"].env.CLAUDE_FLOW_DB_PATH)' \
        "$CLAUDE_CONFIG" > "${CLAUDE_CONFIG}.tmp" && mv "${CLAUDE_CONFIG}.tmp" "$CLAUDE_CONFIG"

    echo "✅ Updated existing config"
else
    echo "📝 Creating new Claude config..."

    # Create base config with MCP servers using TCP bridge
    cat > "$CLAUDE_CONFIG" << 'EOF'
{
  "mcpServers": {
    "claude-flow": {
      "command": "node",
      "args": ["/app/scripts/stdio-to-tcp-bridge.js"],
      "type": "stdio",
      "env": {
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": "9500"
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
    jq '.mcpServers["claude-flow"].command = "node" |
        .mcpServers["claude-flow"].args = ["/app/scripts/stdio-to-tcp-bridge.js"] |
        .mcpServers["claude-flow"].env.MCP_HOST = "127.0.0.1" |
        .mcpServers["claude-flow"].env.MCP_PORT = "9500" |
        del(.mcpServers["claude-flow"].env.CLAUDE_FLOW_DB_PATH)' \
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

# Configure SQLite databases with optimal settings
if [ -f "/app/scripts/sqlite-db-setup.sh" ]; then
    echo "📝 Configuring SQLite databases..."
    /app/scripts/sqlite-db-setup.sh configure
fi

echo "✅ Claude MCP configuration complete"
echo ""
echo "MCP Architecture:"
echo "  - TCP Server:    Running on port 9500 (/workspace/.swarm/tcp-server.db)"
echo "  - Claude MCP:    stdio-to-TCP bridge → port 9500"
echo "  - Hook calls:    /workspace/.swarm/claude-hooks.db"
