#!/bin/bash
# Send client message to MCP TCP server for forwarding to telemetry panel
# Usage: send-client-message.sh "Your message here" [session_id] [agent_id]

set -e

MESSAGE="$1"
SESSION_ID="${2:-}"
AGENT_ID="${3:-}"
MCP_TCP_PORT="${MCP_TCP_PORT:-9500}"
MCP_TCP_HOST="${MCP_TCP_HOST:-localhost}"

if [ -z "$MESSAGE" ]; then
    echo "Usage: send-client-message.sh \"message\" [session_id] [agent_id]"
    exit 1
fi

# Build JSON-RPC notification
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

read -r -d '' JSON_PAYLOAD << EOF || true
{
  "jsonrpc": "2.0",
  "method": "notification",
  "params": {
    "type": "client",
    "message": "$MESSAGE",
    "timestamp": "$TIMESTAMP",
    "session_id": ${SESSION_ID:+\"$SESSION_ID\"},
    "agent_id": ${AGENT_ID:+\"$AGENT_ID\"}
  }
}
EOF

# Send via netcat
echo "$JSON_PAYLOAD" | nc -q 1 "$MCP_TCP_HOST" "$MCP_TCP_PORT" 2>/dev/null || {
    echo "Failed to send message to MCP server at $MCP_TCP_HOST:$MCP_TCP_PORT" >&2
    exit 1
}

echo "✓ Client message sent successfully" >&2
