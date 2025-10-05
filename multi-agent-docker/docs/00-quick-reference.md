# Quick Reference Guide

## One-Page Overview

### Start Container
```bash
docker-compose up -d
```

### Spawn Task
```bash
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "task description" "high")
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID
```

### Check Status
```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID
```

### Get Results
```bash
OUTPUT=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh output-dir $UUID)
ls -la "./workspace/ext/hive-sessions/$UUID/"
```

## Key Ports

| Port | Service |
|------|---------|
| 9500 | MCP TCP Server |
| 3002 | WebSocket Telemetry |
| 5901 | VNC Server |
| 6901 | noVNC (Browser) |

## Important Paths

| Path | Purpose |
|------|---------|
| `/app/scripts/hive-session-manager.sh` | Session management |
| `/workspace/.swarm/sessions/{UUID}/` | Session working directory |
| `/workspace/ext/hive-sessions/{UUID}/` | Session outputs |
| `./logs/` | Persistent logs (host) |

## Session Commands

```bash
# List all sessions
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list | jq

# Get session details
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh get $UUID | jq

# View logs
LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
docker exec multi-agent-container tail -f "$LOG"

# Cleanup old sessions
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 24
```

## MCP Telemetry Query

```bash
# Query via TCP
echo '{
  "jsonrpc":"2.0",
  "id":1,
  "method":"tools/call",
  "params": {
    "name":"session_status",
    "arguments":{"session_id":"'$UUID'"}
  }
}' | nc localhost 9500
```

## Common Troubleshooting

```bash
# Check container status
docker ps | grep multi-agent

# View startup logs
tail -f logs/entrypoint.log

# Check services
docker exec multi-agent-container supervisorctl status

# Restart MCP server
docker exec multi-agent-container \
  supervisorctl restart mcp-tcp-server

# Find database files
docker exec multi-agent-container \
  find /workspace/.swarm -name "memory.db" -ls
```

## Documentation Index

1. [Architecture Overview](01-architecture.md)
2. [Session Isolation](02-session-isolation.md)
3. [MCP Infrastructure](03-mcp-infrastructure.md)
4. [External Integration](04-external-integration.md)
5. [Session API Reference](05-session-api.md)
6. [TCP/MCP Telemetry](06-tcp-mcp-telemetry.md)
7. [Logging System](07-logging.md)
8. [VNC Access](08-vnc-access.md)
9. [Security](09-security.md)
10. [Troubleshooting](10-troubleshooting.md)
11. [Database Troubleshooting](11-database-troubleshooting.md)
