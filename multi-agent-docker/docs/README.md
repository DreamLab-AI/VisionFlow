# Multi-Agent Docker System Documentation

**Last Updated**: 2025-10-06
**Version**: 2.1 (Database-Isolated Architecture)

## ⚠️ Critical: Database Isolation Required

**NEVER run `claude-flow init --force` from `/workspace`** - This creates a shared database that causes container crashes.

The system uses automatic database isolation. No manual initialization is needed.

## Overview

The Multi-Agent Docker System is a containerized AI orchestration platform that provides:

- **Hive-Mind Task Orchestration** with UUID-based session isolation
- **Database Isolation** preventing SQLite lock conflicts
- **MCP (Model Context Protocol)** server infrastructure for AI tool integration
- **GPU-Accelerated Spring Visualization** telemetry streaming
- **VNC Desktop Environment** for GUI-based AI tools (Blender, QGIS, Playwright)
- **Persistent Logging** for debugging and monitoring

## Quick Start

```bash
# Start the container
docker-compose up -d

# Create a hive-mind task session
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "your task here" "high")

# Start the task
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID

# Monitor status
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID

# If experiencing container crashes
docker exec multi-agent-container rm -f /workspace/.swarm/memory.db*
docker-compose restart
```

## Documentation Structure

### Core Concepts
- **[Architecture Overview](01-architecture.md)** - System design and components
- **[Session Isolation](02-session-isolation.md)** - Database isolation and UUID tracking
- **[MCP Infrastructure](03-mcp-infrastructure.md)** - Model Context Protocol servers

### Integration Guides
- **[External Integration](04-external-integration.md)** - Rust/external system integration
- **[Session API Reference](05-session-api.md)** - Complete API documentation
- **[TCP/MCP Telemetry](06-tcp-mcp-telemetry.md)** - Real-time monitoring and visualization

### Operations
- **[Logging System](07-logging.md)** - Persistent logs and debugging
- **[VNC Access](08-vnc-access.md)** - Remote desktop for GUI tools
- **[Security](09-security.md)** - Authentication and access control

### Troubleshooting
- **[Common Issues](10-troubleshooting.md)** - Solutions to known problems
- **[Database Conflicts](11-database-troubleshooting.md)** - SQLite lock resolution

## Key Features

### 1. Session-Based Isolation
Each external task spawn gets:
- Unique UUID identifier
- Isolated SQLite database
- Dedicated output directory
- Separate log file

### 2. Hybrid Architecture
- **Control Plane**: Docker exec for task spawning (reliable, fast)
- **Data Plane**: TCP/MCP for telemetry streaming (rich data)
- **Visualization**: WebSocket for real-time updates

### 3. Concurrent Task Support
- No SQLite lock conflicts
- Unlimited parallel spawns
- Independent task lifecycles
- Resource isolation

## Directory Structure

```
multi-agent-docker/
├── docs/                       # Documentation (you are here)
├── scripts/                    # Utility scripts
│   ├── hive-session-manager.sh # Session lifecycle management
│   └── configure-claude-mcp.sh # MCP configuration
├── core-assets/
│   ├── scripts/
│   │   └── mcp-tcp-server.js   # TCP MCP server
│   └── mcp.json                # MCP server registry
├── logs/                       # Persistent logs (mounted)
│   ├── mcp/                    # MCP server logs
│   ├── supervisor/             # Process manager logs
│   └── entrypoint.log          # Container startup log
├── workspace/                  # Persistent workspace (mounted)
│   ├── .swarm/
│   │   ├── sessions/{UUID}/    # Session working directories
│   │   └── tcp-server-instance/ # TCP server isolation
│   └── ext/
│       └── hive-sessions/{UUID}/ # Session outputs
└── docker-compose.yml
```

## Container Ports

| Port | Service | Purpose |
|------|---------|---------|
| 9500 | TCP MCP Server | Primary MCP communication |
| 9502 | Claude-Flow TCP | Isolated session MCP |
| 3002 | WebSocket Bridge | Real-time telemetry streaming |
| 5901 | VNC Server | Remote desktop access |
| 6901 | noVNC Web | Browser-based VNC |
| 9876-9880 | GUI MCP Servers | Blender, QGIS, Playwright, etc. |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WS_AUTH_TOKEN` | - | WebSocket authentication |
| `TCP_AUTH_TOKEN` | - | TCP MCP authentication |
| `ANTHROPIC_API_KEY` | - | Claude API access |
| `GOOGLE_API_KEY` | - | Google AI Studio access |
| `NODE_ENV` | production | Environment mode |

## Getting Help

- Check the relevant doc in this directory
- Review logs: `tail -f logs/mcp/*.log`
- Inspect sessions: `/app/scripts/hive-session-manager.sh list`
- Report issues with full logs and session UUID

## Important Usage Notes

### Do NOT Use These Commands
- ❌ `claude-flow init --force` (creates shared database conflicts)
- ❌ `claude-flow-init-agents` alias (same issue)
- ❌ Direct `claude-flow hive-mind spawn` from `/workspace` (bypasses isolation)

### DO Use These Patterns
- ✅ Session manager API for all task spawns
- ✅ MCP servers handle initialization automatically
- ✅ Wrapper handles CLI commands transparently
- ✅ Each session gets isolated database automatically

### Database Health Check
```bash
# Should show ONLY isolated databases
docker exec multi-agent-container find /workspace/.swarm -name "memory.db" -ls

# Correct output:
#   tcp-server-instance/.swarm/memory.db
#   sessions/{UUID}/.swarm/memory.db
#   root-cli-instance/.swarm/memory.db

# If you see /workspace/.swarm/memory.db → DELETE IT
```

## Next Steps

1. Read [Architecture Overview](01-architecture.md) to understand the system
2. Follow [External Integration](04-external-integration.md) to integrate with your Rust system
3. Check [Session API Reference](05-session-api.md) for detailed API documentation
