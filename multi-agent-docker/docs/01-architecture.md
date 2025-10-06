# Architecture Overview

## System Design

The Multi-Agent Docker System uses a **hybrid control/data plane architecture** that separates task orchestration from telemetry streaming.

```
External System (Rust)
    ↓
Docker Exec (Control Plane)
    ↓
Hive-Mind Session Manager
    ↓
Isolated Session Directories
    ├── UUID-1: /workspace/.swarm/sessions/{uuid-1}/
    ├── UUID-2: /workspace/.swarm/sessions/{uuid-2}/
    └── UUID-N: /workspace/.swarm/sessions/{uuid-n}/
         ↓
    Each has its own .swarm/memory.db
         ↓
TCP/MCP Server (Data Plane)
    ↓
WebSocket Bridge
    ↓
GPU Spring Visualization System
```

## Core Components

### 1. Hive-Mind Session Manager
**Location**: `/app/scripts/hive-session-manager.sh`

**Purpose**: Manages isolated task execution environments

**Responsibilities**:
- UUID generation for session tracking
- Directory isolation for database separation
- Session lifecycle management (create, start, monitor, cleanup)
- Metadata persistence and indexing

**Key Features**:
- File-based locking for concurrent access
- JSON session registry
- Automatic cleanup of old sessions
- Output directory mapping to `/workspace/ext/`

### 2. MCP TCP Server
**Location**: `/app/core-assets/scripts/mcp-tcp-server.js`

**Purpose**: Provides TCP-based MCP communication for telemetry

**Characteristics**:
- Runs in isolated directory: `/workspace/.swarm/tcp-server-instance/`
- Separate database from hive-mind sessions
- Persistent process managed by supervisord
- Supports multiple concurrent clients
- Authentication optional (configured via env vars)

**Port**: 9500 (primary MCP endpoint)

### 3. WebSocket Bridge
**Location**: `/app/core-assets/scripts/mcp-ws-relay.js`

**Purpose**: Stream telemetry data to external visualization systems

**Features**:
- Real-time data streaming
- Client subscription filtering
- Bandwidth limiting
- Compression support
- Session-based telemetry routing

**Port**: 3002

### 4. Supervisor Process Manager
**Configuration**: `/etc/supervisor/conf.d/supervisord.conf`

**Manages**:
- MCP TCP server
- WebSocket bridge
- Claude-Flow TCP proxy
- VNC services (X11, XFCE, noVNC)
- GUI MCP servers (Playwright, QGIS, Blender, PBR, Web Summary)

**Features**:
- Automatic restart on failure
- Process health monitoring
- Persistent logging
- Resource isolation

## Data Flow

### Task Creation Flow

```
1. External System
   ↓ docker exec
2. hive-session-manager.sh create "task"
   ↓ generates UUID
3. Create session directory
   /workspace/.swarm/sessions/{UUID}/
   ↓
4. Write session metadata
   session.json with UUID, task, status, paths
   ↓
5. Return UUID to external system
```

### Task Execution Flow

```
1. External System
   ↓ docker exec -d (background)
2. hive-session-manager.sh start {UUID}
   ↓ cd to session directory
3. Spawn claude-flow hive-mind
   Working directory: /workspace/.swarm/sessions/{UUID}/
   ↓
4. claude-flow creates database
   /workspace/.swarm/sessions/{UUID}/.swarm/memory.db
   ↓ (isolated, no conflicts)
5. Task executes, writes output
   /workspace/ext/hive-sessions/{UUID}/
   ↓
6. Update session status
   created → starting → running → completed/failed
```

### Telemetry Flow

```
1. Hive-mind task running
   ↓
2. MCP TCP Server queries status
   ↓ port 9500
3. External system connects to TCP MCP
   ↓
4. Query session-specific metrics
   ↓
5. TCP server streams to WebSocket bridge
   ↓ port 3002
6. WebSocket → GPU Spring Visualization
```

## Database Isolation Strategy

### Problem Solved
SQLite databases don't support concurrent writers. Previous architecture had all spawns writing to `/workspace/.swarm/memory.db` → lock conflicts → container crashes.

### Solution: Working Directory Isolation
Each process runs in its own working directory, causing claude-flow to create its own isolated database:

| Process | Working Directory | Database Path |
|---------|------------------|---------------|
| TCP MCP Server | `/workspace/.swarm/tcp-server-instance/` | `tcp-server-instance/.swarm/memory.db` |
| Root CLI Wrapper | `/workspace/.swarm/root-cli-instance/` | `root-cli-instance/.swarm/memory.db` |
| Session UUID-1 | `/workspace/.swarm/sessions/{uuid-1}/` | `sessions/{uuid-1}/.swarm/memory.db` |
| Session UUID-2 | `/workspace/.swarm/sessions/{uuid-2}/` | `sessions/{uuid-2}/.swarm/memory.db` |
| Session UUID-N | `/workspace/.swarm/sessions/{uuid-n}/` | `sessions/{uuid-n}/.swarm/memory.db` |

**Result**: Different files = No lock conflicts = No crashes

### Critical Rule: No Shared Database

**⚠️ NEVER create `/workspace/.swarm/memory.db`**

This path is NOT used by the system. If it exists, it indicates:
- Someone ran `claude-flow init --force` from `/workspace` (DON'T DO THIS)
- Legacy code path creating conflicts
- Container will likely crash due to database locks

**If container keeps restarting**:
```bash
docker exec multi-agent-container rm -f /workspace/.swarm/memory.db*
```

### How claude-flow Wrapper Works

The system installs a wrapper at `/usr/bin/claude-flow` that:
1. Detects if running as root user
2. If root: switches to dev user and changes to `/workspace/.swarm/root-cli-instance/`
3. If dev: runs normally in current directory
4. Prevents accidental shared database creation

## Session Directory Structure

```
/workspace/.swarm/sessions/{UUID}/
├── session.json              # Metadata
│   ├── session_id: UUID
│   ├── task: "task description"
│   ├── status: "running"
│   ├── created: ISO timestamp
│   ├── working_dir: path
│   ├── output_dir: path
│   ├── database: path
│   └── log_file: path
│
├── .swarm/
│   └── memory.db             # Isolated SQLite database
│
└── output/                   # Symlink to /workspace/ext/hive-sessions/{UUID}/
```

## Process Isolation Model

### Control Plane (Synchronous, Reliable)
- **Protocol**: Docker exec
- **Use Cases**: Task spawn, status checks, cleanup
- **Advantages**:
  - Direct process control
  - No network dependencies
  - Immediate feedback
  - Simple error handling

### Data Plane (Asynchronous, High-Bandwidth)
- **Protocol**: TCP/MCP over port 9500
- **Use Cases**: Telemetry streaming, agent metrics, visualization data
- **Advantages**:
  - Rich data structures
  - Real-time updates
  - Multiple concurrent clients
  - Bandwidth efficient

## Logging Architecture

### Persistent Log Locations

```
Host: ./logs/                           Container:
├── entrypoint.log                  →   /var/log/multi-agent/entrypoint.log
├── mcp/
│   ├── tcp-server.log              →   /app/mcp-logs/tcp-server.log
│   ├── tcp-server-error.log        →   /app/mcp-logs/tcp-server-error.log
│   ├── ws-bridge.log               →   /app/mcp-logs/ws-bridge.log
│   └── claude-flow-tcp.log         →   /app/mcp-logs/claude-flow-tcp.log
└── supervisor/
    └── supervisord.log             →   /var/log/supervisor/supervisord.log

Session Logs:
/var/log/multi-agent/hive-{UUID}.log
```

**Features**:
- Survive container restarts
- 50MB per file, 5 backups
- Accessible from host for debugging
- Separate error streams

## Security Model

### Authentication Layers
1. **Docker Socket**: Host-level access control
2. **TCP MCP**: Optional token authentication (`TCP_AUTH_TOKEN`)
3. **WebSocket**: Optional token authentication (`WS_AUTH_TOKEN`)
4. **Container Isolation**: User namespacing, limited capabilities

### Process Ownership
- Supervisor runs as root (required for VNC)
- MCP servers run as `dev` user (uid 1000)
- Session spawns run as `dev` user
- File permissions enforced via chown

## Resource Management

### Session Cleanup
Automatic removal of old sessions:
```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 24
```

Removes sessions older than 24 hours with status `completed` or `failed`.

### Database Vacuuming
Each isolated database can be maintained independently without affecting others.

### Log Rotation
Automatic rotation at 50MB per file, keeping 5 backups.

## Scalability Considerations

### Concurrent Sessions
- **Theoretical Limit**: File system inode count
- **Practical Limit**: Memory and CPU available
- **Tested**: 10+ concurrent sessions without issues

### MCP Client Connections
- **Max Connections**: 50 (configurable in mcp-tcp-server.js)
- **Connection Pooling**: Each client gets dedicated stream
- **Bandwidth**: ~100KB/s per active session telemetry

### Storage Growth
- Session directories: ~50-500MB each depending on task
- Databases: ~1-10MB each
- Logs: 50MB * 5 backups per log file
- Cleanup recommended: Weekly or after major task batches

## Failure Modes and Recovery

### Container Crash
- **Logs persist**: Check `./logs/entrypoint.log` for crash reason
- **Sessions survive**: Session registry in `/workspace/.swarm/sessions/index.json`
- **Recovery**: Container restart, sessions can be resumed

### Database Lock (Legacy, now prevented)
- **Cause**: Multiple processes sharing same DB file
- **Prevention**: Working directory isolation
- **Detection**: Check logs for `SQLITE_BUSY` errors

### Network Partition
- **Control Plane**: Docker exec continues to work (local socket)
- **Data Plane**: TCP/MCP connections fail gracefully
- **Recovery**: Automatic reconnection with exponential backoff

### Zombie Processes
- **Detection**: `ps aux | grep <defunct>`
- **Cleanup**: Session manager cleanup command
- **Prevention**: Proper signal handling in spawned processes

## Design Principles

1. **Separation of Concerns**: Control vs Data plane separation
2. **Single Source of Truth**: Session registry for all state
3. **Fault Isolation**: One session failure doesn't affect others
4. **Idempotency**: Operations safe to retry
5. **Observability**: Comprehensive logging and monitoring
6. **Resource Efficiency**: Cleanup and rotation automation
