# Session Isolation System

## Overview

The session isolation system provides **UUID-tracked, database-isolated execution environments** for concurrent hive-mind tasks.

## Problem Statement

### Before Session Isolation

```
Spawn 1 (docker exec) → /workspace/ → .swarm/memory.db ←┐
Spawn 2 (docker exec) → /workspace/ → .swarm/memory.db ←┼─ CONFLICT!
Spawn 3 (docker exec) → /workspace/ → .swarm/memory.db ←┘

Result: SQLite BUSY error → Process crash → Container exit
```

**Issues**:
- All claude-flow spawns shared `/workspace/.swarm/memory.db`
- SQLite doesn't support concurrent writers
- Lock conflicts caused container crashes
- Lost work when container restarted

### After Session Isolation

```
Spawn 1 → /workspace/.swarm/sessions/{uuid-1}/ → uuid-1/.swarm/memory.db
Spawn 2 → /workspace/.swarm/sessions/{uuid-2}/ → uuid-2/.swarm/memory.db
Spawn 3 → /workspace/.swarm/sessions/{uuid-3}/ → uuid-3/.swarm/memory.db

Result: Different files → No conflicts → Container stable
```

**Benefits**:
- Each session has isolated database
- No SQLite lock conflicts
- Unlimited concurrent spawns
- Container stays alive
- Session state persists across container restarts

## Session Lifecycle

### States

```
created → starting → running → completed
                              ↘ failed
```

| State | Description | Can Transition To |
|-------|-------------|-------------------|
| `created` | Session initialized, not started | `starting` |
| `starting` | Hive-mind spawn beginning | `running`, `failed` |
| `running` | Task actively executing | `completed`, `failed` |
| `completed` | Task finished successfully | (terminal) |
| `failed` | Task encountered error | (terminal) |

### Operations

#### Create Session
```bash
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create \
  "task description" \
  "priority" \
  '{"metadata":"json"}')
```

**What Happens**:
1. Generate UUIDv4
2. Create directory: `/workspace/.swarm/sessions/{UUID}/`
3. Create output directory: `/workspace/ext/hive-sessions/{UUID}/`
4. Write `session.json` metadata
5. Update session registry: `/workspace/.swarm/sessions/index.json`
6. Return UUID to caller

#### Start Session
```bash
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start {UUID}
```

**What Happens**:
1. Change to session directory: `cd /workspace/.swarm/sessions/{UUID}/`
2. Update status: `created` → `starting`
3. Spawn claude-flow as `dev` user in session directory
4. claude-flow creates: `.swarm/memory.db` (isolated!)
5. Update status: `starting` → `running`
6. On completion: `running` → `completed` or `failed`

#### Query Status
```bash
STATUS=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status {UUID})
```

Returns: `created` | `starting` | `running` | `completed` | `failed` | `not_found`

#### Get Metadata
```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh get {UUID}
```

Returns JSON:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "Build Rust CLI tool",
  "priority": "high",
  "created": "2025-10-05T19:45:00+00:00",
  "updated": "2025-10-05T19:46:30+00:00",
  "status": "running",
  "working_dir": "/workspace/.swarm/sessions/550e8400-...",
  "output_dir": "/workspace/ext/hive-sessions/550e8400-...",
  "database": "/workspace/.swarm/sessions/550e8400-.../.swarm/memory.db",
  "log_file": "/var/log/multi-agent/hive-550e8400-....log",
  "metadata": {"custom": "data"}
}
```

## Directory Structure

### Session Directory
```
/workspace/.swarm/sessions/{UUID}/
├── session.json           # Session metadata
├── .swarm/
│   ├── memory.db          # Isolated SQLite database
│   ├── memory.db-shm      # SQLite shared memory
│   └── memory.db-wal      # SQLite write-ahead log
├── .hive-mind/            # Hive-mind working files
│   ├── sessions/
│   └── memory.db          # Hive-mind's own DB (if created)
└── output/                # Symlink → /workspace/ext/hive-sessions/{UUID}/
```

### Output Directory
```
/workspace/ext/hive-sessions/{UUID}/
└── (task artifacts, generated files, etc.)
```

**Mounting**: The `ext/` directory is mounted to the host, making session outputs accessible outside the container.

## Session Registry

**Location**: `/workspace/.swarm/sessions/index.json`

**Purpose**: Fast lookup of all sessions without filesystem traversal

**Format**:
```json
{
  "sessions": {
    "uuid-1": {
      "task": "Task description",
      "status": "completed",
      "created": "2025-10-05T10:00:00Z",
      "updated": "2025-10-05T10:05:00Z",
      "dir": "/workspace/.swarm/sessions/uuid-1"
    },
    "uuid-2": {
      "task": "Another task",
      "status": "running",
      "created": "2025-10-05T11:00:00Z",
      "updated": "2025-10-05T11:02:00Z",
      "dir": "/workspace/.swarm/sessions/uuid-2"
    }
  },
  "created": "2025-10-05T09:00:00Z"
}
```

**Concurrency**: File locking via `flock` prevents race conditions during updates.

## Database Isolation Details

### Why Working Directory Matters

claude-flow creates databases relative to its **current working directory**:

```javascript
// Inside claude-flow/src/memory/shared-memory.js
constructor(options = {}) {
  this.options = {
    directory: options.directory || '.hive-mind',  // Relative to cwd!
    filename: options.filename || 'memory.db',
    // ...
  };
}
```

Database created at: `<cwd>/.swarm/memory.db`

### Isolation Mechanism

```bash
# Session manager changes working directory before spawn
cd /workspace/.swarm/sessions/{UUID}/
su -s /bin/bash dev -c "
  cd '/workspace/.swarm/sessions/{UUID}'  # Ensure correct cwd
  /app/node_modules/.bin/claude-flow hive-mind spawn ...
"
```

**Result**: claude-flow creates database at:
`/workspace/.swarm/sessions/{UUID}/.swarm/memory.db`

### Verification

```bash
# List all database files
docker exec multi-agent-container \
  find /workspace/.swarm -name "memory.db" -ls

# Should see multiple files:
# ./sessions/uuid-1/.swarm/memory.db
# ./sessions/uuid-2/.swarm/memory.db
# ./tcp-server-instance/.swarm/memory.db
```

## Session Cleanup

### Automatic Cleanup

```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup [hours]
```

**Default**: Removes sessions older than 24 hours

**Rules**:
- Only removes `completed` or `failed` sessions
- Keeps `running` sessions indefinitely
- Archives session directory
- Preserves output in `/workspace/ext/hive-sessions/{UUID}/`
- Updates session registry

### Manual Cleanup

```bash
# Remove specific session
docker exec multi-agent-container \
  rm -rf /workspace/.swarm/sessions/{UUID}

# Update registry
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list
```

## Logging Per Session

Each session gets its own log file:
**Location**: `/var/log/multi-agent/hive-{UUID}.log`

**Contents**:
- Session creation timestamp
- Working directory paths
- Hive-mind spawn command
- All stdout/stderr from claude-flow
- Status transitions
- Completion/failure details

**Access**:
```bash
# Get log path
LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log {UUID})

# Tail logs
docker exec multi-agent-container tail -f "$LOG"

# Search for errors
docker exec multi-agent-container grep -i error "$LOG"
```

## Concurrency Safety

### File Locking

```bash
# Session manager uses flock for registry updates
(
  flock -x 200
  jq '.sessions["'$UUID'"] = {...}' index.json > temp.json
  mv temp.json index.json
) 200>/workspace/.swarm/sessions/.lock
```

**Prevents**:
- Race conditions when multiple spawns create sessions
- Corrupted registry from concurrent writes
- Lost session metadata

### Database Locking (Prevented)

Each session's database is only accessed by one claude-flow process, so SQLite's single-writer limitation is not violated.

## Session Recovery After Container Restart

### What Persists
- ✅ Session directory: `/workspace/.swarm/sessions/{UUID}/`
- ✅ Session registry: `index.json`
- ✅ Session metadata: `session.json`
- ✅ Database files: `memory.db`
- ✅ Output files: `/workspace/ext/hive-sessions/{UUID}/`
- ✅ Logs: `/var/log/multi-agent/hive-{UUID}.log`

### What Doesn't Persist
- ❌ Running processes (claude-flow spawns)
- ❌ In-memory state

### Recovery Strategy

```bash
# After container restart, list sessions
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list

# Check status
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status {UUID}

# If status was "running", it's now stale
# Either:
# 1. Resume if possible (not yet implemented)
# 2. Mark as failed
# 3. Create new session with same task
```

## Best Practices

### Session Naming
Use descriptive task descriptions that include context:
```bash
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create \
  "project-X: Build authentication module" \
  "high" \
  '{"project":"project-X","module":"auth","requester":"voice-api"}')
```

### Metadata Usage
Store relevant context in metadata JSON for later queries:
```json
{
  "project": "rust-cli",
  "component": "file-parser",
  "requester": "voice-system",
  "deadline": "2025-10-06T00:00:00Z",
  "priority_reason": "production-blocker"
}
```

### Status Polling
Use exponential backoff for status checks:
```bash
DELAY=1
while true; do
  STATUS=$(docker exec multi-agent-container \
    /app/scripts/hive-session-manager.sh status $UUID)

  [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]] && break

  sleep $DELAY
  DELAY=$((DELAY * 2))
  [ $DELAY -gt 60 ] && DELAY=60
done
```

### Cleanup Strategy
- Run cleanup daily or after major workloads
- Adjust retention based on available storage
- Keep failed sessions longer for debugging (e.g., 48 hours)

## Troubleshooting

### Session Not Found
```bash
# Check if UUID exists in registry
docker exec multi-agent-container \
  jq '.sessions["'$UUID'"]' /workspace/.swarm/sessions/index.json

# Check if directory exists
docker exec multi-agent-container \
  ls -la /workspace/.swarm/sessions/$UUID
```

### Session Stuck in "starting"
```bash
# Check logs for errors
LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
docker exec multi-agent-container tail -50 "$LOG"

# Check if process is still running
docker exec multi-agent-container \
  ps aux | grep $UUID
```

### Database Still Locked
```bash
# Find which processes are using the database
docker exec multi-agent-container \
  lsof /workspace/.swarm/sessions/$UUID/.swarm/memory.db

# If process is stuck, kill it
docker exec multi-agent-container \
  kill -9 <PID>
```

### Output Directory Empty
```bash
# Check if symlink is correct
docker exec multi-agent-container \
  ls -la /workspace/.swarm/sessions/$UUID/output

# Check actual output location
docker exec multi-agent-container \
  ls -la /workspace/ext/hive-sessions/$UUID/
```
