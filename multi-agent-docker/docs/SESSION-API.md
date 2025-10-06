# Hive-Mind Session API for External Systems

**Purpose**: Session-based isolation system for hive-mind spawns with UUID tracking.

## Problem Solved

Previously, multiple `docker exec` spawns would all run in `/workspace`, sharing `/workspace/.swarm/memory.db`:
- ❌ SQLite lock conflicts
- ❌ Container crashes
- ❌ Lost work

Now, each spawn gets its own **isolated session directory** with unique UUID:
- ✅ Separate database: `/workspace/.swarm/sessions/{UUID}/.swarm/memory.db`
- ✅ No SQLite conflicts
- ✅ Output mapped to: `/workspace/ext/hive-sessions/{UUID}/`
- ✅ Container stays alive

## Architecture

```
External Rust System
    ↓ (docker exec)
Container: hive-session-manager.sh
    ↓ creates UUID session
/workspace/.swarm/sessions/{UUID}/
    ├── session.json (metadata)
    ├── .swarm/memory.db (isolated DB)
    └── output/ → /workspace/ext/hive-sessions/{UUID}/
                      ↑
                      External access point
```

## API Reference

### 1. Create Session

Creates a new isolated session with UUID.

```bash
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create \
  "task description" \
  "priority" \
  '{"key":"value"}')
```

**Parameters:**
- `task`: Task description (required)
- `priority`: `high` | `medium` | `low` (optional, default: `medium`)
- `metadata`: JSON object (optional)

**Returns:** UUID string on stdout, session metadata JSON on stderr

**Example:**
```bash
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create \
  "Build a Rust CLI tool for file processing" \
  "high" \
  '{"project":"rust-cli","requester":"voice-system"}')

# UUID: 550e8400-e29b-41d4-a716-446655440000
```

### 2. Start Session

Spawns hive-mind in the session's isolated directory.

```bash
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID [args...]
```

**Parameters:**
- `UUID`: Session UUID (required)
- `args`: Additional hive-mind spawn arguments (optional)

**Runs in background** (`-d` flag recommended)

**Example:**
```bash
# Start in background
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID

# Or with custom arguments
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID \
  --queen-type tactical \
  --max-workers 8
```

### 3. Get Status

Check current status of a session.

```bash
STATUS=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID)
```

**Returns:** One of:
- `created` - Session initialized, not yet started
- `starting` - Hive-mind spawn beginning
- `running` - Actively executing
- `completed` - Finished successfully
- `failed` - Encountered error
- `not_found` - UUID doesn't exist

**Example:**
```bash
STATUS=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID)

if [ "$STATUS" = "completed" ]; then
  echo "Task finished!"
fi
```

### 4. Get Session Metadata

Retrieve full session information.

```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh get $UUID
```

**Returns:** JSON object:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "Build a Rust CLI tool",
  "priority": "high",
  "created": "2025-10-05T19:45:00+00:00",
  "updated": "2025-10-05T19:46:30+00:00",
  "status": "running",
  "working_dir": "/workspace/.swarm/sessions/550e8400-...",
  "output_dir": "/workspace/ext/hive-sessions/550e8400-...",
  "database": "/workspace/.swarm/sessions/550e8400-.../.swarm/memory.db",
  "log_file": "/var/log/multi-agent/hive-550e8400-....log",
  "metadata": {"project": "rust-cli", "requester": "voice-system"}
}
```

### 5. List All Sessions

Get index of all sessions.

```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list
```

**Returns:**
```json
{
  "sessions": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "task": "Build Rust CLI",
      "status": "running",
      "created": "2025-10-05T19:45:00+00:00",
      "updated": "2025-10-05T19:46:30+00:00",
      "dir": "/workspace/.swarm/sessions/550e8400-..."
    },
    "660f9511-f3ac-52e5-b827-557766551111": {
      "task": "Create Python script",
      "status": "completed",
      "created": "2025-10-05T18:30:00+00:00",
      "updated": "2025-10-05T18:35:00+00:00",
      "dir": "/workspace/.swarm/sessions/660f9511-..."
    }
  },
  "created": "2025-10-05T18:00:00+00:00"
}
```

### 6. Get Output Directory

Get path to session's output directory.

```bash
OUTPUT_DIR=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh output-dir $UUID)
```

**Returns:** Path string: `/workspace/ext/hive-sessions/{UUID}/`

**Example:**
```bash
OUTPUT_DIR=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh output-dir $UUID)

# List output files
docker exec multi-agent-container ls -la "$OUTPUT_DIR"

# Copy output to host (if ext/ is mounted)
cp -r "./workspace/ext/hive-sessions/$UUID/" ./results/
```

### 7. Get Log File

Get path to session's log file.

```bash
LOG_FILE=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
```

**Returns:** Path string: `/var/log/multi-agent/hive-{UUID}.log`

**Example:**
```bash
LOG_FILE=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)

# Tail logs
docker exec multi-agent-container tail -f "$LOG_FILE"

# Get recent errors
docker exec multi-agent-container grep -i error "$LOG_FILE"
```

### 8. Cleanup Old Sessions

Remove completed/failed sessions older than specified hours.

```bash
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup [max_age_hours]
```

**Parameters:**
- `max_age_hours`: Age threshold in hours (default: 24)

**Example:**
```bash
# Clean up sessions older than 48 hours
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 48
```

## Complete Workflow Example

```bash
#!/bin/bash
# Complete session lifecycle

# 1. Create session
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create \
  "Build web scraper in Python" \
  "high" \
  '{"requester":"api","timeout":3600}')

echo "Created session: $UUID"

# 2. Start hive-mind spawn (background)
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID

# 3. Monitor status
while true; do
  STATUS=$(docker exec multi-agent-container \
    /app/scripts/hive-session-manager.sh status $UUID)

  echo "Status: $STATUS"

  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
    break
  fi

  sleep 5
done

# 4. Get results
if [ "$STATUS" = "completed" ]; then
  OUTPUT_DIR=$(docker exec multi-agent-container \
    /app/scripts/hive-session-manager.sh output-dir $UUID)

  echo "Task completed! Output at: $OUTPUT_DIR"

  # Copy results from mounted ext/ directory
  cp -r "./workspace/ext/hive-sessions/$UUID/" ./results/

else
  # Get error logs
  LOG_FILE=$(docker exec multi-agent-container \
    /app/scripts/hive-session-manager.sh log $UUID)

  echo "Task failed. Logs:"
  docker exec multi-agent-container tail -50 "$LOG_FILE"
fi
```

## Database Isolation Details

### Before (BROKEN)
```
All spawns → /workspace/.swarm/memory.db
    ↓
SQLite lock conflict
    ↓
Container crashes
```

### After (FIXED)
```
Spawn 1 → /workspace/.swarm/sessions/uuid-1/.swarm/memory.db
Spawn 2 → /workspace/.swarm/sessions/uuid-2/.swarm/memory.db
Spawn 3 → /workspace/.swarm/sessions/uuid-3/.swarm/memory.db
TCP Server → /workspace/.swarm/tcp-server-instance/.swarm/memory.db

Different files = No conflicts!
```

## Integration with Rust docker_hive_mind Module

### Suggested Rust Implementation

```rust
pub struct DockerHiveMind {
    container_name: String,
    session_manager: String, // Path to script in container
}

impl DockerHiveMind {
    pub async fn spawn_task(
        &self,
        task: &str,
        priority: Option<&str>,
        metadata: Option<Value>,
    ) -> Result<SessionHandle, DockerHiveMindError> {
        // 1. Create session
        let metadata_json = metadata
            .map(|m| serde_json::to_string(&m).unwrap())
            .unwrap_or_else(|| "null".to_string());

        let output = Command::new("docker")
            .args(&[
                "exec",
                &self.container_name,
                &self.session_manager,
                "create",
                task,
                priority.unwrap_or("medium"),
                &metadata_json,
            ])
            .output()
            .await?;

        let uuid = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // 2. Start session (background)
        Command::new("docker")
            .args(&[
                "exec", "-d",
                &self.container_name,
                &self.session_manager,
                "start",
                &uuid,
            ])
            .spawn()?;

        Ok(SessionHandle {
            uuid,
            container: self.container_name.clone(),
            session_manager: self.session_manager.clone(),
        })
    }
}

pub struct SessionHandle {
    uuid: String,
    container: String,
    session_manager: String,
}

impl SessionHandle {
    pub async fn get_status(&self) -> Result<SessionStatus, DockerHiveMindError> {
        let output = Command::new("docker")
            .args(&[
                "exec",
                &self.container,
                &self.session_manager,
                "status",
                &self.uuid,
            ])
            .output()
            .await?;

        let status_str = String::from_utf8_lossy(&output.stdout).trim();
        Ok(SessionStatus::from_str(status_str)?)
    }

    pub async fn get_output_dir(&self) -> Result<PathBuf, DockerHiveMindError> {
        let output = Command::new("docker")
            .args(&[
                "exec",
                &self.container,
                &self.session_manager,
                "output-dir",
                &self.uuid,
            ])
            .output()
            .await?;

        Ok(PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
    }

    pub async fn tail_logs(&self) -> Result<mpsc::Receiver<String>, DockerHiveMindError> {
        // Implementation for streaming logs
        // ...
    }
}
```

## File Locations

| Path | Description |
|------|-------------|
| `/app/scripts/hive-session-manager.sh` | Session manager script |
| `/workspace/.swarm/sessions/` | All session directories |
| `/workspace/.swarm/sessions/index.json` | Session registry |
| `/workspace/ext/hive-sessions/{UUID}/` | Output directories (mounted to host) |
| `/var/log/multi-agent/hive-{UUID}.log` | Session log files |

## Session Lifecycle States

```
created → starting → running → completed
                              ↘ failed
```

## Troubleshooting

### Session not found
```bash
# Check if session exists in index
docker exec multi-agent-container \
  jq '.sessions["'$UUID'"]' /workspace/.swarm/sessions/index.json
```

### Database still locked
```bash
# Check which sessions are using databases
docker exec multi-agent-container \
  find /workspace/.swarm/sessions -name "memory.db" -exec lsof {} \;
```

### Logs not appearing
```bash
# Check log file permissions
LOG_FILE=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
docker exec multi-agent-container ls -la "$LOG_FILE"
```

## Next Steps for External Integration

1. **Update Rust `docker_hive_mind.rs`**: Use session manager instead of direct hive-mind spawns
2. **Modify `call_task_orchestrate`**: Return session UUID instead of direct output
3. **Add session monitoring**: Poll status until completion
4. **Output retrieval**: Read from `/workspace/ext/hive-sessions/{UUID}/`
5. **Telemetry integration**: Query MCP using session UUID for metrics
