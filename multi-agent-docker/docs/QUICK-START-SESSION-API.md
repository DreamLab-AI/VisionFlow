# Quick Start: Session-Based Hive-Mind API

## One-Line Summary
**Each external spawn gets a UUID and isolated database to prevent SQLite conflicts.**

## Basic Usage

### Create + Start Session
```bash
# Create and get UUID
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "your task here" "high")

# Start in background
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID

# Monitor
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID
```

### Get Results
```bash
# Get output directory
OUTPUT=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh output-dir $UUID)

# Access from host (if ext/ mounted to ./workspace/)
ls -la "./workspace/ext/hive-sessions/$UUID/"
```

### View Logs
```bash
# Get log path
LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)

# Tail logs
docker exec multi-agent-container tail -f "$LOG"
```

## Quick Reference

| Command | Returns |
|---------|---------|
| `create <task> [priority] [metadata]` | UUID |
| `start <uuid>` | Starts spawn (use with `-d`) |
| `status <uuid>` | `created\|starting\|running\|completed\|failed` |
| `get <uuid>` | Full session JSON |
| `list` | All sessions JSON |
| `output-dir <uuid>` | Path to output directory |
| `log <uuid>` | Path to log file |
| `cleanup [hours]` | Removes old sessions |

## Directory Map

```
Container:
/workspace/.swarm/sessions/<UUID>/
  ├── .swarm/memory.db          # Isolated database
  ├── session.json              # Metadata
  └── output/ → /workspace/ext/hive-sessions/<UUID>/

Host (if mounted):
./workspace/ext/hive-sessions/<UUID>/
  └── (task output files)
```

## Status Flow

```
created → starting → running → completed
                              ↘ failed
```

## Rust Quick Integration

```rust
// Create session
let output = Command::new("docker")
    .args(&["exec", "multi-agent-container",
           "/app/scripts/hive-session-manager.sh",
           "create", "task description", "high"])
    .output().await?;
let uuid = String::from_utf8_lossy(&output.stdout).trim();

// Start
Command::new("docker")
    .args(&["exec", "-d", "multi-agent-container",
           "/app/scripts/hive-session-manager.sh", "start", uuid])
    .spawn()?;

// Check status
let output = Command::new("docker")
    .args(&["exec", "multi-agent-container",
           "/app/scripts/hive-session-manager.sh", "status", uuid])
    .output().await?;
let status = String::from_utf8_lossy(&output.stdout).trim();
```

## Why This Fixes Container Crashes

**Before**: All spawns → `/workspace/.swarm/memory.db` → SQLite lock → crash
**After**: Each spawn → `/workspace/.swarm/sessions/{UUID}/.swarm/memory.db` → No conflict!

## Rebuild Container

```bash
cd multi-agent-docker
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Test It Works

```bash
# Spawn 3 concurrent tasks
for i in 1 2 3; do
  UUID=$(docker exec multi-agent-container \
    /app/scripts/hive-session-manager.sh create "test $i")
  docker exec -d multi-agent-container \
    /app/scripts/hive-session-manager.sh start $UUID
done

# Container should stay UP (not crash)
docker ps | grep multi-agent-container

# No lock errors
grep -i "lock\|SQLITE" logs/mcp/*.log
```

## Full Documentation

- **API Reference**: `SESSION-API.md`
- **Complete Implementation**: `SESSION-ISOLATION-COMPLETE.md`
- **Database Fix Details**: `DATABASE-ISOLATION-FIX.md`
- **Logging Setup**: `LOGGING-SETUP.md`
