# Database Troubleshooting

## Understanding SQLite in Multi-Agent System

SQLite databases are used throughout the system for persistent storage. Understanding their behavior is critical for troubleshooting.

## ⚠️ Critical: Shared Database Problem

**The BIGGEST source of issues is the legacy shared database at `/workspace/.swarm/memory.db`**

This file should NOT exist. If it does, it causes:
- Container crashes and restarts
- SQLite BUSY errors
- Session failures
- Data corruption

**Check for it**:
```bash
docker exec multi-agent-container ls -lah /workspace/.swarm/memory.db
```

**If it exists, delete it immediately**:
```bash
docker exec multi-agent-container rm -f /workspace/.swarm/memory.db*
docker-compose restart
```

**How it gets created**:
- Running `claude-flow init --force` from `/workspace` directory ❌
- Using `claude-flow-init-agents` alias ❌
- Legacy code paths (should be fixed) ❌

**Prevention**: Never run init commands. The system uses automatic database isolation.

## Correct Database Locations

| Database | Path | Purpose | Isolation Level |
|----------|------|---------|-----------------|
| TCP Server DB | `/workspace/.swarm/tcp-server-instance/.swarm/memory.db` | MCP TCP server state | Process-isolated ✅ |
| Root CLI DB | `/workspace/.swarm/root-cli-instance/.swarm/memory.db` | CLI commands run as root | User-isolated ✅ |
| Session DBs | `/workspace/.swarm/sessions/{UUID}/.swarm/memory.db` | Per-session hive-mind state | Session-isolated ✅ |
| **LEGACY SHARED** | `/workspace/.swarm/memory.db` | **SHOULD NOT EXIST** | ❌ CONFLICTS ❌ |

## Common Database Issues

### SQLITE_BUSY Error

**Symptom**: Errors mentioning "database is locked" or "SQLITE_BUSY"

**Cause**: Multiple processes trying to write to the same database file

**Diagnosis**:
```bash
# Find all database files
docker exec multi-agent-container \
  find /workspace/.swarm -name "memory.db" -ls

# Check which processes are accessing each DB
docker exec multi-agent-container bash -c '
  for db in $(find /workspace/.swarm -name "memory.db"); do
    echo "=== $db ==="
    lsof "$db" 2>/dev/null || echo "  (no active locks)"
  done
'
```

**Solution**:

If you see multiple processes accessing the same DB:
```bash
# This should NOT happen with proper session isolation
# Kill the conflicting processes
docker exec multi-agent-container pkill -f "claude-flow"

# Verify isolation is working
# Should see different DB files for each session
docker exec multi-agent-container \
  find /workspace/.swarm/sessions -name "memory.db" -ls | wc -l
```

### Database Corruption

**Symptom**: "database disk image is malformed" or integrity check failures

**Diagnosis**:
```bash
# Check integrity of all databases
docker exec multi-agent-container bash -c '
  for db in $(find /workspace/.swarm -name "memory.db"); do
    echo "Checking: $db"
    sqlite3 "$db" "PRAGMA integrity_check;" || echo "  CORRUPTED!"
  done
'
```

**Recovery**:
```bash
# For session database
UUID=<affected-session-uuid>
docker exec multi-agent-container bash -c "
  cd /workspace/.swarm/sessions/$UUID/.swarm
  # Backup corrupted DB
  mv memory.db memory.db.corrupted
  # SQLite will create new DB on next access
"

# For TCP server database
docker exec multi-agent-container bash -c "
  supervisorctl stop mcp-tcp-server
  mv /workspace/.swarm/tcp-server-instance/.swarm/memory.db{,.corrupted}
  supervisorctl start mcp-tcp-server
"
```

### WAL (Write-Ahead Logging) Files Growing

**Symptom**: Large `-wal` and `-shm` files consuming disk space

**Cause**: Checkpoint not running, uncommitted transactions

**Diagnosis**:
```bash
# Check WAL sizes
docker exec multi-agent-container \
  find /workspace/.swarm -name "*.db-wal" -exec ls -lh {} \;
```

**Solution**:
```bash
# Force checkpoint on session database
UUID=<session-uuid>
docker exec multi-agent-container \
  sqlite3 /workspace/.swarm/sessions/$UUID/.swarm/memory.db "PRAGMA wal_checkpoint(TRUNCATE);"

# For TCP server database
docker exec multi-agent-container \
  sqlite3 /workspace/.swarm/tcp-server-instance/.swarm/memory.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

## Database Performance Issues

### Slow Queries

**Diagnosis**:
```bash
# Enable query logging (requires code modification)
# Check if indexes exist
docker exec multi-agent-container \
  sqlite3 /workspace/.swarm/sessions/$UUID/.swarm/memory.db ".schema"
```

**Optimization**:
```bash
# Vacuum database to reclaim space and optimize
docker exec multi-agent-container \
  sqlite3 /workspace/.swarm/sessions/$UUID/.swarm/memory.db "VACUUM;"

# Analyze to update query planner statistics
docker exec multi-agent-container \
  sqlite3 /workspace/.swarm/sessions/$UUID/.swarm/memory.db "ANALYZE;"
```

### High Disk Usage

**Diagnosis**:
```bash
# Find largest databases
docker exec multi-agent-container \
  find /workspace/.swarm -name "memory.db*" -exec du -h {} \; | sort -hr

# Check for old sessions
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list | \
  jq '.sessions | to_entries | map(select(.value.status == "completed" or .value.status == "failed"))'
```

**Solution**:
```bash
# Cleanup old sessions (removes databases)
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 24

# Manual cleanup of specific session
docker exec multi-agent-container \
  rm -rf /workspace/.swarm/sessions/$UUID
```

## Verifying Database Isolation

### Test Isolation

```bash
# Spawn 3 concurrent sessions
UUID1=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "test 1")
UUID2=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "test 2")
UUID3=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "test 3")

docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID1
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID2
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID3

# Wait a moment
sleep 5

# Check that each has its own database
docker exec multi-agent-container bash -c "
  echo 'Session 1 DB:'
  ls -lh /workspace/.swarm/sessions/$UUID1/.swarm/memory.db
  echo 'Session 2 DB:'
  ls -lh /workspace/.swarm/sessions/$UUID2/.swarm/memory.db
  echo 'Session 3 DB:'
  ls -lh /workspace/.swarm/sessions/$UUID3/.swarm/memory.db
"

# Check for lock errors in logs
grep -i "SQLITE_BUSY\|database.*lock" logs/mcp/*.log
# Should find NOTHING
```

### Verify No Shared Databases

```bash
# This should return EMPTY (no shared memory.db in /workspace/.swarm/)
docker exec multi-agent-container \
  ls -la /workspace/.swarm/memory.db 2>&1 | grep "No such file"

# TCP server should have its own
docker exec multi-agent-container \
  ls -la /workspace/.swarm/tcp-server-instance/.swarm/memory.db
```

## Database Backup and Restore

### Backup Session Database

```bash
UUID=<session-uuid>

# Backup while running (uses SQLite backup API)
docker exec multi-agent-container \
  sqlite3 /workspace/.swarm/sessions/$UUID/.swarm/memory.db \
  ".backup /workspace/ext/backups/session-$UUID-$(date +%Y%m%d).db"

# Or copy from host (if ext/ mounted)
cp workspace/.swarm/sessions/$UUID/.swarm/memory.db \
   backups/session-$UUID-$(date +%Y%m%d).db
```

### Restore Session Database

```bash
UUID=<session-uuid>
BACKUP=<backup-file>

# Stop session if running
# (kill the claude-flow process)

# Restore
docker exec multi-agent-container bash -c "
  cp $BACKUP /workspace/.swarm/sessions/$UUID/.swarm/memory.db
  chown dev:dev /workspace/.swarm/sessions/$UUID/.swarm/memory.db
"
```

## Database Migration

If you need to move sessions between containers:

```bash
# Export session
UUID=<session-uuid>
tar -czf session-$UUID.tar.gz \
  workspace/.swarm/sessions/$UUID \
  workspace/ext/hive-sessions/$UUID

# Import on new container
tar -xzf session-$UUID.tar.gz -C /path/to/new/workspace/

# Update session registry
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list
# Manually edit index.json if needed
```

## Monitoring Database Health

### Regular Health Check Script

```bash
#!/bin/bash
# Save as: scripts/check-db-health.sh

echo "=== Database Health Check ==="
echo "Time: $(date)"
echo ""

# Count databases
DB_COUNT=$(docker exec multi-agent-container \
  find /workspace/.swarm -name "memory.db" | wc -l)
echo "Total databases: $DB_COUNT"

# Check for locks
echo ""
echo "Checking for active locks..."
docker exec multi-agent-container bash -c '
  for db in $(find /workspace/.swarm -name "memory.db"); do
    LOCKS=$(lsof "$db" 2>/dev/null | wc -l)
    if [ $LOCKS -gt 1 ]; then
      echo "  $db: $LOCKS processes (WARNING if > 1)"
      lsof "$db"
    fi
  done
'

# Check for large WAL files
echo ""
echo "Large WAL files (>10MB):"
docker exec multi-agent-container \
  find /workspace/.swarm -name "*.db-wal" -size +10M -exec ls -lh {} \;

# Check integrity
echo ""
echo "Database integrity:"
docker exec multi-agent-container bash -c '
  ERROR_COUNT=0
  for db in $(find /workspace/.swarm -name "memory.db"); do
    if ! sqlite3 "$db" "PRAGMA integrity_check;" > /dev/null 2>&1; then
      echo "  CORRUPT: $db"
      ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
  done
  if [ $ERROR_COUNT -eq 0 ]; then
    echo "  All databases OK"
  else
    echo "  $ERROR_COUNT corrupted databases found"
  fi
'

echo ""
echo "=== Health check complete ==="
```

Run weekly:
```bash
chmod +x scripts/check-db-health.sh
./scripts/check-db-health.sh
```

## Emergency Recovery

If the system is completely broken due to database issues:

```bash
# 1. Stop all processes
docker exec multi-agent-container supervisorctl stop all

# 2. Backup everything
tar -czf workspace-backup-$(date +%Y%m%d).tar.gz workspace/

# 3. Remove all databases
docker exec multi-agent-container bash -c '
  find /workspace/.swarm -name "memory.db*" -delete
'

# 4. Restart services (will create fresh databases)
docker exec multi-agent-container supervisorctl start all

# 5. Sessions are lost, but session metadata might be recoverable
docker exec multi-agent-container \
  cat /workspace/.swarm/sessions/index.json
```
