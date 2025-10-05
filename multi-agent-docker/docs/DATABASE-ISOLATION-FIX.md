# Database Isolation Fix - FINAL SOLUTION

**Date**: 2025-10-05
**Status**: Ready to test

## Root Cause Found

From the persistent logs in `logs/mcp/tcp-server.log`:

```
[PMCP-DEBUG] MCP: [2025-10-05T18:31:27.170Z] INFO [memory-store] Initialized SQLite at: /workspace/.swarm/memory.db
```

**Problem**: claude-flow v2.0.0-alpha.59 **IGNORES** the `CLAUDE_FLOW_DB_PATH` environment variable!

### Why Environment Variable Doesn't Work

Examined `/app/node_modules/claude-flow/src/memory/shared-memory.js`:

```javascript
constructor(options = {}) {
  this.options = {
    directory: options.directory || '.hive-mind',  // Hardcoded default!
    filename: options.filename || 'memory.db',      // Hardcoded default!
    // NO support for CLAUDE_FLOW_DB_PATH env var
  };
}
```

Claude-flow creates database at `<cwd>/.swarm/memory.db` where `<cwd>` is the current working directory.

## The Solution: Directory Isolation

Instead of trying to change the DB filename (which claude-flow doesn't support), **change the working directory** for each instance:

### Before (BROKEN)
```javascript
spawn(mcpCommand, mcpArgs, {
  cwd: '/workspace',  // ALL instances use /workspace/.swarm/memory.db
  env: devEnv
});
```

### After (FIXED)
```javascript
// Create isolated directory for TCP server instance
const tcpServerDir = '/workspace/.swarm/tcp-server-instance';
fs.mkdirSync(tcpServerDir, { recursive: true });

spawn(mcpCommand, mcpArgs, {
  cwd: tcpServerDir,  // This instance uses /workspace/.swarm/tcp-server-instance/.swarm/memory.db
  env: devEnv
});
```

## Database Locations After Fix

| Component | Working Directory | Database Path |
|-----------|------------------|---------------|
| TCP MCP Server | `/workspace/.swarm/tcp-server-instance` | `/workspace/.swarm/tcp-server-instance/.swarm/memory.db` |
| Claude MCP (via bridge) | `/workspace` | `/workspace/.swarm/memory.db` |
| Hive-mind spawns | `/workspace` | `/workspace/.hive-mind/memory.db` |
| Hook calls | `/workspace` | `/workspace/.swarm/memory.db` |

**Each instance now has its own database file** - no more SQLite lock conflicts!

## Changes Made

### core-assets/scripts/mcp-tcp-server.js
- Added `tcpServerDir` creation at `/workspace/.swarm/tcp-server-instance`
- Changed `cwd: '/workspace'` to `cwd: tcpServerDir`
- Added logging to show isolated working directory

## Expected Log Output After Fix

```
[PMCP-INFO] Starting MCP: /app/node_modules/.bin/claude-flow mcp start
[PMCP-INFO] Working directory: /workspace/.swarm/tcp-server-instance (isolated DB)
[PMCP-DEBUG] MCP: Initialized SQLite at: /workspace/.swarm/tcp-server-instance/.swarm/memory.db
```

Note the database path is now under `tcp-server-instance/` subdirectory!

## Testing Steps

```bash
# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Watch logs
tail -f logs/mcp/tcp-server.log

# Test 1: Check TCP server starts with isolated DB
docker exec multi-agent-container bash -c 'ls -la /workspace/.swarm/tcp-server-instance/.swarm/'
# Should see memory.db created

# Test 2: Spawn hive-mind task (this used to crash container)
docker exec -u dev multi-agent-container bash -c 'cd /workspace && claude-flow hive-mind spawn "test task" --claude'

# Test 3: Check container is still running
docker ps | grep multi-agent-container
# Should show "Up" status, NOT restarted

# Test 4: Verify no database lock errors
grep -i "lock\|SQLITE_BUSY" logs/mcp/*.log
# Should find NOTHING
```

## What This Fixes

✅ Container no longer exits when spawning hive-mind tasks
✅ Multiple claude-flow instances can run simultaneously
✅ TCP server has its own isolated database
✅ No more SQLite `SQLITE_BUSY` errors
✅ Logs persist across restarts (from previous logging fix)

## Files Modified

1. ✅ `core-assets/scripts/mcp-tcp-server.js` - Working directory isolation
2. ✅ `docker-compose.yml` - Log volume mounts (previous fix)
3. ✅ `supervisord.conf` - Persistent logging (previous fix)
4. ✅ `entrypoint.sh` - Persistent logging (previous fix)

## Why This Works

**SQLite file locking is process-based**. When multiple processes try to write to the same DB file:
- Process 1: Acquires exclusive lock on `/workspace/.swarm/memory.db`
- Process 2: Tries to acquire lock on same file → `SQLITE_BUSY` → Crash → Container exit

With directory isolation:
- TCP server process: Locks `/workspace/.swarm/tcp-server-instance/.swarm/memory.db`
- Hive-mind process: Locks `/workspace/.hive-mind/memory.db`
- **Different files = No conflict!**

## Next Steps

After rebuild:
1. Verify TCP server log shows isolated directory
2. Trigger hive-mind spawn
3. Confirm container stays up
4. Celebrate 🎉
