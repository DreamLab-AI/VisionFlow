# Persistent Logging Setup

**Status**: Ready to test
**Date**: 2025-10-05

## What Changed

Added persistent logging that survives container restarts, so we can diagnose what causes the container to exit.

## Changes Made

### 1. docker-compose.yml
Added three log volume mounts:
```yaml
volumes:
  - ./logs:/var/log/multi-agent:rw
  - ./logs/mcp:/app/mcp-logs:rw
  - ./logs/supervisor:/var/log/supervisor:rw
```

### 2. supervisord.conf
Updated all log paths to use persistent files instead of stdout/stderr:

**Before:**
```ini
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
```

**After:**
```ini
stdout_logfile=/app/mcp-logs/tcp-server.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
stderr_logfile=/app/mcp-logs/tcp-server-error.log
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=5
```

Updated for:
- `mcp-tcp-server`
- `mcp-ws-bridge`
- `claude-flow-tcp`
- `supervisord` main log

### 3. entrypoint.sh
Added persistent logging to entrypoint:
```bash
ENTRYPOINT_LOG="/var/log/multi-agent/entrypoint.log"
exec 1> >(tee -a "$ENTRYPOINT_LOG")
exec 2>&1
```

## How to Use

### Start Container with Logging
```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Monitor Logs in Real-Time
```bash
# Watch all errors
tail -f logs/mcp/*-error.log logs/supervisor/*.log

# Watch TCP server
tail -f logs/mcp/tcp-server.log

# Watch container startup
tail -f logs/entrypoint.log

# Watch everything
tail -f logs/**/*.log
```

### Trigger the Container Exit Issue
```bash
# From host
docker exec -u dev multi-agent-container bash -c 'cd /workspace && claude-flow hive-mind spawn "test task" --claude'

# Monitor in another terminal
tail -f logs/mcp/tcp-server.log logs/entrypoint.log
```

### Check What Happened After Exit
```bash
# See if container restarted
grep "Starting entrypoint.sh" logs/entrypoint.log | tail -5

# Check for database errors
grep -r "SQLITE\|lock\|database" logs/

# Check for crashes
grep -i "exit\|crash\|fatal\|error" logs/entrypoint.log | tail -20

# Check MCP server errors
cat logs/mcp/tcp-server-error.log

# Check supervisor errors
cat logs/supervisor/supervisord.log | tail -50
```

## Expected Behavior

### Normal Startup
```
logs/entrypoint.log:
[2025-10-05 XX:XX:XX] Starting entrypoint.sh
[2025-10-05 XX:XX:XX] Starting supervisord in background...
[2025-10-05 XX:XX:XX] Supervisord started with PID: XXX

logs/supervisor/supervisord.log:
INFO spawned: 'mcp-tcp-server' with pid XXX
INFO success: mcp-tcp-server entered RUNNING state

logs/mcp/tcp-server.log:
[MCP-TCP] Server started on port 9500
[MCP-TCP] Database: /workspace/.swarm/tcp-server.db
```

### Container Exit (What We're Debugging)
The logs will now show:
1. What process crashed first
2. Database lock errors if any
3. Supervisor restart attempts
4. Exact error messages before exit

## Benefits

✅ **Logs survive restarts** - See what happened before crash
✅ **No more lost logs** - Everything persisted to host
✅ **Easy to grep** - All logs in one place
✅ **Timestamped** - Track sequence of events
✅ **Rotated** - Won't fill disk (50MB max, 5 backups)

## Next Steps

After rebuild:
1. Start container and verify logs are being written to `./logs/`
2. Trigger the hive-mind spawn that causes the exit
3. Check `./logs/` to see exactly what happened
4. Fix the root cause based on the evidence

## Troubleshooting

### Logs not appearing?
```bash
# Check permissions
ls -la logs/
# Should be writable

# Check mount
docker inspect multi-agent-container | grep -A 10 "Mounts"

# Check inside container
docker exec multi-agent-container ls -la /var/log/multi-agent /app/mcp-logs
```

### Too many logs?
```bash
# Clear old logs
find logs/ -name "*.log*" -mtime +7 -delete
```
