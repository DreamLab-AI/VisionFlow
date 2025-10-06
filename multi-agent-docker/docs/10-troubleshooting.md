# Troubleshooting Guide

## Common Issues

### Container Exits/Restarts Repeatedly

**Symptoms**: Container restarts every few minutes, loses active sessions

**Primary Cause**: Shared database `/workspace/.swarm/memory.db` causing SQLite lock conflicts

**Diagnosis**:
```bash
# Check if legacy shared database exists
docker exec multi-agent-container ls -lah /workspace/.swarm/memory.db

# If file exists → THIS IS THE PROBLEM
```

**Solution**:
```bash
# Remove the shared database
docker exec multi-agent-container rm -f /workspace/.swarm/memory.db*

# Verify proper isolation
docker exec multi-agent-container \
  find /workspace/.swarm -name "memory.db" -ls

# Should show ONLY isolated databases:
#   /workspace/.swarm/tcp-server-instance/.swarm/memory.db
#   /workspace/.swarm/sessions/{UUID}/.swarm/memory.db
#   /workspace/.swarm/root-cli-instance/.swarm/memory.db
# Should NOT show: /workspace/.swarm/memory.db

# Restart container
docker-compose restart
```

**Prevention**:
- NEVER run `claude-flow init --force` from `/workspace`
- Don't use `claude-flow-init-agents` alias (it creates shared DB)
- Use session manager API for all task spawns
- The wrapper handles isolation automatically for CLI commands

### Session Stuck in "starting"

**Symptoms**: Session status never progresses past "starting"

**Diagnosis**:
```bash
# Check session log
LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
docker exec multi-agent-container tail -50 "$LOG"

# Check if process is running
docker exec multi-agent-container ps aux | grep $UUID
```

**Solutions**:
```bash
# If no errors in log, may need to wait longer (hive-mind startup is slow)

# If process died, check for OOM
docker stats multi-agent-container

# Restart session (not yet implemented, create new one)
```

### MCP Server Not Responding

**Symptoms**: TCP connections to port 9500 fail

**Diagnosis**:
```bash
# Check if process is running
docker exec multi-agent-container \
  supervisorctl status mcp-tcp-server

# Check if port is listening
docker exec multi-agent-container ss -tlnp | grep 9500

# Check logs
tail -50 logs/mcp/tcp-server-error.log
```

**Solutions**:
```bash
# Restart MCP server
docker exec multi-agent-container \
  supervisorctl restart mcp-tcp-server

# If still failing, rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Output Directory Empty

**Symptoms**: `/workspace/ext/hive-sessions/{UUID}/` is empty after task completion

**Possible Causes**:
1. Task hasn't generated output yet
2. Task failed before generating output
3. Output written to wrong directory

**Diagnosis**:
```bash
# Check session status
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID

# Check logs for file creation
LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
docker exec multi-agent-container grep -i "write\|save\|output" "$LOG"

# Check working directory
docker exec multi-agent-container \
  ls -la /workspace/.swarm/sessions/$UUID/
```

### High Memory Usage

**Symptoms**: Container using excessive RAM

**Diagnosis**:
```bash
# Check container stats
docker stats multi-agent-container

# Check process list
docker exec multi-agent-container ps aux --sort=-%mem | head -20

# Count active sessions
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list | jq '.sessions | length'
```

**Solutions**:
```bash
# Cleanup old sessions
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 12

# Kill stuck processes
docker exec multi-agent-container pkill -f "claude-flow.*defunct"

# Restart container
docker-compose restart
```

### VNC Desktop Frozen

**Symptoms**: VNC connection works but desktop doesn't respond

**Solutions**:
```bash
# Restart XFCE
docker exec multi-agent-container supervisorctl restart xfce

# Kill screensaver
docker exec multi-agent-container \
  bash -c "DISPLAY=:1 killall xfce4-screensaver xfce4-power-manager"

# Reset session
docker exec multi-agent-container bash -c "
  rm -rf /home/dev/.cache/sessions
  rm -rf /home/dev/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-session.xml
"
docker exec multi-agent-container supervisorctl restart xfce
```

### Logs Not Appearing

**Symptoms**: Log files empty or not created

**Diagnosis**:
```bash
# Check log directory permissions
docker exec multi-agent-container ls -la /var/log/multi-agent/
docker exec multi-agent-container ls -la /app/mcp-logs/

# Check if supervisord is writing logs
docker exec multi-agent-container supervisorctl tail mcp-tcp-server
```

**Solutions**:
```bash
# Fix permissions
docker exec multi-agent-container \
  chown -R dev:dev /var/log/multi-agent /app/mcp-logs

# Restart supervisord
docker exec multi-agent-container \
  supervisorctl restart all
```

## Debugging Workflows

### Debug Container Startup

```bash
# Watch container startup in real-time
tail -f logs/entrypoint.log

# Check supervisor initialization
tail -f logs/supervisor/supervisord.log

# Verify all services started
docker exec multi-agent-container supervisorctl status
```

### Debug Session Execution

```bash
# Create session with verbose metadata
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create \
  "debug test" "high" '{"debug":true}')

# Start and immediately tail log
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID

LOG=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh log $UUID)
docker exec multi-agent-container tail -f "$LOG"
```

### Debug MCP Communication

```bash
# Test TCP connection
echo '{"jsonrpc":"2.0","method":"ping","id":1}' | nc localhost 9500

# Monitor MCP traffic
docker exec multi-agent-container tcpdump -i any port 9500 -A

# Check MCP process
docker exec multi-agent-container \
  ps aux | grep mcp-tcp-server

# Check file descriptors
docker exec multi-agent-container \
  lsof -p $(docker exec multi-agent-container pgrep -f mcp-tcp-server)
```

## Recovery Procedures

### Container Won't Start

```bash
# Check docker logs
docker logs multi-agent-container

# Remove container and recreate
docker-compose down
docker-compose up -d

# If still failing, rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Lost Session Data

```bash
# Sessions should persist in /workspace/.swarm/sessions/
# Check if directory exists
ls -la workspace/.swarm/sessions/

# Restore from registry
docker exec multi-agent-container \
  cat /workspace/.swarm/sessions/index.json

# Manually recreate session if needed
UUID=<lost-uuid>
docker exec multi-agent-container bash -c "
  cd /workspace/.swarm/sessions/$UUID
  /app/scripts/hive-session-manager.sh update-status $UUID failed
"
```

### Database Corruption

```bash
# Identify corrupted database
docker exec multi-agent-container \
  find /workspace/.swarm -name "*.db" -exec sqlite3 {} "PRAGMA integrity_check;" \;

# Remove corrupted session
docker exec multi-agent-container \
  rm -rf /workspace/.swarm/sessions/$UUID

# Update registry
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 0
```

## Performance Tuning

### Reduce Memory Usage

```bash
# Limit concurrent sessions
# (Implement rate limiting in external system)

# Increase cleanup frequency
# Add to cron:
0 * * * * docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh cleanup 6
```

### Improve Response Time

```bash
# Use persistent MCP connections (connection pooling)

# Batch session queries
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh list | \
  jq '.sessions | to_entries | .[0:10]'
```

## Getting Help

When reporting issues, include:

```bash
# System info
docker version
docker-compose version

# Container status
docker ps -a | grep multi-agent

# Recent logs
tar -czf debug-logs.tar.gz logs/

# Session registry
docker exec multi-agent-container \
  cat /workspace/.swarm/sessions/index.json > sessions-dump.json

# Supervisor status
docker exec multi-agent-container \
  supervisorctl status > supervisor-status.txt
```

Share these files with your issue report.
