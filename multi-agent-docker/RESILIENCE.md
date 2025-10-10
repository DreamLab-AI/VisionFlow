# Multi-Agent Docker Resilience Configuration

This document describes the resilience mechanisms preventing system shutdown on error states.

## Container-Level Resilience

### Docker Compose Configuration ([docker-compose.yml](docker-compose.yml))

1. **Init System** (line 159)
   ```yaml
   init: true
   ```
   - Reaps zombie processes
   - Prevents PID 1 issues

2. **Auto-restart Policy** (line 180)
   ```yaml
   restart: unless-stopped
   ```
   - Container restarts automatically unless manually stopped
   - Survives daemon restarts

3. **Graceful Shutdown** (lines 162-163)
   ```yaml
   stop_signal: SIGTERM
   stop_grace_period: 30s
   ```
   - Clean shutdown with 30s grace period
   - Prevents data corruption

4. **Keep-alive Configuration** (lines 155-156)
   ```yaml
   stdin_open: true
   tty: true
   ```
   - Prevents premature exit
   - Maintains interactive session

## Process-Level Resilience

### Supervisord Configuration ([supervisord.conf](supervisord.conf))

1. **Resource Limits** (lines 12-14)
   ```ini
   minfds=1024
   minprocs=200
   silent=false
   ```
   - Ensures sufficient file descriptors
   - Allows 200+ concurrent processes
   - Logs all failures

2. **Critical Services Restart Strategy**

   **MCP Core Services** (lines 28-67):
   - `autorestart=unexpected` - Only restart on unexpected exits
   - `startretries=999` - Virtually unlimited retry attempts
   - `exitcodes=0` - Only exit code 0 is "expected"
   - Applies to:
     - mcp-ws-bridge
     - mcp-tcp-server
     - claude-flow-tcp

3. **GUI Services Restart Strategy** (lines 90-147):
   - `autorestart=true` - Always restart
   - Applies to:
     - playwright-mcp-server
     - qgis-mcp-server
     - pbr-mcp-server
     - web-summary-mcp-server
     - vnc, xfce, novnc

4. **Long-running Services** (lines 298-313):
   - Wrapped in infinite loops with error suppression
   - Example: `while true; do /script.sh || true; sleep 600; done`
   - Applies to:
     - session-cleanup

## Entrypoint Resilience

### Entry Script Hardening ([entrypoint.sh](entrypoint.sh))

1. **Error Handling** (line 3)
   ```bash
   trap 'echo "[ERROR] Caught error at line $LINENO, continuing..."' ERR
   ```
   - Logs errors without exiting
   - Continues execution on non-fatal errors

2. **Supervisord Monitor** (lines 95-108)
   ```bash
   while true; do
       sleep 60
       if ! kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
           /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &
           SUPERVISORD_PID=$!
       fi
   done
   ```
   - Monitors supervisord health every 60s
   - Auto-restarts if supervisord dies
   - Runs in background

3. **Background Jobs with Error Suppression** (lines 102-159)
   - Setup jobs use `|| true` for non-critical operations
   - Permission fixes run asynchronously
   - Failed setup doesn't block container startup

## Recovery Mechanisms

### Supervisor Recovery Script ([scripts/supervisor-recovery.sh](scripts/supervisor-recovery.sh))

Manual recovery tool for severe failures:

```bash
# Check supervisor health
/app/scripts/supervisor-recovery.sh

# Can be added to cron for automated checks
*/5 * * * * /app/scripts/supervisor-recovery.sh
```

**Features:**
- Detects supervisor process failures
- Validates socket responsiveness
- Auto-restarts with exponential backoff
- Monitors critical service states
- Comprehensive logging

### Session Cleanup ([scripts/session-cleanup.sh](scripts/session-cleanup.sh))

Prevents resource exhaustion:
- Removes stale session data
- Cleans WAL/SHM database files
- Marks abandoned sessions as completed
- Runs every 10 minutes via supervisord

## Exposed Ports (Minimized)

Only essential ports are exposed to reduce attack surface and simplify networking:

- **5901** - VNC GUI access
- **8080** - VS Code Server (code-server)
- **9500** - MCP TCP Server (primary communication)

All other services (MCP tool servers, WebSocket bridge, UI servers, health endpoints) operate internally within the container and do not need external exposure.

## Failure Modes & Recovery

| Failure Type | Detection | Recovery | Downtime |
|-------------|-----------|----------|----------|
| Single MCP server crash | Supervisord | Auto-restart (unlimited) | <5s |
| Supervisord crash | Entrypoint monitor | Auto-restart | <60s |
| Container exit | Docker restart policy | Container restart | <10s |
| Database lock | Session cleanup | Lock removal every 10min | N/A |
| Resource exhaustion | Supervisord minfds/minprocs | Graceful rejection | N/A |
| Process zombie accumulation | Init system | Automatic reaping | N/A |

## Monitoring

### Log Locations

```bash
# Container logs
docker logs multi-agent-container

# Entrypoint logs
tail -f /var/log/multi-agent/entrypoint.log

# Supervisor logs
tail -f /var/log/supervisor/supervisord.log

# Individual service logs
tail -f /app/mcp-logs/tcp-server.log
tail -f /app/mcp-logs/ws-bridge.log

# Recovery logs
tail -f /var/log/multi-agent/supervisor-recovery.log
```

### Health Check Commands

```bash
# Check supervisord status
docker exec multi-agent-container supervisorctl status

# Check specific service
docker exec multi-agent-container supervisorctl status mcp-tcp-server

# Restart specific service
docker exec multi-agent-container supervisorctl restart mcp-ws-bridge

# View logs
docker exec multi-agent-container supervisorctl tail -f mcp-tcp-server
```

## Best Practices

1. **Never use `docker-compose down`** - Use `docker-compose stop` to preserve state
2. **Monitor logs regularly** - Check for repeated restart patterns
3. **Review session cleanup** - Ensure old sessions are being pruned
4. **Database maintenance** - Enable `VACUUM_DBS=true` periodically
5. **Resource monitoring** - Watch for file descriptor exhaustion

## Limitations

The system is designed for maximum availability but cannot recover from:
- Host system crashes
- Docker daemon failures
- Disk space exhaustion
- OOM killer events
- Manual `docker-compose down`
- Corrupted SQLite databases (requires manual intervention)

## Future Enhancements

Potential improvements:
- [ ] External health monitoring (Prometheus/Grafana)
- [ ] Distributed tracing for request flows
- [ ] Automated database corruption detection/repair
- [ ] Circuit breaker patterns in Node.js services
- [ ] Leader election for multi-container deployments
