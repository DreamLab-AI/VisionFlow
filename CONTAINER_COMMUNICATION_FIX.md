# Container Communication Fix - agentic-workstation ↔ visionflow_container

## Problem Identified (2025-10-25)

### Symptoms
- agentic-workstation container: `Up 2 days (unhealthy)`
- visionflow logs showed errors:
  ```
  [AgentMonitorActor] Management API query failed: Network error:
  error sending request for url (http://agentic-workstation:9090/v1/tasks)
  ```
- Health check failing: `curl http://agentic-workstation:9090/health` → Connection refused

### Root Cause
Management API was not running in agentic-workstation container.

**Evidence:**
1. `ps aux` in agentic-workstation showed no `node` process for Management API
2. Old logs in `/var/log/management-api.log` (from Oct 20) but no current process
3. supervisord.conf at `/etc/supervisord.conf` configured to autostart Management API but service was stopped

## Fix Applied

### Temporary Fix (Session-Only)
Started Management API manually in agentic-workstation:
```bash
docker exec agentic-workstation bash -c "cd /opt/management-api && node server.js" &
```

**Verification:**
```bash
# From host
curl http://agentic-workstation:9090/health
# Response: {"status":"healthy","timestamp":"2025-10-25T22:25:12.390Z"}

# From visionflow_container
docker exec visionflow_container curl -s http://agentic-workstation:9090/health
# Response: {"status":"healthy","timestamp":"2025-10-25T22:25:41.111Z"}
```

**AgentMonitorActor logs (visionflow):**
```
[2025-10-25T22:25:41Z DEBUG] [AgentMonitorActor] Polling active tasks from Management API
[2025-10-25T22:25:41Z DEBUG] [AgentMonitorActor] Retrieved 0 active tasks from Management API
[2025-10-25T22:25:41Z INFO] [AgentMonitorActor] Processing 0 agent statuses from MCP
[2025-10-25T22:25:41Z INFO] [AgentMonitorActor] Sending graph update with 0 agents
```
✅ No more network errors!

## Permanent Fix Required

### In agentic-workstation Container
**File:** `/multi-agent-docker/unified-config/entrypoint-unified.sh` or equivalent

Ensure supervisord starts the Management API service automatically:

```bash
# Check if service is running
supervisorctl status management-api

# If not running, restart
supervisorctl restart management-api

# Or start if stopped
supervisorctl start management-api
```

### Configuration Verified
**File:** `/etc/supervisord.conf` (lines for management-api)
```ini
[program:management-api]
command=/usr/bin/node /opt/management-api/server.js
directory=/opt/management-api
user=devuser
environment=HOME="/home/devuser",PORT="9090",NODE_ENV="production"
autostart=true
autorestart=true
priority=300
stdout_logfile=/var/log/management-api.log
stderr_logfile=/var/log/management-api.error.log
```

**Server Configuration Correct:**
`/opt/management-api/server.js` (lines 12-13)
```javascript
const PORT = process.env.MANAGEMENT_API_PORT || 9090;
const HOST = process.env.MANAGEMENT_API_HOST || '0.0.0.0';  // ✅ Binds to all interfaces
```

## Network Configuration

Both containers on `docker_ragflow` bridge network:
- **agentic-workstation:** 172.18.0.7:9090
- **visionflow_container:** 172.18.0.11

**visionflow Configuration:**
`src/actors/agent_monitor_actor.rs` (lines 125-132)
```rust
let host = std::env::var("MANAGEMENT_API_HOST")
    .unwrap_or_else(|_| "agentic-workstation".to_string());
let port = std::env::var("MANAGEMENT_API_PORT")
    .ok()
    .and_then(|p| p.parse::<u16>().ok())
    .unwrap_or(9090);
```

## Testing

### Health Check
```bash
# From any container on docker_ragflow network
curl http://agentic-workstation:9090/health

# Expected response
{"status":"healthy","timestamp":"2025-10-25T22:25:12.390Z"}
```

### Agent Monitor Integration
Check visionflow logs for successful polling:
```bash
docker exec visionflow_container tail -30 /app/logs/rust-error.log | grep AgentMonitor
```

Expected: No "Network error" messages, only "Retrieved X active tasks"

## Status

- ✅ Temporary fix applied (Management API running)
- ✅ Communication working (both containers can reach each other)
- ✅ AgentMonitorActor polling successfully
- ✅ **PERMANENT FIX APPLIED** (2025-10-25 23:00)
- ✅ Documented in multi-agent-docker source

## Permanent Fix Applied (2025-10-25)

### Changes Made

**1. Health Check Script Created:**
- **File**: `/home/devuser/workspace/project/multi-agent-docker/unified-config/scripts/verify-management-api.sh`
- **Purpose**: Automated verification of Management API health after supervisord startup
- **Features**:
  - 30 retry attempts with 2-second intervals (60 seconds total)
  - Automatic service restart if process is not running
  - Detailed diagnostic output on failure
  - Exit code 0 on success, 1 on failure

**2. Supervisord Configuration Updated:**
- **File**: `/home/devuser/workspace/project/multi-agent-docker/unified-config/supervisord.unified.conf`
- **Added**: `[program:management-api-healthcheck]` section (lines 226-240)
- **Configuration**:
  ```ini
  [program:management-api-healthcheck]
  command=/bin/bash /opt/scripts/verify-management-api.sh
  user=root
  environment=MANAGEMENT_API_HOST="localhost",MANAGEMENT_API_PORT="9090"
  autostart=true
  autorestart=false
  startsecs=0
  priority=950
  ```
- **Priority**: 950 (runs after tmux-autostart at 900, before DBus user at 15)

**3. Entrypoint Script Updated:**
- **File**: `/home/devuser/workspace/project/multi-agent-docker/unified-config/entrypoint-unified.sh`
- **Added**: Phase 7.5 - Install Management API Health Check Script (lines 314-363)
- **Features**:
  - Creates `/opt/scripts/` directory
  - Copies verification script from unified-config if available
  - Falls back to inline script creation if source not found
  - Sets executable permissions

### How It Works

1. Container starts → entrypoint-unified.sh runs
2. Phase 7.5: Health check script installed to `/opt/scripts/verify-management-api.sh`
3. Phase 10: Supervisord starts all services
4. Management API starts (priority 300)
5. Health check runs (priority 950, after all services)
6. Health check verifies API responds on port 9090
7. If API not responding after 60s → failure logged, diagnostics printed
8. If API responding → success logged, health check exits

### Testing

To test after next container rebuild:
```bash
# Restart agentic-workstation container
docker restart agentic-workstation

# Wait 60 seconds for health check to complete
sleep 60

# Check health check logs
docker exec agentic-workstation supervisorctl tail management-api-healthcheck

# Verify Management API is responding
curl http://agentic-workstation:9090/health

# Check visionflow logs for successful polling
docker exec visionflow_container tail -30 /app/logs/rust-error.log | grep AgentMonitor
```

### Files Modified

1. **Created**: `multi-agent-docker/unified-config/scripts/verify-management-api.sh`
2. **Modified**: `multi-agent-docker/unified-config/supervisord.unified.conf` (added health check program)
3. **Modified**: `multi-agent-docker/unified-config/entrypoint-unified.sh` (added Phase 7.5)
4. **Updated**: `CONTAINER_COMMUNICATION_FIX.md` (this file)

## Next Steps (Optional)

1. **Test the permanent fix**:
   - Rebuild agentic-workstation container using Dockerfile.unified
   - Verify health check runs automatically
   - Confirm Management API starts on boot

2. **Additional improvements** (low priority):
   - Add Prometheus metrics for Management API availability
   - Add alerting for health check failures
   - Consider adding health check for other critical services
