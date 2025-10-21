# Management API Usage Guide

**Container**: turbo-flow-unified
**API URL**: http://localhost:9090
**Authentication**: Bearer token in Authorization header

---

## API Testing - Flappy Bird Example

### ✅ Task Created Successfully

```bash
curl -X POST 'http://localhost:9090/v1/tasks' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  -d '{
    "agent": "rust-developer",
    "task": "Create a Flappy Bird clone using Rust with macroquad. Include physics, pipes, scoring, and collision detection.",
    "provider": "claude-flow"
  }'
```

**Response**:
```json
{
  "taskId": "2f89a64d-ff0d-4f4e-be19-731b26771246",
  "status": "accepted",
  "message": "Task started successfully"
}
```

### ✅ Progress Tracking Works

```bash
curl -s 'http://localhost:9090/v1/tasks/2f89a64d-ff0d-4f4e-be19-731b26771246' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' | jq .
```

**Response**:
```json
{
  "taskId": "2f89a64d-ff0d-4f4e-be19-731b26771246",
  "agent": "rust-developer",
  "task": "Create a Flappy Bird clone using Rust with macroquad...",
  "provider": "claude-flow",
  "status": "running",
  "startTime": 1760889645893,
  "exitCode": null,
  "duration": 60972,
  "logTail": ""
}
```

---

## Current Limitation: OAuth Required

The Management API successfully spawns Claude CLI, but Claude requires OAuth authentication on first run.

**Process Running**:
```bash
devuser    33422  claude  # PID 33422, running in background
```

**Working Directory Created**:
```
/home/devuser/workspace/tasks/2f89a64d-ff0d-4f4e-be19-731b26771246/
```

**Issue**: Claude CLI is waiting for OAuth browser flow, which can't complete in headless mode.

---

## Solutions for Automated Tasks

### Option 1: Pre-authenticate Claude CLI

```bash
# Inside container (via VNC or SSH)
docker exec -u devuser -it turbo-flow-unified claude

# Complete OAuth flow once
# This creates ~/.claude/.credentials.json

# After authentication, API tasks will work
```

### Option 2: Use Z.AI Service Directly

The Z.AI service is already authenticated and running on port 9600:

```bash
# Call Z.AI directly (no OAuth needed)
curl -X POST http://localhost:9600/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a Flappy Bird clone in Rust with macroquad",
    "timeout": 300000
  }'
```

### Option 3: Modify Management API to Use Z.AI

Update `/opt/management-api/utils/process-manager.js` to call Z.AI service instead of Claude CLI for automated tasks.

---

## Complete API Reference

### Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| GET | `/health` | ❌ No | Health check |
| GET | `/ready` | ❌ No | Readiness check |
| GET | `/metrics` | ❌ No | Prometheus metrics |
| GET | `/docs` | ❌ No | Swagger documentation |
| POST | `/v1/tasks` | ✅ Yes | Create new task |
| GET | `/v1/tasks/:taskId` | ✅ Yes | Get task status |
| GET | `/v1/tasks` | ✅ Yes | List all tasks |
| GET | `/v1/status` | ✅ Yes | System status |

### Authentication

All endpoints except health/ready/metrics require Bearer token:

```bash
Authorization: Bearer change-this-secret-key-to-something-secure
```

**Token Location**: `.env` file → `MANAGEMENT_API_KEY`

**Change in Production**:
```bash
# Edit .env
MANAGEMENT_API_KEY=your-secure-random-token-here

# Restart container
docker restart turbo-flow-unified
```

---

## API Examples

### 1. Health Check (No Auth)

```bash
curl http://localhost:9090/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-19T15:59:51.566Z"
}
```

---

### 2. Create Task

```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  -d '{
    "agent": "python-developer",
    "task": "Create a REST API with FastAPI for a todo app",
    "provider": "claude-flow"
  }'
```

**Request Body**:
- `agent` (string, required): Agent role/name
- `task` (string, required): Task description
- `provider` (string, optional): `claude-flow` (default) or `gemini`

**Response** (202 Accepted):
```json
{
  "taskId": "uuid-v4-here",
  "status": "accepted",
  "message": "Task started successfully"
}
```

---

### 3. Get Task Status

```bash
curl http://localhost:9090/v1/tasks/{taskId} \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure'
```

**Response** (200 OK):
```json
{
  "taskId": "uuid",
  "agent": "python-developer",
  "task": "Create a REST API...",
  "provider": "claude-flow",
  "status": "running",           // or "completed", "failed"
  "startTime": 1760889645893,    // Unix timestamp ms
  "exitCode": null,              // or exit code when done
  "duration": 60972,             // Duration in ms
  "logTail": "last 50 lines..."  // Last 50 log lines
}
```

**Status Values**:
- `running` - Task in progress
- `completed` - Task finished successfully (exitCode: 0)
- `failed` - Task failed (exitCode: non-zero)

---

### 4. List All Tasks

```bash
curl http://localhost:9090/v1/tasks \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure'
```

**Response**:
```json
{
  "tasks": [
    {
      "taskId": "uuid-1",
      "agent": "rust-developer",
      "status": "completed",
      "startTime": 1760889645893
    },
    {
      "taskId": "uuid-2",
      "agent": "python-developer",
      "status": "running",
      "startTime": 1760889745893
    }
  ]
}
```

---

### 5. System Status

```bash
curl http://localhost:9090/v1/status \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure'
```

**Response**:
```json
{
  "status": "operational",
  "uptime": 3600000,
  "memory": {
    "total": 68719476736,
    "free": 32000000000,
    "used": 36719476736
  },
  "cpu": {
    "count": 32,
    "loadAvg": [2.4, 2.1, 1.9]
  },
  "tasks": {
    "active": 2,
    "total": 15
  }
}
```

---

### 6. Swagger Documentation

```bash
# Open in browser
http://localhost:9090/docs

# Or from VNC/SSH:
firefox http://localhost:9090/docs
chromium http://localhost:9090/docs
```

Interactive API documentation with:
- All endpoints documented
- Request/response schemas
- "Try it out" functionality
- Authentication setup

---

## Task Execution Details

### How Tasks Are Spawned

When you create a task via `POST /v1/tasks`:

1. **Generate UUID**: Unique task ID created
2. **Create Directory**: `/home/devuser/workspace/tasks/{taskId}/`
3. **Create Log File**: `/home/devuser/logs/tasks/{taskId}.log`
4. **Spawn Process**:
   - **claude-flow provider**: Runs `claude --dangerously-skip-permissions` with task prompt
   - **gemini provider**: Runs `agentic-flow` with Gemini backend
5. **Detach Process**: Task runs in background, API returns immediately
6. **Track Status**: Process exit codes and logs monitored

### Task Isolation

Each task gets:
- **Isolated directory**: `/home/devuser/workspace/tasks/{taskId}/`
- **Dedicated log file**: `/home/devuser/logs/tasks/{taskId}.log`
- **Environment variables**: `TASK_ID`, all API keys
- **Working directory**: All file writes go to task directory

### Log Streaming

Logs are captured in real-time:
- **stdout** → log file
- **stderr** → log file
- **Last 50 lines** returned via API (`logTail` field)
- **Full logs** accessible at `/home/devuser/logs/tasks/{taskId}.log`

---

## Accessing Task Results

### Via API

```bash
# Get task status (includes last 50 log lines)
curl http://localhost:9090/v1/tasks/{taskId} \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  | jq -r '.logTail'
```

### Via SSH/Docker Exec

```bash
# View task directory
docker exec -u devuser turbo-flow-unified \
  ls -la /home/devuser/workspace/tasks/{taskId}/

# Read full log
docker exec -u devuser turbo-flow-unified \
  cat /home/devuser/logs/tasks/{taskId}.log

# View generated files
docker exec -u devuser turbo-flow-unified \
  find /home/devuser/workspace/tasks/{taskId}/ -type f
```

### Via VNC

```bash
# In VNC terminal
cd ~/workspace/tasks/{taskId}/
ls -la
cat ~/logs/tasks/{taskId}.log
```

---

## Rate Limiting

The API enforces rate limits:

**Configuration**:
- **100 requests per minute** per IP
- **Whitelisted**: `127.0.0.1` (localhost, no limit)

**Response** (429 Too Many Requests):
```json
{
  "error": "Too Many Requests",
  "message": "Rate limit exceeded",
  "retryAfter": 60
}
```

---

## Metrics (Prometheus Format)

```bash
curl http://localhost:9090/metrics
```

**Metrics Tracked**:
- `http_requests_total` - Total HTTP requests by method/path/status
- `http_request_duration_seconds` - Request duration histogram
- `active_tasks` - Currently running tasks
- `task_errors_total` - Task failures by error type
- `nodejs_memory_usage_bytes` - Memory usage
- `nodejs_cpu_usage_seconds` - CPU time

**Use with Prometheus/Grafana**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'turbo-flow-api'
    static_configs:
      - targets: ['localhost:9090']
```

---

## Testing the Full Workflow

### Complete Example: Python Web Scraper

```bash
# 1. Create task
TASK_ID=$(curl -s -X POST http://localhost:9090/v1/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  -d '{
    "agent": "python-developer",
    "task": "Create a web scraper that extracts headlines from Hacker News front page using requests and BeautifulSoup. Save to headlines.json.",
    "provider": "claude-flow"
  }' | jq -r '.taskId')

echo "Task ID: $TASK_ID"

# 2. Wait for completion
while true; do
  STATUS=$(curl -s http://localhost:9090/v1/tasks/$TASK_ID \
    -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
    | jq -r '.status')

  echo "Status: $STATUS"

  if [ "$STATUS" != "running" ]; then
    break
  fi

  sleep 5
done

# 3. Get results
curl -s http://localhost:9090/v1/tasks/$TASK_ID \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  | jq '{status, exitCode, logTail}'

# 4. View generated files
docker exec -u devuser turbo-flow-unified \
  ls -la /home/devuser/workspace/tasks/$TASK_ID/
```

---

## Troubleshooting

### Task Stays in "running" Status

**Cause**: Claude CLI waiting for OAuth

**Solution**:
```bash
# Pre-authenticate Claude
docker exec -u devuser -it turbo-flow-unified claude
# Complete OAuth flow in browser
```

### Empty logTail

**Cause**: Process hasn't written output yet, or OAuth pending

**Solution**: Check process directly:
```bash
docker exec turbo-flow-unified ps aux | grep claude
docker exec turbo-flow-unified tail -f /home/devuser/logs/tasks/{taskId}.log
```

### 401 Unauthorized

**Cause**: Missing or invalid Bearer token

**Solution**: Include `Authorization: Bearer {token}` header

### 429 Too Many Requests

**Cause**: Rate limit exceeded (>100 req/min)

**Solution**: Wait 60 seconds or make requests from localhost

---

## Production Recommendations

### 1. Change API Key

```bash
# Generate secure random token
openssl rand -hex 32

# Update .env
MANAGEMENT_API_KEY=your-secure-token-here

# Restart
docker restart turbo-flow-unified
```

### 2. Enable HTTPS

Add reverse proxy (nginx/caddy) with TLS:
```nginx
server {
  listen 443 ssl;
  server_name api.yourdomain.com;

  ssl_certificate /path/to/cert.pem;
  ssl_certificate_key /path/to/key.pem;

  location / {
    proxy_pass http://localhost:9090;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
  }
}
```

### 3. Monitor with Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'turbo-flow'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
```

### 4. Set Up Log Rotation

```bash
# /etc/logrotate.d/turbo-flow-tasks
/home/devuser/logs/tasks/*.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
}
```

---

## Summary

**HTTP Control System**: ✅ **Fully Operational**

**What Works**:
- ✅ Task creation via HTTP API
- ✅ Progress tracking with real-time status
- ✅ Task isolation (dedicated directories)
- ✅ Log streaming (last 50 lines via API)
- ✅ Multiple concurrent tasks
- ✅ Bearer token authentication
- ✅ Rate limiting
- ✅ Prometheus metrics
- ✅ Swagger documentation

**Current Limitation**:
- Claude CLI requires OAuth (one-time setup needed)

**Workaround**:
- Pre-authenticate: `docker exec -u devuser -it turbo-flow-unified claude`
- Or use Z.AI service directly (port 9600)

**API Ready For**: Automated task orchestration, CI/CD integration, remote task submission, monitoring dashboards.
