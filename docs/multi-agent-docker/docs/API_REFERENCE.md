# API Reference

Complete reference documentation for the Management API and Claude-ZAI service.

## Management API

**Base URL**: `http://localhost:9090`

**Authentication**: Bearer token (except health endpoints)

**Rate Limiting**: 100 requests/minute per IP

### Authentication

All endpoints except `/health`, `/ready`, and `/metrics` require Bearer token authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:9090/v1/status
```

Set the API key in `.env`:
```bash
MANAGEMENT_API_KEY=your-secure-token
```

---

## Endpoints

### `GET /` - API Information

Get API metadata and available endpoints.

**Authentication**: Required

**Response**: `200 OK`

```json
{
  "name": "Agentic Flow Management API",
  "version": "2.1.0",
  "description": "HTTP API for managing AI agent workflows and MCP tools",
  "endpoints": {
    "tasks": {
      "create": "POST /v1/tasks",
      "get": "GET /v1/tasks/:taskId",
      "list": "GET /v1/tasks",
      "delete": "DELETE /v1/tasks/:taskId"
    },
    "monitoring": {
      "status": "GET /v1/status",
      "health": "GET /health",
      "ready": "GET /ready",
      "metrics": "GET /metrics"
    }
  },
  "documentation": "http://localhost:9090/docs"
}
```

---

### `POST /v1/tasks` - Create Task

Create a new isolated agentic-flow task.

**Authentication**: Required

**Request Body**:

```json
{
  "agent": "coder",
  "task": "Build a REST API with Express",
  "provider": "gemini",
  "timeout": 300000
}
```

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent` | string | Yes | Agent type to use (coder, reviewer, tester, etc.) |
| `task` | string | Yes | Task description |
| `provider` | string | No | AI provider (gemini, openai, claude, openrouter) |
| `timeout` | number | No | Task timeout in milliseconds (default: 300000) |

**Response**: `202 Accepted`

```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted",
  "message": "Task started successfully",
  "taskDir": "/home/devuser/workspace/tasks/550e8400-e29b-41d4-a716-446655440000",
  "logFile": "/home/devuser/logs/tasks/550e8400-e29b-41d4-a716-446655440000.log",
  "startTime": 1704110400000
}
```

**Error Responses**:

`400 Bad Request` - Invalid parameters
```json
{
  "statusCode": 400,
  "error": "Bad Request",
  "message": "agent and task are required"
}
```

`401 Unauthorized` - Missing or invalid API key
```json
{
  "statusCode": 401,
  "error": "Unauthorized",
  "message": "Invalid or missing API key"
}
```

`429 Too Many Requests` - Rate limit exceeded
```json
{
  "statusCode": 429,
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Try again later."
}
```

**Example**:

```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Create a Python script to parse JSON",
    "provider": "gemini"
  }'
```

---

### `GET /v1/tasks/:taskId` - Get Task Status

Retrieve status and log output for a specific task.

**Authentication**: Required

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `taskId` | string | UUID of the task |

**Response**: `200 OK`

```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "agent": "coder",
  "task": "Build a REST API with Express",
  "provider": "gemini",
  "status": "running",
  "startTime": 1704110400000,
  "exitTime": null,
  "exitCode": null,
  "duration": 45000,
  "taskDir": "/home/devuser/workspace/tasks/550e8400-e29b-41d4-a716-446655440000",
  "logFile": "/home/devuser/logs/tasks/550e8400-e29b-41d4-a716-446655440000.log",
  "logTail": "... last 50 lines of log output ..."
}
```

**Status Values**:

- `running`: Task is currently executing
- `completed`: Task finished successfully (exitCode 0)
- `failed`: Task exited with error (exitCode > 0)

**Error Responses**:

`404 Not Found` - Task not found
```json
{
  "statusCode": 404,
  "error": "Not Found",
  "message": "Task not found"
}
```

**Example**:

```bash
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/550e8400-e29b-41d4-a716-446655440000
```

---

### `GET /v1/tasks` - List Active Tasks

Get all currently running tasks.

**Authentication**: Required

**Query Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (running, completed, failed) |
| `limit` | number | Max number of tasks to return (default: 100) |

**Response**: `200 OK`

```json
{
  "activeTasks": [
    {
      "taskId": "550e8400-e29b-41d4-a716-446655440000",
      "agent": "coder",
      "task": "Build a REST API",
      "provider": "gemini",
      "status": "running",
      "startTime": 1704110400000,
      "duration": 45000
    }
  ],
  "count": 1,
  "totalTasks": 15
}
```

**Example**:

```bash
# List all active tasks
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks

# Filter by status
curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:9090/v1/tasks?status=running"
```

---

### `DELETE /v1/tasks/:taskId` - Stop Task

Stop a running task.

**Authentication**: Required

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `taskId` | string | UUID of the task |

**Response**: `200 OK`

```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "stopped",
  "message": "Task stopped successfully"
}
```

**Error Responses**:

`404 Not Found` - Task not found or already completed

**Example**:

```bash
curl -X DELETE \
  -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/550e8400-e29b-41d4-a716-446655440000
```

---

### `GET /v1/status` - System Status

Comprehensive system health check including GPU, providers, and resources.

**Authentication**: Required

**Response**: `200 OK`

```json
{
  "timestamp": "2025-01-12T17:00:00.000Z",
  "api": {
    "uptime": 3600,
    "version": "2.1.0",
    "pid": 1234,
    "nodeVersion": "v20.10.0"
  },
  "tasks": {
    "active": 2,
    "total": 15,
    "completed": 12,
    "failed": 1
  },
  "gpu": {
    "available": true,
    "gpus": [
      {
        "index": 0,
        "name": "NVIDIA RTX 4090",
        "uuid": "GPU-xxxxx",
        "utilization": 45.5,
        "memory": {
          "used": 8192,
          "total": 24576,
          "free": 16384,
          "percentUsed": "33.33"
        },
        "temperature": 65,
        "powerDraw": 350,
        "powerLimit": 450
      }
    ]
  },
  "providers": {
    "gemini": {
      "status": "configured",
      "enabled": true,
      "priority": 1
    },
    "openai": {
      "status": "configured",
      "enabled": true,
      "priority": 2
    },
    "claude": {
      "status": "configured",
      "enabled": true,
      "priority": 3
    },
    "openrouter": {
      "status": "configured",
      "enabled": true,
      "priority": 4
    }
  },
  "system": {
    "platform": "linux",
    "arch": "x64",
    "cpu": {
      "cores": 32,
      "model": "AMD Ryzen 9 7950X",
      "loadAverage": {
        "load1": 2.5,
        "load5": 2.2,
        "load15": 1.8
      }
    },
    "memory": {
      "total": 65536,
      "used": 32768,
      "free": 32768,
      "percentUsed": "50.00"
    },
    "disk": {
      "size": "1.0T",
      "used": "500G",
      "available": "500G",
      "percentUsed": "50%",
      "mountPoint": "/"
    }
  },
  "services": {
    "claudeZai": {
      "status": "healthy",
      "url": "http://claude-zai-service:9600"
    }
  }
}
```

**Example**:

```bash
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq
```

---

### `GET /health` - Health Check

Simple health check endpoint (no authentication required).

**Authentication**: Not required

**Response**: `200 OK`

```json
{
  "status": "healthy",
  "timestamp": "2025-01-12T17:00:00.000Z",
  "uptime": 3600
}
```

**Example**:

```bash
curl http://localhost:9090/health
```

---

### `GET /ready` - Readiness Probe

Kubernetes-style readiness check (no authentication required).

**Authentication**: Not required

**Response**: `200 OK` if ready, `503 Service Unavailable` if not ready

```json
{
  "ready": true,
  "activeTasks": 2,
  "timestamp": "2025-01-12T17:00:00.000Z"
}
```

**Example**:

```bash
curl http://localhost:9090/ready
```

---

### `GET /metrics` - Prometheus Metrics

Prometheus-compatible metrics endpoint (no authentication required).

**Authentication**: Not required

**Response**: `200 OK` (text/plain)

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/v1/status",status="200"} 150

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 120
http_request_duration_seconds_bucket{le="0.5"} 145
http_request_duration_seconds_bucket{le="1.0"} 150

# HELP tasks_created_total Total tasks created
# TYPE tasks_created_total counter
tasks_created_total 15

# HELP tasks_active Current active tasks
# TYPE tasks_active gauge
tasks_active 2
```

**Example**:

```bash
curl http://localhost:9090/metrics
```

---

## Claude-ZAI Service API

**Base URL**: `http://localhost:9600`

**Authentication**: None (internal service)

**Note**: This service should not be exposed externally. It's designed for internal use only.

---

### `POST /prompt` - Send Prompt to Claude

Send a prompt to Claude AI via Z.AI.

**Request Body**:

```json
{
  "prompt": "Explain Docker in 3 sentences",
  "timeout": 15000,
  "model": "claude-3-sonnet-20240229"
}
```

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | The prompt to send to Claude |
| `timeout` | number | No | Request timeout in milliseconds (default: 30000) |
| `model` | string | No | Claude model to use (default: claude-3-sonnet) |

**Response**: `200 OK`

```json
{
  "response": "Docker is a platform that uses containerization...",
  "model": "claude-3-sonnet-20240229",
  "duration": 1250,
  "timestamp": "2025-01-12T17:00:00.000Z"
}
```

**Error Responses**:

`400 Bad Request` - Missing prompt
```json
{
  "error": "prompt is required"
}
```

`503 Service Unavailable` - All workers busy
```json
{
  "error": "All workers busy. Queue full.",
  "queueSize": 50,
  "maxQueueSize": 50
}
```

`504 Gateway Timeout` - Request timeout
```json
{
  "error": "Request timeout",
  "timeout": 30000
}
```

**Example**:

```bash
curl -X POST http://localhost:9600/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci",
    "timeout": 30000
  }'
```

---

### `GET /health` - Claude-ZAI Health Check

Health check for Claude-ZAI service.

**Response**: `200 OK`

```json
{
  "status": "healthy",
  "workers": {
    "total": 4,
    "available": 2,
    "busy": 2
  },
  "queue": {
    "size": 5,
    "maxSize": 50
  },
  "timestamp": "2025-01-12T17:00:00.000Z"
}
```

**Example**:

```bash
curl http://localhost:9600/health
```

---

## OpenAPI/Swagger Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:9090/docs
- **OpenAPI Spec**: http://localhost:9090/docs/json

The Swagger UI provides:
- Interactive API testing
- Request/response examples
- Parameter descriptions
- Authentication testing

---

## SDK Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:9090';
const API_KEY = process.env.MANAGEMENT_API_KEY;

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  }
});

// Create task
async function createTask(agent, task, provider = 'gemini') {
  const response = await client.post('/v1/tasks', {
    agent,
    task,
    provider
  });
  return response.data;
}

// Get task status
async function getTask(taskId) {
  const response = await client.get(`/v1/tasks/${taskId}`);
  return response.data;
}

// Poll until complete
async function waitForTask(taskId, pollInterval = 5000) {
  while (true) {
    const task = await getTask(taskId);
    if (task.status !== 'running') {
      return task;
    }
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }
}

// Usage
(async () => {
  const task = await createTask('coder', 'Build a REST API');
  console.log(`Task created: ${task.taskId}`);

  const result = await waitForTask(task.taskId);
  console.log(`Task ${result.status}`);
  console.log(result.logTail);
})();
```

### Python

```python
import requests
import time
import os

API_URL = "http://localhost:9090"
API_KEY = os.getenv("MANAGEMENT_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def create_task(agent, task, provider="gemini"):
    response = requests.post(
        f"{API_URL}/v1/tasks",
        headers=headers,
        json={"agent": agent, "task": task, "provider": provider}
    )
    response.raise_for_status()
    return response.json()

def get_task(task_id):
    response = requests.get(
        f"{API_URL}/v1/tasks/{task_id}",
        headers=headers
    )
    response.raise_for_status()
    return response.json()

def wait_for_task(task_id, poll_interval=5):
    while True:
        task = get_task(task_id)
        if task["status"] != "running":
            return task
        time.sleep(poll_interval)

# Usage
if __name__ == "__main__":
    task = create_task("coder", "Build a REST API")
    print(f"Task created: {task['taskId']}")

    result = wait_for_task(task["taskId"])
    print(f"Task {result['status']}")
    print(result["logTail"])
```

### cURL

```bash
#!/bin/bash

API_URL="http://localhost:9090"
API_KEY="${MANAGEMENT_API_KEY}"

# Create task
RESPONSE=$(curl -s -X POST "$API_URL/v1/tasks" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Build a REST API",
    "provider": "gemini"
  }')

TASK_ID=$(echo "$RESPONSE" | jq -r '.taskId')
echo "Task created: $TASK_ID"

# Poll for completion
while true; do
  STATUS=$(curl -s -H "Authorization: Bearer $API_KEY" \
    "$API_URL/v1/tasks/$TASK_ID" | jq -r '.status')

  echo "Status: $STATUS"

  if [ "$STATUS" != "running" ]; then
    break
  fi

  sleep 5
done

# Get final result
curl -s -H "Authorization: Bearer $API_KEY" \
  "$API_URL/v1/tasks/$TASK_ID" | jq
```

---

## Rate Limiting

The Management API implements rate limiting to prevent abuse:

- **Default Limit**: 100 requests/minute per IP
- **Whitelist**: 127.0.0.1 (localhost) is exempt
- **Headers**: Rate limit information included in response headers

**Response Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704110460000
```

**Rate Limit Exceeded Response**:
```json
{
  "statusCode": 429,
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Try again later.",
  "retryAfter": 60
}
```

---

## Error Handling

All error responses follow this format:

```json
{
  "statusCode": 400,
  "error": "Bad Request",
  "message": "Detailed error message"
}
```

**Common Status Codes**:

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 202 | Accepted | Task accepted for processing |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

---

## Versioning

Current API version: **v2.1.0**

The API uses URL versioning:
- Current: `/v1/*`
- Future: `/v2/*` (when breaking changes introduced)

Version information available at `GET /` endpoint.
