# Programmatic Control Guide

---
**Version:** 1.0.0
**Last Updated:** 2025-10-12
**Status:** Active
**Category:** Guide
**Tags:** [api, control, automation, integration]
---

## Overview

This guide explains how external codebases can programmatically control and interact with the Agentic Flow Docker environment through the Management API. The Management API is the **sole entry point** for all external communication and provides secure, authenticated access to spawn, monitor, and terminate agent workers.

## Architecture

### Communication Method

All external communication with the agentic-flow Docker container is handled through the **Management API**, a secure RESTful HTTP service exposed on port **9090**.

```
┌─────────────────────┐
│  External Codebase  │
│   (Your System)     │
└──────────┬──────────┘
           │ HTTP/HTTPS
           │ Port 9090
           ↓
┌──────────────────────────────┐
│   Management API             │
│   (agentic-flow-cachyos)     │
├──────────────────────────────┤
│ • Authentication             │
│ • Task Spawning              │
│ • Status Monitoring          │
│ • Log Streaming              │
│ • Task Termination           │
└──────────┬───────────────────┘
           │
           ↓
┌──────────────────────────────┐
│   Process Manager            │
│   • Worker Isolation         │
│   • Task Tracking            │
│   • Resource Management      │
└──────────┬───────────────────┘
           │
           ↓
┌──────────────────────────────┐
│   Agent Workers              │
│   • gemini-flow              │
│   • claude-flow              │
│   • Custom Agents            │
└──────────────────────────────┘
```

### Key Components

**Management API**
- **Location:** `docker/cachyos/management-api/`
- **Process:** Managed by supervisord (`config/supervisord.conf`)
- **Port:** 9090 (exposed on Docker host)
- **Protocol:** RESTful HTTP with JSON payloads
- **Documentation:** OpenAPI 3.0 at `/docs`

**Authentication**
- **Method:** Bearer token authentication
- **Header:** `Authorization: Bearer <token>`
- **Configuration:** `MANAGEMENT_API_KEY` environment variable in `docker-compose.yml`
- **Scope:** Required for all endpoints except `/health`, `/ready`, and `/metrics`

---

## Authentication

### Setting Up Authentication

1. **Configure API Key in Docker Compose:**

```yaml
# docker/cachyos/docker-compose.yml
services:
  agentic-flow:
    environment:
      - MANAGEMENT_API_KEY=your-secure-token-here
```

2. **Include Token in Requests:**

```bash
# All authenticated requests
curl -H "Authorization: Bearer your-secure-token-here" \
     -H "Content-Type: application/json" \
     http://localhost:9090/v1/tasks
```

### Example: Node.js Client Setup

```javascript
const axios = require('axios');

const client = axios.create({
  baseURL: 'http://localhost:9090',
  headers: {
    'Authorization': `Bearer ${process.env.MANAGEMENT_API_KEY}`,
    'Content-Type': 'application/json'
  },
  timeout: 30000
});

module.exports = client;
```

### Example: Python Client Setup

```python
import os
import requests

class AgenticFlowClient:
    def __init__(self, base_url='http://localhost:9090'):
        self.base_url = base_url
        self.headers = {
            'Authorization': f"Bearer {os.getenv('MANAGEMENT_API_KEY')}",
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
```

---

## Spawning Agent Workers

### Overview

Spawn isolated agent workers to execute tasks. Each worker runs in its own dedicated directory to prevent database locking and file conflicts between concurrent tasks.

### Endpoint

**`POST /v1/tasks`**

### Request Schema

```json
{
  "agent": "string",      // Required: Agent strategy (e.g., "coder", "researcher")
  "task": "string",       // Required: High-level objective
  "provider": "string"    // Optional: AI provider (default: "gemini")
}
```

### Response Schema

**Status:** `202 Accepted`

```json
{
  "taskId": "uuid",               // Unique task identifier
  "status": "accepted",            // Task status
  "message": "string",             // Status message
  "taskDir": "string",             // Isolated working directory
  "logFile": "string"              // Dedicated log file path
}
```

### Example: Spawn a Coding Task

**cURL:**
```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Refactor the authentication module to use JWT and add unit tests.",
    "provider": "gemini"
  }'
```

**Response:**
```json
{
  "taskId": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "accepted",
  "message": "Task started successfully",
  "taskDir": "/home/devuser/workspace/tasks/a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "logFile": "/home/devuser/logs/tasks/a1b2c3d4-e5f6-7890-1234-567890abcdef.log"
}
```

### Example: Node.js Integration

```javascript
const client = require('./agentic-flow-client');

async function spawnCodingTask(taskDescription) {
  try {
    const response = await client.post('/v1/tasks', {
      agent: 'coder',
      task: taskDescription,
      provider: 'claude'
    });

    console.log(`Task spawned: ${response.data.taskId}`);
    return response.data.taskId;
  } catch (error) {
    console.error('Failed to spawn task:', error.message);
    throw error;
  }
}

// Usage
const taskId = await spawnCodingTask(
  'Implement OAuth2 authentication with Google and GitHub providers'
);
```

### Example: Python Integration

```python
client = AgenticFlowClient()

def spawn_research_task(topic):
    try:
        response = client.request('POST', '/v1/tasks', json={
            'agent': 'researcher',
            'task': f'Research and summarize: {topic}',
            'provider': 'gemini'
        })

        task_id = response['taskId']
        print(f"Task spawned: {task_id}")
        return task_id
    except requests.HTTPError as e:
        print(f"Failed to spawn task: {e}")
        raise

# Usage
task_id = spawn_research_task('Latest developments in quantum computing')
```

### Available Agent Types

| Agent | Description | Best For |
|-------|-------------|----------|
| `coder` | Code generation, refactoring, optimization | Implementation tasks |
| `researcher` | Information gathering, analysis, summarization | Research, documentation |
| `system-architect` | System design, architecture decisions | Design tasks |
| `reviewer` | Code review, quality assurance | Quality checks |
| `tester` | Test generation, validation | Testing workflows |
| `debugger` | Error analysis, troubleshooting | Bug fixing |

### Available Providers

| Provider | Models | Strengths |
|----------|--------|-----------|
| `gemini` | Gemini 1.5 Pro, Flash | Fast, cost-effective |
| `claude` | Claude Sonnet 4 | High quality, reasoning |
| `openrouter` | 100+ models | Flexibility, fallbacks |
| `openai` | GPT-4, GPT-3.5 | General purpose |

---

## Monitoring Task Status

### Endpoint

**`GET /v1/tasks/:taskId`**

### Response Schema

```json
{
  "taskId": "uuid",
  "agent": "string",
  "task": "string",
  "provider": "string",
  "status": "running|completed|failed|stopped",
  "startTime": "number",       // Unix timestamp (ms)
  "exitTime": "number|null",   // Unix timestamp (ms)
  "exitCode": "number|null",   // Process exit code
  "duration": "number",         // Duration in ms
  "logTail": "string",          // Last 50 lines of log
  "error": "string|undefined"   // Error message if failed
}
```

### Example: Poll Task Status

**cURL:**
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:9090/v1/tasks/a1b2c3d4-e5f6-7890-1234-567890abcdef
```

**Node.js:**
```javascript
async function pollTaskStatus(taskId, intervalMs = 5000) {
  return new Promise((resolve, reject) => {
    const poll = setInterval(async () => {
      try {
        const response = await client.get(`/v1/tasks/${taskId}`);
        const status = response.data;

        console.log(`Task ${taskId}: ${status.status} (${status.duration}ms)`);

        if (status.status !== 'running') {
          clearInterval(poll);

          if (status.status === 'completed') {
            resolve(status);
          } else {
            reject(new Error(`Task ${status.status}: ${status.error || 'Unknown error'}`));
          }
        }
      } catch (error) {
        clearInterval(poll);
        reject(error);
      }
    }, intervalMs);
  });
}

// Usage
try {
  const result = await pollTaskStatus(taskId);
  console.log('Task completed successfully');
  console.log('Log tail:', result.logTail);
} catch (error) {
  console.error('Task failed:', error.message);
}
```

**Python:**
```python
import time

def poll_task_status(task_id, interval=5):
    while True:
        try:
            response = client.request('GET', f'/v1/tasks/{task_id}')
            status = response['status']

            print(f"Task {task_id}: {status} ({response['duration']}ms)")

            if status != 'running':
                if status == 'completed':
                    return response
                else:
                    error = response.get('error', 'Unknown error')
                    raise Exception(f"Task {status}: {error}")

            time.sleep(interval)
        except requests.HTTPError as e:
            raise Exception(f"Failed to get task status: {e}")

# Usage
try:
    result = poll_task_status(task_id)
    print("Task completed successfully")
    print(f"Log tail: {result['logTail']}")
except Exception as e:
    print(f"Task failed: {e}")
```

---

## Real-Time Log Streaming

### Overview

Stream task logs in real-time using **Server-Sent Events (SSE)**. This provides live visibility into agent operations as they occur.

### Endpoint

**`GET /v1/tasks/:taskId/logs/stream`**

### Response Format

**Content-Type:** `text/event-stream`

**Event Types:**
- `data` - Log line with timestamp
- `task-complete` - Task finished (includes status and exit code)
- `error` - Error occurred

### Example: Node.js SSE Client

```javascript
const EventSource = require('eventsource');

function streamTaskLogs(taskId, onLog, onComplete, onError) {
  const url = `http://localhost:9090/v1/tasks/${taskId}/logs/stream`;
  const eventSource = new EventSource(url, {
    headers: {
      'Authorization': `Bearer ${process.env.MANAGEMENT_API_KEY}`
    }
  });

  eventSource.on('message', (event) => {
    const data = JSON.parse(event.data);
    onLog(data.line, data.timestamp);
  });

  eventSource.addEventListener('task-complete', (event) => {
    const data = JSON.parse(event.data);
    eventSource.close();
    onComplete(data);
  });

  eventSource.addEventListener('error', (event) => {
    const data = JSON.parse(event.data);
    eventSource.close();
    onError(data);
  });

  eventSource.onerror = (error) => {
    eventSource.close();
    onError(error);
  };

  return eventSource;
}

// Usage
const stream = streamTaskLogs(
  taskId,
  (line, timestamp) => {
    console.log(`[${new Date(timestamp).toISOString()}] ${line}`);
  },
  (result) => {
    console.log(`Task completed with status: ${result.status}`);
  },
  (error) => {
    console.error('Stream error:', error);
  }
);

// Stop streaming when needed
// stream.close();
```

### Example: Python SSE Client

```python
import sseclient
import requests
import json

def stream_task_logs(task_id, on_log, on_complete, on_error):
    url = f"http://localhost:9090/v1/tasks/{task_id}/logs/stream"
    headers = {
        'Authorization': f"Bearer {os.getenv('MANAGEMENT_API_KEY')}",
        'Accept': 'text/event-stream'
    }

    response = requests.get(url, headers=headers, stream=True)
    client = sseclient.SSEClient(response)

    try:
        for event in client.events():
            if event.event == 'message' or event.event == '':
                data = json.loads(event.data)
                on_log(data['line'], data['timestamp'])
            elif event.event == 'task-complete':
                data = json.loads(event.data)
                on_complete(data)
                break
            elif event.event == 'error':
                data = json.loads(event.data)
                on_error(data)
                break
    except Exception as e:
        on_error({'message': str(e)})
    finally:
        response.close()

# Usage
def handle_log(line, timestamp):
    print(f"[{timestamp}] {line}")

def handle_complete(result):
    print(f"Task completed: {result['status']}")

def handle_error(error):
    print(f"Stream error: {error['message']}")

stream_task_logs(task_id, handle_log, handle_complete, handle_error)
```

### Example: Browser JavaScript

```html
<!DOCTYPE html>
<html>
<head>
  <title>Task Log Viewer</title>
</head>
<body>
  <div id="logs"></div>
  <script>
    const taskId = 'a1b2c3d4-e5f6-7890-1234-567890abcdef';
    const token = 'your-token-here';

    const eventSource = new EventSource(
      `http://localhost:9090/v1/tasks/${taskId}/logs/stream`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const logDiv = document.getElementById('logs');
      logDiv.innerHTML += `<div>[${new Date(data.timestamp).toISOString()}] ${data.line}</div>`;
      logDiv.scrollTop = logDiv.scrollHeight;
    };

    eventSource.addEventListener('task-complete', (event) => {
      const data = JSON.parse(event.data);
      console.log('Task completed:', data);
      eventSource.close();
    });

    eventSource.onerror = (error) => {
      console.error('Stream error:', error);
      eventSource.close();
    };
  </script>
</body>
</html>
```

---

## Stopping Running Tasks

### Endpoint

**`DELETE /v1/tasks/:taskId`**

### Behaviour

1. Sends `SIGTERM` to the task process
2. Updates task status to `stopped`
3. If process doesn't exit within 10 seconds, sends `SIGKILL`

### Response Schema

**Status:** `200 OK`

```json
{
  "taskId": "uuid",
  "status": "stopped",
  "message": "Task stop signal sent successfully"
}
```

**Status:** `404 Not Found` (if task doesn't exist)

**Status:** `409 Conflict` (if task already stopped)

### Example: Stop a Task

**cURL:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer your-token" \
  http://localhost:9090/v1/tasks/a1b2c3d4-e5f6-7890-1234-567890abcdef
```

**Node.js:**
```javascript
async function stopTask(taskId) {
  try {
    const response = await client.delete(`/v1/tasks/${taskId}`);
    console.log(`Task ${taskId} stopped:`, response.data.message);
    return true;
  } catch (error) {
    if (error.response && error.response.status === 404) {
      console.error('Task not found');
    } else if (error.response && error.response.status === 409) {
      console.error('Task already stopped');
    } else {
      console.error('Failed to stop task:', error.message);
    }
    return false;
  }
}

// Usage
await stopTask(taskId);
```

**Python:**
```python
def stop_task(task_id):
    try:
        response = client.request('DELETE', f'/v1/tasks/{task_id}')
        print(f"Task {task_id} stopped: {response['message']}")
        return True
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print("Task not found")
        elif e.response.status_code == 409:
            print("Task already stopped")
        else:
            print(f"Failed to stop task: {e}")
        return False

# Usage
stop_task(task_id)
```

---

## Listing Active Tasks

### Endpoint

**`GET /v1/tasks`**

### Response Schema

```json
{
  "activeTasks": [
    {
      "taskId": "uuid",
      "agent": "string",
      "startTime": "number",
      "duration": "number"
    }
  ],
  "count": "number"
}
```

### Example

**cURL:**
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:9090/v1/tasks
```

**Response:**
```json
{
  "activeTasks": [
    {
      "taskId": "task-1",
      "agent": "coder",
      "startTime": 1704110400000,
      "duration": 125000
    },
    {
      "taskId": "task-2",
      "agent": "researcher",
      "startTime": 1704110500000,
      "duration": 25000
    }
  ],
  "count": 2
}
```

---

## Complete Integration Example

### Node.js Task Manager

```javascript
const client = require('./agentic-flow-client');
const EventSource = require('eventsource');

class TaskManager {
  constructor() {
    this.activeTasks = new Map();
  }

  async spawn(agent, task, provider = 'gemini') {
    const response = await client.post('/v1/tasks', {
      agent,
      task,
      provider
    });

    const taskId = response.data.taskId;
    this.activeTasks.set(taskId, {
      agent,
      task,
      status: 'running',
      startTime: Date.now()
    });

    return taskId;
  }

  async getStatus(taskId) {
    const response = await client.get(`/v1/tasks/${taskId}`);
    return response.data;
  }

  streamLogs(taskId, callback) {
    const url = `http://localhost:9090/v1/tasks/${taskId}/logs/stream`;
    const stream = new EventSource(url, {
      headers: {
        'Authorization': `Bearer ${process.env.MANAGEMENT_API_KEY}`
      }
    });

    stream.on('message', (event) => {
      const data = JSON.parse(event.data);
      callback('log', data);
    });

    stream.addEventListener('task-complete', (event) => {
      const data = JSON.parse(event.data);
      callback('complete', data);
      stream.close();
    });

    return stream;
  }

  async stop(taskId) {
    await client.delete(`/v1/tasks/${taskId}`);
    this.activeTasks.delete(taskId);
  }

  async waitForCompletion(taskId, timeoutMs = 300000) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const status = await this.getStatus(taskId);

      if (status.status !== 'running') {
        this.activeTasks.delete(taskId);

        if (status.status === 'completed') {
          return status;
        } else {
          throw new Error(`Task ${status.status}: ${status.error || 'Unknown error'}`);
        }
      }

      await new Promise(resolve => setTimeout(resolve, 5000));
    }

    throw new Error('Task timeout');
  }
}

// Usage
const manager = new TaskManager();

async function main() {
  // Spawn task
  const taskId = await manager.spawn(
    'coder',
    'Implement user authentication with JWT'
  );

  // Stream logs
  const stream = manager.streamLogs(taskId, (event, data) => {
    if (event === 'log') {
      console.log(data.line);
    } else if (event === 'complete') {
      console.log('Task completed:', data.status);
    }
  });

  // Wait for completion
  try {
    const result = await manager.waitForCompletion(taskId);
    console.log('Success:', result);
  } catch (error) {
    console.error('Failed:', error.message);
  }
}

main();
```

---

## Error Handling

### Common Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `401` | Unauthorized | Check API key in Authorization header |
| `404` | Task not found | Verify task ID is correct |
| `409` | Conflict | Task already in terminal state |
| `429` | Rate limited | Reduce request frequency |
| `500` | Internal error | Check logs, retry with backoff |

### Example: Robust Error Handling

```javascript
async function robustTaskExecution(agent, task) {
  const maxRetries = 3;
  let attempt = 0;

  while (attempt < maxRetries) {
    try {
      // Spawn task
      const taskId = await manager.spawn(agent, task);

      // Wait for completion
      const result = await manager.waitForCompletion(taskId);

      return result;
    } catch (error) {
      attempt++;

      if (error.response && error.response.status === 429) {
        // Rate limited - exponential backoff
        const delay = Math.pow(2, attempt) * 1000;
        console.log(`Rate limited. Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      } else if (attempt >= maxRetries) {
        throw error;
      } else {
        console.log(`Attempt ${attempt} failed. Retrying...`);
      }
    }
  }

  throw new Error('Max retries exceeded');
}
```

---

## Next Steps

- [Telemetry and Monitoring Guide](telemetry-monitoring.md) - System-wide metrics and observability
- [API Reference](../api/management-api.md) - Complete API documentation
- [Deployment Guide](deployment.md) - Production deployment strategies

---

**Questions or Issues?**
- [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)
- [API Documentation](http://localhost:9090/docs)
