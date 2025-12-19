---
title: ComfyUI MCP Server Integration with Management API
description: The ComfyUI MCP server will integrate with the existing Management API (port 9090) to provide unified task management, metrics collection, and real-time status updates for image/video generation wo...
category: reference
tags:
  - architecture
  - design
  - structure
  - api
  - api
related-docs:
  - ARCHITECTURE_COMPLETE.md
  - ARCHITECTURE_OVERVIEW.md
  - ASCII_DEPRECATION_COMPLETE.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# ComfyUI MCP Server Integration with Management API

## Architecture Overview

The ComfyUI MCP server will integrate with the existing Management API (port 9090) to provide unified task management, metrics collection, and real-time status updates for image/video generation workflows.

## Integration Components

### 1. API Routes (`/home/devuser/workspace/project/multi-agent-docker/multi-agent-docker/management-api/routes/comfyui.js`)

New Fastify routes for ComfyUI operations:

```javascript
/**
 * ComfyUI workflow management routes
 * Integrates with existing Management API architecture
 */

async function comfyuiRoutes(fastify, options) {
  const { logger, metrics, comfyuiManager } = options;

  // Submit workflow for execution
  fastify.post('/v1/comfyui/workflow', {
    schema: {
      description: 'Submit a ComfyUI workflow for execution',
      tags: ['comfyui'],
      body: {
        type: 'object',
        required: ['workflow'],
        properties: {
          workflow: {
            type: 'object',
            description: 'ComfyUI workflow JSON'
          },
          priority: {
            type: 'string',
            enum: ['low', 'normal', 'high'],
            default: 'normal'
          },
          gpu: {
            type: 'string',
            enum: ['local', 'salad'],
            default: 'local'
          }
        }
      },
      response: {
        202: {
          type: 'object',
          properties: {
            workflowId: { type: 'string' },
            status: { type: 'string' },
            queuePosition: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { workflow, priority, gpu } = request.body;

    logger.info({ priority, gpu }, 'Submitting ComfyUI workflow');

    try {
      const result = await comfyuiManager.submitWorkflow(workflow, { priority, gpu });

      reply.code(202).send({
        workflowId: result.workflowId,
        status: 'queued',
        queuePosition: result.queuePosition
      });
    } catch (error) {
      logger.error({ error: error.message }, 'Failed to submit workflow');
      reply.code(500).send({
        error: 'Internal Server Error',
        message: error.message
      });
    }
  });

  // Get workflow status
  fastify.get('/v1/comfyui/workflow/:workflowId', {
    schema: {
      description: 'Get workflow execution status',
      tags: ['comfyui'],
      params: {
        type: 'object',
        properties: {
          workflowId: { type: 'string' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            workflowId: { type: 'string' },
            status: { type: 'string' },
            progress: { type: 'number' },
            currentNode: { type: 'string' },
            startTime: { type: 'number' },
            completionTime: { type: ['number', 'null'] },
            outputs: { type: 'array' },
            error: { type: ['string', 'null'] }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { workflowId } = request.params;

    const status = await comfyuiManager.getWorkflowStatus(workflowId);

    if (!status) {
      return reply.code(404).send({
        error: 'Not Found',
        message: `Workflow ${workflowId} not found`
      });
    }

    reply.send(status);
  });

  // List available models
  fastify.get('/v1/comfyui/models', {
    schema: {
      description: 'List available ComfyUI models',
      tags: ['comfyui'],
      querystring: {
        type: 'object',
        properties: {
          type: {
            type: 'string',
            enum: ['checkpoints', 'loras', 'vae', 'controlnet', 'upscale']
          }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            models: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  type: { type: 'string' },
                  size: { type: 'number' },
                  hash: { type: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { type } = request.query;

    const models = await comfyuiManager.listModels(type);

    reply.send({ models });
  });

  // List workflow outputs
  fastify.get('/v1/comfyui/outputs', {
    schema: {
      description: 'List generated outputs',
      tags: ['comfyui'],
      querystring: {
        type: 'object',
        properties: {
          workflowId: { type: 'string' },
          limit: { type: 'number', default: 50 }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            outputs: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  filename: { type: 'string' },
                  workflowId: { type: 'string' },
                  type: { type: 'string' },
                  size: { type: 'number' },
                  createdAt: { type: 'number' },
                  url: { type: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { workflowId, limit } = request.query;

    const outputs = await comfyuiManager.listOutputs({ workflowId, limit });

    reply.send({ outputs });
  });

  // Cancel workflow
  fastify.delete('/v1/comfyui/workflow/:workflowId', {
    schema: {
      description: 'Cancel a running or queued workflow',
      tags: ['comfyui'],
      params: {
        type: 'object',
        properties: {
          workflowId: { type: 'string' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            workflowId: { type: 'string' },
            status: { type: 'string' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { workflowId } = request.params;

    logger.info({ workflowId }, 'Cancelling workflow');

    const success = await comfyuiManager.cancelWorkflow(workflowId);

    if (!success) {
      const status = await comfyuiManager.getWorkflowStatus(workflowId);

      if (!status) {
        return reply.code(404).send({
          error: 'Not Found',
          message: `Workflow ${workflowId} not found`
        });
      }

      return reply.code(409).send({
        error: 'Conflict',
        message: 'Workflow cannot be cancelled',
        currentStatus: status.status
      });
    }

    reply.send({
      workflowId,
      status: 'cancelled'
    });
  });

  // WebSocket for real-time updates
  fastify.get('/v1/comfyui/stream', { websocket: true }, (connection, request) => {
    const clientId = Date.now().toString();
    logger.info({ clientId }, 'WebSocket client connected');

    // Subscribe to workflow events
    const unsubscribe = comfyuiManager.subscribe((event) => {
      try {
        connection.socket.send(JSON.stringify(event));
      } catch (error) {
        logger.error({ error: error.message }, 'Failed to send WebSocket message');
      }
    });

    connection.socket.on('message', (message) => {
      try {
        const data = JSON.parse(message.toString());

        // Handle ping/pong
        if (data.type === 'ping') {
          connection.socket.send(JSON.stringify({ type: 'pong' }));
        }

        // Handle workflow subscription
        if (data.type === 'subscribe' && data.workflowId) {
          comfyuiManager.subscribeToWorkflow(data.workflowId, clientId);
        }

        if (data.type === 'unsubscribe' && data.workflowId) {
          comfyuiManager.unsubscribeFromWorkflow(data.workflowId, clientId);
        }
      } catch (error) {
        logger.error({ error: error.message }, 'Failed to parse WebSocket message');
      }
    });

    connection.socket.on('close', () => {
      logger.info({ clientId }, 'WebSocket client disconnected');
      unsubscribe();
    });
  });
}

module.exports = comfyuiRoutes;
```

### 2. ComfyUI Manager (`/home/devuser/workspace/project/multi-agent-docker/multi-agent-docker/management-api/utils/comfyui-manager.js`)

Manages ComfyUI workflow lifecycle:

```javascript
/**
 * ComfyUI Workflow Manager
 * Manages workflow submission, execution tracking, and event broadcasting
 */

const { v4: uuidv4 } = require('uuid');
const EventEmitter = require('events');
const fs = require('fs');
const path = require('path');

class ComfyUIManager extends EventEmitter {
  constructor(logger, metrics) {
    super();
    this.logger = logger;
    this.metrics = metrics;
    this.workflows = new Map(); // workflowId -> workflow info
    this.queue = [];
    this.subscribers = new Map(); // workflowId -> Set of clientIds
    this.outputsDir = process.env.COMFYUI_OUTPUTS || '/home/devuser/comfyui/output';

    // Ensure output directory exists
    if (!fs.existsSync(this.outputsDir)) {
      fs.mkdirSync(this.outputsDir, { recursive: true });
    }
  }

  /**
   * Submit workflow for execution
   */
  async submitWorkflow(workflow, options = {}) {
    const workflowId = uuidv4();
    const { priority = 'normal', gpu = 'local' } = options;

    const workflowInfo = {
      workflowId,
      workflow,
      priority,
      gpu,
      status: 'queued',
      progress: 0,
      currentNode: null,
      startTime: null,
      completionTime: null,
      outputs: [],
      error: null,
      queuePosition: this.queue.length
    };

    this.workflows.set(workflowId, workflowInfo);

    // Add to queue based on priority
    if (priority === 'high') {
      this.queue.unshift(workflowId);
    } else {
      this.queue.push(workflowId);
    }

    this.logger.info({ workflowId, priority, gpu }, 'Workflow queued');
    this.emit('workflow:queued', workflowInfo);
    this.metrics.recordComfyUIWorkflow('queued');

    // Process queue
    this._processQueue();

    return {
      workflowId,
      queuePosition: workflowInfo.queuePosition
    };
  }

  /**
   * Get workflow status
   */
  async getWorkflowStatus(workflowId) {
    return this.workflows.get(workflowId) || null;
  }

  /**
   * Cancel workflow
   */
  async cancelWorkflow(workflowId) {
    const workflowInfo = this.workflows.get(workflowId);

    if (!workflowInfo) {
      return false;
    }

    if (workflowInfo.status === 'completed' || workflowInfo.status === 'failed') {
      return false;
    }

    workflowInfo.status = 'cancelled';
    workflowInfo.completionTime = Date.now();

    // Remove from queue if queued
    const queueIndex = this.queue.indexOf(workflowId);
    if (queueIndex !== -1) {
      this.queue.splice(queueIndex, 1);
    }

    this.logger.info({ workflowId }, 'Workflow cancelled');
    this.emit('workflow:cancelled', workflowInfo);
    this.metrics.recordComfyUIWorkflow('cancelled');

    return true;
  }

  /**
   * List available models
   */
  async listModels(type) {
    // This would integrate with actual ComfyUI model directory scanning
    // For now, return mock structure
    const modelTypes = {
      checkpoints: '/home/devuser/comfyui/models/checkpoints',
      loras: '/home/devuser/comfyui/models/loras',
      vae: '/home/devuser/comfyui/models/vae',
      controlnet: '/home/devuser/comfyui/models/controlnet',
      upscale: '/home/devuser/comfyui/models/upscale_models'
    };

    const modelsDir = type ? modelTypes[type] : null;
    const models = [];

    if (modelsDir && fs.existsSync(modelsDir)) {
      const files = fs.readdirSync(modelsDir);

      for (const file of files) {
        const fullPath = path.join(modelsDir, file);
        const stats = fs.statSync(fullPath);

        if (stats.isFile()) {
          models.push({
            name: file,
            type: type || 'unknown',
            size: stats.size,
            hash: null // Could add hash calculation
          });
        }
      }
    }

    return models;
  }

  /**
   * List outputs
   */
  async listOutputs(options = {}) {
    const { workflowId, limit = 50 } = options;
    const outputs = [];

    if (!fs.existsSync(this.outputsDir)) {
      return outputs;
    }

    const files = fs.readdirSync(this.outputsDir);

    for (const file of files.slice(0, limit)) {
      const fullPath = path.join(this.outputsDir, file);
      const stats = fs.statSync(fullPath);

      if (stats.isFile()) {
        // Extract workflow ID from filename if present
        const fileWorkflowId = file.match(/^([a-f0-9-]+)_/)?.[1];

        if (!workflowId || fileWorkflowId === workflowId) {
          outputs.push({
            filename: file,
            workflowId: fileWorkflowId || 'unknown',
            type: path.extname(file).slice(1),
            size: stats.size,
            createdAt: stats.mtimeMs,
            url: `/v1/comfyui/output/${file}`
          });
        }
      }
    }

    return outputs.sort((a, b) => b.createdAt - a.createdAt);
  }

  /**
   * Subscribe to all workflow events
   */
  subscribe(callback) {
    this.on('workflow:*', callback);

    return () => {
      this.off('workflow:*', callback);
    };
  }

  /**
   * Subscribe to specific workflow
   */
  subscribeToWorkflow(workflowId, clientId) {
    if (!this.subscribers.has(workflowId)) {
      this.subscribers.set(workflowId, new Set());
    }

    this.subscribers.get(workflowId).add(clientId);
  }

  /**
   * Unsubscribe from specific workflow
   */
  unsubscribeFromWorkflow(workflowId, clientId) {
    const subs = this.subscribers.get(workflowId);
    if (subs) {
      subs.delete(clientId);
    }
  }

  /**
   * Process workflow queue
   */
  async _processQueue() {
    // This would integrate with actual ComfyUI execution
    // For now, simulate execution
    if (this.queue.length === 0) {
      return;
    }

    const workflowId = this.queue.shift();
    const workflowInfo = this.workflows.get(workflowId);

    if (!workflowInfo || workflowInfo.status !== 'queued') {
      return;
    }

    workflowInfo.status = 'running';
    workflowInfo.startTime = Date.now();

    this.logger.info({ workflowId }, 'Workflow started');
    this.emit('workflow:started', workflowInfo);
    this.metrics.recordComfyUIWorkflow('started');

    // Simulate progress updates
    // In real implementation, this would listen to ComfyUI API events
    this._simulateProgress(workflowId);
  }

  /**
   * Simulate workflow progress (replace with actual ComfyUI integration)
   */
  _simulateProgress(workflowId) {
    const workflowInfo = this.workflows.get(workflowId);
    if (!workflowInfo) return;

    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      workflowInfo.progress = progress;
      workflowInfo.currentNode = `node_${Math.floor(progress / 10)}`;

      this.emit('workflow:progress', {
        workflowId,
        progress,
        currentNode: workflowInfo.currentNode
      });

      if (progress >= 100) {
        clearInterval(interval);
        this._completeWorkflow(workflowId);
      }
    }, 1000);
  }

  /**
   * Complete workflow
   */
  _completeWorkflow(workflowId) {
    const workflowInfo = this.workflows.get(workflowId);
    if (!workflowInfo) return;

    workflowInfo.status = 'completed';
    workflowInfo.progress = 100;
    workflowInfo.completionTime = Date.now();

    const duration = (workflowInfo.completionTime - workflowInfo.startTime) / 1000;

    this.logger.info({ workflowId, duration }, 'Workflow completed');
    this.emit('workflow:completed', workflowInfo);
    this.metrics.recordComfyUIWorkflow('completed', duration);

    // Process next in queue
    this._processQueue();
  }
}

module.exports = ComfyUIManager;
```

### 3. Metrics Extensions (`/home/devuser/workspace/project/multi-agent-docker/multi-agent-docker/management-api/utils/metrics.js`)

Add to existing metrics.js:

```javascript
// Add these metrics to the existing file

const comfyuiWorkflowTotal = new client.Counter({
  name: 'comfyui_workflow_total',
  help: 'Total number of ComfyUI workflows',
  labelNames: ['status']
});

const comfyuiWorkflowDuration = new client.Histogram({
  name: 'comfyui_workflow_duration_seconds',
  help: 'Duration of ComfyUI workflow execution',
  labelNames: ['gpu_type'],
  buckets: [1, 5, 10, 30, 60, 120, 300, 600]
});

const comfyuiWorkflowErrors = new client.Counter({
  name: 'comfyui_workflow_errors_total',
  help: 'Total number of ComfyUI workflow errors',
  labelNames: ['error_type']
});

const comfyuiGpuUtilization = new client.Gauge({
  name: 'comfyui_gpu_utilization',
  help: 'GPU utilization percentage for ComfyUI',
  labelNames: ['gpu_id']
});

const comfyuiVramUsage = new client.Gauge({
  name: 'comfyui_vram_usage_bytes',
  help: 'VRAM usage in bytes for ComfyUI',
  labelNames: ['gpu_id']
});

const comfyuiQueueLength = new client.Gauge({
  name: 'comfyui_queue_length',
  help: 'Number of workflows in queue'
});

// Register metrics
register.registerMetric(comfyuiWorkflowTotal);
register.registerMetric(comfyuiWorkflowDuration);
register.registerMetric(comfyuiWorkflowErrors);
register.registerMetric(comfyuiGpuUtilization);
register.registerMetric(comfyuiVramUsage);
register.registerMetric(comfyuiQueueLength);

// Helper function
function recordComfyUIWorkflow(status, duration) {
  comfyuiWorkflowTotal.inc({ status });
  if (duration !== undefined) {
    comfyuiWorkflowDuration.observe({ gpu_type: 'local' }, duration);
  }
}

function setComfyUIGpuMetrics(gpuId, utilization, vramUsage) {
  comfyuiGpuUtilization.set({ gpu_id: gpuId }, utilization);
  comfyuiVramUsage.set({ gpu_id: gpuId }, vramUsage);
}

function setComfyUIQueueLength(length) {
  comfyuiQueueLength.set(length);
}

// Export additions
module.exports = {
  // ... existing exports
  recordComfyUIWorkflow,
  setComfyUIGpuMetrics,
  setComfyUIQueueLength
};
```

### 4. Server Integration (`/home/devuser/workspace/project/multi-agent-docker/multi-agent-docker/management-api/server.js`)

Add to existing server.js:

```javascript
// After existing managers
const ComfyUIManager = require('./utils/comfyui-manager');
const comfyuiManager = new ComfyUIManager(logger, metrics);

// Register WebSocket support
app.register(require('@fastify/websocket'));

// Register ComfyUI routes
app.register(require('./routes/comfyui'), {
  prefix: '',
  comfyuiManager,
  logger,
  metrics
});

// Update root endpoint to include ComfyUI
const rootEndpoints = {
  // ... existing endpoints
  comfyui: {
    submitWorkflow: 'POST /v1/comfyui/workflow',
    getStatus: 'GET /v1/comfyui/workflow/:id',
    listModels: 'GET /v1/comfyui/models',
    listOutputs: 'GET /v1/comfyui/outputs',
    stream: 'WS /v1/comfyui/stream'
  }
};
```

### 5. Service Registration Pattern

ComfyUI MCP server registers with Management API on startup:

```javascript
// In ComfyUI MCP server startup
async function registerWithManagementAPI() {
  const config = {
    service: 'comfyui-mcp',
    version: '1.0.0',
    endpoints: {
      health: 'http://localhost:8188/health',
      api: 'http://localhost:8188'
    },
    capabilities: ['text2img', 'img2img', 'video', 'upscale']
  };

  try {
    await fetch('http://localhost:9090/v1/services/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.MANAGEMENT_API_KEY
      },
      body: JSON.stringify(config)
    });

    console.log('Registered with Management API');
  } catch (error) {
    console.error('Failed to register with Management API:', error);
  }
}
```

## WebSocket Protocol

### Client → Server Messages

```json
// Subscribe to workflow updates
{
  "type": "subscribe",
  "workflowId": "uuid"
}

// Unsubscribe from workflow
{
  "type": "unsubscribe",
  "workflowId": "uuid"
}

// Ping for keepalive
{
  "type": "ping"
}
```

### Server → Client Messages

```json
// Workflow queued
{
  "type": "workflow:queued",
  "workflowId": "uuid",
  "queuePosition": 2
}

// Workflow started
{
  "type": "workflow:started",
  "workflowId": "uuid",
  "timestamp": 1234567890
}

// Progress update
{
  "type": "workflow:progress",
  "workflowId": "uuid",
  "progress": 45,
  "currentNode": "KSampler",
  "timestamp": 1234567890
}

// Workflow completed
{
  "type": "workflow:completed",
  "workflowId": "uuid",
  "outputs": [
    {
      "filename": "output_00001.png",
      "type": "image",
      "url": "/v1/comfyui/output/output_00001.png"
    }
  ],
  "timestamp": 1234567890
}

// Error
{
  "type": "workflow:error",
  "workflowId": "uuid",
  "error": "Out of memory",
  "timestamp": 1234567890
}
```

## Prometheus Metrics

### Exposed Metrics

```
# Workflow metrics
comfyui_workflow_total{status="queued"} 150
comfyui_workflow_total{status="completed"} 142
comfyui_workflow_total{status="failed"} 5
comfyui_workflow_total{status="cancelled"} 3

# Duration histogram
comfyui_workflow_duration_seconds_bucket{gpu_type="local",le="10"} 45
comfyui_workflow_duration_seconds_bucket{gpu_type="local",le="30"} 89
comfyui_workflow_duration_seconds_sum{gpu_type="local"} 3456.78
comfyui_workflow_duration_seconds_count{gpu_type="local"} 142

# Error tracking
comfyui_workflow_errors_total{error_type="oom"} 3
comfyui_workflow_errors_total{error_type="timeout"} 2

# Resource metrics
comfyui_gpu_utilization{gpu_id="0"} 87.5
comfyui_vram_usage_bytes{gpu_id="0"} 8589934592

# Queue metrics
comfyui_queue_length 5
```

## Dependencies to Add

Add to `management-api/package.json`:

```json
{
  "dependencies": {
    "@fastify/websocket": "^10.0.0"
  }
}
```

## Environment Variables

Add to `.env`:

```bash
COMFYUI_API_URL=http://localhost:8188
COMFYUI_OUTPUTS=/home/devuser/comfyui/output
COMFYUI_MODELS=/home/devuser/comfyui/models
```

## Integration Flow

1. **Startup**:
   - Management API starts on port 9090
   - ComfyUI MCP server starts on port 8188
   - ComfyUI registers with Management API

2. **Workflow Submission**:
   - Client sends POST to `/v1/comfyui/workflow`
   - ComfyUIManager queues workflow
   - Returns workflow ID and queue position

3. **Execution**:
   - ComfyUIManager processes queue
   - Sends workflow to ComfyUI backend
   - Tracks progress via ComfyUI API

4. **Real-time Updates**:
   - WebSocket clients receive progress events
   - Metrics updated in real-time
   - Prometheus scrapes `/metrics` endpoint

5. **Completion**:
   - Workflow completes or fails
   - Final status broadcast via WebSocket
   - Outputs available via `/v1/comfyui/outputs`

## File Structure

```
management-api/
├── server.js                    # Main server (update)
├── package.json                 # Add @fastify/websocket
├── routes/
│   ├── tasks.js                 # Existing
│   ├── status.js                # Existing
│   └── comfyui.js              # NEW - ComfyUI routes
├── utils/
│   ├── logger.js                # Existing
│   ├── metrics.js               # Update with ComfyUI metrics
│   ├── process-manager.js       # Existing
│   ├── system-monitor.js        # Existing
│   └── comfyui-manager.js      # NEW - ComfyUI workflow manager
└── middleware/
    └── auth.js                  # Existing
```

## Testing

```bash
# Submit workflow
curl -X POST http://localhost:9090/v1/comfyui/workflow \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-this-secret-key" \
  -d '{"workflow": {...}}'

# Get status
curl http://localhost:9090/v1/comfyui/workflow/uuid \
  -H "X-API-Key: change-this-secret-key"

# WebSocket connection
wscat -c ws://localhost:9090/v1/comfyui/stream \
  -H "X-API-Key: change-this-secret-key"

# Metrics
curl http://localhost:9090/metrics
```

---

---

## Related Documentation

- [ComfyUI Management API Integration - Summary](comfyui-management-api-integration-summary.md)
- [REST API Architecture Documentation](diagrams/server/api/rest-api-architecture.md)
- [VisionFlow Client Architecture - Deep Analysis](archive/analysis/client-architecture-analysis-2025-12.md)
- [VisionFlow Testing Infrastructure Architecture](diagrams/infrastructure/testing/test-architecture.md)
- [VisionFlow Architecture Cross-Reference Matrix](diagrams/cross-reference-matrix.md)

## Future Enhancements

1. **Salad Cloud GPU Integration**: Automatic failover to Salad when local GPU busy
2. **Priority Queue**: Advanced scheduling based on workflow complexity
3. **Batch Processing**: Group similar workflows for efficiency
4. **Model Management**: Automatic model download and caching
5. **Cost Tracking**: Track GPU costs per workflow
