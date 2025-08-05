# Docker Integration Guide for MCP Observability

This guide explains how to integrate the MCP Observability Server into your Docker-based agent project.

## Overview

The MCP Observability Server provides real-time monitoring and control for bot swarms using spring-physics visualization. This guide covers integration into a Docker container environment.

## Integration Architecture

```
┌─────────────────────────────────────────┐
│          Host Docker Container           │
├─────────────────────────────────────────┤
│  ┌───────────────────────────────────┐  │
│  │      Your Agent Application       │  │
│  └────────────┬──────────────────────┘  │
│               │ stdio                    │
│  ┌────────────▼──────────────────────┐  │
│  │    MCP Observability Server       │  │
│  │  • Agent Management               │  │
│  │  • Spring Physics Engine          │  │
│  │  • Message Flow Tracking          │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Step 1: Dockerfile Configuration

Add the MCP Observability Server to your Dockerfile:

```dockerfile
# Multi-stage build for MCP Observability
FROM node:20-alpine AS mcp-builder

# Copy MCP server files
COPY mcp-observability /app/mcp-observability
WORKDIR /app/mcp-observability

# Install dependencies
RUN npm ci --only=production

# Main application stage
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    supervisor \
    bash

# Copy MCP server from builder
COPY --from=mcp-builder /app/mcp-observability /opt/mcp-observability

# Copy your application files
COPY . /app
WORKDIR /app

# Set environment variables
ENV MCP_OBSERVABILITY_PATH=/opt/mcp-observability
ENV MCP_PHYSICS_UPDATE_RATE=60
ENV MCP_MAX_AGENTS=1000
ENV NODE_ENV=production

# Expose ports if needed
EXPOSE 3000 8080

# Start script
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
```

## Step 2: Docker Entrypoint Script

Create `docker-entrypoint.sh`:

```bash
#!/bin/bash
set -e

# Start MCP Observability Server in background
echo "Starting MCP Observability Server..."
cd /opt/mcp-observability
node src/index.js &
MCP_PID=$!

# Wait for MCP server to be ready
sleep 2

# Start your main application
echo "Starting main application..."
cd /app
exec node your-app.js

# Cleanup on exit
trap "kill $MCP_PID" EXIT
```

## Step 3: Docker Compose Configuration

For multi-container setups, use Docker Compose:

```yaml
version: '3.8'

services:
  agent-system:
    build: .
    environment:
      - MCP_OBSERVABILITY_PATH=/opt/mcp-observability
      - MCP_PHYSICS_UPDATE_RATE=60
      - MCP_MAX_AGENTS=1000
      - MCP_MEMORY_LIMIT=512M
    volumes:
      - agent-data:/app/data
      - mcp-memory:/opt/mcp-observability/memory
    networks:
      - agent-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "node", "/opt/mcp-observability/healthcheck.js"]
      interval: 30s
      timeout: 10s
      retries: 3

  visualization:
    image: visionflow:latest
    ports:
      - "3000:3000"
    environment:
      - MCP_SERVER_URL=ws://agent-system:8080
    networks:
      - agent-network
    depends_on:
      - agent-system

volumes:
  agent-data:
  mcp-memory:

networks:
  agent-network:
    driver: bridge
```

## Step 4: Environment Variables

Configure the MCP server using environment variables:

```bash
# Core Configuration
MCP_OBSERVABILITY_PATH=/opt/mcp-observability
MCP_LOG_LEVEL=info

# Physics Configuration
MCP_PHYSICS_UPDATE_RATE=60           # FPS for physics updates
MCP_SPRING_STRENGTH=0.1              # Spring force strength
MCP_DAMPING=0.95                     # Velocity damping
MCP_LINK_DISTANCE=8.0                # Ideal link distance
MCP_NODE_REPULSION=500.0             # Node repulsion force
MCP_MAX_VELOCITY=2.0                 # Maximum node velocity

# Performance Configuration
MCP_MAX_AGENTS=1000                  # Maximum supported agents
MCP_MEMORY_LIMIT=512M                # Memory limit for MCP server
MCP_MESSAGE_HISTORY_SIZE=1000        # Message history buffer size

# Memory Persistence
MCP_MEMORY_PERSIST=true              # Enable memory persistence
MCP_MEMORY_PATH=/opt/mcp-observability/memory
MCP_MEMORY_SYNC_INTERVAL=300         # Sync interval in seconds
```

## Step 5: Supervisor Configuration

For production deployments, use Supervisor to manage processes:

Create `/etc/supervisor/conf.d/mcp-observability.conf`:

```ini
[program:mcp-observability]
command=node /opt/mcp-observability/src/index.js
directory=/opt/mcp-observability
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/mcp-observability.err.log
stdout_logfile=/var/log/mcp-observability.out.log
environment=NODE_ENV="production",MCP_LOG_LEVEL="info"
user=node

[program:agent-application]
command=node /app/your-app.js
directory=/app
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/agent-app.err.log
stdout_logfile=/var/log/agent-app.out.log
environment=NODE_ENV="production"
user=node

[group:agent-system]
programs=mcp-observability,agent-application
```

## Step 6: Integration Code

### Node.js Integration Example

```javascript
// mcp-client.js
import { spawn } from 'child_process';

class MCPObservabilityClient {
  constructor() {
    this.mcpProcess = null;
    this.messageQueue = [];
    this.ready = false;
  }

  async connect() {
    const mcpPath = process.env.MCP_OBSERVABILITY_PATH || '/opt/mcp-observability';
    
    this.mcpProcess = spawn('node', [`${mcpPath}/src/index.js`], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    this.mcpProcess.stdout.on('data', (data) => {
      this.handleResponse(data.toString());
    });

    this.mcpProcess.stderr.on('data', (data) => {
      console.error('MCP Error:', data.toString());
    });

    // Initialize connection
    await this.sendRequest('initialize', {
      protocolVersion: '0.1.0',
      clientInfo: {
        name: 'docker-agent',
        version: '1.0.0'
      }
    });

    this.ready = true;
  }

  async sendRequest(method, params) {
    const request = {
      jsonrpc: '2.0',
      id: Date.now().toString(),
      method,
      params
    };

    return new Promise((resolve, reject) => {
      this.messageQueue.push({ id: request.id, resolve, reject });
      this.mcpProcess.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  handleResponse(data) {
    try {
      const lines = data.split('\n').filter(line => line.trim());
      
      lines.forEach(line => {
        const response = JSON.parse(line);
        const pending = this.messageQueue.find(m => m.id === response.id);
        
        if (pending) {
          if (response.error) {
            pending.reject(new Error(response.error.message));
          } else {
            pending.resolve(response.result);
          }
          
          this.messageQueue = this.messageQueue.filter(m => m.id !== response.id);
        }
      });
    } catch (error) {
      console.error('Failed to parse MCP response:', error);
    }
  }

  // Tool wrappers
  async createAgent(params) {
    return this.sendRequest('tools/call', {
      name: 'agent.create',
      arguments: params
    });
  }

  async initializeSwarm(params) {
    return this.sendRequest('tools/call', {
      name: 'swarm.initialize',
      arguments: params
    });
  }

  async sendMessage(params) {
    return this.sendRequest('tools/call', {
      name: 'message.send',
      arguments: params
    });
  }

  async getVisualizationSnapshot() {
    return this.sendRequest('tools/call', {
      name: 'visualization.snapshot',
      arguments: {
        includePositions: true,
        includeVelocities: true,
        includeConnections: true
      }
    });
  }
}

// Usage
const mcp = new MCPObservabilityClient();
await mcp.connect();

// Initialize swarm
await mcp.initializeSwarm({
  topology: 'hierarchical',
  agentConfig: {
    coordinatorCount: 1,
    workerTypes: [
      { type: 'coder', count: 5 },
      { type: 'tester', count: 3 }
    ]
  }
});
```

## Step 7: Health Checks

Create `/opt/mcp-observability/healthcheck.js`:

```javascript
#!/usr/bin/env node

const net = require('net');
const { spawn } = require('child_process');

async function checkMCPHealth() {
  return new Promise((resolve) => {
    const child = spawn('node', ['/opt/mcp-observability/src/index.js'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let timeout = setTimeout(() => {
      child.kill();
      resolve(false);
    }, 5000);

    child.stdout.on('data', (data) => {
      if (data.toString().includes('MCP Observability Server started')) {
        clearTimeout(timeout);
        child.kill();
        resolve(true);
      }
    });

    // Send test request
    child.stdin.write(JSON.stringify({
      jsonrpc: '2.0',
      id: '1',
      method: 'initialize',
      params: {}
    }) + '\n');
  });
}

checkMCPHealth().then(healthy => {
  process.exit(healthy ? 0 : 1);
});
```

## Step 8: Monitoring and Logging

### Prometheus Metrics

Add Prometheus metrics endpoint:

```javascript
// metrics.js
import { register, Counter, Gauge, Histogram } from 'prom-client';

export const metrics = {
  agentCount: new Gauge({
    name: 'mcp_agent_count',
    help: 'Total number of agents',
    labelNames: ['type', 'status']
  }),
  
  messagesSent: new Counter({
    name: 'mcp_messages_sent_total',
    help: 'Total messages sent',
    labelNames: ['type', 'priority']
  }),
  
  physicsUpdateDuration: new Histogram({
    name: 'mcp_physics_update_duration_seconds',
    help: 'Physics update duration',
    buckets: [0.001, 0.005, 0.01, 0.05, 0.1]
  })
};
```

### Log Aggregation

Configure structured logging:

```javascript
// logger-config.js
export const loggerConfig = {
  transports: [
    {
      type: 'console',
      format: 'json',
      level: process.env.MCP_LOG_LEVEL || 'info'
    },
    {
      type: 'file',
      filename: '/var/log/mcp-observability.log',
      format: 'json',
      maxsize: 10485760, // 10MB
      maxFiles: 5
    }
  ]
};
```

## Step 9: Production Optimizations

### Memory Management

```dockerfile
# Set Node.js memory limits
ENV NODE_OPTIONS="--max-old-space-size=512"

# Enable memory monitoring
ENV MCP_ENABLE_MEMORY_MONITOR=true
ENV MCP_MEMORY_ALERT_THRESHOLD=0.8
```

### CPU Optimization

```javascript
// Use worker threads for physics calculations
import { Worker } from 'worker_threads';

const physicsWorker = new Worker('./physics-worker.js');
physicsWorker.on('message', (result) => {
  // Update agent positions
});
```

## Step 10: Troubleshooting

### Common Issues

1. **MCP Server Not Starting**
   - Check logs: `docker logs <container-id>`
   - Verify Node.js version: `node --version` (requires 18+)
   - Check file permissions

2. **Memory Issues**
   - Increase memory limit: `NODE_OPTIONS="--max-old-space-size=1024"`
   - Enable memory persistence to reduce RAM usage
   - Use memory profiling: `node --inspect`

3. **Performance Issues**
   - Reduce physics update rate: `MCP_PHYSICS_UPDATE_RATE=30`
   - Limit max agents: `MCP_MAX_AGENTS=500`
   - Enable performance profiling

### Debug Mode

Enable debug logging:

```bash
docker run -e MCP_LOG_LEVEL=debug -e NODE_ENV=development your-image
```

## Best Practices

1. **Resource Limits**: Always set memory and CPU limits in production
2. **Persistence**: Enable memory persistence for crash recovery
3. **Monitoring**: Use Prometheus/Grafana for production monitoring
4. **Security**: Run as non-root user in container
5. **Updates**: Use specific version tags, not `latest`

## Conclusion

This integration provides a robust foundation for bot swarm observability in Docker environments. The spring-physics visualization enables intuitive understanding of agent interactions and system behavior.