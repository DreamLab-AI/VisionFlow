# Docker and MCP Integration Guide

## Overview

This guide consolidates all Docker container setup and MCP (Model Context Protocol) integration documentation for deploying the agent visualization system.

## Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Network: docker_ragflow               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐     ┌───────────────────────────┐     │
│  │  Rust Backend      │     │  Agent Container         │     │
│  │  (logseq)          │     │  (multi-agent-container) │     │
│  │                    │     │                          │     │
│  │  - UI Server       │     │  - Agent Control System  │     │
│  │  - WebSocket       │ TCP │  - MCP Server (stdio)    │     │
│  │  - TCP Client  ─────▶│  - TCP Server (:9500)    │     │
│  │  - GPU Physics     │     │  - Physics Engine        │     │
│  │                    │     │  - Claude Flow Tools     │     │
│  └─────────────────────┘     └───────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ext

# Create environment file
cp .env.example .env

# Edit environment variables
vim .env
```

### 2. Build and Deploy

```bash
# Build all containers
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Verify Deployment

```bash
# Check Rust backend
curl http://localhost:3001/api/health

# Check agent control system
curl http://localhost:9500/health

# Test WebSocket
wscat -c ws://localhost:3001/ws
```

## Docker Compose Configuration

### Minimal Setup (docker-compose.yml)

```yaml
version: '3.8'

networks:
  docker_ragflow:
    external: true

services:
  # Rust Backend (UI Server)
  webxr:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: logseq-webxr
    ports:
      - "3001:3001"
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - AGENT_CONTROL_URL=multi-agent-container:9500
      - ENABLE_GPU_PHYSICS=true
    volumes:
      - ./client/dist:/app/static
    networks:
      - docker_ragflow
    depends_on:
      - multi-agent-container

  # Agent Control Container
  multi-agent-container:
    build:
      context: ./agent-control-system
      dockerfile: Dockerfile
    container_name: multi-agent-container
    ports:
      - "9500:9500"
    environment:
      - TCP_SERVER_ENABLED=true
      - TCP_SERVER_PORT=9500
      - MAX_AGENTS=100
      - PHYSICS_UPDATE_RATE=60
    volumes:
      - agent-data:/var/lib/agents
    networks:
      - docker_ragflow

volumes:
  agent-data:
```

## Agent Control System Setup

### Dockerfile for Agent Container

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Install claude-flow globally
RUN npm install -g claude-flow@alpha

# Copy package files
COPY package*.json ./
RUN npm ci --production

# Copy application code
COPY . .

# Environment
ENV NODE_ENV=production

EXPOSE 9500

CMD ["node", "index.js"]
```

### Agent Control Configuration

```javascript
// config.js
module.exports = {
  tcp: {
    enabled: process.env.TCP_SERVER_ENABLED !== 'false',
    port: parseInt(process.env.TCP_SERVER_PORT || '9500', 10)
  },
  physics: {
    updateRate: parseInt(process.env.PHYSICS_UPDATE_RATE || '60', 10),
    springStrength: 0.3,
    damping: 0.95,
    centerForce: 0.001
  },
  agents: {
    maxCount: parseInt(process.env.MAX_AGENTS || '100', 10),
    defaultCapabilities: ['base_agent']
  },
  mcp: {
    stdio: true,
    tools: [
      'multi-agent_init',
      'agent_spawn',
      'agent_list',
      'agent_metrics',
      'task_orchestrate'
    ]
  }
};
```

## MCP Integration

### MCP Server Implementation

```javascript
// mcp-server.js
const { spawn } = require('child_process');
const { EventEmitter } = require('events');

class MCPServer extends EventEmitter {
  constructor() {
    super();
    this.process = null;
    this.messageId = 0;
    this.pendingRequests = new Map();
  }

  async start() {
    this.process = spawn('npx', ['claude-flow@alpha', 'mcp', 'start', '--stdio'], {
      stdio: ['pipe', 'pipe', 'inherit']
    });

    // Handle stdout (responses)
    this.process.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(line => line.trim());
      for (const line of lines) {
        try {
          const message = JSON.parse(line);
          this.handleMessage(message);
        } catch (e) {
          // Not JSON, ignore
        }
      }
    });

    // Initialize connection
    await this.initialize();
  }

  async initialize() {
    const response = await this.sendRequest('initialize', {
      clientInfo: {
        name: 'agent-control-system',
        version: '1.0.0'
      }
    });

    this.emit('ready', response.result);
    return response.result;
  }

  async callTool(toolName, args) {
    return this.sendRequest('tools/call', {
      name: toolName,
      arguments: args
    });
  }

  sendRequest(method, params) {
    return new Promise((resolve, reject) => {
      const id = `req-${++this.messageId}`;
      const request = {
        jsonrpc: '2.0',
        id,
        method,
        params
      };

      this.pendingRequests.set(id, { resolve, reject });
      this.process.stdin.write(JSON.stringify(request) + '\n');

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 30000);
    });
  }

  handleMessage(message) {
    if (message.id && this.pendingRequests.has(message.id)) {
      const { resolve, reject } = this.pendingRequests.get(message.id);
      this.pendingRequests.delete(message.id);

      if (message.error) {
        reject(new Error(message.error.message));
      } else {
        resolve(message);
      }
    } else if (message.method) {
      // Server-initiated message (notification)
      this.emit('notification', message);
    }
  }
}

module.exports = MCPServer;
```

### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `multi-agent_init` | Initialize Multi Agent | topology, maxAgents, strategy |
| `agent_spawn` | Create new agent | name, type, capabilities |
| `agent_list` | List active agents | filter, includeMetrics |
| `agent_metrics` | Get performance metrics | agentId, timeRange |
| `task_orchestrate` | Coordinate tasks | taskType, agents, parameters |
| `memory_usage` | Store/retrieve data | action, key, value |
| `neural_train` | Train neural patterns | modelType, trainingData |

## Rust Backend Configuration

### Environment Variables

```bash
# Agent Control
AGENT_CONTROL_URL=multi-agent-container:9500

# GPU Physics
ENABLE_GPU_PHYSICS=true
PHYSICS_UPDATE_RATE=60
MAX_VISIBLE_AGENTS=500

# WebSocket
WEBSOCKET_PORT=8080
WEBSOCKET_COMPRESSION=true

# Performance
RUST_LOG=info
WORKER_THREADS=4
```

### TCP Client Configuration

```rust
// src/services/agent_control_client.rs
pub struct AgentControlClient {
    addr: String,
    stream: Option<TcpStream>,
    message_id: u64,
}

impl AgentControlClient {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            stream: None,
            message_id: 0,
        }
    }

    pub async fn connect(&mut self) -> Result<()> {
        let stream = TcpStream::connect(&self.addr).await?;
        self.stream = Some(stream);
        Ok(())
    }

    pub async fn initialize_multi-agent(&mut self, params: initializeMultiAgentParams) -> Result<multi-agentInfo> {
        let request = json!({
            "method": "initialize_multi-agent",
            "params": params,
            "id": self.next_id()
        });

        self.send_request(request).await
    }
}
```

## Production Deployment

### Security Hardening

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  webxr:
    image: your-registry/logseq-webxr:latest
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    secrets:
      - jwt_secret
      - db_password
    environment:
      - NODE_ENV=production
      - JWT_SECRET_FILE=/run/secrets/jwt_secret

secrets:
  jwt_secret:
    external: true
  db_password:
    external: true
```

### Health Checks

```yaml
services:
  multi-agent-container:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9500/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Monitoring Stack

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
```

## Troubleshooting

### Common Issues

#### 1. Agent Control Connection Failed
```bash
# Check if container is running
docker ps | grep multi-agent-container

# Check TCP port
telnet multi-agent-container 9500

# View logs
docker logs -f multi-agent-container
```

#### 2. MCP Tools Not Working
```bash
# Test MCP server directly
docker exec -it multi-agent-container \
  npx claude-flow@alpha tools list

# Check stdio communication
docker exec -it multi-agent-container \
  node -e "require('./mcp-server').test()"
```

#### 3. WebSocket Issues
```bash
# Test WebSocket endpoint
wscat -c ws://localhost:8080/ws

# Check binary protocol
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://localhost:8080/ws
```

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug
export NODE_ENV=development
export DEBUG=mcp:*,agent:*

# Run with verbose output
docker-compose up
```

## Scaling Considerations

### Horizontal Scaling

```yaml
# docker-multi-agent.yml
version: '3.8'

services:
  webxr:
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
```

### Load Balancing

```nginx
# nginx.conf
upstream agent_backend {
    least_conn;
    server webxr1:3001 weight=5;
    server webxr2:3001 weight=5;
    server webxr3:3001 weight=5;
}

upstream agent_ws {
    ip_hash; # Sticky sessions for WebSocket
    server webxr1:8080;
    server webxr2:8080;
    server webxr3:8080;
}
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

# Backup agent data
docker run --rm \
  -v agent-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/agent-data-$(date +%Y%m%d).tar.gz -C /data .

# Backup configuration
tar czf backups/config-$(date +%Y%m%d).tar.gz \
  docker-compose.yml \
  .env \
  config/
```

### Restore Procedure

```bash
#!/bin/bash
# restore.sh

# Stop services
docker-compose down

# Restore data
docker run --rm \
  -v agent-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/agent-data-20240810.tar.gz -C /data

# Start services
docker-compose up -d
```

## Performance Tuning

### Container Resources

```yaml
services:
  multi-agent-container:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '2.0'
          memory: 2G
    sysctls:
      - net.core.somaxconn=1024
      - net.ipv4.tcp_syncookies=0
```

### Network Optimization

```yaml
networks:
  docker_ragflow:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000 # Jumbo frames
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## References

- [Agent Control System Documentation](../../agent-control-system/README.md)
- [MCP Protocol Specification](https://github.com/anthropics/mcp)
- [Claude Flow Documentation](../server/features/claude-flow-mcp-integration.md)
- [WebSocket Protocols](../api/websocket-protocols.md)
- [GPU Migration Guide](../architecture/visionflow-gpu-migration.md)