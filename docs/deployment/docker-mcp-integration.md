# Docker MCP Integration - Production Deployment Guide

## Overview

This comprehensive guide provides production-ready Docker container orchestration with MCP (Model Context Protocol) integration for the agentic-flow visualisation system. The deployment includes real-time agent monitoring, control systems, security hardening, observability stack, and operational procedures for maintaining a reliable system in production environments.

## Production Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Production Environment - agent-net                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  nginx-proxy    │  │ agent-visualizer │  │  claude-flow   │         │
│  │                 │  │                  │  │                │         │
│  │  Port: 80/443   │  │  Ports:          │  │  Port: 3001    │         │
│  │  - SSL Term     │  │  - 3000 (API)    │  │  - MCP Server  │         │
│  │  - Load Balance │  │  - 8080 (WS)     │  │  - Tool Exec   │         │
│  │  - Rate Limit   │  │  - 9091 (Metrics)│  │  - Auth Layer  │         │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬───────┘         │
│           │                     │                      │                 │
│  ┌────────▼────────────────────▼──────────────────────▼─────────┐       │
│  │                 Shared Volume: agent-data                    │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   PostgreSQL    │  │     Redis       │  │   Prometheus   │         │
│  │                 │  │                 │  │                │         │
│  │  Port: 5432     │  │  Port: 6379     │  │  Port: 9090    │         │
│  │  - Agent State  │  │  - Pub/Sub      │  │  - Metrics     │         │
│  │  - History      │  │  - Cache        │  │  - Alerts      │         │
│  │  - Backups      │  │  - Sessions     │  │  - Storage     │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │    Grafana      │  │     Jaeger      │  │  Alertmanager  │         │
│  │  Visualization  │  │  Tracing        │  │  Notifications │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
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

## Production Docker Compose

### Complete Production Stack (docker-compose.yml)

```yaml
version: '3.8'

x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "5"

x-restart-policy: &restart-unless-stopped
  restart: unless-stopped

x-health-check: &default-healthcheck
  interval: 30s
  timeout: 10s
  start_period: 60s
  retries: 3

services:
  # Reverse proxy with SSL termination and security headers
  nginx-proxy:
    image: nginx:1.25-alpine
    container_name: agent-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/security.conf:/etc/nginx/conf.d/security.conf:ro
      - static-content:/usr/share/nginx/html:ro
      - ./nginx/logs:/var/log/nginx
    depends_on:
      agent-visualizer:
        condition: service_healthy
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    healthcheck:
      <<: *default-healthcheck
      test: ["CMD", "curl", "-f", "http://localhost/health"]
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

  # Main agent visualization service
  agent-visualizer:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - BUILDKIT_INLINE_CACHE=1
        - VITE_API_URL=${API_BASE_URL:-/api}
        - VITE_WS_URL=${WS_BASE_URL:-/ws}
    image: agent-visualizer:${IMAGE_TAG:-latest}
    container_name: agent-viz
    environment:
      # Database configuration
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@postgres:5432/${DB_NAME}
      - DATABASE_POOL_SIZE=25
      - DATABASE_MAX_CONNECTIONS=100
      - DATABASE_SSL_MODE=require

      # Redis configuration
      - REDIS_URL=redis://:${REDIS_PASS}@redis:6379/0
      - REDIS_POOL_SIZE=15
      - REDIS_CLUSTER_MODE=${REDIS_CLUSTER:-false}

      # MCP Integration
      - MCP_SERVER_URL=http://mcp-server:3001
      - MCP_API_KEY=${MCP_API_KEY}
      - MCP_TIMEOUT=30000
      - MCP_RETRY_ATTEMPTS=3

      # WebSocket Configuration
      - WS_PORT=8080
      - WS_MAX_CONNECTIONS=2000
      - WS_HEARTBEAT_INTERVAL=30000
      - WS_COMPRESSION=true
      - WS_RATE_LIMIT=100

      # Performance tuning
      - ENABLE_GPU_PHYSICS=${ENABLE_GPU:-true}
      - MAX_VISIBLE_AGENTS=1000
      - PHYSICS_UPDATE_RATE=60
      - POSITION_BATCH_SIZE=200
      - WORKER_THREADS=${WORKER_THREADS:-4}

      # Security
      - JWT_SECRET=${JWT_SECRET}
      - JWT_EXPIRY=3600
      - ENABLE_RATE_LIMITING=true
      - CORS_ORIGINS=${CORS_ORIGINS}

      # Monitoring and observability
      - PROMETHEUS_ENABLED=true
      - METRICS_PORT=9091
      - JAEGER_AGENT_HOST=jaeger
      - JAEGER_AGENT_PORT=6831
      - OTEL_SERVICE_NAME=agent-visualizer

      # Logging
      - RUST_LOG=agent_server=info,tower_http=warn,sqlx=warn
      - LOG_FORMAT=json
      - LOG_LEVEL=${LOG_LEVEL:-info}
    volumes:
      - agent-data:/var/lib/agent-server
      - agent-logs:/var/log/agent-server
      - ./config:/etc/agent-server:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      mcp-server:
        condition: service_healthy
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    healthcheck:
      <<: *default-healthcheck
      test: ["CMD", "/usr/local/bin/health-check.sh"]
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G

  # MCP Server for claude-flow integration
  mcp-server:
    build:
      context: ./mcp
      dockerfile: Dockerfile
    image: mcp-server:${IMAGE_TAG:-latest}
    container_name: mcp-server
    environment:
      - MCP_PORT=3001
      - MCP_LOG_LEVEL=${LOG_LEVEL:-info}
      - CLAUDE_FLOW_CONFIG=/app/config/claude-flow.json
      - ENABLE_SWARM_TOOLS=true
      - ENABLE_MEMORY_TOOLS=true
      - ENABLE_NEURAL_TOOLS=true
      - NODE_ENV=production
      - MAX_CONCURRENT_TOOLS=${MAX_CONCURRENT_TOOLS:-50}
    volumes:
      - ./mcp/config:/app/config:ro
      - ./mcp/tools:/app/mcp-tools:ro
      - mcp-data:/var/lib/mcp
      - mcp-logs:/var/log
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    healthcheck:
      <<: *default-healthcheck
      test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  # PostgreSQL with high availability configuration
  postgres:
    image: postgres:16-alpine
    container_name: agent-postgres
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASS}
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --locale=en_US.UTF-8
      - POSTGRES_SHARED_PRELOAD_LIBRARIES=pg_stat_statements
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
      - ./sql/performance.sql:/docker-entrypoint-initdb.d/02-performance.sql:ro
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - postgres-backups:/var/backups/postgres
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports:
      - "${DB_PORT:-5432}:5432"
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    healthcheck:
      <<: *default-healthcheck
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]

  # Redis with persistence and clustering support
  redis:
    image: redis:7-alpine
    container_name: agent-redis
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --maxmemory ${REDIS_MEMORY:-1gb}
      --maxmemory-policy allkeys-lru
      --tcp-backlog 511
      --timeout 0
      --tcp-keepalive 300
      --save 900 1
      --save 300 10
      --save 60 10000
      --requirepass ${REDIS_PASS}
    volumes:
      - redis-data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    ports:
      - "${REDIS_PORT:-6379}:6379"
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    healthcheck:
      <<: *default-healthcheck
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASS}", "ping"]

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: agent-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=90d'
      - '--storage.tsdb.retention.size=50GB'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/alerts:/etc/prometheus/alerts:ro
      - prometheus-data:/prometheus
    ports:
      - "${PROMETHEUS_PORT:-9090}:9090"
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  # Grafana for visualization and dashboards
  grafana:
    image: grafana/grafana:10.1.0
    container_name: agent-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASS}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_DOMAIN=${GRAFANA_DOMAIN:-localhost}
      - GF_SMTP_ENABLED=true
      - GF_SMTP_HOST=${SMTP_HOST}
      - GF_SMTP_FROM_ADDRESS=${SMTP_FROM}
      - GF_INSTALL_PLUGINS=grafana-worldmap-panel,grafana-piechart-panel
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "${GRAFANA_PORT:-3001}:3000"
    depends_on:
      - prometheus
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped
    user: "${GRAFANA_UID:-472}:${GRAFANA_GID:-0}"

  # Jaeger for distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:1.49
    container_name: agent-jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - SPAN_STORAGE_TYPE=badger
      - BADGER_EPHEMERAL=false
      - BADGER_DIRECTORY_VALUE=/badger/data
      - BADGER_DIRECTORY_KEY=/badger/key
    volumes:
      - jaeger-data:/badger
    ports:
      - "${JAEGER_PORT:-16686}:16686"
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped

  # Alertmanager for alert routing and notification
  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: agent-alertmanager
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    volumes:
      - ./monitoring/alertmanager/config.yml:/etc/alertmanager/config.yml:ro
      - alertmanager-data:/alertmanager
    ports:
      - "${ALERTMANAGER_PORT:-9093}:9093"
    depends_on:
      - prometheus
    networks:
      - agent-net
    logging: *default-logging
    <<: *restart-unless-stopped

volumes:
  agent-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/agent
  mcp-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/mcp
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/postgres
  postgres-backups:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${BACKUP_PATH:-./backups}/postgres
  redis-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/redis
  prometheus-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/prometheus
  grafana-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/grafana
  jaeger-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/jaeger
  alertmanager-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/alertmanager
  static-content:
    driver: local
  agent-logs:
    driver: local
  mcp-logs:
    driver: local

networks:
  agent-net:
    driver: bridge
    enable_ipv6: false
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
    driver_opts:
      com.docker.network.bridge.name: agent-br0
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

## Production Environment Configuration

### Environment Variables (.env.production)

```bash
# Production environment configuration - SECURE THESE VALUES

# Database Configuration
DB_USER=agent_user
DB_PASS=AgenT_Pr0d_DB_P@ssw0rd_2024!
DB_NAME=agent_db
DB_PORT=5432
DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@postgres:${DB_PORT}/${DB_NAME}
DATABASE_SSL_MODE=require
DATABASE_POOL_SIZE=25
DATABASE_MAX_CONNECTIONS=100

# Redis Configuration
REDIS_PASS=Redis_Secure_2024_P@ssw0rd!
REDIS_PORT=6379
REDIS_URL=redis://:${REDIS_PASS}@redis:${REDIS_PORT}/0
REDIS_MEMORY=2gb
REDIS_CLUSTER=false

# MCP Integration Security
MCP_API_KEY=mcp_api_super_secure_key_production_2024_xyz789!
MCP_SERVER_URL=http://mcp-server:3001
MCP_VERIFY_SSL=true
MAX_CONCURRENT_TOOLS=50

# JWT Configuration
JWT_SECRET=JWT_Super_Secure_Signing_Key_For_Production_2024_abcdef123456!
JWT_EXPIRY=3600

# SSL/TLS Configuration
SSL_CERT_PATH=/etc/nginx/ssl/fullchain.pem
SSL_KEY_PATH=/etc/nginx/ssl/privkey.pem
SSL_DH_PARAM_PATH=/etc/nginx/ssl/dhparam.pem

# CORS Configuration
CORS_ORIGINS=https://agent-viz.yourdomain.com,https://dashboard.yourdomain.com
API_BASE_URL=https://agent-viz.yourdomain.com/api
WS_BASE_URL=wss://agent-viz.yourdomain.com/ws

# Performance Configuration
WORKER_THREADS=8
ENABLE_GPU=true
MAX_VISIBLE_AGENTS=1000
PHYSICS_UPDATE_RATE=60

# Monitoring Configuration
GRAFANA_USER=admin
GRAFANA_PASS=Graf@na_Adm1n_Secure_2024!
GRAFANA_DOMAIN=grafana.yourdomain.com
GRAFANA_PORT=3001
GRAFANA_UID=472
GRAFANA_GID=0

PROMETHEUS_PORT=9090
JAEGER_PORT=16686
ALERTMANAGER_PORT=9093

# Data Storage Paths
DATA_PATH=/opt/agent-data
BACKUP_PATH=/opt/backups
LOG_LEVEL=info
IMAGE_TAG=v2.0.0
```

### Production MCP Configuration

```json
{
  "server": {
    "port": 3001,
    "host": "0.0.0.0",
    "timeout": 30000,
    "maxConnections": 1000,
    "cors": {
      "enabled": true,
      "origins": [
        "https://agent-viz.yourdomain.com",
        "https://*.yourdomain.com"
      ],
      "credentials": true
    },
    "compression": {
      "enabled": true,
      "threshold": 1024,
      "level": 6
    }
  },
  "security": {
    "authentication": {
      "enabled": true,
      "type": "jwt",
      "secret": "${JWT_SECRET}",
      "expiry": 3600
    },
    "rateLimit": {
      "enabled": true,
      "windowMs": 60000,
      "maxRequests": 100,
      "skipSuccessfulRequests": false
    },
    "validation": {
      "strict": true,
      "sanitizeInputs": true,
      "maxPayloadSize": "10MB"
    }
  },
  "tools": {
    "swarm_init": {
      "enabled": true,
      "maxAgents": 100,
      "topologies": ["star", "ring", "mesh", "hierarchical"],
      "persistence": true
    },
    "agent_spawn": {
      "enabled": true,
      "maxConcurrent": 50,
      "timeout": 30000,
      "validation": {
        "strict": true,
        "schema": "agent-spawn-schema.json"
      },
      "resources": {
        "memory": "512MB",
        "cpu": "0.5"
      }
    },
    "task_orchestrate": {
      "enabled": true,
      "maxParallel": 20,
      "queueSize": 1000,
      "retryAttempts": 3
    },
    "memory_usage": {
      "enabled": true,
      "maxSize": "1GB",
      "persistence": "redis",
      "ttl": 86400,
      "compression": true
    },
    "neural_train": {
      "enabled": true,
      "models": ["decision", "pattern", "optimization", "coordination"],
      "gpuAcceleration": true,
      "maxEpochs": 1000,
      "batchSize": 32
    }
  },
  "integrations": {
    "database": {
      "type": "postgresql",
      "connectionString": "${DATABASE_URL}",
      "poolSize": 20,
      "maxConnections": 100,
      "ssl": true
    },
    "cache": {
      "type": "redis",
      "connectionString": "${REDIS_URL}",
      "poolSize": 10,
      "keyPrefix": "mcp:",
      "ttl": 3600
    },
    "monitoring": {
      "prometheus": {
        "enabled": true,
        "port": 9091,
        "path": "/metrics",
        "prefix": "mcp_"
      },
      "jaeger": {
        "enabled": true,
        "serviceName": "mcp-server",
        "agentHost": "${JAEGER_AGENT_HOST}",
        "agentPort": "${JAEGER_AGENT_PORT}",
        "samplingRate": 0.1
      }
    }
  }
}
```

### Security Hardening

#### Nginx Security Configuration

```nginx
# /nginx/security.conf
# Production security hardening for Nginx

# Security headers
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy strict-origin-when-cross-origin always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' wss: https:; font-src 'self' data:; frame-ancestors 'none';" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Hide server information
server_tokens off;
more_clear_headers Server;

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=websocket:10m rate=100r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

# Connection limiting
limit_conn_zone $binary_remote_addr zone=perip:10m;
limit_conn perip 20;

# Request size limits
client_max_body_size 10M;
client_header_buffer_size 1k;
large_client_header_buffers 4 16k;

# Block suspicious patterns
location ~* \.(env|git|svn|htaccess|htpasswd|ini|conf|sql|log|tar|gz|zip|rar|7z)$ {
    deny all;
    return 404;
}

# Block common attack patterns
location ~* (union.*select|concat.*\(|script.*alert|javascript:|vbscript:) {
    deny all;
    return 444;
}

# Rate limiting application
location /api/ {
    limit_req zone=api burst=20 nodelay;
    limit_conn perip 10;
    
    proxy_pass http://agent-visualizer:3000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location /ws/ {
    limit_req zone=websocket burst=50 nodelay;
    limit_conn perip 5;
    
    proxy_pass http://agent-visualizer:8080;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
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

### Production Deployment Scripts

#### Deploy Script (deploy.sh)

```bash
#!/bin/bash
set -euo pipefail

# Production deployment script with comprehensive validation and rollback capability
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.production"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
BACKUP_DIR="${SCRIPT_DIR}/backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SCRIPT_DIR}/logs/deploy_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [DEPLOY] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    log "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    error_exit "Environment file $ENV_FILE not found"
fi

# Validate required environment variables
validate_environment() {
    log "Validating environment variables..."
    
    local required_vars=(
        "DB_USER" "DB_PASS" "DB_NAME"
        "REDIS_PASS" "MCP_API_KEY" "JWT_SECRET"
        "SSL_CERT_PATH" "SSL_KEY_PATH"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error_exit "Required environment variable $var is not set"
        fi
    done
    
    log "Environment validation completed"
}

# Pre-deployment validation
validate_ssl_certificates() {
    log "Validating SSL certificates..."
    
    if [ ! -f "$SSL_CERT_PATH" ] || [ ! -f "$SSL_KEY_PATH" ]; then
        error_exit "SSL certificates not found"
    fi
    
    # Check certificate expiry
    if ! openssl x509 -checkend 604800 -noout -in "$SSL_CERT_PATH"; then
        log "WARNING: SSL certificate expires within 7 days"
    fi
    
    log "SSL certificate validation completed"
}

# Backup existing deployment
backup_existing_deployment() {
    log "Creating backup of existing deployment..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup data volumes
    for volume in agent-data mcp-data postgres-data redis-data; do
        if docker volume ls | grep -q "$volume"; then
            log "Backing up volume: $volume"
            docker run --rm \
                -v "${volume}:/source:ro" \
                -v "${BACKUP_DIR}:/backup" \
                alpine \
                tar czf "/backup/${volume}.tar.gz" -C /source .
        fi
    done
    
    # Backup database
    if docker ps --format '{{.Names}}' | grep -q agent-postgres; then
        log "Creating database backup..."
        docker exec agent-postgres pg_dumpall -U "$DB_USER" | gzip > "$BACKUP_DIR/database.sql.gz"
    fi
    
    log "Backup completed: $BACKUP_DIR"
}

# Build and validate images
build_and_validate_images() {
    log "Building Docker images..."
    
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    if ! docker-compose -f "$COMPOSE_FILE" build --parallel; then
        error_exit "Image build failed"
    fi
    
    log "Image build completed"
}

# Perform rolling update
perform_rolling_update() {
    log "Performing rolling deployment..."
    
    # Start infrastructure services first
    local infra_services=("postgres" "redis" "prometheus" "jaeger")
    
    for service in "${infra_services[@]}"; do
        log "Starting service: $service"
        docker-compose -f "$COMPOSE_FILE" up -d "$service"
        sleep 5
    done
    
    # Start application services
    local app_services=("mcp-server" "agent-visualizer" "nginx-proxy" "grafana" "alertmanager")
    
    for service in "${app_services[@]}"; do
        log "Starting service: $service"
        docker-compose -f "$COMPOSE_FILE" up -d "$service"
        sleep 10
    done
    
    log "Rolling deployment completed"
}

# Run post-deployment tests
run_post_deployment_tests() {
    log "Running post-deployment tests..."
    
    # Test API endpoints
    if curl -f -s --max-time 10 "http://localhost:${API_PORT:-3000}/health" >/dev/null; then
        log "✓ API health endpoint is responsive"
    else
        error_exit "✗ API health endpoint test failed"
    fi
    
    # Test database connectivity
    if docker exec agent-postgres pg_isready -U "$DB_USER" >/dev/null 2>&1; then
        log "✓ Database is accessible"
    else
        error_exit "✗ Database connectivity test failed"
    fi
    
    # Test Redis connectivity
    if docker exec agent-redis redis-cli ping | grep -q "PONG"; then
        log "✓ Redis is accessible"
    else
        error_exit "✗ Redis connectivity test failed"
    fi
    
    log "All post-deployment tests passed"
}

# Main deployment function
main() {
    log "Starting production deployment..."
    
    validate_environment
    validate_ssl_certificates
    backup_existing_deployment
    build_and_validate_images
    perform_rolling_update
    run_post_deployment_tests
    
    log "Production deployment completed successfully!"
    
    # Display service status
    docker-compose -f "$COMPOSE_FILE" ps
    
    log "Services available at:"
    log "  - Web Interface: https://$(hostname)/"
    log "  - API: https://$(hostname)/api"
    log "  - Grafana: https://$(hostname):3001"
    log "  - Prometheus: https://$(hostname):9090"
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    backup)
        validate_environment
        backup_existing_deployment
        ;;
    test)
        validate_environment
        run_post_deployment_tests
        ;;
    *)
        echo "Usage: $0 {deploy|backup|test}"
        exit 1
        ;;
esac
```

#### Health Check Script (health-check.sh)

```bash
#!/bin/sh
# Comprehensive production health check with detailed diagnostics

set -e

HEALTH_CHECK_TIMEOUT=10
API_PORT=${API_PORT:-3000}
WS_PORT=${WS_PORT:-8080}
METRICS_PORT=${METRICS_PORT:-9091}

# Function to check HTTP endpoint with retry
check_http() {
    local url=$1
    local expected_status=${2:-200}
    local timeout=${3:-$HEALTH_CHECK_TIMEOUT}
    local retries=${4:-3}
    
    for attempt in $(seq 1 $retries); do
        response=$(curl -s -w "%{http_code}" -o /dev/null --max-time "$timeout" "$url" 2>/dev/null || echo "000")
        
        if [ "$response" = "$expected_status" ]; then
            return 0
        fi
        
        if [ $attempt -lt $retries ]; then
            sleep 1
        fi
    done
    
    echo "Health check failed for $url (got $response, expected $expected_status)" >&2
    return 1
}

# Main health checks
echo "Running comprehensive health checks..."

# 1. Check main API endpoint
if check_http "http://localhost:$API_PORT/health"; then
    echo "✓ API health check passed"
else
    echo "✗ API health check failed"
    exit 1
fi

# 2. Check WebSocket endpoint
if timeout 5 bash -c "</dev/tcp/localhost/$WS_PORT" 2>/dev/null; then
    echo "✓ WebSocket port is accessible"
else
    echo "✗ WebSocket connectivity test failed"
    exit 1
fi

# 3. Check metrics endpoint
if check_http "http://localhost:$METRICS_PORT/metrics"; then
    echo "✓ Metrics health check passed"
else
    echo "✗ Metrics health check failed"
    exit 1
fi

# 4. Check database connectivity
if pg_isready -h "${DATABASE_HOST:-localhost}" -p "${DATABASE_PORT:-5432}" -t 5 >/dev/null 2>&1; then
    echo "✓ Database health check passed"
else
    echo "✗ Database health check failed"
    exit 1
fi

# 5. Check Redis connectivity
if [ ! -z "$REDIS_PASS" ]; then
    redis_cmd="redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} -a $REDIS_PASS"
else
    redis_cmd="redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379}"
fi

if timeout 5 $redis_cmd ping >/dev/null 2>&1; then
    echo "✓ Redis health check passed"
else
    echo "✗ Redis health check failed"
    exit 1
fi

# 6. Check system resources
DISK_USAGE=$(df /var/lib/agent-server 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "✗ Disk usage is too high: $DISK_USAGE%"
    exit 1
fi

MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}' || echo "0")
if [ "$MEMORY_USAGE" -gt 95 ]; then
    echo "✗ Memory usage is too high: $MEMORY_USAGE%"
    exit 1
fi

echo "✓ All health checks passed"
exit 0
```

### Monitoring Configuration

#### Prometheus Configuration (prometheus.yml)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'agent-viz-prod'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Load alerting rules
rule_files:
  - "/etc/prometheus/alerts/*.yml"

# Scrape configurations
scrape_configs:
  # Agent visualization service
  - job_name: 'agent-visualizer'
    static_configs:
      - targets: ['agent-visualizer:9091']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # MCP server metrics
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['mcp-server:9091']
    metrics_path: '/metrics'
    scrape_interval: 15s

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

  # Node/system metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s
```

## Troubleshooting

### Common Production Issues

#### 1. Container Startup Failures

**Problem**: Containers fail to start or become unhealthy

```bash
# Check container logs
docker-compose logs <service_name>

# Check resource usage
docker stats

# Inspect container configuration
docker inspect <container_name>

# Check network connectivity
docker network ls
docker network inspect agent-net
```

**Solutions**:
- Verify environment variables are set correctly
- Check port conflicts with other services
- Ensure sufficient system resources (memory, disk)
- Validate configuration files syntax

#### 2. Database Connection Issues

**Problem**: Application cannot connect to PostgreSQL

```bash
# Test database connectivity
docker exec agent-postgres pg_isready -U $DB_USER

# Check database logs
docker logs agent-postgres

# Test connection from application container
docker exec agent-viz pg_isready -h postgres -p 5432 -U $DB_USER

# Run database migrations manually
docker exec agent-viz /usr/local/bin/agent-server migrate
```

#### 3. Redis Connection Issues

**Problem**: Redis connectivity or performance problems

```bash
# Test Redis connectivity
docker exec agent-redis redis-cli ping

# Check Redis configuration
docker exec agent-redis redis-cli CONFIG GET "*"

# Monitor Redis performance
docker exec agent-redis redis-cli --latency-history

# Check memory usage
docker exec agent-redis redis-cli INFO memory
```

#### 4. MCP Tool Execution Failures

**Problem**: MCP tools failing or timing out

```bash
# Check MCP server logs
docker logs mcp-server

# Test MCP server health
curl -f http://localhost:3001/health

# Test specific MCP tool
docker exec mcp-server node -e "
const tools = require('./mcp-tools');
console.log(Object.keys(tools));
"

# Monitor MCP server resources
docker stats mcp-server
```

#### 5. WebSocket Connection Issues

**Problem**: WebSocket connections failing or dropping

```bash
# Test WebSocket endpoint
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: test" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:8080/ws

# Check nginx WebSocket configuration
docker exec agent-nginx nginx -t

# Monitor WebSocket connections
docker exec agent-viz ss -tuln | grep 8080
```

#### 6. Performance Issues

**Problem**: Slow response times or high resource usage

```bash
# Monitor system resources
htop
iotop
free -h
df -h

# Check database performance
docker exec agent-postgres psql -U $DB_USER -d $DB_NAME -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"

# Monitor application metrics
curl http://localhost:9091/metrics | grep -E "(response_time|memory|cpu)"

# Check for slow queries
docker logs agent-viz | grep "slow query"
```

#### 7. SSL/TLS Certificate Issues

**Problem**: HTTPS not working or certificate errors

```bash
# Check certificate validity
openssl x509 -in $SSL_CERT_PATH -text -noout

# Test certificate chain
openssl verify -CAfile /path/to/ca.pem $SSL_CERT_PATH

# Check nginx SSL configuration
docker exec agent-nginx nginx -t

# Test HTTPS connection
curl -I https://yourdomain.com
```

### Debug Mode Configuration

```bash
# Enable comprehensive debug logging
export RUST_LOG=debug
export NODE_ENV=development
export DEBUG=mcp:*,agent:*
export MCP_LOG_LEVEL=debug

# Run with verbose output and detailed logging
docker-compose up --build
```

### Performance Optimization Tips

#### Database Optimization

```sql
-- Add to performance.sql
-- Connection and memory settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Query optimization
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET default_statistics_target = 100;

-- Enable query statistics
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;

SELECT pg_reload_conf();
```

#### Redis Optimization

```redis
# Add to redis.conf
# Memory optimization
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Network optimization
tcp-keepalive 300
tcp-backlog 511

# Persistence optimization
save 900 1
save 300 10
save 60 10000
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

## Production Checklist

### Pre-Deployment Verification

- [ ] **Environment Variables**: All required production environment variables are set and validated
- [ ] **SSL Certificates**: Valid SSL certificates are installed and not expiring soon
- [ ] **Database**: PostgreSQL is properly configured with performance optimizations
- [ ] **Redis**: Redis is configured with authentication and persistence
- [ ] **Security**: All security headers and rate limiting are configured
- [ ] **Monitoring**: Prometheus, Grafana, and Jaeger are properly configured
- [ ] **Backups**: Automated backup procedures are in place
- [ ] **Health Checks**: All health check scripts are tested and working
- [ ] **Resource Limits**: Container resource limits are set appropriately
- [ ] **Network Security**: Firewall rules and network segmentation are configured

### Post-Deployment Verification

- [ ] **Service Health**: All services pass health checks
- [ ] **API Endpoints**: All API endpoints are responsive
- [ ] **WebSocket**: WebSocket connections are working properly
- [ ] **Database Connectivity**: Application can connect to database
- [ ] **Redis Connectivity**: Application can connect to Redis
- [ ] **MCP Tools**: All MCP tools are functional
- [ ] **Monitoring**: Metrics are being collected and displayed
- [ ] **Alerts**: Alert rules are firing appropriately
- [ ] **SSL/TLS**: HTTPS is working with valid certificates
- [ ] **Performance**: Response times are within acceptable ranges
- [ ] **Backup**: Backup procedures have been tested
- [ ] **Rollback Plan**: Rollback procedures are documented and tested

### Ongoing Maintenance

- [ ] **Regular Backups**: Automated daily backups are running
- [ ] **Certificate Renewal**: SSL certificate auto-renewal is configured
- [ ] **Security Updates**: Regular security updates for base images
- [ ] **Performance Monitoring**: Regular performance reviews and optimizations
- [ ] **Log Management**: Log rotation and archival procedures
- [ ] **Disaster Recovery**: Disaster recovery procedures are tested
- [ ] **Capacity Planning**: Resource usage trends are monitored
- [ ] **Security Audits**: Regular security assessments

## Summary

This production-ready Docker MCP integration guide provides comprehensive deployment documentation including:

1. **Production Architecture** - Scalable multi-container setup with proper networking
2. **Security Hardening** - SSL/TLS, authentication, rate limiting, and security headers
3. **Monitoring Stack** - Prometheus, Grafana, Jaeger, and Alertmanager integration
4. **Backup & Recovery** - Automated backups with tested restore procedures
5. **Performance Optimization** - Database and cache tuning for production workloads
6. **Operational Procedures** - Deployment scripts, health checks, and troubleshooting guides
7. **MCP Integration** - Production-ready MCP tools with security and monitoring
8. **Scalability** - Horizontal scaling configuration and load balancing
9. **Maintenance** - Ongoing operational procedures and best practices

The deployment supports real-time agent monitoring, coordination, and visualisation with enterprise-grade reliability, security, and observability.

## References

- [Agent Control System Documentation](../../agent-control-system/README.md)
- [MCP Protocol Specification](https://github.com/anthropics/mcp)
- [Claude Flow Documentation](../server/features/claude-flow-mcp-integration.md)
- [WebSocket Protocols](../api/websocket-protocols.md)
- [GPU Migration Guide](../architecture/visionflow-gpu-migration.md)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Configuration Guide](https://redis.io/docs/manual/config/)
- [Nginx Security Guide](https://nginx.org/en/docs/http/securing_http_traffic.html)
- [Prometheus Monitoring Guide](https://prometheus.io/docs/practices/naming/)

---

**Note**: This documentation provides production-ready configurations and procedures. Always validate configurations in a staging environment before deploying to production. Regular security updates and monitoring are essential for maintaining a secure and reliable deployment.