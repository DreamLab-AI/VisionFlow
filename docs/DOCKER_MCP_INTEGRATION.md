# Docker Container and MCP Integration Guide

## Overview

This document provides detailed instructions for setting up the Docker containers and integrating MCP (Model Context Protocol) hooks required for the agentic-flow visualization system. The setup involves multiple containers working together to provide real-time agent monitoring and control.

## Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Docker Network: agent-net               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  nginx-proxy    │  │ agent-visualizer │  │  claude-flow   │ │
│  │                 │  │                  │  │                │ │
│  │  Port: 80/443   │  │  Ports:          │  │  Port: 3001    │ │
│  │  - SSL Term     │  │  - 3000 (API)    │  │  - MCP Server  │ │
│  │  - Load Balance │  │  - 8080 (WS)     │  │  - Tool Exec   │ │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬───────┘ │
│           │                     │                      │         │
│  ┌────────▼────────────────────▼──────────────────────▼───────┐ │
│  │                    Shared Volume: agent-data                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   PostgreSQL    │  │     Redis       │  │   Prometheus   │ │
│  │                 │  │                 │  │                │ │
│  │  Port: 5432     │  │  Port: 6379     │  │  Port: 9090    │ │
│  │  - Agent State  │  │  - Pub/Sub      │  │  - Metrics     │ │
│  │  - History      │  │  - Cache        │  │  - Alerts      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Stage Dockerfile

### Agent Visualization Container

```dockerfile
# Build stage for frontend
FROM node:20-alpine AS frontend-builder

# Install build dependencies
RUN apk add --no-cache python3 make g++ git

WORKDIR /app/client

# Copy package files
COPY client/package*.json ./
COPY client/.npmrc* ./

# Install dependencies with frozen lockfile
RUN npm ci --frozen-lockfile

# Copy source code
COPY client/ ./

# Build frontend with production optimizations
ARG VITE_API_URL=/api
ARG VITE_WS_URL=/ws
ENV NODE_ENV=production
RUN npm run build

# Analyze bundle size
RUN npm run analyze || true

# Build stage for Rust backend
FROM rust:1.75-alpine AS backend-builder

# Install build dependencies
RUN apk add --no-cache musl-dev openssl-dev pkgconfig

WORKDIR /app

# Copy cargo files for dependency caching
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies only
RUN cargo build --release
RUN rm -rf src

# Copy actual source code
COPY src ./src
COPY migrations ./migrations

# Build the application
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    openssl \
    libgcc \
    tini \
    curl \
    jq

# Create non-root user
RUN addgroup -g 1000 agent && \
    adduser -u 1000 -G agent -s /bin/sh -D agent

# Install Node.js for MCP integration
RUN apk add --no-cache nodejs npm

# Copy built artifacts
COPY --from=backend-builder /app/target/release/agent-server /usr/local/bin/
COPY --from=frontend-builder /app/client/dist /usr/share/nginx/html

# Copy configuration files
COPY config /etc/agent-server/
COPY scripts/entrypoint.sh /usr/local/bin/

# Create necessary directories
RUN mkdir -p /var/lib/agent-server /var/log/agent-server && \
    chown -R agent:agent /var/lib/agent-server /var/log/agent-server

# Switch to non-root user
USER agent

# Environment variables
ENV RUST_LOG=info
ENV NODE_ENV=production
ENV MCP_INTEGRATION=enabled

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000 8080

ENTRYPOINT ["tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
```

### MCP Integration Layer

```dockerfile
# MCP Integration Service
FROM node:20-alpine AS mcp-service

WORKDIR /app

# Install claude-flow globally
RUN npm install -g claude-flow@alpha

# Copy MCP configuration
COPY mcp-config.json ./
COPY mcp-tools ./mcp-tools/

# Install custom MCP tools
WORKDIR /app/mcp-tools
RUN npm install

WORKDIR /app

# Create MCP server wrapper
RUN cat > mcp-server.js << 'EOF'
const { MCPServer } = require('claude-flow');
const config = require('./mcp-config.json');

const server = new MCPServer(config);

// Register custom tools
const tools = require('./mcp-tools');
Object.entries(tools).forEach(([name, handler]) => {
  server.registerTool(name, handler);
});

// Start server
server.listen(process.env.MCP_PORT || 3001);
console.log(`MCP Server listening on port ${process.env.MCP_PORT || 3001}`);
EOF

# Environment variables
ENV MCP_PORT=3001
ENV MCP_MODE=server
ENV ENABLE_AGENT_TOOLS=true

EXPOSE 3001

CMD ["node", "mcp-server.js"]
```

## Docker Compose Configuration

### Complete Stack Setup

```yaml
version: '3.8'

x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"

services:
  # Reverse proxy with SSL termination
  nginx-proxy:
    image: nginx:alpine
    container_name: agent-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - static-content:/usr/share/nginx/html:ro
    depends_on:
      - agent-visualizer
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped

  # Main agent visualization service
  agent-visualizer:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - BUILDKIT_INLINE_CACHE=1
    image: agent-visualizer:latest
    container_name: agent-viz
    environment:
      # Database
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_db
      - DATABASE_POOL_SIZE=20

      # Redis
      - REDIS_URL=redis://redis:6379/0
      - REDIS_POOL_SIZE=10

      # MCP Integration
      - MCP_SERVER_URL=http://mcp-server:3001
      - MCP_API_KEY=${MCP_API_KEY}
      - MCP_TIMEOUT=30000

      # WebSocket Configuration
      - WS_PORT=8080
      - WS_MAX_CONNECTIONS=1000
      - WS_HEARTBEAT_INTERVAL=30000
      - WS_COMPRESSION=true

      # Performance
      - ENABLE_GPU_PHYSICS=true
      - MAX_VISIBLE_AGENTS=500
      - PHYSICS_UPDATE_RATE=60
      - POSITION_BATCH_SIZE=100

      # Monitoring
      - PROMETHEUS_ENABLED=true
      - METRICS_PORT=9091
      - JAEGER_AGENT_HOST=jaeger
      - JAEGER_AGENT_PORT=6831

      # Logging
      - RUST_LOG=agent_server=debug,tower_http=info
      - LOG_FORMAT=json
    volumes:
      - agent-data:/var/lib/agent-server
      - ./config:/etc/agent-server:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      mcp-server:
        condition: service_started
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  # MCP Server for claude-flow integration
  mcp-server:
    build:
      context: ./mcp
      dockerfile: Dockerfile
    image: mcp-server:latest
    container_name: mcp-server
    environment:
      - MCP_PORT=3001
      - MCP_LOG_LEVEL=debug
      - CLAUDE_FLOW_CONFIG=/app/config/claude-flow.json
      - ENABLE_multi-agent_TOOLS=true
      - ENABLE_MEMORY_TOOLS=true
      - ENABLE_NEURAL_TOOLS=true
    volumes:
      - ./mcp/config:/app/config:ro
      - ./mcp/tools:/app/mcp-tools:ro
      - mcp-data:/var/lib/mcp
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped

  # PostgreSQL for persistent storage
  postgres:
    image: postgres:16-alpine
    container_name: agent-postgres
    environment:
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=agent_pass
      - POSTGRES_DB=agent_db
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --locale=en_US.UTF-8
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent_user -d agent_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for pub/sub and caching
  redis:
    image: redis:7-alpine
    container_name: agent-redis
    command: >
      redis-server
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --tcp-backlog 511
      --timeout 0
      --tcp-keepalive 300
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: agent-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: agent-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped

  # Jaeger for distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: agent-jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    networks:
      - agent-net
    logging: *default-logging
    restart: unless-stopped

volumes:
  agent-data:
    driver: local
  mcp-data:
    driver: local
  postgres-data:
    driver: local
  redis-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  static-content:
    driver: local

networks:
  agent-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## MCP Tool Configuration

### MCP Configuration File (mcp-config.json)

```json
{
  "server": {
    "port": 3001,
    "host": "0.0.0.0",
    "cors": {
      "enabled": true,
      "origins": ["http://localhost:3000", "https://agent-viz.example.com"]
    }
  },
  "tools": {
    "agent_spawn": {
      "enabled": true,
      "maxConcurrent": 50,
      "timeout": 30000,
      "validation": {
        "strict": true,
        "schema": "agent-spawn-schema.json"
      }
    },
    "agent_coordinate": {
      "enabled": true,
      "patterns": ["hierarchical", "mesh", "pipeline", "consensus"],
      "maxParticipants": 100
    },
    "multi-agent_init": {
      "enabled": true,
      "maxAgents": 100,
      "topologies": ["star", "ring", "mesh", "hierarchical"]
    },
    "memory_usage": {
      "enabled": true,
      "maxSize": "100MB",
      "persistence": "redis",
      "ttl": 86400
    },
    "neural_train": {
      "enabled": true,
      "models": ["decision", "pattern", "optimization"],
      "gpuAcceleration": true
    }
  },
  "hooks": {
    "preExecution": [
      {
        "name": "validate-permissions",
        "script": "/app/hooks/validate-permissions.js"
      },
      {
        "name": "rate-limit",
        "script": "/app/hooks/rate-limit.js"
      }
    ],
    "postExecution": [
      {
        "name": "log-metrics",
        "script": "/app/hooks/log-metrics.js"
      },
      {
        "name": "update-state",
        "script": "/app/hooks/update-state.js"
      }
    ]
  },
  "integrations": {
    "database": {
      "type": "postgresql",
      "connectionString": "${DATABASE_URL}"
    },
    "cache": {
      "type": "redis",
      "connectionString": "${REDIS_URL}"
    },
    "monitoring": {
      "prometheus": {
        "enabled": true,
        "port": 9091,
        "path": "/metrics"
      },
      "jaeger": {
        "enabled": true,
        "serviceName": "mcp-server",
        "agentHost": "${JAEGER_AGENT_HOST}",
        "agentPort": "${JAEGER_AGENT_PORT}"
      }
    }
  }
}
```

### Custom MCP Tools Implementation

```javascript
// mcp-tools/index.js
const { z } = require('zod');

// Agent visualization specific tools
module.exports = {
  // Update agent position in 3D space
  update_agent_position: {
    description: 'Update agent position in 3D visualization',
    inputSchema: z.object({
      agentId: z.string(),
      position: z.object({
        x: z.number(),
        y: z.number(),
        z: z.number()
      }),
      velocity: z.object({
        x: z.number(),
        y: z.number(),
        z: z.number()
      }).optional()
    }),
    handler: async ({ agentId, position, velocity }) => {
      // Implementation to update agent position
      const redis = require('redis').createClient(process.env.REDIS_URL);

      await redis.hSet(`agent:${agentId}`, {
        posX: position.x,
        posY: position.y,
        posZ: position.z,
        ...(velocity && {
          velX: velocity.x,
          velY: velocity.y,
          velZ: velocity.z
        })
      });

      // Publish position update event
      await redis.publish('agent:position:update', JSON.stringify({
        agentId,
        position,
        velocity,
        timestamp: Date.now()
      }));

      return { success: true, agentId };
    }
  },

  // Coordinate multi-agent formation
  coordinate_multi-agent_formation: {
    description: 'Coordinate agents into specific formation patterns',
    inputSchema: z.object({
      pattern: z.enum(['circle', 'grid', 'sphere', 'helix', 'custom']),
      agents: z.array(z.string()),
      centerPoint: z.object({
        x: z.number(),
        y: z.number(),
        z: z.number()
      }),
      spacing: z.number().default(5),
      customPattern: z.any().optional()
    }),
    handler: async ({ pattern, agents, centerPoint, spacing, customPattern }) => {
      const positions = calculateFormationPositions(
        pattern,
        agents.length,
        centerPoint,
        spacing,
        customPattern
      );

      // Update all agent positions
      const updates = agents.map((agentId, index) => ({
        agentId,
        position: positions[index]
      }));

      // Batch update positions
      await Promise.all(updates.map(update =>
        module.exports.update_agent_position.handler(update)
      ));

      return {
        success: true,
        formation: pattern,
        agentCount: agents.length,
        positions
      };
    }
  },

  // Real-time performance monitoring
  monitor_agent_performance: {
    description: 'Monitor and report agent performance metrics',
    inputSchema: z.object({
      agentId: z.string(),
      metrics: z.object({
        tasksCompleted: z.number(),
        successRate: z.number(),
        averageResponseTime: z.number(),
        resourceUtilization: z.number()
      })
    }),
    handler: async ({ agentId, metrics }) => {
      // Store metrics in time-series database
      const timestamp = Date.now();

      // Update current metrics
      await redis.hSet(`agent:${agentId}:metrics`, {
        ...metrics,
        lastUpdate: timestamp
      });

      // Store historical data
      await redis.zAdd(`agent:${agentId}:metrics:history`, {
        score: timestamp,
        value: JSON.stringify({ timestamp, ...metrics })
      });

      // Calculate trends
      const trends = await calculatePerformanceTrends(agentId);

      // Publish metrics update
      await redis.publish('agent:metrics:update', JSON.stringify({
        agentId,
        metrics,
        trends,
        timestamp
      }));

      return {
        success: true,
        agentId,
        metrics,
        trends
      };
    }
  }
};

// Helper functions
function calculateFormationPositions(pattern, count, center, spacing, custom) {
  const positions = [];

  switch (pattern) {
    case 'circle':
      for (let i = 0; i < count; i++) {
        const angle = (i / count) * Math.PI * 2;
        positions.push({
          x: center.x + Math.cos(angle) * spacing,
          y: center.y,
          z: center.z + Math.sin(angle) * spacing
        });
      }
      break;

    case 'grid':
      const gridSize = Math.ceil(Math.sqrt(count));
      for (let i = 0; i < count; i++) {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        positions.push({
          x: center.x + (col - gridSize / 2) * spacing,
          y: center.y,
          z: center.z + (row - gridSize / 2) * spacing
        });
      }
      break;

    case 'sphere':
      // Fibonacci sphere algorithm
      const phi = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < count; i++) {
        const y = 1 - (i / (count - 1)) * 2;
        const radius = Math.sqrt(1 - y * y);
        const theta = phi * i;
        positions.push({
          x: center.x + radius * Math.cos(theta) * spacing,
          y: center.y + y * spacing,
          z: center.z + radius * Math.sin(theta) * spacing
        });
      }
      break;

    case 'custom':
      if (custom && custom.positions) {
        return custom.positions;
      }
      break;
  }

  return positions;
}

async function calculatePerformanceTrends(agentId) {
  // Fetch historical metrics
  const history = await redis.zRange(
    `agent:${agentId}:metrics:history`,
    -100,  // Last 100 data points
    -1,
    { withScores: true }
  );

  // Calculate trends
  const metrics = history.map(h => JSON.parse(h.value));

  return {
    successRateTrend: calculateTrend(metrics.map(m => m.successRate)),
    responseTrend: calculateTrend(metrics.map(m => m.averageResponseTime)),
    utilizationTrend: calculateTrend(metrics.map(m => m.resourceUtilization))
  };
}

function calculateTrend(values) {
  if (values.length < 2) return 'stable';

  const recent = values.slice(-10);
  const older = values.slice(-20, -10);

  const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
  const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;

  const change = (recentAvg - olderAvg) / olderAvg;

  if (change > 0.1) return 'increasing';
  if (change < -0.1) return 'decreasing';
  return 'stable';
}
```

## Startup Script

### entrypoint.sh

```bash
#!/bin/sh
set -e

# Wait for dependencies
echo "Waiting for PostgreSQL..."
until pg_isready -h ${DATABASE_HOST:-postgres} -p ${DATABASE_PORT:-5432}; do
  sleep 1
done

echo "Waiting for Redis..."
until redis-cli -h ${REDIS_HOST:-redis} ping; do
  sleep 1
done

echo "Waiting for MCP Server..."
until curl -f http://${MCP_HOST:-mcp-server}:${MCP_PORT:-3001}/health; do
  sleep 1
done

# Run database migrations
echo "Running database migrations..."
/usr/local/bin/agent-server migrate

# Start the server
echo "Starting agent visualization server..."
exec /usr/local/bin/agent-server
```

## Database Schema

### PostgreSQL Initialization (init.sql)

```sql
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Agent table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL DEFAULT 'IDLE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',

    -- Indexes
    INDEX idx_agents_type (type),
    INDEX idx_agents_state (state),
    INDEX idx_agents_metadata_gin (metadata gin_trgm_ops)
);

-- Agent positions (for historical tracking)
CREATE TABLE agent_positions (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    position_x REAL NOT NULL,
    position_y REAL NOT NULL,
    position_z REAL NOT NULL,
    velocity_x REAL DEFAULT 0,
    velocity_y REAL DEFAULT 0,
    velocity_z REAL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Partitioning by month
    INDEX idx_positions_agent_timestamp (agent_id, timestamp DESC)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE agent_positions_2024_01 PARTITION OF agent_positions
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Performance metrics
CREATE TABLE agent_metrics (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tasks_completed INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0,
    avg_response_time REAL DEFAULT 0,
    resource_utilization REAL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_metrics_agent_timestamp (agent_id, timestamp DESC)
);

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_agent UUID NOT NULL REFERENCES agents(id),
    to_agent UUID NOT NULL REFERENCES agents(id),
    message_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) DEFAULT 'NORMAL',
    content JSONB NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    received_at TIMESTAMP WITH TIME ZONE,
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,

    INDEX idx_messages_from_to (from_agent, to_agent),
    INDEX idx_messages_timestamp (sent_at DESC)
);

-- Coordination instances
CREATE TABLE coordinations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'FORMING',
    progress REAL DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',

    INDEX idx_coordinations_status (status),
    INDEX idx_coordinations_pattern (pattern)
);

-- Coordination participants
CREATE TABLE coordination_participants (
    coordination_id UUID REFERENCES coordinations(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'PARTICIPANT',
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (coordination_id, agent_id)
);

-- Functions and triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Performance views
CREATE MATERIALIZED VIEW agent_performance_summary AS
SELECT
    a.id,
    a.name,
    a.type,
    a.state,
    COUNT(DISTINCT m.id) as total_messages,
    AVG(m.latency_ms) as avg_message_latency,
    COUNT(DISTINCT c.id) as coordination_count,
    MAX(am.tasks_completed) as tasks_completed,
    AVG(am.success_rate) as avg_success_rate
FROM agents a
LEFT JOIN messages m ON a.id IN (m.from_agent, m.to_agent)
LEFT JOIN coordination_participants cp ON a.id = cp.agent_id
LEFT JOIN coordinations c ON cp.coordination_id = c.id
LEFT JOIN agent_metrics am ON a.id = am.agent_id
GROUP BY a.id, a.name, a.type, a.state;

-- Refresh materialized view every 5 minutes
CREATE OR REPLACE FUNCTION refresh_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY agent_performance_summary;
END;
$$ LANGUAGE plpgsql;
```

## Monitoring Configuration

### Prometheus Configuration (prometheus.yml)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'agent-viz'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Load rules
rule_files:
  - "alerts/*.yml"

# Scrape configurations
scrape_configs:
  # Agent visualization metrics
  - job_name: 'agent-visualizer'
    static_configs:
      - targets: ['agent-visualizer:9091']
    metrics_path: '/metrics'
    scrape_interval: 5s

  # MCP server metrics
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['mcp-server:9091']
    metrics_path: '/metrics'

  # PostgreSQL exporter
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Node exporter for system metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

## Security Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://agent_user:secure_password@postgres:5432/agent_db
DATABASE_SSL_MODE=require

# Redis
REDIS_URL=redis://:redis_password@redis:6379/0
REDIS_TLS=true

# MCP Integration
MCP_API_KEY=your-secure-api-key-here
MCP_SERVER_URL=http://mcp-server:3001
MCP_VERIFY_SSL=true

# JWT Configuration
JWT_SECRET=your-very-secure-jwt-secret
JWT_EXPIRY=3600

# OAuth2 (optional)
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-client-secret
OAUTH_REDIRECT_URI=https://your-domain.com/auth/callback

# Monitoring
GRAFANA_PASSWORD=secure_admin_password

# Rate Limiting
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=100
```

## Deployment Scripts

### deploy.sh

```bash
#!/bin/bash
set -e

# Load environment variables
source .env

# Build images
docker-compose build --parallel

# Run database migrations
docker-compose run --rm agent-visualizer /usr/local/bin/agent-server migrate

# Start services
docker-compose up -d

# Wait for health checks
echo "Waiting for services to be healthy..."
sleep 10

# Verify deployment
docker-compose ps
docker-compose logs --tail=50

echo "Deployment complete!"
```

## Troubleshooting

### Common Issues and Solutions

1. **WebSocket Connection Failures**
   ```bash
   # Check WebSocket service logs
   docker logs agent-viz -f | grep "WebSocket"

   # Verify port is accessible
   curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8080/ws
   ```

2. **MCP Tool Execution Errors**
   ```bash
   # Check MCP server logs
   docker logs mcp-server -f

   # Test MCP tool directly
   docker exec mcp-server claude-flow tools test agent_spawn
   ```

3. **Performance Issues**
   ```bash
   # Monitor resource usage
   docker stats

   # Check Prometheus metrics
   curl http://localhost:9090/metrics | grep agent_
   ```

This completes the comprehensive Docker and MCP integration setup for the agent visualization system.