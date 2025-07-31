# Agentic-Flow Integration Guide

## Overview

This document provides a comprehensive guide for integrating the rich data streams and control mechanisms from the agentic-flow system into our multi-agent visualization platform. The integration brings advanced real-time agent monitoring, coordination visualization, and control capabilities.

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Three.js)              │
├─────────────────────────────────────────────────────────────┤
│  - Enhanced Agent Nodes (Multi-layered 3D visualization)     │
│  - Floating Dashboard Panels                                 │
│  - Message Flow Visualization                                │
│  - WebSocket Client (Binary + JSON protocols)               │
└─────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        ┌───────────▼────────────┐ ┌─────────▼──────────┐
        │   WebSocket Server     │ │   REST API Server   │
        ├────────────────────────┤ ├────────────────────┤
        │ - Binary position data │ │ - Agent management  │
        │ - JSON control msgs    │ │ - Swarm init        │
        │ - Real-time events     │ │ - MCP tool calls    │
        └────────────────────────┘ └────────────────────┘
                    │                         │
        ┌───────────▼─────────────────────────▼──────────┐
        │            Agent Orchestration Layer            │
        ├─────────────────────────────────────────────────┤
        │ - Claude-Flow MCP Integration                   │
        │ - Agent State Management                        │
        │ - Message Bus & Coordination                    │
        │ - Performance Metrics Collection                │
        └─────────────────────────────────────────────────┘
```

## Data Streams

### 1. Agent State Stream

**WebSocket Endpoint**: `/api/visualization/agents/ws`  
**Protocol**: Binary (position/velocity) + JSON (state updates)  
**Frequency**: 60 FPS for positions, event-based for state

#### Binary Protocol Format (28 bytes per agent)
```
Offset  Size  Type      Description
0       4     uint32    Agent ID
4       4     float32   Position X
8       4     float32   Position Y
12      4     float32   Position Z
16      4     float32   Velocity X
20      4     float32   Velocity Y
24      4     float32   Velocity Z
```

#### JSON State Update Schema
```typescript
interface AgentStateUpdate {
  type: 'agent_update';
  agent: {
    id: string;
    name: string;
    type: AgentType;
    state: AgentState;
    performance: {
      tasksCompleted: number;
      successRate: number;
      averageResponseTime: number;
      resourceUtilization: number;
    };
    capabilities: AgentCapability[];
    goals: Goal[];
    metadata: {
      lastActivity: Date;
      processingLogs: string[];
      teamId?: string;
      teamRole?: string;
    };
  };
}
```

### 2. Message Flow Stream

**Event Type**: `message_event`  
**Update Frequency**: Real-time as messages occur

```typescript
interface MessageFlowUpdate {
  type: 'message_event';
  message: {
    id: string;
    from: AgentId;
    to: AgentId | AgentId[];
    type: MessageType;
    priority: MessagePriority;
    content: any;
    timestamp: Date;
    latency: number;
    success: boolean;
  };
}
```

### 3. Coordination Pattern Stream

**Event Type**: `coordination_event`  
**Patterns**: Hierarchical, Mesh, Pipeline, Consensus, Barrier

```typescript
interface CoordinationUpdate {
  type: 'coordination_event';
  coordination: {
    id: string;
    pattern: CoordinationPattern;
    participants: AgentId[];
    status: 'forming' | 'active' | 'completing' | 'completed';
    progress: number;
    metadata: {
      consensusThreshold?: number;
      barrierCount?: number;
      pipelineStages?: string[];
    };
  };
}
```

### 4. System Metrics Stream

**Event Type**: `system_metrics`  
**Update Frequency**: Every 1 second

```typescript
interface SystemMetricsUpdate {
  type: 'system_metrics';
  metrics: {
    activeAgents: number;
    totalAgents: number;
    messageRate: number;
    averageLatency: number;
    errorRate: number;
    networkHealth: number;
    resourceUtilization: {
      cpu: number;
      memory: number;
      gpu?: number;
    };
  };
}
```

## Control Mechanisms

### 1. Agent Control API

#### Spawn Agent
```typescript
POST /api/agents/spawn
{
  "name": "Research Agent",
  "type": "researcher",
  "capabilities": ["web_search", "document_analysis"],
  "goals": [{
    "description": "Analyze market trends",
    "type": "ACHIEVE",
    "priority": "HIGH"
  }]
}
```

#### Send Agent Command
```typescript
POST /api/agents/:agentId/command
{
  "command": "execute_task",
  "params": {
    "task": "analyze_document",
    "document_url": "https://example.com/doc.pdf"
  }
}
```

#### Update Agent State
```typescript
PUT /api/agents/:agentId/state
{
  "state": "EXECUTING",
  "metadata": {
    "currentTask": "Document analysis in progress"
  }
}
```

### 2. MCP Tool Integration

The system integrates with Claude-Flow MCP tools through a standardized interface:

```typescript
interface MCPToolCall {
  tool: string;
  arguments: Record<string, any>;
  agentId: string;
  requestId: string;
}

// Example: Agent spawning through MCP
{
  "tool": "agent_spawn",
  "arguments": {
    "name": "Data Analyzer",
    "type": "analyzer",
    "capabilities": ["data_processing", "visualization"]
  },
  "agentId": "coordinator-001",
  "requestId": "req-12345"
}
```

### 3. Swarm Coordination Controls

#### Initialize Swarm
```typescript
POST /api/swarms/initialize
{
  "topology": "hierarchical",
  "agents": [
    { "type": "coordinator", "count": 1 },
    { "type": "executor", "count": 5 },
    { "type": "monitor", "count": 2 }
  ],
  "goals": [{
    "description": "Complete project implementation",
    "deadline": "2024-01-15T00:00:00Z"
  }]
}
```

#### Coordinate Pipeline
```typescript
POST /api/coordination/pipeline
{
  "stages": [
    { "agentId": "researcher-001", "task": "gather_requirements" },
    { "agentId": "architect-001", "task": "design_system" },
    { "agentId": "coder-001", "task": "implement_features" },
    { "agentId": "tester-001", "task": "validate_implementation" }
  ],
  "timeoutMs": 300000
}
```

## Docker Integration Requirements

### 1. Container Configuration

```dockerfile
# Multi-stage build for agent visualization system
FROM node:20-alpine AS frontend-builder
WORKDIR /app/client
COPY client/package*.json ./
RUN npm ci --frozen-lockfile
COPY client/ ./
RUN npm run build

FROM rust:1.75 AS backend-builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts
COPY --from=backend-builder /app/target/release/agent-server /usr/local/bin/
COPY --from=frontend-builder /app/client/dist /usr/share/nginx/html

# Environment variables
ENV RUST_LOG=info
ENV MCP_SERVER_URL=http://claude-flow-mcp:3001
ENV WEBSOCKET_PORT=8080
ENV API_PORT=3000

EXPOSE 3000 8080

CMD ["agent-server"]
```

### 2. Docker Compose Services

```yaml
version: '3.8'

services:
  agent-visualization:
    build: .
    ports:
      - "3000:3000"     # REST API
      - "8080:8080"     # WebSocket
    environment:
      - MCP_SERVER_URL=http://claude-flow:3001
      - DATABASE_URL=postgresql://postgres:password@db:5432/agents
      - REDIS_URL=redis://redis:6379
      - ENABLE_GPU_PHYSICS=true
    volumes:
      - agent-data:/var/lib/agents
    depends_on:
      - claude-flow
      - db
      - redis
    networks:
      - agent-network

  claude-flow:
    image: claude-flow:latest
    ports:
      - "3001:3001"
    environment:
      - MCP_MODE=server
      - ENABLE_AGENT_TOOLS=true
    volumes:
      - claude-flow-data:/var/lib/claude-flow
    networks:
      - agent-network

  db:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=agents
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - agent-network

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    networks:
      - agent-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - agent-visualization
    networks:
      - agent-network

volumes:
  agent-data:
  claude-flow-data:
  postgres-data:
  redis-data:

networks:
  agent-network:
    driver: bridge
```

### 3. Nginx Configuration

```nginx
upstream api_backend {
    server agent-visualization:3000;
}

upstream ws_backend {
    server agent-visualization:8080;
}

server {
    listen 80;
    server_name _;

    # Upgrade to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    # REST API
    location /api {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket connections
    location /ws {
        proxy_pass http://ws_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # WebSocket specific timeouts
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

## Environment Variables

### Required Environment Variables

```bash
# MCP Integration
MCP_SERVER_URL=http://claude-flow:3001
MCP_API_KEY=your-api-key

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Redis (for pub/sub and caching)
REDIS_URL=redis://redis:6379

# WebSocket Configuration
WEBSOCKET_PORT=8080
WEBSOCKET_MAX_CONNECTIONS=1000
WEBSOCKET_HEARTBEAT_INTERVAL=30000

# Performance
ENABLE_GPU_PHYSICS=true
MAX_VISIBLE_AGENTS=500
PHYSICS_UPDATE_RATE=60

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

## Implementation Checklist

### Backend Requirements

- [ ] WebSocket server with binary protocol support
- [ ] MCP client integration for claude-flow
- [ ] Agent state management with persistence
- [ ] Message bus implementation
- [ ] Performance metrics collection
- [ ] GraphQL subscriptions (optional)
- [ ] Authentication and authorization
- [ ] Rate limiting and DDoS protection

### Frontend Requirements

- [ ] WebSocket client with auto-reconnect
- [ ] Binary protocol parser
- [ ] React state management (Zustand/Redux)
- [ ] Three.js scene optimization
- [ ] GPU physics simulation
- [ ] Performance monitoring
- [ ] Error boundary implementation
- [ ] Progressive loading for large swarms

### Infrastructure Requirements

- [ ] Docker multi-stage builds
- [ ] Kubernetes deployment manifests (optional)
- [ ] Prometheus metrics export
- [ ] Grafana dashboards
- [ ] Log aggregation (ELK stack)
- [ ] SSL/TLS certificates
- [ ] CDN for static assets
- [ ] Load balancer configuration

## Performance Optimization

### GPU Physics Acceleration

The system uses GPU-accelerated physics for agent positioning:

```glsl
// Vertex shader for spring physics
attribute vec3 position;
attribute vec3 velocity;
uniform float deltaTime;
uniform float springStrength;
uniform float damping;

void main() {
    vec3 force = calculateSpringForce(position);
    velocity = velocity * damping + force * deltaTime;
    position = position + velocity * deltaTime;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

### Binary Protocol Optimization

For 1000+ agents, the binary protocol reduces bandwidth by 85%:

- JSON: ~500 bytes per agent update
- Binary: 28 bytes per agent update
- Compression: Additional 30% reduction with zlib

### Connection Pooling

```javascript
const wsPool = new WebSocketPool({
    maxConnections: 4,
    loadBalancer: 'round-robin',
    reconnectStrategy: 'exponential-backoff'
});
```

## Security Considerations

### Authentication

```typescript
// JWT-based authentication for WebSocket
const token = await getAuthToken();
const ws = new WebSocket(`wss://api.example.com/ws?token=${token}`);
```

### Rate Limiting

```typescript
// Rate limit configuration
const rateLimiter = {
    windowMs: 60000, // 1 minute
    maxRequests: {
        spawn_agent: 10,
        send_command: 100,
        update_state: 200
    }
};
```

### Input Validation

```typescript
// Zod schema for agent commands
const AgentCommandSchema = z.object({
    command: z.enum(['execute_task', 'update_goal', 'coordinate']),
    params: z.record(z.any()),
    timestamp: z.string().datetime()
});
```

## Monitoring and Observability

### Metrics to Track

1. **Agent Metrics**
   - Active agent count
   - Agent state distribution
   - Task completion rate
   - Average response time

2. **System Metrics**
   - WebSocket connection count
   - Message throughput
   - CPU/Memory usage
   - GPU utilization

3. **Network Metrics**
   - Bandwidth usage
   - Latency percentiles
   - Connection drops
   - Error rates

### Prometheus Metrics

```typescript
// Custom metrics
const agentGauge = new prometheus.Gauge({
    name: 'active_agents_total',
    help: 'Total number of active agents',
    labelNames: ['type', 'state']
});

const messageCounter = new prometheus.Counter({
    name: 'messages_processed_total',
    help: 'Total messages processed',
    labelNames: ['type', 'priority']
});

const latencyHistogram = new prometheus.Histogram({
    name: 'message_latency_seconds',
    help: 'Message processing latency',
    buckets: [0.001, 0.01, 0.1, 0.5, 1, 5]
});
```

## Next Steps

1. **Phase 1**: Implement WebSocket server with binary protocol
2. **Phase 2**: Integrate MCP tools and agent management
3. **Phase 3**: Add performance monitoring and optimization
4. **Phase 4**: Deploy with Docker and monitoring stack
5. **Phase 5**: Scale testing and optimization

This integration brings enterprise-grade agent visualization and control capabilities to our platform, enabling real-time monitoring and coordination of complex multi-agent systems.