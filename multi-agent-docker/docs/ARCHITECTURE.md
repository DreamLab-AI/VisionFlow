# Architecture

## Overview

The Agentic Flow Docker environment is a multi-container system designed for GPU-accelerated AI agent development with intelligent multi-model orchestration, comprehensive MCP tool integration, and production-grade task management.

## System Components

### 1. Main Workstation Container (agentic-flow-cachyos)

**Base Image**: ArchLinux (CachyOS optimized)

**Purpose**: Primary development and execution environment

**Key Components**:
- Management API (Node.js/Fastify)
- Multi-model router
- MCP tool ecosystem
- Optional desktop environment (XFCE4 + VNC)
- Optional VS Code server
- Development toolchain

**Resource Allocation**:
```yaml
Limits:
  Memory: 64GB
  CPUs: 32 cores
  GPU: All available NVIDIA GPUs

Reservations:
  Memory: 16GB
  CPUs: 8 cores
  GPU: All with compute+utility capabilities
```

**Volumes**:
- `workspace`: User workspace files (`/home/devuser/workspace`)
- `model-cache`: Downloaded AI models (`/home/devuser/models`)
- `agent-memory`: Agent session data (`/home/devuser/.agentic-flow`)
- `config-persist`: Configuration files (`/home/devuser/.config`)
- `management-logs`: API and task logs (`/home/devuser/logs`)

**Exposed Ports**:
- `9090`: Management API (primary interface)
- `5901`: VNC server (if desktop enabled)
- `6901`: noVNC web interface (if desktop enabled)
- `8080`: VS Code server (if code server enabled)

### 2. Claude-ZAI Service

**Base Image**: Node.js (Alpine-based)

**Purpose**: High-performance Claude AI wrapper with Z.AI integration

**Key Features**:
- Worker pool for concurrent requests
- Queue management
- Automatic retry logic
- Health monitoring

**Configuration**:
```javascript
Environment:
  ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
  ANTHROPIC_AUTH_TOKEN: ${ZAI_API_KEY}
  CLAUDE_WORKER_POOL_SIZE: 4 (default)
  CLAUDE_MAX_QUEUE_SIZE: 50 (default)
```

**Exposed Ports**:
- `9600`: HTTP API for Claude requests

**Restart Policy**: `unless-stopped`

### 3. Management API

**Framework**: Fastify (Node.js)

**Purpose**: RESTful HTTP API for external task management and system monitoring

**Architecture**:
```
Management API Server
├── Authentication Middleware
│   └── Bearer token validation
├── Rate Limiting
│   └── 100 requests/minute per IP
├── Task Manager
│   ├── Process spawning
│   ├── Task isolation
│   └── Log management
├── System Monitor
│   ├── GPU monitoring (nvidia-smi)
│   ├── Provider status
│   └── System resources
└── Metrics Collector
    ├── HTTP metrics
    ├── Task metrics
    └── Performance tracking
```

**Task Isolation Strategy**:
```
Each task runs in isolated environment:
/home/devuser/workspace/tasks/{taskId}/
  ├── .db files (SQLite databases)
  ├── generated code
  └── task artifacts

/home/devuser/logs/tasks/{taskId}.log
  └── stdout/stderr output
```

**Process Management**: Supervised by pm2

## Multi-Model Router

**Purpose**: Intelligent routing across multiple AI providers with automatic fallback

### Router Modes

1. **Performance Mode** (default)
   - Routes to fastest available provider
   - Optimizes for latency
   - Best for interactive tasks

2. **Quality Mode**
   - Routes to highest quality model
   - Optimizes for output quality
   - Best for critical tasks

3. **Cost Mode**
   - Routes to most economical provider
   - Optimizes for API costs
   - Best for batch processing

4. **Balanced Mode**
   - Weighted routing across all factors
   - Configurable priorities
   - Best for general use

### Supported Providers

| Provider | Models | Features | Priority |
|----------|--------|----------|----------|
| **Gemini** | gemini-2.5-flash, gemini-2.5-pro | 1M context, streaming, tool calling | 1 |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo | 128K context, streaming, tool calling | 2 |
| **Claude** | claude-3-opus, claude-3-sonnet, claude-3-haiku | 200K context, streaming, tool calling | 3 |
| **OpenRouter** | Multiple models | Universal access, various models | 4 |

### Fallback Chain

```
Request → Primary Provider (e.g., Gemini)
           ↓ (if fails)
         Fallback #1 (e.g., OpenAI)
           ↓ (if fails)
         Fallback #2 (e.g., Claude)
           ↓ (if fails)
         Fallback #3 (e.g., OpenRouter)
           ↓ (if fails)
         Error Response
```

### Router Configuration

Located at: `/home/devuser/.config/agentic-flow/router.config.json`

```json
{
  "version": "1.0.0",
  "mode": "performance",
  "providers": {
    "gemini": {
      "enabled": true,
      "priority": 1,
      "metrics": {
        "speed": 95,
        "quality": 85,
        "cost": 98
      }
    }
  }
}
```

## MCP (Model Context Protocol) Integration

**Purpose**: Provide Claude with external tools and capabilities

### MCP Architecture

```
Claude Instance
    ↓ (stdio)
MCP Server (per tool)
    ↓
Tool Implementation
    ↓
External Resource
```

### Available MCP Tools

| Tool | Purpose | Protocol |
|------|---------|----------|
| **claude-flow** | Agentic workflow orchestration | stdio |
| **context7** | Up-to-date code documentation | stdio |
| **playwright** | Browser automation | stdio |
| **filesystem** | Workspace file operations | stdio |
| **git** | Git version control | stdio |
| **github** | GitHub API operations | stdio |
| **fetch** | HTTP requests | stdio |
| **brave-search** | Web search | stdio |
| **web-summary** | Web content summarization | stdio |

### MCP Configuration

Located at: `/home/devuser/.config/claude/mcp.json`

**Tool Categories**:
- **documentation**: context7
- **automation**: playwright
- **filesystem**: filesystem, git
- **web**: fetch, brave-search, web-summary
- **github**: github
- **workflows**: claude-flow

**On-Demand Spawning**:
- Each worker session spawns its own tool instances
- Tools only run when needed
- stdio communication (no HTTP servers)
- Automatic cleanup on session end

## Network Architecture

### Internal Network: `agentic-network`

```
Docker Bridge Network (agentic-network)
├── agentic-flow-cachyos (172.20.0.2)
│   ├── Management API: 9090
│   ├── Desktop: 5901, 6901
│   └── Code Server: 8080
└── claude-zai-service (172.20.0.3)
    └── API: 9600
```

### External Network Integration

**RAGFlow Integration** (optional):
- Connects to `docker_ragflow` network if available
- Enables communication with RAGFlow services
- Automatic connection on startup

### Port Mapping

```
Host → Container
9090 → 9090 (Management API)
9600 → 9600 (Claude-ZAI)
5901 → 5901 (VNC) *
6901 → 6901 (noVNC) *
8080 → 8080 (Code Server) *

* Only if optional service enabled
```

## GPU Architecture

### GPU Access Strategy

**Runtime**: NVIDIA Container Runtime

**Device Access**:
```yaml
devices:
  - /dev/dri:/dev/dri
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
  - /dev/nvidia-modeset:/dev/nvidia-modeset
```

**Capabilities**:
- `gpu`: GPU device access
- `compute`: CUDA compute
- `utility`: nvidia-smi and monitoring

### GPU Monitoring

**nvidia-smi Integration**:
```javascript
{
  "gpu": {
    "available": true,
    "gpus": [
      {
        "index": 0,
        "name": "NVIDIA RTX 4090",
        "utilization": 45.5,
        "memory": {
          "used": 8192,
          "total": 24576,
          "percentUsed": "33.33"
        },
        "temperature": 65
      }
    ]
  }
}
```

## Task Execution Architecture

### Task Lifecycle

```
1. Client submits task via API
   POST /v1/tasks
   ↓
2. Management API validates request
   - Check authentication
   - Validate parameters
   - Check system resources
   ↓
3. Process Manager creates isolated task
   - Generate unique task ID
   - Create task directory
   - Setup log file
   ↓
4. Spawn agentic-flow process
   - Set environment variables
   - Set working directory
   - Redirect output to log
   ↓
5. Task executes
   - Load model router
   - Initialize MCP tools
   - Execute agent logic
   ↓
6. Task completes
   - Capture exit code
   - Finalize logs
   - Update task status
   ↓
7. Client retrieves results
   GET /v1/tasks/{taskId}
```

### Task Isolation Benefits

1. **Database Isolation**: Each task has its own SQLite database
2. **Process Isolation**: Tasks run in separate processes
3. **Log Isolation**: Separate log files for debugging
4. **Resource Control**: Per-task resource limits
5. **Cleanup**: Automatic cleanup on task completion

## Data Flow Architecture

### Request Flow

```
External Client
    ↓ HTTP
Management API (9090)
    ├─→ Task Manager
    │   ├─→ Process Spawner
    │   └─→ Log Manager
    ├─→ System Monitor
    │   ├─→ GPU Monitor
    │   └─→ Provider Status
    └─→ Metrics Collector
        └─→ Performance Tracking

Task Process
    ↓
Model Router
    ├─→ Gemini API
    ├─→ OpenAI API
    ├─→ Claude API (via ZAI)
    └─→ OpenRouter API

MCP Tools
    ├─→ Claude Flow
    ├─→ Context7
    ├─→ Playwright
    └─→ Filesystem/Git
```

### Claude-ZAI Request Flow

```
Application
    ↓ HTTP POST
Claude-ZAI Service (9600)
    ↓
Worker Pool
    ├─→ Worker 1 (idle)
    ├─→ Worker 2 (busy)
    ├─→ Worker 3 (busy)
    └─→ Worker 4 (idle)
    ↓
Z.AI API
    ↓
Anthropic Claude API
```

## Storage Architecture

### Volume Strategy

All persistent data stored in Docker named volumes:

```
volumes:
  workspace:           # User files and projects
  model-cache:         # Downloaded AI models
  agent-memory:        # Session and state data
  config-persist:      # Configuration files
  management-logs:     # API and task logs
```

### Directory Structure

```
/home/devuser/
├── workspace/             # workspace volume
│   ├── projects/
│   └── tasks/            # Task isolation directories
│       └── {taskId}/
├── models/               # model-cache volume
│   ├── gemini/
│   ├── openai/
│   └── local/
├── .agentic-flow/        # agent-memory volume
│   ├── memory/
│   ├── metrics/
│   └── sessions/
├── .config/              # config-persist volume
│   ├── agentic-flow/
│   │   └── router.config.json
│   └── claude/
│       └── mcp.json
└── logs/                 # management-logs volume
    ├── management-api.log
    └── tasks/
        └── {taskId}.log
```

## Security Architecture

### Authentication

**Management API**:
- Bearer token authentication
- Rate limiting (100 req/min)
- Exempt endpoints: `/health`, `/ready`, `/metrics`

**Claude-ZAI**:
- No authentication (internal network only)
- Should not be exposed externally

### Network Security

1. **Internal Network**: Services communicate via Docker bridge network
2. **External Access**: Only required ports exposed to host
3. **Firewall**: Recommended to restrict access to 9090, 9600
4. **TLS**: Should use reverse proxy for HTTPS in production

### Container Security

```yaml
security_opt:
  - seccomp:unconfined    # Required for GPU access
  - apparmor:unconfined   # Required for GPU access

cap_add:
  - SYS_ADMIN            # Required for certain operations
  - SYS_PTRACE           # Required for debugging
```

## Monitoring Architecture

### Health Checks

**Management API**:
```yaml
healthcheck:
  test: curl -f http://localhost:9090/health
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

**Claude-ZAI**:
```yaml
healthcheck:
  test: curl -f http://localhost:9600/health
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 30s
```

### Metrics Collection

**Available Metrics**:
- HTTP request metrics (method, path, status, duration)
- Task metrics (created, completed, failed)
- GPU metrics (utilization, memory, temperature)
- System metrics (CPU, memory, disk)
- Provider metrics (requests, errors, latency)

### Logging Architecture

**Structured Logging**:
```json
{
  "level": "info",
  "time": "2025-01-12T17:00:00.000Z",
  "reqId": "req-123",
  "msg": "Request completed",
  "responseTime": 45
}
```

**Log Destinations**:
- **Container stdout**: Docker logs
- **Log files**: Persistent volume storage
- **pm2 logs**: Process manager logs

## Scalability Considerations

### Horizontal Scaling

**Current**: Single-container deployment

**Future**:
- Multiple workstation containers
- Load balancer for Management API
- Shared storage for tasks
- Distributed task queue

### Vertical Scaling

**CPU/Memory**:
```yaml
deploy:
  resources:
    limits:
      memory: 64G    # Adjustable
      cpus: '32'     # Adjustable
```

**GPU**:
```yaml
CUDA_VISIBLE_DEVICES: all  # or specific GPUs: 0,1,2
```

### Performance Optimization

1. **Model Caching**: Cache downloaded models
2. **Connection Pooling**: Reuse HTTP connections
3. **Worker Pools**: Multiple Claude-ZAI workers
4. **Task Queuing**: Queue tasks when system busy
5. **Resource Limits**: Prevent resource exhaustion

## Deployment Modes

### 1. Standalone (Production)

**Characteristics**:
- Installs `agentic-flow` from npm
- Minimal dependencies
- Production-ready
- Self-contained

**Use Cases**:
- Production deployment
- User installations
- Cloud deployment

### 2. Development

**Characteristics**:
- Uses local source code
- Full repository required
- Hot reload support
- Debug capabilities

**Use Cases**:
- Development
- Testing
- Contributing

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Base OS** | ArchLinux (CachyOS) |
| **Container Runtime** | Docker + NVIDIA Runtime |
| **Management API** | Node.js + Fastify |
| **Process Manager** | pm2 |
| **Desktop** | XFCE4 + TigerVNC |
| **GPU** | NVIDIA CUDA |
| **AI Providers** | Gemini, OpenAI, Claude, OpenRouter |
| **Protocol** | MCP (Model Context Protocol) |
| **Shell** | zsh + Oh My Zsh |

## Design Principles

1. **Isolation**: Tasks run in isolated environments
2. **Resilience**: Automatic restart and recovery
3. **Observability**: Comprehensive logging and metrics
4. **Flexibility**: Multiple providers and fallback chains
5. **Security**: Authentication and rate limiting
6. **Scalability**: Resource limits and GPU support
7. **Developer Experience**: Full development environment

## Future Architecture Enhancements

### Planned
- Distributed task queue (Redis/RabbitMQ)
- Horizontal scaling support
- Enhanced monitoring (Prometheus/Grafana)
- Service mesh integration
- Multi-region deployment
- Advanced load balancing

### Under Consideration
- Kubernetes deployment
- Event-driven architecture
- Microservices decomposition
- Real-time streaming API
- WebSocket support for task updates
