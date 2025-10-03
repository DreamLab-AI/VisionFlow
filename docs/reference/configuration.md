# Configuration Reference

[← Knowledge Base](../index.md) > [Reference](./index.md) > Configuration

Complete configuration reference for the VisionFlow AR-AI Knowledge Graph system, covering environment variables, service parameters, GPU/CUDA settings, multi-agent configuration, and network ports.

## Table of Contents

- [System Configuration](#system-configuration)
- [Database Configuration](#database-configuration)
- [GPU & CUDA Configuration](#gpu--cuda-configuration)
- [Multi-Agent Configuration](#multi-agent-configuration)
- [Network & Port Configuration](#network--port-configuration)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Configuration Files](#configuration-files)
- [Best Practices](#best-practices)

## System Configuration

### Core System Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ENV` | `development` | Runtime environment (`development`, `staging`, `production`) |
| `PORT` | `3000` | Main application port |
| `HOST` | `0.0.0.0` | Host binding address |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warn`, `error`) |
| `ENABLE_CLUSTERING` | `false` | Enable Node.js cluster mode |
| `WORKERS` | CPU cores | Number of worker processes |

### Docker Resource Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_MEMORY` | `16g` | Memory limit for containers |
| `DOCKER_CPUS` | `4` | CPU core allocation |

### External Services

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | - | GitHub API token |
| `GITHUB_WEBHOOK_SECRET` | - | GitHub webhook secret |
| `GITHUB_CLIENT_ID` | - | OAuth client ID |
| `GITHUB_CLIENT_SECRET` | - | OAuth client secret |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (required for agents) |
| `RAGFLOW_URL` | - | RAGFlow service URL |
| `NOSTR_RELAY_URL` | - | Nostr relay URL |

## Database Configuration

### PostgreSQL Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string (required) |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `visionflow` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | - | Database password |
| `DB_POOL_SIZE` | `20` | Connection pool size |
| `DB_POOL_TIMEOUT` | `10000` | Pool timeout in milliseconds |

### PostgreSQL Optimisation

```sql
-- Recommended production settings
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
```

### Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Redis password |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_KEY_PREFIX` | `vf:` | Key prefix for namespacing |

### Redis Optimisation

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save ""
appendonly no
tcp-keepalive 60
timeout 300
```

## GPU & CUDA Configuration

### GPU Compute Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_GPU` | `true` | Enable GPU acceleration |
| `CUDA_DEVICE` | `0` | CUDA device index |
| `GPU_MEMORY_LIMIT` | `4096` | GPU memory limit in MB |
| `FORCE_ITERATIONS` | `500` | Force-directed layout iterations |
| `ENABLE_SPATIAL_HASHING` | `true` | Enable spatial hashing optimisation |
| `HASH_GRID_SIZE` | `128` | Spatial hash grid dimensions |

### CUDA Kernel Parameters

#### Core Physics Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `restLength` | float | 50.0 | Natural spring rest length for force calculations |
| `repulsionCutoff` | float | 50.0 | Maximum distance for repulsion force calculations |
| `repulsionSofteningEpsilon` | float | 0.0001 | Prevents division by zero in force calculations |
| `centerGravityK` | float | 0.0 | Gravity strength towards centre (0 = disabled) |
| `gridCellSize` | float | 50.0 | Spatial grid resolution for neighbour searches |

#### Warmup Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmupIterations` | integer | 100 | Number of warmup simulation steps |
| `coolingRate` | float | 0.001 | Rate of cooling during warmup phase |

#### Boundary Behaviour Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boundaryExtremeMultiplier` | float | 2.0 | Multiplier for extreme boundary detection |
| `boundaryExtremeForceMultiplier` | float | 10.0 | Force multiplier for extreme positions |
| `boundaryVelocityDamping` | float | 0.5 | Velocity reduction on boundary collision |

### CUDA Configuration File

Server administrators can override CUDA defaults in `/ext/data/dev_config.toml`:

```toml
[cuda]
rest_length = 50.0
repulsion_cutoff = 50.0
repulsion_softening_epsilon = 0.0001
center_gravity_k = 0.0
grid_cell_size = 50.0
warmup_iterations = 100
cooling_rate = 0.001
```

### GPU Kernel Configuration (YAML)

```yaml
gpu:
  compute_capability: 7.0
  max_threads_per_block: 1024
  shared_memory_size: 49152

  kernels:
    force_directed:
      block_size: 256
      gravity: 0.1
      repulsion: 50.0
      spring_constant: 0.01
      damping: 0.95

    sssp:
      block_size: 512
      max_iterations: 1000

    collision_detection:
      block_size: 128
      cell_size: 100.0
```

### CUDA Parameter API

#### GET /api/settings
Returns current settings including CUDA parameters in camelCase format.

#### POST /api/settings
Updates settings with new CUDA parameter values.

Example request:
```json
{
  "visualisation": {
    "graphs": {
      "logseq": {
        "physics": {
          "restLength": 60.0,
          "repulsionCutoff": 75.0,
          "gridCellSize": 40.0,
          "coolingRate": 0.002
        }
      }
    }
  }
}
```

### Performance Considerations

- **gridCellSize**: Smaller values increase precision but require more computation
- **repulsionCutoff**: Larger values include more nodes in force calculations
- **warmupIterations**: Higher values provide better initial layout at the cost of startup time
- **coolingRate**: Lower values provide smoother transitions but slower convergence

## Multi-Agent Configuration

### Agent System Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_CONTAINER_URL` | `http://localhost:8080` | Multi-agent container URL |
| `MCP_SERVER_PORT` | `3001` | MCP server port |
| `MAX_AGENTS` | `50` | Maximum concurrent agents |
| `AGENT_TIMEOUT` | `300000` | Agent task timeout (ms) |
| `ENABLE_AGENT_TELEMETRY` | `true` | Enable agent performance monitoring |
| `TELEMETRY_INTERVAL` | `5000` | Telemetry reporting interval (ms) |

### MCP Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TCP_PORT` | `9500` | Main MCP TCP server port |
| `MCP_TCP_AUTOSTART` | `true` | Auto-start MCP TCP server |
| `CLAUDE_FLOW_TCP_PORT` | `9502` | Isolated session proxy port |
| `CLAUDE_FLOW_MAX_SESSIONS` | `10` | Maximum isolated sessions |

### Actor System Configuration (TOML)

```toml
[actor_system]
name = "visionflow"
threads = 8
message_queue_size = 10000
max_actors = 10000

[actor_system.timeouts]
message = 5000  # ms
spawn = 1000    # ms
shutdown = 30000 # ms

[actor_system.persistence]
enabled = true
backend = "postgresql"
checkpoint_interval = 60 # seconds
```

### Agent Types Configuration (JSON)

```json
{
  "agents": {
    "types": ["planner", "coder", "researcher", "reviewer", "tester"],
    "coordination": {
      "mode": "hierarchical",
      "consensus_threshold": 0.7
    }
  }
}
```

## Network & Port Configuration

### Application Ports

| Port | Service | Protocol | Access | Purpose |
|------|---------|----------|--------|---------|
| **3000** | Claude Flow UI | HTTP | Public | Web interface for Claude Flow |
| **3001** | WebSocket Server | WebSocket | Public | Real-time communication |
| **3002** | MCP WebSocket Bridge | WebSocket | Public | WebSocket-to-stdio MCP bridge |

### MCP Server Ports

| Port | Service | Protocol | Access | Purpose | Status |
|------|---------|----------|--------|---------|--------|
| **9500** | MCP TCP Server | TCP/MCP | Public | **PRIMARY** - Shared claude-flow MCP server | ✅ **CRITICAL** |
| **9502** | Claude-Flow TCP Proxy | TCP/MCP | Public | Isolated claude-flow sessions (one per client) | ✅ Optional |
| **9503** | CF-TCP Health Check | HTTP | Localhost | Health endpoint for port 9502 | ✅ Optional |

### GUI Container Ports (gui-tools-service)

| Port | Service | Protocol | Access | Purpose |
|------|---------|----------|--------|---------|
| **5901** | VNC Server | VNC | Public | Remote desktop access to GUI tools |
| **9876** | Blender MCP | TCP/MCP | Internal | Blender 3D modelling bridge |
| **9877** | QGIS MCP | TCP/MCP | Internal | QGIS geospatial analysis bridge |
| **9878** | PBR Generator | TCP/MCP | Internal | PBR texture generation service |
| **9879** | Playwright MCP | TCP/MCP | Internal | Browser automation service |

### Monitoring & Metrics Ports

| Port | Service | Protocol | Access | Purpose |
|------|---------|----------|--------|---------|
| **9090** | Prometheus Metrics | HTTP | Internal | Metrics endpoint |

### WebSocket Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_PORT` | `3001` | WebSocket server port |
| `WS_PATH` | `/ws` | WebSocket endpoint path |
| `WS_PING_INTERVAL` | `30000` | Ping interval (ms) |
| `WS_MAX_PAYLOAD` | `104857600` | Max message size (100MB) |
| `ENABLE_COMPRESSION` | `true` | Enable WebSocket compression |
| `BINARY_PROTOCOL` | `true` | Use binary protocol |

### WebSocket Optimisation

```json
{
  "websocket": {
    "perMessageDeflate": {
      "zlibDeflateOptions": {
        "level": 1
      },
      "threshold": 1024
    },
    "maxBackpressure": 1048576,
    "compression": 1,
    "maxPayloadLength": 104857600
  }
}
```

### Port 9500 - MCP TCP Server (PRIMARY)

**Status**: ✅ **CRITICAL - ALWAYS REQUIRED**

**Service**: `mcp-tcp-server` (managed by supervisord)
**Script**: `/app/core-assets/scripts/mcp-tcp-server.js`
**Listen Address**: `0.0.0.0:9500`

**Purpose**:
- Main MCP TCP server for external system integration
- Provides JSON-RPC 2.0 over TCP for MCP protocol
- Spawns and manages shared claude-flow instance
- All external clients connect through this port

**Usage**:
```bash
# Test connection
echo '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}' | nc localhost 9500
```

### Port 9502 - Claude-Flow TCP Proxy (Optional)

**Status**: ✅ Optional (for isolated sessions)

**Service**: `claude-flow-tcp` (managed by supervisord)
**Script**: `/app/core-assets/scripts/claude-flow-tcp-proxy.js`
**Listen Address**: `0.0.0.0:9502`

**Purpose**:
- Provides **isolated claude-flow processes** per TCP connection
- Each client gets their own separate claude-flow instance
- Prevents state sharing between different external projects
- Max concurrent sessions configurable (default: 10)

**Difference from Port 9500**:
- **9500**: All clients share **one** claude-flow instance (shared state)
- **9502**: Each client gets **their own** claude-flow instance (isolated state)

**When to use**:
- Multiple independent external projects
- Need session isolation and separate state
- Running tests that shouldn't interfere

**Usage**:
```bash
# Connect to get isolated session
nc localhost 9502
{"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}
```

### Port 9503 - Claude-Flow TCP Health (Optional)

**Status**: ✅ Optional (monitors 9502)

**Service**: Built into `claude-flow-tcp-proxy.js`
**Listen Address**: `127.0.0.1:9503` (localhost only)

**Purpose**:
- HTTP health check endpoint for port 9502
- Returns JSON with active sessions and capacity
- Automatically serves on `CLAUDE_FLOW_TCP_PORT + 1`

**Response Format**:
```json
{
  "active_sessions": 2,
  "max_sessions": 10,
  "port": 9502
}
```

**Usage**:
```bash
# Check health
curl http://localhost:9503
```

### Essential vs Optional Ports

**Essential Ports (Cannot Disable)**:
- **9500** - MCP TCP Server

**Optional Ports (Can Disable If Not Needed)**:
- **3000** - Claude Flow UI (if not using web interface)
- **3002** - WebSocket Bridge (if only using TCP)
- **9502** - Isolated sessions (if shared state is acceptable)
- **9503** - Health check (if not monitoring 9502)

**External Dependencies (GUI Container)**:
- **5901** - VNC (for visual access to GUI tools)
- **9876-9879** - GUI tool bridges (Blender, QGIS, PBR, Playwright)

## Security Configuration

### Authentication & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | - | JWT signing secret (required in production) |
| `JWT_EXPIRY` | `7d` | JWT token expiry |
| `REFRESH_TOKEN_EXPIRY` | `30d` | Refresh token expiry |
| `BCRYPT_ROUNDS` | `10` | Password hashing rounds |
| `ENABLE_MFA` | `false` | Enable multi-factor authentication |
| `SESSION_SECRET` | - | Session encryption secret |
| `CORS_ORIGIN` | `*` | CORS allowed origins |

### TLS/SSL Configuration

```yaml
tls:
  enabled: true
  cert: /etc/ssl/certs/server.crt
  key: /etc/ssl/private/server.key
  ca: /etc/ssl/certs/ca-bundle.crt
  minVersion: "TLSv1.2"
  ciphers:
    - "ECDHE-RSA-AES128-GCM-SHA256"
    - "ECDHE-RSA-AES256-GCM-SHA384"
```

### Authentication Configuration

```json
{
  "auth": {
    "providers": {
      "local": {
        "enabled": true,
        "passwordPolicy": {
          "minLength": 12,
          "requireUppercase": true,
          "requireLowercase": true,
          "requireNumbers": true,
          "requireSymbols": true
        }
      },
      "oauth": {
        "github": {
          "enabled": true,
          "clientId": "${GITHUB_CLIENT_ID}",
          "clientSecret": "${GITHUB_CLIENT_SECRET}",
          "callbackURL": "/auth/github/callback"
        }
      }
    },
    "session": {
      "duration": "7d",
      "sliding": true,
      "httpOnly": true,
      "secure": true,
      "sameSite": "strict"
    }
  }
}
```

### Rate Limiting & Security Headers

```json
{
  "security": {
    "helmet": {
      "enabled": true
    },
    "rateLimit": {
      "windowMs": 60000,
      "max": 100
    },
    "forceHTTPS": true,
    "hsts": {
      "maxAge": 31536000,
      "includeSubDomains": true,
      "preload": true
    }
  }
}
```

## Performance Tuning

### Performance & Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `METRICS_PORT` | `9090` | Metrics endpoint port |
| `ENABLE_TRACING` | `false` | Enable OpenTelemetry tracing |
| `OTLP_ENDPOINT` | - | OTLP collector endpoint |
| `CACHE_TTL` | `3600` | Default cache TTL (seconds) |
| `RATE_LIMIT_WINDOW` | `60000` | Rate limit window (ms) |
| `RATE_LIMIT_MAX` | `100` | Max requests per window |

### Node.js Optimisation

```bash
# Recommended Node.js flags for production
NODE_OPTIONS="--max-old-space-size=4096 --optimise-for-size"

# V8 optimisation flags
--turbo-inline-threshold=1000
--max-inlined-bytecode-size=2000
```

## Configuration Files

### Main Configuration (`config/default.json`)

```json
{
  "app": {
    "name": "VisionFlow",
    "version": "2.0.0",
    "environment": "development"
  },
  "server": {
    "port": 3000,
    "host": "0.0.0.0",
    "cors": {
      "enabled": true,
      "credentials": true
    }
  },
  "database": {
    "client": "postgresql",
    "pool": {
      "min": 2,
      "max": 20
    },
    "migrations": {
      "directory": "./migrations"
    }
  },
  "redis": {
    "retry_strategy": {
      "times": 5,
      "interval": 1000
    }
  },
  "gpu": {
    "enabled": true,
    "fallbackToCPU": true,
    "kernels": {
      "force": "kernels/force.cu",
      "sssp": "kernels/sssp.cu",
      "collision": "kernels/collision.cu"
    }
  },
  "agents": {
    "types": ["planner", "coder", "researcher", "reviewer", "tester"],
    "coordination": {
      "mode": "hierarchical",
      "consensus_threshold": 0.7
    }
  },
  "security": {
    "helmet": {
      "enabled": true
    },
    "rateLimit": {
      "windowMs": 60000,
      "max": 100
    }
  }
}
```

### Production Configuration (`config/production.json`)

```json
{
  "app": {
    "environment": "production"
  },
  "server": {
    "cors": {
      "origin": ["https://your-domain.com"]
    }
  },
  "database": {
    "pool": {
      "min": 10,
      "max": 50
    }
  },
  "security": {
    "forceHTTPS": true,
    "hsts": {
      "maxAge": 31536000,
      "includeSubDomains": true,
      "preload": true
    }
  },
  "logging": {
    "level": "warn",
    "format": "json",
    "destination": "/var/log/visionflow"
  }
}
```

### Docker Compose Environment

```yaml
# docker-compose.yml environment section
services:
  app:
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/visionflow
      - REDIS_URL=redis://redis:6379
      - ENABLE_GPU=true
      - JWT_SECRET=${JWT_SECRET}

  multi-agent:
    environment:
      - MCP_SERVER_PORT=3001
      - MAX_AGENTS=100
      - AGENT_TIMEOUT=600000
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

### Docker Profiles

```bash
# Development profile
COMPOSE_PROFILES=dev docker-compose up

# Production profile with monitoring
COMPOSE_PROFILES=prod,monitoring docker-compose up

# GPU-enabled profile
COMPOSE_PROFILES=gpu docker-compose up
```

### Environment File Template (.env)

```bash
# System
NODE_ENV=production
PORT=3000
LOG_LEVEL=info

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/visionflow
DB_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=

# Security (REQUIRED)
JWT_SECRET=your-secret-key-min-32-chars
SESSION_SECRET=your-session-secret

# GPU
ENABLE_GPU=true
CUDA_DEVICE=0
GPU_MEMORY_LIMIT=4096

# Multi-Agent
ANTHROPIC_API_KEY=sk-ant-...
MAX_AGENTS=50
MCP_TCP_PORT=9500

# Docker
DOCKER_MEMORY=16g
DOCKER_CPUS=4

# External Services (Optional)
GITHUB_TOKEN=
OPENAI_API_KEY=
RAGFLOW_URL=
```

## Configuration Precedence

Configuration is loaded in the following order (later sources override earlier ones):

1. Default configuration (`config/default.json`)
2. Environment-specific configuration (`config/{NODE_ENV}.json`)
3. Local configuration (`config/local.json`) - not in version control
4. Environment variables
5. Command-line arguments

## Configuration Validation

The system validates configuration on startup:

```typescript
// Configuration schema validation
const configSchema = {
  type: 'object',
  required: ['JWT_SECRET', 'DATABASE_URL'],
  properties: {
    JWT_SECRET: {
      type: 'string',
      minLength: 32
    },
    DATABASE_URL: {
      type: 'string',
      format: 'uri'
    },
    PORT: {
      type: 'number',
      minimum: 1,
      maximum: 65535
    }
  }
};
```

## Best Practices

1. **Security**: Never commit secrets to version control
   - Use `.env` files and add to `.gitignore`
   - Use environment variables for sensitive data
   - Rotate secrets regularly

2. **Environment-specific**: Use different configurations for dev/staging/production
   - Maintain separate config files per environment
   - Use Docker profiles for different deployment scenarios
   - Test configuration changes in staging first

3. **Validation**: Always validate configuration on startup
   - Use schema validation for all config
   - Fail fast with clear error messages
   - Log configuration issues prominently

4. **Documentation**: Document all configuration options
   - Keep this reference up to date
   - Add inline comments in config files
   - Document default values and ranges

5. **Defaults**: Provide sensible defaults for optional settings
   - Production-safe defaults
   - Performance-optimised values
   - Clear indication of required vs optional

6. **Hot-reload**: Support configuration updates without restarts where possible
   - Use file watchers for config files
   - Implement graceful reload mechanisms
   - Log configuration changes

## Troubleshooting

Common configuration issues:

- **Missing required variables**: Check that all required environment variables are set
  - Review `.env` file against template
  - Check `JWT_SECRET` and `DATABASE_URL` are present
  - Verify API keys for external services

- **Connection failures**: Verify service URLs and credentials
  - Test database connectivity with `psql`
  - Verify Redis with `redis-cli ping`
  - Check MCP ports are accessible

- **Performance issues**: Review performance tuning settings
  - Adjust pool sizes for database connections
  - Tune CUDA parameters for GPU workload
  - Monitor memory usage and adjust limits

- **Security warnings**: Ensure production security settings are enabled
  - Enable HTTPS/TLS in production
  - Set appropriate CORS origins
  - Use strong JWT secrets (32+ characters)

- **Port conflicts**: Verify no services conflict on the same port
  - Check all port mappings in `docker-compose.yml`
  - Use `netstat` or `lsof` to identify port usage
  - Adjust port numbers if conflicts exist

## Related Documentation

- [Installation Guide](../getting-started/01-installation.md) - Initial setup and installation
- [Deployment Guide](../guides/01-deployment.md) - Production deployment procedures
- [Security Guide](../guides/security.md) - Security best practices
- [Troubleshooting Guide](../guides/06-troubleshooting.md) - Common issues and solutions
- [Development Workflow](../guides/02-development-workflow.md) - Local development configuration

## Case Conversion Notes

The API automatically handles conversion between:
- **Server-side**: snake_case (Rust convention)
- **Client-side**: camelCase (JavaScript/TypeScript convention)

When configuring CUDA parameters via the API, use camelCase in JSON requests. The system will convert to snake_case internally for Rust components.

## Logging

All services log to stdout/stderr for Docker logs visibility:

```bash
# View all logs
docker logs multi-agent-container

# Follow logs
docker logs -f multi-agent-container

# View last 50 lines
docker logs --tail 50 multi-agent-container

# View supervisor service logs
docker exec multi-agent-container supervisorctl status
```
