# Configuration Reference

This comprehensive reference covers all configuration options for the AR-AI Knowledge Graph system, including environment variables, configuration files, and runtime settings.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Service Configuration](#service-configuration)
- [Docker Configuration](#docker-configuration)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)

## Environment Variables

### Core System Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ENV` | `development` | Runtime environment (`development`, `staging`, `production`) |
| `PORT` | `3000` | Main application port |
| `HOST` | `0.0.0.0` | Host binding address |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warn`, `error`) |
| `ENABLE_CLUSTERING` | `false` | Enable Node.js cluster mode |
| `WORKERS` | CPU cores | Number of worker processes |

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `visionflow` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | - | Database password |
| `DB_POOL_SIZE` | `20` | Connection pool size |
| `DB_POOL_TIMEOUT` | `10000` | Pool timeout in milliseconds |

### Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Redis password |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_KEY_PREFIX` | `vf:` | Key prefix for namespacing |

### GPU Compute Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_GPU` | `true` | Enable GPU acceleration |
| `CUDA_DEVICE` | `0` | CUDA device index |
| `GPU_MEMORY_LIMIT` | `4096` | GPU memory limit in MB |
| `FORCE_ITERATIONS` | `500` | Force-directed layout iterations |
| `ENABLE_SPATIAL_HASHING` | `true` | Enable spatial hashing optimisation |
| `HASH_GRID_SIZE` | `128` | Spatial hash grid dimensions |

### Multi-Agent Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_CONTAINER_URL` | `http://localhost:8080` | Multi-agent container URL |
| `MCP_SERVER_PORT` | `3001` | MCP server port |
| `MAX_AGENTS` | `50` | Maximum concurrent agents |
| `AGENT_TIMEOUT` | `300000` | Agent task timeout (ms) |
| `ENABLE_AGENT_TELEMETRY` | `true` | Enable agent performance monitoring |
| `TELEMETRY_INTERVAL` | `5000` | Telemetry reporting interval (ms) |

### Authentication & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | - | JWT signing secret (required) |
| `JWT_EXPIRY` | `7d` | JWT token expiry |
| `REFRESH_TOKEN_EXPIRY` | `30d` | Refresh token expiry |
| `BCRYPT_ROUNDS` | `10` | Password hashing rounds |
| `ENABLE_MFA` | `false` | Enable multi-factor authentication |
| `SESSION_SECRET` | - | Session encryption secret |
| `CORS_ORIGIN` | `*` | CORS allowed origins |

### External Services

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | - | GitHub API token |
| `GITHUB_WEBHOOK_SECRET` | - | GitHub webhook secret |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `RAGFLOW_URL` | - | RAGFlow service URL |
| `NOSTR_RELAY_URL` | - | Nostr relay URL |

### WebSocket Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_PORT` | `3001` | WebSocket server port |
| `WS_PATH` | `/ws` | WebSocket endpoint path |
| `WS_PING_INTERVAL` | `30000` | Ping interval (ms) |
| `WS_MAX_PAYLOAD` | `104857600` | Max message size (100MB) |
| `ENABLE_COMPRESSION` | `true` | Enable WebSocket compression |
| `BINARY_PROTOCOL` | `true` | Use binary protocol |

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

## Configuration Files

### Main Configuration (`config/default.json`)

```json
{
  "app": {
    "name": "AR-AI Knowledge Graph",
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

## Service Configuration

### Actor System Configuration

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

### GPU Kernel Configuration

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

## Docker Configuration

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

## Security Configuration

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

## Performance Tuning

### Node.js Optimization

```bash
# Recommended Node.js flags for production
NODE_OPTIONS="--max-old-space-size=4096 --optimise-for-size"

# V8 optimisation flags
--turbo-inline-threshold=1000
--max-inlined-bytecode-size=2000
```

### Database Optimization

```sql
-- PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
```

### Redis Optimization

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save ""
appendonly no
tcp-keepalive 60
timeout 300
```

### WebSocket Optimization

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
2. **Environment-specific**: Use different configurations for dev/staging/prod
3. **Validation**: Always validate configuration on startup
4. **Documentation**: Document all configuration options
5. **Defaults**: Provide sensible defaults for optional settings
6. **Hot-reload**: Support configuration updates without restarts where possible

## Troubleshooting

Common configuration issues:

- **Missing required variables**: Check that all required environment variables are set
- **Connection failures**: Verify service URLs and credentials
- **Performance issues**: Review performance tuning settings
- **Security warnings**: Ensure production security settings are enabled

For detailed troubleshooting, see the [Troubleshooting Guide](../guides/06-troubleshooting.md).