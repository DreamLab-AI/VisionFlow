---
title: Configuration Reference
description: Complete reference for all VisionFlow configuration options across environment variables, YAML files, and runtime settings
category: reference
tags:
  - rest
  - websocket
  - docker
  - rust
updated-date: 2025-12-18
difficulty-level: advanced
version: 2.0
---


# Configuration Reference

**Version**: 2.0
**Last Updated**: December 18, 2025

This comprehensive reference documents all configuration options for VisionFlow including environment variables, YAML settings, and runtime configuration.

---

## Table of Contents

1. [Configuration Files](#configuration-files)
2. [Environment Variables](#environment-variables)
3. [YAML Configuration](#yaml-configuration)
4. [Runtime Settings](#runtime-settings)
5. [Feature Flags](#feature-flags)
6. [Performance Tuning](#performance-tuning)
7. [Security Configuration](#security-configuration)

---

## Configuration Files

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| `.env` | Project root | Environment variables |
| `data/settings.yaml` | Data directory | Application settings |
| `docker-compose.yml` | Project root | Container configuration |
| `config/database.yaml` | Config directory | Database settings |
| `config/security.yaml` | Config directory | Security policies |

### Configuration Precedence

**Priority Order** (highest to lowest):
1. Runtime API calls (`POST /api/config`)
2. Environment variables (`.env`)
3. YAML configuration files (`data/settings.yaml`)
4. Default values (hardcoded)

---

## Environment Variables

### Core System Variables

#### Application Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | string | `development` | Environment mode: `development`, `staging`, `production` |
| `DEBUG_MODE` | boolean | `false` | Enable debug logging |
| `RUST_LOG` | string | `info` | Rust log level: `trace`, `debug`, `info`, `warn`, `error` |
| `HOST_PORT` | integer | `3001` | HTTP server port |
| `API_BASE_URL` | string | `http://localhost:9090` | API base URL |

**Example**:
```bash
ENVIRONMENT=production
DEBUG_MODE=false
RUST_LOG=warn
HOST_PORT=3030
```

#### Network Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MCP_TCP_PORT` | integer | `9500` | MCP protocol TCP port |
| `WS_PORT` | integer | `9090` | WebSocket server port |
| `METRICS_PORT` | integer | `9090` | Prometheus metrics port |
| `MAX_CONCURRENT_REQUESTS` | integer | `5000` | Maximum concurrent HTTP requests |
| `WS_CONNECTION_LIMIT` | integer | `1000` | Maximum WebSocket connections |

**Example**:
```bash
MCP_TCP_PORT=9500
WS_PORT=9090
MAX_CONCURRENT_REQUESTS=10000
WS_CONNECTION_LIMIT=2000
```

### Authentication & Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JWT_SECRET` | string | *required* | JWT signing secret (256-bit recommended) |
| `AUTH_PROVIDER` | string | `nostr` | Authentication provider: `jwt`, `nostr`, `oauth` |
| `AUTH_REQUIRED` | boolean | `true` | Require authentication for API access |
| `SESSION_TIMEOUT` | integer | `86400` | Session timeout in seconds (24 hours) |
| `API_KEYS_ENABLED` | boolean | `true` | Enable API key authentication |

**Example**:
```bash
JWT_SECRET=$(openssl rand -hex 32)
AUTH_PROVIDER=nostr
AUTH_REQUIRED=true
SESSION_TIMEOUT=86400
```

### Database Configuration

#### PostgreSQL

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `POSTGRES_HOST` | string | `postgres` | PostgreSQL host |
| `POSTGRES_PORT` | integer | `5432` | PostgreSQL port |
| `POSTGRES_DB` | string | `visionflow` | Database name |
| `POSTGRES_USER` | string | `visionflow` | Database user |
| `POSTGRES_PASSWORD` | string | *required* | Database password |
| `POSTGRES_MAX_CONNECTIONS` | integer | `100` | Maximum connection pool size |
| `POSTGRES_SSL_MODE` | string | `prefer` | SSL mode: `disable`, `prefer`, `require` |

**Example**:
```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=visionflow
POSTGRES_USER=visionflow
POSTGRES_PASSWORD=$(openssl rand -hex 24)
POSTGRES_MAX_CONNECTIONS=200
```

#### Redis Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_HOST` | string | `redis` | Redis host |
| `REDIS_PORT` | integer | `6379` | Redis port |
| `REDIS_PASSWORD` | string | `""` | Redis password (empty = no auth) |
| `REDIS_MAX_MEMORY` | string | `512mb` | Maximum Redis memory |
| `REDIS_MAX_CONNECTIONS` | integer | `100` | Maximum connection pool size |

**Example**:
```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_MAX_MEMORY=2gb
REDIS_MAX_CONNECTIONS=200
```

### Resource Limits

#### Memory Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMORY_LIMIT` | string | `16g` | Container memory limit |
| `SHARED_MEMORY_SIZE` | string | `2g` | Shared memory size |
| `POSTGRES_SHARED_BUFFERS` | string | `1GB` | PostgreSQL shared buffers |
| `HEAP_SIZE` | string | `8g` | JVM heap size (if applicable) |

**Example**:
```bash
# For 32GB system
MEMORY_LIMIT=24g
SHARED_MEMORY_SIZE=4g
POSTGRES_SHARED_BUFFERS=2GB
```

#### CPU Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CPU_LIMIT` | float | `8.0` | Maximum CPU cores |
| `CPU_RESERVATION` | float | `4.0` | Reserved CPU cores |
| `WORKER_THREADS` | integer | `8` | Worker thread count |
| `MAX_AGENTS` | integer | `20` | Maximum concurrent agents |

**Example**:
```bash
# For 16-core system
CPU_LIMIT=16.0
CPU_RESERVATION=8.0
WORKER_THREADS=16
MAX_AGENTS=50
```

### GPU Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_GPU` | boolean | `false` | Enable GPU acceleration |
| `NVIDIA_VISIBLE_DEVICES` | string | `0` | GPU device IDs (comma-separated) |
| `CUDA_ARCH` | integer | `89` | CUDA architecture (86=RTX 30xx, 89=RTX 40xx) |
| `GPU_MEMORY_LIMIT` | string | `8g` | GPU memory limit |
| `NVIDIA_DRIVER_CAPABILITIES` | string | `compute,utility` | Driver capabilities |

**Example**:
```bash
ENABLE_GPU=true
NVIDIA_VISIBLE_DEVICES=0,1
CUDA_ARCH=89
GPU_MEMORY_LIMIT=16g
```

### Feature Flags

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_XR` | boolean | `false` | Enable XR/VR features |
| `ENABLE_VOICE` | boolean | `false` | Enable voice interaction |
| `ENABLE_NEURAL_ENHANCEMENT` | boolean | `false` | Enable neural acceleration |
| `ENABLE_WASM_ACCELERATION` | boolean | `false` | Enable WASM acceleration |
| `ENABLE_GITHUB_SYNC` | boolean | `true` | Enable GitHub synchronization |
| `ENABLE_METRICS` | boolean | `true` | Enable Prometheus metrics |

**Example**:
```bash
ENABLE_GPU=true
ENABLE_XR=true
ENABLE_NEURAL_ENHANCEMENT=true
ENABLE_METRICS=true
```

### AI Service Configuration

#### OpenAI

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | string | `""` | OpenAI API key |
| `DEFAULT_LLM_MODEL` | string | `gpt-4o` | Default model |
| `LLM_TEMPERATURE` | float | `0.7` | Model temperature |
| `LLM_MAX_TOKENS` | integer | `4096` | Maximum tokens per request |
| `LLM_TIMEOUT` | integer | `30` | Request timeout in seconds |

**Example**:
```bash
OPENAI_API_KEY=sk-proj-your-key
DEFAULT_LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=8192
```

#### Anthropic Claude

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ANTHROPIC_API_KEY` | string | `""` | Anthropic API key |
| `CLAUDE_MODEL` | string | `claude-3-5-sonnet-20241022` | Claude model version |
| `CLAUDE_MAX_TOKENS` | integer | `4096` | Maximum tokens |

#### Perplexity

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PERPLEXITY_API_KEY` | string | `""` | Perplexity API key |
| `PERPLEXITY_MODEL` | string | `llama-3.1-sonar-small-128k-online` | Model version |

### Voice Services

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VOICE_LANGUAGE` | string | `en-GB` | Voice language code |
| `STT_PROVIDER` | string | `whisper` | Speech-to-text provider |
| `TTS_PROVIDER` | string | `kokoro` | Text-to-speech provider |
| `KOKORO_DEFAULT_VOICE` | string | `af-heart` | Kokoro voice ID |
| `WHISPER_MODEL` | string | `base` | Whisper model size |

### Logging & Monitoring

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | string | `info` | Logging level |
| `LOG_FORMAT` | string | `json` | Log format: `json`, `plain` |
| `LOG_FILE` | string | `/app/logs/visionflow.log` | Log file path |
| `LOG_ROTATION` | string | `daily` | Rotation policy |
| `LOG_MAX_SIZE` | string | `100MB` | Maximum log file size |
| `TRACK_PERFORMANCE` | boolean | `true` | Enable performance tracking |

---

## YAML Configuration

### Main Settings File (`data/settings.yaml`)

#### System Configuration

```yaml
system:
  # Agent Management
  agent-management:
    max-concurrent-agents: 50
    agent-spawn-timeout: 60
    agent-heartbeat-interval: 30
    enable-agent-persistence: true

  # Performance Settings
  performance:
    enable-load-balancing: true
    task-distribution-strategy: "adaptive"  # adaptive, round-robin, random
    resource-monitoring: true
    auto-scaling:
      enabled: true
      min-agents: 5
      max-agents: 50
      scale-up-threshold: 0.8
      scale-down-threshold: 0.2

  # Monitoring
  monitoring:
    enable-metrics: true
    metrics-port: 9090
    metrics-interval: 15
    health-check-interval: 30
    health-check-timeout: 10
    health-check-retries: 3

  # Disaster Recovery
  disaster-recovery:
    enable-automatic-backup: true
    backup-interval: 86400
    backup-retention: 2592000
    enable-auto-recovery: true
    max-recovery-attempts: 3
```

#### Visualization Settings

```yaml
visualization:
  graphs:
    # Default graph settings
    default:
      physics:
        enabled: true
        spring-strength: 0.005
        repulsion-strength: 50.0
        damping: 0.9
        max-velocity: 1.0
        iterations: 200
        enable-gpu: true

      nodes:
        base-color: '#3498db'
        node-size: 1.5
        enable-hologram: true
        hologram-opacity: 0.3
        label-font-size: 1.0

      edges:
        default-color: '#7f8c8d'
        width: 0.1
        opacity: 0.6
        animate-on-hover: true

    # Ontology-specific settings
    ontology:
      physics:
        spring-strength: 0.003
        repulsion-strength: 80.0

      nodes:
        class-color: '#e74c3c'
        individual-color: '#2ecc71'
        property-color: '#f39c12'
```

#### XR Configuration

```yaml
xr:
  enabled: true
  client-side-enable-xr: true
  mode: "immersive-vr"  # immersive-vr, immersive-ar
  space-type: "local-floor"  # local, local-floor, unbounded
  quality: high  # low, medium, high, ultra
  render-scale: 1.2

  # Hand Tracking
  enable-hand-tracking: true
  hand-mesh-enabled: true
  gesture-smoothing: 0.8

  # Comfort Settings
  locomotion-method: teleport  # teleport, smooth, snap
  enable-passthrough-portal: true
  passthrough-opacity: 0.8
  comfort-mode: true
  reduce-motion: false
```

#### Authentication Settings

```yaml
auth:
  enabled: true
  provider: nostr  # jwt, nostr, oauth
  required: true
  session-duration: 86400

  # Nostr Configuration
  nostr:
    relay-urls:
      - "wss://relay.damus.io"
      - "wss://nos.lol"
      - "wss://relay.snort.social"
    event-kinds: [1, 30023]
    max-event-size: 65536

  # JWT Configuration
  jwt:
    algorithm: "HS256"
    issuer: "visionflow"
    audience: "visionflow-api"
```

#### AI Services

```yaml
# OpenAI Configuration
openai:
  model: "gpt-4o"
  max-tokens: 4096
  temperature: 0.7
  timeout: 30
  rate-limit: 1000

# Perplexity Configuration
perplexity:
  model: "llama-3.1-sonar-small-128k-online"
  max-tokens: 4096
  temperature: 0.5
  timeout: 30
  rate-limit: 100

# RAGFlow Configuration
ragflow:
  agent-id: "your-agent-id"
  timeout: 30
  max-retries: 3
  chunk-size: 512
  max-chunks: 100
```

---

## Runtime Settings

### API Configuration Endpoint

Update configuration at runtime:

```http
POST /api/config/update
Content-Type: application/json
Authorization: Bearer {token}

{
  "physics": {
    "enabled": true,
    "gpuAcceleration": true
  },
  "rendering": {
    "quality": "high",
    "shadows": true
  }
}
```

### Dynamic Feature Toggles

Enable/disable features without restart:

```http
POST /api/features/toggle
Content-Type: application/json

{
  "feature": "gpu-acceleration",
  "enabled": true
}
```

---

## Feature Flags

### System Features

| Flag | Default | Description |
|------|---------|-------------|
| `gpu-acceleration` | `false` | GPU-accelerated physics |
| `neural-enhancement` | `false` | Neural network optimization |
| `wasm-acceleration` | `false` | WebAssembly SIMD |
| `ontology-validation` | `true` | OWL ontology validation |
| `github-sync` | `true` | GitHub repository sync |

### Experimental Features

| Flag | Default | Description |
|------|---------|-------------|
| `delta-encoding` | `false` | WebSocket delta encoding (V4) |
| `quantum-resistant-auth` | `false` | Post-quantum cryptography |
| `distributed-reasoning` | `false` | Multi-node reasoning |

---

## Performance Tuning

### Recommended Configurations

#### Small Deployment (< 10K nodes)

```yaml
# data/settings.yaml
system:
  agent-management:
    max-concurrent-agents: 10

visualization:
  graphs:
    default:
      physics:
        iterations: 100
        enable-gpu: false
```

```bash
# .env
MEMORY_LIMIT=8g
CPU_LIMIT=4.0
WORKER_THREADS=4
ENABLE_GPU=false
```

#### Medium Deployment (10K-100K nodes)

```yaml
system:
  agent-management:
    max-concurrent-agents: 30

visualization:
  graphs:
    default:
      physics:
        iterations: 200
        enable-gpu: true
```

```bash
MEMORY_LIMIT=16g
CPU_LIMIT=8.0
WORKER_THREADS=8
ENABLE_GPU=true
GPU_MEMORY_LIMIT=8g
```

#### Large Deployment (100K+ nodes)

```yaml
system:
  agent-management:
    max-concurrent-agents: 50

visualization:
  graphs:
    default:
      physics:
        iterations: 300
        enable-gpu: true
```

```bash
MEMORY_LIMIT=32g
CPU_LIMIT=16.0
WORKER_THREADS=16
ENABLE_GPU=true
GPU_MEMORY_LIMIT=16g
WS_CONNECTION_LIMIT=2000
```

---

## Security Configuration

### Production Security Checklist

```bash
# Strong authentication
AUTH_REQUIRED=true
JWT_SECRET=$(openssl rand -hex 32)

# HTTPS enforcement
HSTS_MAX_AGE=31536000
FORCE_HTTPS=true

# CORS configuration
CORS_ORIGINS="https://yourdomain.com"
CORS_CREDENTIALS=true

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX=1000
RATE_LIMIT_WINDOW=900

# Security headers
X_FRAME_OPTIONS=DENY
CONTENT_SECURITY_POLICY="default-src 'self'"

# Database security
POSTGRES_SSL_MODE=require
REDIS_PASSWORD=$(openssl rand -hex 24)

# Audit logging
ENABLE_AUDIT_LOGGING=true
LOG_SENSITIVE_DATA=false
```

---

## Configuration Validation

### Validation Script

```bash
#!/bin/bash
# scripts/validate-config.sh

# Check required variables
required_vars=(
  "JWT_SECRET"
  "POSTGRES_PASSWORD"
  "OPENAI_API_KEY"
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set"
    exit 1
  fi
done

# Validate ports
if [ "$HOST_PORT" -lt 1024 ] || [ "$HOST_PORT" -gt 65535 ]; then
  echo "Error: Invalid HOST_PORT"
  exit 1
fi

echo "Configuration valid"
```

### YAML Schema Validation

```bash
# Validate YAML syntax
yamllint data/settings.yaml

# Validate against JSON schema
ajv validate -s config/schema.json -d data/settings.yaml
```

---

---

---

## Related Documentation

- [Complete API Reference](API_REFERENCE.md)
- [Error Reference and Troubleshooting](ERROR_REFERENCE.md)
- [WebSocket Binary Protocol Reference](websocket-protocol.md)
- [WebSocket Binary Protocol - Complete System Documentation](../diagrams/infrastructure/websocket/binary-protocol-complete.md)
- [VisionFlow Binary WebSocket Protocol](protocols/binary-websocket.md)

## Cross-Reference Index

### Related Documentation

| Topic | Documentation | Link |
|-------|---------------|------|
| Environment Setup | Configuration Guide | [configuration.md](../guides/configuration.md) |
| Deployment | Deployment Guide | [deployment.md](../guides/deployment.md) |
| Performance | Performance Benchmarks | [performance-benchmarks.md](./performance-benchmarks.md) |
| Security | Security Guide | [security.md](../guides/security.md) |

---

**Configuration Reference Version**: 2.0
**VisionFlow Version**: v0.1.0
**Maintainer**: VisionFlow Configuration Team
**Last Updated**: December 18, 2025
