# Configuration Guide

*[Guides](index.md)*

This practical guide covers common configuration scenarios and use cases for VisionFlow. For comprehensive technical reference, see the [Configuration Reference](../reference/configuration.md).

## Quick Configuration Setup

### Basic Development Setup

Get VisionFlow running locally in development mode:

```bash
# 1. Copy environment template
cp .env_template .env

# 2. Set essential variables
cat >> .env << 'EOF'
# Core Settings
CLAUDE_FLOW_HOST=multi-agent-container
MCP_TCP_PORT=9500
HOST_PORT=3001

# Security (generate a strong secret!)
JWT_SECRET=your_very_secure_256_bit_secret_key_here_please_change_this

# Database
POSTGRES_PASSWORD=dev_password_change_in_production

# Development flags
DEBUG_MODE=true
RUST_LOG=debug
HOT_RELOAD=true
EOF

# 3. Start development environment
docker-compose --profile dev up
```

### Production Deployment

Configure for production deployment:

```bash
# 1. Generate secure secrets
JWT_SECRET=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 24)
CLOUDFLARE_TUNNEL_TOKEN="your_cloudflare_tunnel_token"

# 2. Set production environment
cat > .env << EOF
# Production Configuration
ENVIRONMENT=production
DEBUG_MODE=false
RUST_LOG=info

# Security
JWT_SECRET=$JWT_SECRET
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
CLOUDFLARE_TUNNEL_TOKEN=$CLOUDFLARE_TUNNEL_TOKEN

# Performance
MEMORY_LIMIT=16g
CPU_LIMIT=8.0
ENABLE_GPU=true
MAX_AGENTS=20

# Network
HOST_PORT=3001
DOMAIN=your-domain.com
EOF

# 3. Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile production up -d
```

## Common Configuration Scenarios

### GPU-Accelerated Setup

Enable GPU acceleration for better performance:

```bash
# Environment variables for GPU
ENABLE_GPU=true
NVIDIA_VISIBLE_DEVICES=0              # Use first GPU
CUDA_ARCH=89                          # RTX 40xx series
GPU_MEMORY_LIMIT=8g

# Start with GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Verify GPU is working:
```bash
# Check GPU status
docker exec -it visionflow_container nvidia-smi

# Check GPU utilisation
curl http://localhost:3001/api/analytics/gpu-metrics
```

### Multi-Agent System Optimisation

Configure for large-scale multi-agent deployments:

```bash
# High-performance agent configuration
MAX_AGENTS=50
AGENT_TIMEOUT=600
TASK_QUEUE_SIZE=10000
WORKER_THREADS=16

# Memory allocation
MEMORY_LIMIT=32g
SHARED_MEMORY_SIZE=4g

# Neural enhancement
ENABLE_NEURAL_ENHANCEMENT=true
ENABLE_WASM_ACCELERATION=true
NEURAL_BATCH_SIZE=128
```

Update agent settings via API (requires power user permission):
```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr <your_pubkey>" \
  -H "Content-Type: application/json" \
  -d '{
    "system": {
      "agentManagement": {
        "maxConcurrentAgents": 50,
        "agentSpawnTimeout": 60,
        "agentHeartbeatInterval": 30,
        "enableAgentPersistence": true
      },
      "performance": {
        "enableLoadBalancing": true,
        "taskDistributionStrategy": "adaptive",
        "resourceMonitoring": true
      }
    }
  }'
```

### XR/VR Configuration

Enable extended reality features for Quest 3 and other XR devices:

```bash
# XR Environment variables
ENABLE_XR=true
QUEST3_SUPPORT=true
HAND_TRACKING=true
XR_RENDER_SCALE=1.2
XR_REFRESH_RATE=90
```

Configure XR settings via API:
```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr <your_pubkey>" \
  -H "Content-Type: application/json" \
  -d '{
    "xr": {
      "enabled": true,
      "clientSideEnableXr": true,
      "mode": "immersive-vr",
      "spaceType": "local-floor",
      "quality": "high",
      "renderScale": 1.2,
      "enableHandTracking": true,
      "handMeshEnabled": true,
      "gestureSmoothing": 0.8,
      "locomotionMethod": "teleport",
      "enablePassthroughPortal": true,
      "passthroughOpacity": 0.8
    }
  }'
```

### Knowledge Graph Integration

Configure for Logseq and GitHub integration:

```bash
# GitHub integration
GITHUB_TOKEN=ghp_your_github_personal_access_token
GITHUB_SYNC_INTERVAL=300
ENABLE_GITHUB_WEBHOOKS=true

# Logseq configuration
LOGSEQ_GRAPH_PATH=/data/logseq
LOGSEQ_SYNC_MODE=auto
ENABLE_BLOCK_REFERENCES=true
ENABLE_PAGE_PROPERTIES=true
```

Set up graph-specific visualisation via API:
```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr <your_pubkey>" \
  -H "Content-Type: application/json" \
  -d '{
    "visualisation": {
      "graphs": {
        "logseq": {
          "physics": {
            "enabled": true,
            "springK": 5.0,
            "repelK": 50.0,
            "damping": 0.9,
            "maxVelocity": 50.0,
            "iterations": 200
          },
          "nodes": {
            "baseColor": "#a06522",
            "nodeSize": 1.8,
            "enableHologram": true
          },
          "labels": {
            "enableLabels": true,
            "showMetadata": true,
            "desktopFontSize": 1.2
          }
        }
      }
    }
  }'
```

## Performance Tuning

### Memory Optimisation

Configure memory usage for your available resources:

```bash
# For 16GB system
MEMORY_LIMIT=12g
SHARED_MEMORY_SIZE=2g
POSTGRES_SHARED_BUFFERS=1GB
REDIS_MAX_MEMORY=512mb

# For 32GB system
MEMORY_LIMIT=24g
SHARED_MEMORY_SIZE=4g
POSTGRES_SHARED_BUFFERS=2GB
REDIS_MAX_MEMORY=2gb

# For 64GB+ system
MEMORY_LIMIT=48g
SHARED_MEMORY_SIZE=8g
POSTGRES_SHARED_BUFFERS=4GB
REDIS_MAX_MEMORY=4gb
```

### CPU Configuration

Optimise CPU usage based on your hardware:

```bash
# For 4-core system
CPU_LIMIT=4.0
CPU_RESERVATION=2.0
WORKER_THREADS=4

# For 8-core system
CPU_LIMIT=8.0
CPU_RESERVATION=4.0
WORKER_THREADS=8

# For 16+ core system
CPU_LIMIT=16.0
CPU_RESERVATION=8.0
WORKER_THREADS=16
```

### Network Performance

Configure for high-throughput scenarios:

```bash
# High-performance networking
MAX_CONCURRENT_REQUESTS=5000
WS_CONNECTION_LIMIT=1000
RATE_LIMIT_MAX=10000

# WebSocket optimisation
WS_BINARY_CHUNK_SIZE=4096
WS_UPDATE_RATE=120
COMPRESSION_ENABLED=true
```

## Security Configuration

### Authentication Setup

Configure Nostr-based authentication:

```bash
# Nostr authentication
AUTH_PROVIDER=nostr
AUTH_REQUIRED=true
SESSION_TIMEOUT=86400
```

Update authentication settings via API:
```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr <your_pubkey>" \
  -H "Content-Type: application/json" \
  -d '{
    "auth": {
      "enabled": true,
      "provider": "nostr",
      "required": true
    }
  }'
```

Note: Nostr relay configuration is handled through environment variables for security.

### Security Hardening

Production security configuration:

```bash
# Security headers
HSTS_MAX_AGE=31536000
X_FRAME_OPTIONS=DENY
CONTENT_SECURITY_POLICY="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

# CORS configuration
CORS_ORIGINS="https://your-domain.com,https://www.your-domain.com"
CORS_CREDENTIALS=true

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX=1000
RATE_LIMIT_WINDOW=900

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/visionflow.crt
SSL_KEY_PATH=/etc/ssl/private/visionflow.key
```

### Database Security

Secure database configuration:

```bash
# PostgreSQL security
POSTGRES_SSL_MODE=require
POSTGRES_SSL_CERT=/certs/client.crt
POSTGRES_SSL_KEY=/certs/client.key
POSTGRES_CONNECTION_TIMEOUT=30
POSTGRES_STATEMENT_TIMEOUT=30000

# Redis security
REDIS_PASSWORD=your_secure_redis_password
REDIS_MAX_CONNECTIONS=100
REDIS_TIMEOUT=5
```

## AI Service Configuration

### Language Model Setup

Configure AI services for multi-agent capabilities:

```bash
# Primary AI services
OPENAI_API_KEY=sk-proj-your_openai_api_key
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key
PERPLEXITY_API_KEY=pplx-your_perplexity_api_key

# Model configuration
DEFAULT_LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
LLM_TIMEOUT=30
```

Configure model preferences via API:
```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr <your_pubkey>" \
  -H "Content-Type: application/json" \
  -d '{
    "openai": {
      "timeout": 30,
      "rateLimit": 1000
    },
    "perplexity": {
      "model": "llama-3.1-sonar-small-128k-online",
      "maxTokens": 4096,
      "temperature": 0.5,
      "timeout": 30,
      "rateLimit": 100
    },
    "ragflow": {
      "agentId": "your_ragflow_agent_id",
      "timeout": 30,
      "maxRetries": 3
    }
  }'
```

### Voice and Audio Services

Enable voice interaction capabilities:

```bash
# Voice services
ENABLE_VOICE=true
VOICE_LANGUAGE=en-GB
STT_PROVIDER=whisper
TTS_PROVIDER=kokoro

# Kokoro TTS
KOKORO_DEFAULT_VOICE=af_heart
KOKORO_DEFAULT_FORMAT=mp3
KOKORO_SAMPLE_RATE=24000

# Whisper STT
WHISPER_MODEL=base
WHISPER_LANGUAGE=en
WHISPER_TEMPERATURE=0.0
```

## Environment-Specific Configurations

### Development Environment

Optimised for local development:

```bash
# .env.development
ENVIRONMENT=development
DEBUG_MODE=true
RUST_LOG=debug
HOT_RELOAD=true

# Relaxed security for development
CORS_ALLOW_ALL=true
DISABLE_HTTPS_REDIRECT=true
AUTH_REQUIRED=false

# Development services
MOCK_SERVICES=true
ENABLE_PROFILING=true
LOG_LEVEL=debug
```

### Staging Environment

Pre-production testing configuration:

```bash
# .env.staging
ENVIRONMENT=staging
DEBUG_MODE=false
RUST_LOG=info

# Staging-specific settings
ENABLE_METRICS=true
ENABLE_PROFILING=true
LOG_LEVEL=info

# Moderate security
AUTH_REQUIRED=true
RATE_LIMIT_MAX=5000
```

### Production Environment

Production-ready configuration:

```bash
# .env.production
ENVIRONMENT=production
DEBUG_MODE=false
RUST_LOG=warn

# Production optimisation
ENABLE_GPU=true
MAX_AGENTS=50
MEMORY_LIMIT=32g
CPU_LIMIT=16.0

# Strict security
AUTH_REQUIRED=true
RATE_LIMIT_MAX=10000
ENABLE_AUDIT_LOGGING=true
LOG_SENSITIVE_DATA=false
```

## Monitoring and Observability

### Metrics Configuration

Enable comprehensive monitoring:

```bash
# Metrics and monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
PROMETHEUS_ENDPOINT=true

# Performance tracking
TRACK_PERFORMANCE=true
TRACK_USAGE=true
TRACK_ERRORS=true
TRACK_SECURITY_EVENTS=true
```

Configure monitoring via API:
```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr <your_pubkey>" \
  -H "Content-Type: application/json" \
  -d '{
    "system": {
      "network": {
        "enableMetrics": true,
        "metricsPort": 9090
      }
    }
  }'
```

### Logging Configuration

Configure structured logging:

```bash
# Logging configuration
LOG_LEVEL=info
LOG_FORMAT=json
LOG_FILE=/app/logs/visionflow.log
LOG_ROTATION=daily
LOG_MAX_SIZE=100MB
LOG_MAX_FILES=10

# Structured logging options
LOG_JSON_PRETTY=false
LOG_INCLUDE_LOCATION=true
LOG_INCLUDE_THREAD=true
```

## Backup and Recovery

### Configuration Backup

Set up automatic configuration backups:

```bash
# Backup configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"           # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_LOCATION=/opt/backups/visionflow
BACKUP_COMPRESSION=gzip

# What to backup
BACKUP_DATABASE=true
BACKUP_REDIS=true
BACKUP_USER_DATA=true
BACKUP_CONFIGURATION=true
```

### Disaster Recovery

Disaster recovery is configured through environment variables and system configuration:

```bash
# Backup configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
BACKUP_LOCATION=/opt/backups/visionflow

# Recovery settings
ENABLE_AUTO_RECOVERY=true
MAX_RECOVERY_ATTEMPTS=3
RECOVERY_TIMEOUT=300
```

## Troubleshooting Common Issues

### Port Conflicts

Resolve port conflicts:
```bash
# Check for port conflicts
sudo netstat -tulpn | grep :3001
sudo lsof -i :3001

# Change ports if needed
HOST_PORT=3002
MCP_TCP_PORT=9501
METRICS_PORT=9091
```

### Memory Issues

Fix memory-related problems:
```bash
# Check memory usage
free -h
docker system df

# Adjust memory limits
MEMORY_LIMIT=8g                    # Reduce if insufficient
SHARED_MEMORY_SIZE=1g
POSTGRES_SHARED_BUFFERS=512MB
REDIS_MAX_MEMORY=256mb
```

### GPU Configuration Issues

Resolve GPU problems:
```bash
# Disable GPU if unavailable
ENABLE_GPU=false

# Or fix GPU configuration
NVIDIA_VISIBLE_DEVICES=all
CUDA_ARCH=86                       # Adjust for your GPU
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Database Connection Issues

Fix database connectivity:
```bash
# Check database status
docker-compose exec postgres pg_isready

# Reset database connection
docker-compose restart postgres
docker-compose restart webxr

# Verify connection settings
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_CONNECTION_TIMEOUT=30
```

## Configuration Validation

### Automated Validation

Use validation scripts to verify configuration:

```bash
# Validate environment variables
./scripts/validate-env.sh

# Validate settings via API
curl -X POST http://localhost:4000/api/settings/validate \
  -H "Content-Type: application/json" \
  -d @proposed_settings.json

# Test Docker configuration
./scripts/test-docker-config.sh

# Full configuration test
./scripts/test-full-config.sh
```

### Manual Verification

Manually verify key components:

```bash
# Test API health
curl http://localhost:3001/api/health

# Test WebSocket connection
wscat -c ws://localhost:3001/ws

# Test MCP connection
telnet localhost 9500

# Test GPU (if enabled)
docker exec -it visionflow_container nvidia-smi

# Test database connection
docker-compose exec postgres psql -U visionflow -d visionflow -c "SELECT version();"
```

## Best Practices

### Configuration Management

1. **Use version control**: Keep configuration files in version control
2. **Environment separation**: Use different files for dev/staging/production
3. **Secret management**: Never commit secrets; use environment variables
4. **Documentation**: Document all custom configuration choices
5. **Validation**: Always validate configuration changes before deployment

### Performance Guidelines

1. **Start conservative**: Begin with lower resource limits and scale up
2. **Monitor resources**: Use monitoring to understand actual usage
3. **Profile regularly**: Enable profiling in staging to identify bottlenecks
4. **Optimise incrementally**: Make small changes and measure impact
5. **Plan for growth**: Configure with future scaling in mind

### Security Guidelines

1. **Principle of least privilege**: Only grant necessary permissions
2. **Regular updates**: Keep all components and dependencies updated
3. **Secure communications**: Use TLS/SSL for all external communications
4. **Audit trails**: Enable comprehensive logging and auditing
5. **Regular reviews**: Periodically review and update security configuration

---

This guide covers the most common configuration scenarios for VisionFlow. For advanced configuration options and complete technical reference, see the [Configuration Reference](../reference/configuration.md).

## Related Topics

- [Settings System Architecture](../settings-system.md) - SQLite-backed settings system
- [Settings API Reference](../settings-api.md) - Complete API documentation
- [Settings Validation Rules](../settings-validation.md) - All validation constraints
- [Settings Migration Guide](../settings-migration-guide.md) - Developer guide
- [Configuration Reference](../reference/configuration.md) - Comprehensive technical reference
- [Installation Guide](../getting-started/installation.md) - Initial setup instructions
- [Deployment Guide](../deployment/index.md) - Production deployment strategies
- [Troubleshooting](../troubleshooting/configuration.md) - Configuration problem solving
- [Security Guide](../security/index.md) - Security best practices